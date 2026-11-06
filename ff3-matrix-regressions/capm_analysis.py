import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f, linregress

OUTPUT_DIR_NAME = "."
FILE_NAME = "data/asset_pricing_class_data.xlsx"

# current script location
script_directory = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(script_directory, OUTPUT_DIR_NAME)

file_path = os.path.join(output_dir, FILE_NAME)
# create directory if missing
os.makedirs(output_dir, exist_ok=True)

RISK_FACTORS = ["RMRF", "SMB", "HML"]

# --- Helper Functions ---


def as_2d(a):
    """Turns a 1D vector into a 2D array to simplify transposing later"""
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def add_constant(F):
    """Adds a column of 1s to the beginning of a matrix to include the intercept later
    X = [1, F]  (T x K+1)."""
    F = as_2d(F)
    return np.column_stack([np.ones(F.shape[0]), F])


def excess_returns(rawrets, rf):
    """ret = Rawret - rf*1
    Note: Use rf*1 because multiplying rf by 1' expands each month's risk-free rate across
    all asset columns to match the dimensions of rawret.
    """
    return as_2d(rawrets) - as_2d(rf)


def summary_stats(vec, rfvec=None):
    """Mean, sample variance, std dev, and Sharpe ratio of the inputted vector"""
    vec = as_2d(vec)
    mean = vec.mean(axis=0)
    var = vec.var(axis=0, ddof=1)
    sd = np.sqrt(var)
    rfrate = 0 if rfvec is None else np.asarray(rfvec).mean()

    return {
        "mean": mean,
        "var": var,
        "std": sd,
        "rf rate": rfrate,
        "sr": (mean - rfrate) / sd,
    }


# --- Time Series Regressions ----
def time_series_regression(R, F):
    """Run N time-series OLS regressions simultaneously.

    B    = (X'X)^-1 X'R           Coefficient matrix (K+1 x N)
    E    = R - XB                  Residual matrix (T x N)
    s2   = sum(e_i^2)/(T-K-1)     Residual variance for each asset
    SE   = sqrt(diag((X'X)^-1) outer s2)

    """
    # Ensure inputs are 2D arrays
    R, F = as_2d(R), as_2d(F)

    # Number of observations (T) and factors (K)
    T, K = F.shape

    # Add intercept column to factor matrix
    X = add_constant(F)

    # Estimate OLS coefficients
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    XtX_inv = np.linalg.inv(X.T @ X)

    E = R - X @ B

    # Estimate residual variance for each asset
    s2 = (E**2).sum(axis=0) / (T - K - 1)

    # Calculate standard errors and t-statistics
    SE = np.sqrt(np.outer(np.diag(XtX_inv), s2))
    tstat = B / SE
    # Return regression results as a dictionary
    return {
        "alpha": B[0],
        "beta": B[1:],
        "se_alpha": SE[0],
        "se_beta": SE[1:],
        "t_alpha": tstat[0],
        "t_beta": tstat[1:],
        "resids": E,
    }


# --- CAPM TS GRS F-TEST ---
def resid_covmat(E):
    """Calculate residuals, demeaned residuals, and residual covariance matrix.
    E = R - XB                      Residuals (T x N)
    Ec = E - mean(E)                Demeaned residuals (T x N)
    Sigma = Ec' Ec / (T-1)          Residual covariance matrix (N x N)

    """
    # Dimensionality checks:
    E = as_2d(E)

    T = E.shape[0]

    emat = E - E.mean(axis=0)
    ecovmat = emat.T @ emat / (T - 1)

    return {"resids": E, "residemat": emat, "residcovmat": ecovmat}


def get_capmalphas(regression_results):
    """Return the CAPM alpha estimates as a row vector"""
    return regression_results["alpha"].reshape(1, -1)


def grs_test(T, N, K, CAPMALPHAS, RMRF, residcovmat):
    """Calculate the Gibbons-Ross-Shanken (GRS) F-statistic to jointly test
    whether all CAPM regression alphas are zero.
    """
    first_term = (T - N - K) / N
    second_term_numerator = (
        CAPMALPHAS @ np.linalg.inv(residcovmat) @ CAPMALPHAS.T
    ).item()
    second_term_denom = 1 + (RMRF.mean() ** 2) / RMRF.var(ddof=1)
    fgrs = first_term * (second_term_numerator / second_term_denom)
    return fgrs


def fgrs_hypothesis(fgrs, f_critical):
    if fgrs > f_critical:
        print(
            "Reject the Null Hypothesis that the alphas are jointly zero. \n Evidence against CAPM."
        )
    else:
        print(
            "Fail to reject the Null Hypothesis that the alphas are jointly zero. \n No conclusive evidence."
        )


# --- CAPM Cross Sectional One Shot ---
def get_capmbetas(regression_results):
    """Return the CAPM betas as a row vector"""
    return regression_results["beta"].reshape(1, -1)


def get_gamma(R, CAPMBETAS):
    """Calculate the cross-sectional regression slope (gamme) for each time period."""
    R = as_2d(R)
    betas = np.asarray(CAPMBETAS).ravel()
    gamma = np.array([np.polyfit(betas, returns, 1)[0] for returns in R])
    return gamma


def capm_cross_sectional(T, gamma):
    avg_gamma = gamma.mean()
    se_gamma = gamma.std(ddof=1) / np.sqrt(T)
    t_stat = (avg_gamma - 0) / se_gamma
    return {"avg_gamma": avg_gamma, "se_gamma": se_gamma, "t_gamma": t_stat}


# --- Main ---
""" ChatGPT prompt: Given this implemented excel file and this implemented python file, write def main() to put it all together and organize the results: 
"""


def main():
    """
    Run the CAPM analysis, GRS test, and Fama-Macbeth regressions.
    Organize & export results
    """
    # Load & Prep data
    data = pd.read_excel(file_path, sheet_name="Data", header=0, nrows=360)
    print("Column names:", data.columns.tolist())
    print(data.head())
    portfolio_cols = [f"ME{i} BM{j}" for i in range(1, 6) for j in range(1, 6)]
    data.columns = data.columns.str.strip()

    F_all = data[RISK_FACTORS].to_numpy(dtype=float)
    RMRF = data["RMRF"].to_numpy(dtype=float)
    rf = data["RF"].to_numpy(dtype=float)
    rawrets = data[portfolio_cols].to_numpy(dtype=float)

    # raw returns -> excess returns
    R = excess_returns(rawrets, rf)
    F = as_2d(RMRF)
    T, N = R.shape
    K = F.shape[1]

    print(f"Observations (T): {T}")
    print(f"Portfolios (N): {N}")
    print(f"CAPM factors (K): {K}")

    # Descriptive stats
    portfolio_stats = summary_stats(rawrets, rf)
    summary_df = pd.DataFrame(
        {
            "Portfolio": portfolio_cols,
            "Mean": portfolio_stats["mean"],
            "Variance": portfolio_stats["var"],
            "Std Dev": portfolio_stats["std"],
            "Sharpe Ratio": portfolio_stats["sr"],
        }
    )

    # CAPM time-series regression
    ts_results = time_series_regression(R, F)

    CAPMALPHAS = get_capmalphas(ts_results)
    CAPMBETAS = get_capmbetas(ts_results)

    regression_df = pd.DataFrame(
        {
            "Portfolio": portfolio_cols,
            "Alpha": ts_results["alpha"],
            "Beta": ts_results["beta"][0],
            "SE Alpha": ts_results["se_alpha"],
            "SE Beta": ts_results["se_beta"][0],
            "t Alpha": ts_results["t_alpha"],
            "t Beta": ts_results["t_beta"][0],
        }
    )

    # GRS test
    residual_results = resid_covmat(ts_results["resids"])

    RESIDCOVMAT = residual_results["residcovmat"]
    fgrs = grs_test(T, N, K, CAPMALPHAS, RMRF, RESIDCOVMAT)
    f_critical = f.isf(0.05, N, T - N - K)
    # p_grs = f.sf(fgrs, N, T - N - K)
    fgrs_hypothesis(fgrs, f_critical)
    grs_df = pd.DataFrame(
        {
            "Metric": [
                "GRS F-statistic",
                "F Critical Value (5%)",
                "p-value",
            ],
            "Value": [fgrs, f_critical, fgrs_hypothesis(fgrs, f_critical)],
        }
    )

    # Fama-Macbest cross sectional regressions
    gamma = get_gamma(R, CAPMBETAS)
    cs_results = capm_cross_sectional(T, gamma)
    fm_df = pd.DataFrame(
        {
            "Metric": ["Average Gamma", "SE Gamma", "t Gamma"],
            "Value": [
                cs_results["avg_gamma"],
                cs_results["se_gamma"],
                cs_results["t_gamma"],
            ],
        }
    )

    # Output
    print("\n========== SUMMARY STATISTICS (Raw Returns) ==========")
    print(summary_df.round(4).to_string(index=False))

    print("\n========== CAPM TIME-SERIES REGRESSIONS ==========")
    print(regression_df.round(4).to_string(index=False))

    print("\n========== GRS F-TEST ==========")
    print(grs_df.to_string(index=False))
    fgrs_hypothesis(fgrs, f_critical)

    print("\n========== FAMA-MACBETH REGRESSION ==========")
    print(fm_df.to_string(index=False))


if __name__ == "__main__":
    main()
