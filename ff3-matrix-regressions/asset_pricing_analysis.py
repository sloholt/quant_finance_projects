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

# =====================================
# SHARED UTILITY FUNCTIONS
# =====================================

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


def summary_stats(vec, rfvec=None, excess_returns=False):
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


def resid_covmat(E, num_factors):
    """Calculate residuals, demeaned residuals, and residual covariance matrix.
    E = R - XB                      Residuals (T x N)
    Ec = E - mean(E)                Demeaned residuals (T x N)
    Sigma = Ec' Ec / (T-1)          Residual covariance matrix (N x N)

    """
    # Dimensionality checks:
    E = as_2d(E)

    T = E.shape[0]

    emat = E - E.mean(axis=0)
    ecovmat = emat.T @ emat / (T - num_factors - 1)

    return {"resids": E, "residemat": emat, "residcovmat": ecovmat}


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


# --- Extraction Functions ---


def get_alphas(regression_results):
    """Return the CAPM alpha estimates as a row vector"""
    return regression_results["alpha"].reshape(1, -1)


def get_betas(regression_results):
    """Return factor betas as a K x N matrix."""
    return regression_results["beta"]


def get_gamma(R, betas):
    """Calculate cross-sectional risk premia for each time period."""
    R = as_2d(R)
    betas = as_2d(betas)
    X = add_constant(betas.T)
    gamma = np.linalg.lstsq(X, R.T, rcond=None)[0].T
    return gamma


# --- TS GRS F-TEST ---
"""Chat prompt: Where is the error in this function that is causing '0-dimensional
array give."""


def grs_test(T, N, num_factors, alphas, factors, residcovmat):
    """Calculate the Gibbons-Ross-Shanken (GRS) F-statistic to test CAPM or Fama-French."""
    factors = as_2d(factors)
    alphas = np.asarray(alphas).reshape(-1, 1)

    factor_means = factors.mean(axis=0).reshape(-1, 1)
    factor_covmat = np.atleast_2d(np.cov(factors, rowvar=False, ddof=0))

    first_term = ((T - N - num_factors)) / N * (T / (T - num_factors - 1))
    numerator = (alphas.T @ np.linalg.solve(residcovmat, alphas)).item()
    denominator = (
        1 + (factor_means.T @ np.linalg.solve(factor_covmat, factor_means)).item()
    )
    # print("Alpha shape:", np.asarray(alphas).shape)
    # print("Covariance shape:", np.asarray(residcovmat).shape)
    # print("Covariance type:", type(residcovmat))
    return first_term * (numerator / denominator)


def fgrs_hypothesis(fgrs, f_critical, model_name):
    if fgrs > f_critical:
        print(
            f"Reject H0: Alphas are jointly zero.\n" f"Evidence against {model_name}."
        )
    else:
        print(
            f"Fail to reject H0: Alphas are jointly zero.\n"
            f"No conclusive evidence against {model_name}."
        )


# --- Cross Sectional One Shot ---
def cross_sectional(gamma):
    gamma = as_2d(gamma)
    T = gamma.shape[0]
    avg_gamma = gamma.mean(axis=0)
    se_gamma = gamma.std(axis=0, ddof=1) / np.sqrt(T)
    t_stat = (avg_gamma - 0) / se_gamma
    return {"avg_gamma": avg_gamma, "se_gamma": se_gamma, "t_gamma": t_stat}


# =====================================
# CAPM & FAMA-FRENCH MODEL BUILDING
# =====================================


# --- Full Model Run ---
"""Chat prompt: add a section to run model that uses summary_stats to calculate the values for each of the
factors given as input"""


def run_model(data, portfolio, factors):
    """Run asset pricing analysis for selected factors"""
    F = data[factors].to_numpy(dtype=float)
    rf = data["RF"].to_numpy(dtype=float)
    rawrets = data[portfolio].to_numpy(dtype=float)

    rets = excess_returns(rawrets, rf)
    T, N = rawrets.shape
    num_factors = F.shape[1]

    # Factor Summary Statistics
    factor_stats = summary_stats(F, rfvec=rf)
    factor_summary_df = pd.DataFrame(
        {
            "Mean": factor_stats["mean"],
            "Variance": factor_stats["var"],
            "Std Dev": factor_stats["std"],
            "Sharpe Ratio": factor_stats["sr"],
        },
        index=factors,
    )
    # Portfolio Excess Return Summary Statistics
    portfolio_stats = summary_stats(rets)
    expected_excess_returns = portfolio_stats["mean"]

    portfolio_summary_df = pd.DataFrame(
        {
            "Mean": expected_excess_returns,
            "Variance": portfolio_stats["var"],
            "Std Dev": portfolio_stats["std"],
            "Sharpe Ratio": portfolio_stats["sr"],
        },
        index=portfolio,
    )

    results = time_series_regression(rets, F)
    alphas = get_alphas(results)
    residuals = resid_covmat(results["resids"], num_factors=num_factors)
    RESIDCOVMAT = residuals["residcovmat"]

    fgrs = grs_test(T, N, num_factors, alphas, F, RESIDCOVMAT)
    p_grs = f.sf(fgrs, N, T - N - num_factors)

    # Fama-MacBeth regressions
    betas = get_betas(results)
    gamma = get_gamma(rets, betas)
    fm_results = cross_sectional(gamma)

    return {
        "factor_summary": factor_summary_df,
        "portfolio_summary": portfolio_summary_df,
        "regression": results,
        "grs_stat": fgrs,
        "grs_pvalue": p_grs,
        "fama_macbeth": fm_results,
    }


# --- Main ---
"""Chat prompt: Add a section to main that organizes and displays the results for each model type"""


def main():
    """
    Run the analysis, GRS test, and Fama-Macbeth regressions for CAPM and Fama-French.
    Organize & export results
    """
    # Load & Prep data
    data = pd.read_excel(file_path, sheet_name="Data", header=0, nrows=360)
    # print("Column names:", data.columns.tolist())
    # print(data.head())
    portfolio_cols = [f"ME{i} BM{j}" for i in range(1, 6) for j in range(1, 6)]
    data.columns = data.columns.str.strip()

    # Run CAPM
    capm_results = run_model(data, portfolio_cols, ["RMRF"])

    # Run Fama-French 3-factor
    ff3_results = run_model(data, portfolio_cols, ["RMRF", "SMB", "HML"])

    # Organize and display results
    models = {
        "CAPM": {"results": capm_results, "factors": ["RMRF"]},
        "Fama-French 3-Factor": {
            "results": ff3_results,
            "factors": ["RMRF", "SMB", "HML"],
        },
    }

    output_tables = {}

    for model_name, model in models.items():
        results = model["results"]
        factors = model["factors"]
        regression = results["regression"]
        fm = results["fama_macbeth"]

        # Time-series regression results
        regression_df = pd.DataFrame(
            {"Alpha": regression["alpha"]}, index=portfolio_cols
        )

        for i, factor in enumerate(factors):
            regression_df[f"Beta_{factor}"] = regression["beta"][i]
            regression_df[f"t_Beta_{factor}"] = regression["t_beta"][i]

        regression_df["t_Alpha"] = regression["t_alpha"]

        # Fama-MacBeth cross-sectional results
        fm_df = pd.DataFrame(
            {
                "Avg Gamma": fm["avg_gamma"],
                "SE Gamma": fm["se_gamma"],
                "t Gamma": fm["t_gamma"],
            },
            index=["Intercept"] + factors,
        )

        # GRS test results
        grs_df = pd.DataFrame(
            {
                "GRS F-statistic": [results["grs_stat"]],
                "p-value": [results["grs_pvalue"]],
                "Reject H0 (5%)": [results["grs_pvalue"] < 0.05],
            },
            index=[model_name],
        )

        factor_summary_df = results["factor_summary"]

        output_tables[model_name] = {
            "Factor Summary": factor_summary_df,
            "Regression": regression_df,
            "GRS": grs_df,
            "Fama-MacBeth": fm_df,
        }

        # --- Display Results ---
        print(f"\n{'=' * 60}")
        print(f"{model_name.upper()} RESULTS")
        print(f"{'=' * 60}")

        # Factor summary statistics
        print("\nFactor Summary Statistics:")
        print(factor_summary_df.round(4).to_string())

        # Portfolio expected returns
        print("\nCAPM Portfolio Excess Return Summary:")
        print(capm_results["portfolio_summary"].round(6))

        print("\nTime-Series Regression:")
        print(regression_df.round(4).to_string())

        print("\nGRS Joint Alpha Test:")
        print(grs_df.round(4).to_string())

        print("\nFama-MacBeth Cross-Sectional Regression:")
        print(fm_df.round(4).to_string())

    # --- Compare Models ---
    comparison_df = pd.concat([output_tables[name]["GRS"] for name in output_tables])

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(comparison_df.round(4).to_string())


if __name__ == "__main__":
    main()

"""Chat prompt: Where is the error in this file that is giving ValueError: not enough values to unpack (expected 2, got 0)"""
# Said line 259 but there is no line 259
