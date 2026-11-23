import pandas as pd
import numpy as np
import validation as v
from scipy.stats import f
from pathlib import Path
from openpyxl import load_workbook

from reporting import (
    export_results_to_excel,
    plot_portfolio_returns,
    plot_sml,
    generate_summary,
    display_results,
    export_validation_table,
)
from data_loader import (
    load_data,
    PORTFOLIO_COLS,
    DATE_COL,
    RF_COL,
    load_excel_data,
    read_class_reference,
    read_part1_reference,
    load_part1_data,
)

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
    """ret = Rawret - rf*1"""
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
    Sigma = Ec' Ec / (T-K-1)          Residual covariance matrix (N x N)

    """
    # Dimensionality checks:
    E = as_2d(E)

    T = E.shape[0]

    emat = E - E.mean(axis=0)
    residcovmat = emat.T @ emat / (T - num_factors - 1)

    return {"resids": E, "residemat": emat, "residcovmat": residcovmat}


# --- Time Series Regressions ----
def time_series_regression(R, F):
    """N time-series OLS regressions simultaneously.

    B    = (X'X)^-1 X'R           Coefficient matrix (K+1 x N)
    E    = R - XB                  Residual matrix (T x N)
    s2   = sum(e_i^2)/(T-K-1)     Residual variance for each asset
    SE   = sqrt(diag((X'X)^-1) outer s2)

    """
    R, F = as_2d(R), as_2d(F)

    # Number of observations (T) and factors (K)
    T, K = F.shape
    X = add_constant(F)

    # OLS coefficients
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    v.check_matrix_condition(X.T @ X)
    XtX_inv = np.linalg.inv(X.T @ X)

    E = R - X @ B

    # Residual variance for each asset
    s2 = (E**2).sum(axis=0) / (T - K - 1)

    # Standard errors and t-statistics
    SE = np.sqrt(np.outer(np.diag(XtX_inv), s2))
    tstat = B / SE
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


def grs_test(
    T,
    N,
    num_factors,
    alphas,
    factors,
    residcovmat,
):
    """Calculate the Gibbons-Ross-Shanken (GRS) F-statistic."""

    factors = as_2d(factors)
    alphas = np.asarray(alphas).reshape(-1, 1)
    residcovmat = np.atleast_2d(residcovmat)

    # Factor means
    factor_means = factors.mean(axis=0).reshape(-1, 1)
    # Factor covariance
    factor_demeaned = factors - factors.mean(axis=0)

    # factor_covmat = factor_demeaned.T @ factor_demeaned / (T - 1)
    # first_term = (T - N - num_factors) / N
    factor_covmat = factor_demeaned.T @ factor_demeaned / T  # MLE, not T-1
    first_term = (T - N - num_factors) / N * T / (T - num_factors - 1)

    # Check matrix conditioning
    v.check_matrix_condition(residcovmat)
    v.check_matrix_condition(factor_covmat)

    # GRS numerator
    numerator = (alphas.T @ np.linalg.solve(residcovmat, alphas)).item()

    # GRS denominator
    denominator = (
        1 + factor_means.T @ np.linalg.solve(factor_covmat, factor_means)
    ).item()

    grs_stat = first_term * numerator / denominator
    return grs_stat


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


def one_shot_cross_sectional(avg_rets, betas):
    y = np.asarray(avg_rets, dtype=float).reshape(-1)
    X = add_constant(betas.T)  # (N, K+1)

    N, P = X.shape  # N portfolios, P estimated coefficients

    # Cross-sectional OLS
    gamma = np.linalg.lstsq(X, y, rcond=None)[0]

    # Fitted returns and residuals
    fitted = X @ gamma
    residuals = y - fitted

    # Cross-sectional R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # OLS standard errors
    if N <= P:
        raise ValueError("Insufficient portfolios for OLS standard errors.")

    v.check_matrix_condition(X.T @ X)

    sigma2 = ss_res / (N - P)
    cov_gamma = sigma2 * np.linalg.inv(X.T @ X)

    se_gamma = np.sqrt(np.diag(cov_gamma))

    # t-stats
    t_gamma = np.divide(
        gamma, se_gamma, out=np.full_like(gamma, np.nan), where=se_gamma > 0
    )

    return {
        "gamma": gamma,
        "se_gamma": se_gamma,
        "t_gamma": t_gamma,
        "fitted": fitted,
        "residuals": residuals,
        "r_squared": r_squared,
    }


def cross_sectional(gamma):
    gamma = as_2d(gamma)
    T = gamma.shape[0]
    avg_gamma = gamma.mean(axis=0)
    se_gamma = gamma.std(axis=0, ddof=1) / np.sqrt(T)
    t_stat = (avg_gamma - 0) / se_gamma
    return {"avg_gamma": avg_gamma, "se_gamma": se_gamma, "t_gamma": t_stat}


def model_consistency_checks(factor_means, one_shot, fm_results):
    """
    Compare estimated factor risk premia with factor means
    and evaluate whether pricing intercepts are near zero.
    """
    factor_means = np.asarray(factor_means, dtype=float)

    one_shot_gamma = np.asarray(one_shot["gamma"])
    fm_gamma = np.asarray(fm_results["avg_gamma"])

    return {
        "factor_means": factor_means,
        "one_shot_premia": one_shot_gamma[1:],
        "fm_premia": fm_gamma[1:],
        "one_shot_diff": one_shot_gamma[1:] - factor_means,
        "fm_diff": fm_gamma[1:] - factor_means,
        "one_shot_intercept": one_shot_gamma[0],
        "fm_intercept": fm_gamma[0],
        "fm_intercept_t": fm_results["t_gamma"][0],
    }


# --- Full Model Run ---
"""Chat prompt: add a section to run model that uses summary_stats to calculate the values for each of the
factors given as input"""


def run_model(data, portfolio, factors, export=True):
    """Run asset pricing analysis for selected factors"""

    # Validate required columns
    required_columns = [DATE_COL, RF_COL] + portfolio + factors
    v.validate_data_schema(data, required_columns)

    # Validate missing values in selected columns
    v.validate_missing_values(data[required_columns])

    F = data[factors].to_numpy(dtype=float)
    rf = data["RF"].to_numpy(dtype=float)
    rawrets = data[portfolio].to_numpy(dtype=float)

    rets = excess_returns(rawrets, rf)
    T, N = rawrets.shape
    num_factors = F.shape[1]
    v.validate_sample_size(T, N, K=num_factors)

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
    portfolio_stats = summary_stats(rawrets, rfvec=rf)
    expected_excess_returns = rets.mean(axis=0)

    portfolio_summary_df = pd.DataFrame(
        {
            "Mean": portfolio_stats["mean"],
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

    fgrs = grs_test(
        T=T,
        N=N,
        num_factors=len(factors),
        alphas=results["alpha"],
        factors=F,
        residcovmat=RESIDCOVMAT,
    )
    p_grs = f.sf(fgrs, N, T - N - num_factors)
    f_crit = f.ppf(0.95, N, T - N - num_factors)

    # Fama-MacBeth regressions
    betas = get_betas(results)
    gamma = get_gamma(rets, betas)
    fm_results = cross_sectional(gamma)

    # One-shot cross-sectional
    one_shot_results = one_shot_cross_sectional(expected_excess_returns, betas)
    # Model consistency checks
    consistency_results = model_consistency_checks(
        factor_stats["mean"], one_shot_results, fm_results
    )

    model_results = {
        "factor_summary": factor_summary_df,
        "portfolio_summary": portfolio_summary_df,
        "regression": results,
        "alphas": alphas,
        "betas": betas,
        "grs_stat": fgrs,
        "grs_pvalue": p_grs,
        "grs_fcrit": f_crit,
        "fama_macbeth": fm_results,
        "residual_covariance": RESIDCOVMAT,
        "one_shot": one_shot_results,
        "consistency": consistency_results,
    }
    if export:
        export_results_to_excel(
            model_results=model_results, factors=factors, portfolio=portfolio
        )

    return model_results


def available_factors(data):
    """Every column that is not the date, the risk-free rate, or a test portfolio."""
    excluded = {DATE_COL, RF_COL, *PORTFOLIO_COLS}
    return [c for c in data.columns if c not in excluded]


def fmt_month(yyyymm):  # Month formatting
    return f"{yyyymm // 100}-{yyyymm % 100:02d}"


def min_obs_required(n_portfolios, n_factors):
    """GRS needs T - N - K > 0 degrees of freedom, so T >= N + K + 1."""
    return n_portfolios + n_factors + 1


def prompt_date_range(data, max_factors):
    """Ask for a start and end month and return the rows in that (inclusive) range."""

    months = data[DATE_COL].tolist()
    first, last = min(months), max(months)

    N = len(PORTFOLIO_COLS)
    K = max_factors

    print(
        f"\nData covers {fmt_month(first)} to {fmt_month(last)} "
        f"({len(months)} observations)."
    )

    while True:
        try:
            start_in = input(f"Start month [YYYYMM, Enter = {first}]: ").strip()
            end_in = input(f"End month   [YYYYMM, Enter = {last}]: ").strip()

            # Validate individual dates
            start = v.validate_date(start_in) if start_in else first
            end = v.validate_date(end_in) if end_in else last

            # Validate date ordering and minimum calendar window
            v.validate_date_range(start, end, N=N, K=K)

            # Validate dates against available dataset
            v.validate_data_range(start, end, months)

            # Filter data to selected window
            subset = data[(data[DATE_COL] >= start) & (data[DATE_COL] <= end)]

            # Validate actual sample size
            v.validate_sample_size(len(subset), N, K)

            return subset.reset_index(drop=True)

        except ValueError as err:
            print(f"Invalid range: {err} Please try again.\n")

        except (KeyboardInterrupt, EOFError):
            print("\nInput cancelled. Exiting.")
            raise SystemExit(0)


def model_name_for(factors):
    """Return a unique readable name for each factor combination."""

    factor_set = set(factors)

    if factor_set == {"RMRF"}:
        return "CAPM"

    if factor_set == {"RMRF", "SMB", "HML"}:
        return "Fama-French 3-Factor"

    if factor_set == {"RMRF", "SMB", "HML", "MOM"}:
        return "Carhart 4-Factor"

    return f"{len(factors)}-Factor ({', '.join(sorted(factors))})"


def prompt_factor_models(factor_options):
    """Select factor combinations or run all possible models."""

    default = "RMRF; RMRF SMB HML; RMRF SMB HML MOM"

    print(f"\nAvailable factors: {', '.join(factor_options)}")
    print("Enter factor combinations separated by ';'")
    print("Enter ALL to run every possible combination.")
    print("Examples:")
    print("  RMRF MOM")
    print("  RMRF; RMRF SMB HML; RMRF SMB HML MOM")
    print("  ALL")

    while True:
        raw = input(f"Models [Enter = {default}]: ").strip()

        if not raw:
            raw = default

        if raw.upper() == "ALL":
            return [list(factor_options)]

        try:
            models = []

            for chunk in raw.split(";"):
                tokens = chunk.replace(",", " ").split()

                if not tokens:
                    continue

                factors = v.validate_factors(tokens, factor_options)

                if not any(set(factors) == set(m) for m in models):
                    models.append(factors)

            if not models:
                raise ValueError("Enter at least one factor.")

            return models

        except ValueError as err:
            print(f"Invalid input: {err} Please try again.\n")


def run_table6_validation(
    part1_workbook_path,
    class_workbook_path,
):
    """Run Part 1 and historical Table A2 validation."""

    factors_ff3 = ["RMRF", "SMB", "HML"]

    # Part 1: 30-year validation period
    part1_data = load_part1_data(part1_workbook_path)

    part1_data = part1_data[
        (part1_data[DATE_COL] >= 199610) & (part1_data[DATE_COL] <= 202609)
    ].reset_index(drop=True)

    if len(part1_data) != 360:
        raise ValueError(f"Part 1 requires 360 months, found {len(part1_data)}.")

    part1_results = run_model(part1_data, PORTFOLIO_COLS, factors_ff3, export=False)

    wb = load_workbook(
        part1_workbook_path,
        data_only=True,
        read_only=True,
    )

    try:
        ws = wb["Tables"]
        table_values = list(
            ws.iter_rows(
                min_row=1,
                max_row=101,
                min_col=1,
                max_col=15,
                values_only=True,
            )
        )
    finally:
        wb.close()

    part1_reference = read_part1_reference(table_values)

    # print("Reference keys:", part1_reference.keys())
    # print("Calculated keys:", part1_results.keys())
    part1_diffs = v.validate_model_results(
        part1_results,
        part1_reference,
    )

    class_data = load_excel_data(class_workbook_path)

    class_sample = class_data[
        (class_data[DATE_COL] >= 196302) & (class_data[DATE_COL] <= 199301)
    ].reset_index(drop=True)

    if len(class_sample) != 360:
        raise ValueError(
            f"Class benchmark requires 360 months, " f"found {len(class_sample)}."
        )

    capm_results = run_model(
        class_sample,
        PORTFOLIO_COLS,
        ["RMRF"],
        export=False,
    )

    ff3_results = run_model(
        class_sample,
        PORTFOLIO_COLS,
        factors_ff3,
        export=False,
    )

    class_reference = read_class_reference()

    class_diffs = v.validate_class_benchmark(
        capm_results,
        ff3_results,
        class_reference,
    )
    print("\n" + "=" * 60)
    print("TABLE 6: VALIDATION")
    print("=" * 60)

    print("\nPART 1 WORKBOOK COMPARISON")
    for category, difference in part1_diffs.items():
        print(f"{category}: {difference:.10e}")

    print("\nCLASS TABLE A2 COMPARISON")
    for category, difference in class_diffs.items():
        if difference is not None:
            print(f"{category}: {difference:.10e}")

    export_validation_table(
        part1_diffs=part1_diffs,
        class_diffs=class_diffs,
        output_path="results/Table_6_Validation.xlsx",
    )

    return part1_diffs, class_diffs


# --- Main ---
"""Chat prompt: Add a section to main that organizes and displays the results for each model type"""


def main():
    """
    Prompt for a date range and factor combinations, then run the time-series
    regressions, GRS test, and Fama-MacBeth regressions for each model.
    """
    data = load_data()
    portfolio_cols = PORTFOLIO_COLS

    factor_models = prompt_factor_models(available_factors(data))
    max_k = max(len(f) for f in factor_models)
    sample = prompt_date_range(data, max_k)

    print(
        f"\nSample: {fmt_month(sample[DATE_COL].iloc[0])} to "
        f"{fmt_month(sample[DATE_COL].iloc[-1])} (T = {len(sample)} months, "
        f"N = {len(portfolio_cols)} portfolios)"
    )

    # Run each requested model
    models = {
        model_name_for(factors): {
            "results": run_model(sample, portfolio_cols, factors),
            "factors": factors,
        }
        for factors in factor_models
    }

    output_tables = {}

    for model_name, model in models.items():
        results = model["results"]
        factors = model["factors"]
        regression = results["regression"]
        fm = results["fama_macbeth"]
        one_shot = results["one_shot"]
        consistency = results["consistency"]

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
                "F-critical (5%)": [results["grs_fcrit"]],
                "p-value": [results["grs_pvalue"]],
                "Reject H0 (5%)": [results["grs_pvalue"] < 0.05],
            },
            index=[model_name],
        )
        # One-shot cross-sectional results
        one_shot_df = pd.DataFrame(
            {
                "Gamma": one_shot["gamma"],
                "SE Gamma": one_shot["se_gamma"],
                "t Gamma": one_shot["t_gamma"],
            },
            index=["Intercept"] + factors,
        )
        # Factor risk premium consistency
        consistency_df = pd.DataFrame(
            {
                "Factor Mean": consistency["factor_means"],
                "One-Shot Gamma": consistency["one_shot_premia"],
                "One-Shot Difference": consistency["one_shot_diff"],
                "FM Gamma": consistency["fm_premia"],
                "FM Difference": consistency["fm_diff"],
            },
            index=factors,
        )

        factor_summary_df = results["factor_summary"]

        output_tables[model_name] = {
            "Factor Summary": factor_summary_df,
            "Regression": regression_df,
            "GRS": grs_df,
            "Fama-MacBeth": fm_df,
            "One-Shot Cross-Sectional": one_shot_df,
            "Consistency": consistency_df,
        }

        display_results(model_name, results, output_tables)

        generate_summary(results, model_name)

    # Plots
    rawrets = sample[portfolio_cols].to_numpy(dtype=float)
    rf = sample["RF"].to_numpy(dtype=float).reshape(-1, 1)

    rets = excess_returns(rawrets, rf)

    plot_portfolio_returns(rets, portfolio_cols)

    # SML scatter plot (CAPM only)
    if "CAPM" in models:
        betas = models["CAPM"]["results"]["regression"]["beta"]
        expected_returns = rets.mean(axis=0)

        plot_sml(betas, expected_returns, portfolio_cols)

    # Model comparison

    comparison_df = pd.concat([output_tables[name]["GRS"] for name in output_tables])

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(comparison_df.round(4).to_string())

    # Table 6

    part1_workbook = Path("data/project2_part1.xlsx")

    class_workbook = Path("data/CAPM_FF3F_Raw_Data_SizeBM.xlsx")

    run_table6_validation(
        part1_workbook_path=part1_workbook,
        class_workbook_path=class_workbook,
    )


if __name__ == "__main__":
    main()

"""Chat prompt: Where is the error in this file that is giving ValueError: not enough values to unpack (expected 2, got 0)"""
# Said line 259 but there is no line 259
