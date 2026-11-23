import warnings
from datetime import datetime
import pandas as pd
import numpy as np

CATEGORIES = [
    "Regression coefficients",
    "Standard errors",
    "t-statistics",
    "GRS F statistic and p-value",
    "One-shot gammas and standard errors",
    "Fama-MacBeth gammas, standard errors, t-statistics",
]


def validate_sample_size(T, N, K):
    """Check that the sample has enough observations for the GRS test.

    Raises a value error if the sample is too small and a warning if there are less than 60
    months available.
    """
    # Min sample size for GRS:
    if T <= N + K:
        raise ValueError(
            f"Insufficient sample size for GRS test: T={T}, N={N}, K={K}."
            f"Requires T > N+K ({N+K})."
        )
    # Warning
    if T < 60:
        warnings.warn(
            f"Only {T} months of data available."
            "GRS test results will have low statistical power.",
            UserWarning,
        )
    return True


def validate_date(date_str):
    """Validate YYYYMM, YYYY-MM, or YYYY/MM and return integer YYYYMM."""
    if not isinstance(date_str, str) or not date_str.strip():
        raise ValueError("Date cannot be empty. Use YYYYMM or YYYY-MM format.")

    date_str = date_str.strip()
    digits = date_str.replace("-", "").replace("/", "")

    if len(digits) != 6 or not digits.isdigit():
        raise ValueError(f"Invalid date '{date_str}'. Use YYYYMM or YYYY-MM format.")

    year = int(digits[:4])
    month = int(digits[4:])

    if year < 1 or not 1 <= month <= 12:
        raise ValueError(f"Invalid year or month: '{date_str}'.")

    return year * 100 + month


def validate_date_range(start, end, N=25, K=3):
    """Validate date order and minimum sample length"""
    if start > end:
        raise ValueError("Start month must be on or before end month.")

    start_year, start_month = divmod(start, 100)
    end_year, end_month = divmod(end, 100)

    T = (end_year - start_year) * 12 + (end_month - start_month) + 1

    validate_sample_size(T, N, K)

    return T


def validate_data_range(start, end, available_dates):
    """Check that the requested dates are available in the dataset"""
    dates = list(available_dates)

    if not dates:
        raise ValueError("No available dates found in dataset.")

    first = min(dates)
    last = max(dates)

    if start < first or end > last:
        raise ValueError(
            f"Requested range {start}–{end} is outside "
            f"available data range {first}–{last}."
        )

    return True


def validate_factors(factors, valid_factors):
    """Validate selected risk factors against available factors"""
    # Check for empty input
    if not factors:
        raise ValueError("At least one factor must be selected.")

    # Create case-insensitive lookup
    lookup = {f.upper(): f for f in valid_factors}

    # Normalize factor names
    normalized = [f.strip().upper() for f in factors]

    # Check for empty factor names
    if any(not f for f in normalized):
        raise ValueError("Factor names cannot be empty.")

    # Check for unknown factors
    unknown = [f for f in normalized if f not in lookup]

    if unknown:
        raise ValueError(
            f"Unknown factor(s): {', '.join(unknown)}. "
            f"Valid factors: {', '.join(valid_factors)}."
        )

    # Check for duplicate factors
    duplicates = [f for f in set(normalized) if normalized.count(f) > 1]

    if duplicates:
        raise ValueError(f"Duplicate factor(s): {', '.join(sorted(duplicates))}.")

    return [lookup[f] for f in normalized]


def validate_data_schema(data, required_cols):
    """Check dataset has all required columns"""
    # Check for empty dataset
    if data is None or data.empty:
        raise ValueError("Dataset is empty or could not be loaded.")

    # Check for duplicate column names
    duplicates = data.columns[data.columns.duplicated()].tolist()

    if duplicates:
        raise ValueError(f"Duplicate columns found: {', '.join(duplicates)}.")

    # Check for missing required columns
    missing = [col for col in required_cols if col not in data.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Available columns: {', '.join(data.columns)}."
        )

    return True


def validate_missing_values(data, date_col="Month"):
    """Check for missing values"""
    # Identify missing values
    missing = data.isna()

    if not missing.any().any():
        return True

    # Count missing values by column
    missing_counts = missing.sum()
    affected_columns = missing_counts[missing_counts > 0]

    # Build detailed error message
    details = []

    for col, count in affected_columns.items():
        # Identify months with missing values
        affected_months = data.loc[missing[col], date_col].tolist()

        # Format dates for readability
        months = [
            f"{int(m) // 100}-{int(m) % 100:02d}" if pd.notna(m) else "Unknown"
            for m in affected_months
        ]

        details.append(
            f"  {col}: {count} missing value(s) " f"in month(s): {', '.join(months)}"
        )

    raise ValueError(
        "Missing values detected in dataset:\n"
        + "\n".join(details)
        + "\nPlease resolve missing values before running the analysis."
    )


def check_matrix_condition(matrix, threshold=1e12):
    """Check for single or ill-conditiioned matrices"""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains NaN or infinite values.")

    condition = np.linalg.cond(matrix)

    if not np.isfinite(condition) or condition > threshold:
        raise np.linalg.LinAlgError(
            f"Matrix is singular or ill-conditioned "
            f"(condition number: {condition:.2e})."
        )


def max_absolute_difference(reference, calculated):
    """Calculate maximum absolute difference and identify invalid values."""

    reference = np.asarray(reference, dtype=float)
    calculated = np.asarray(calculated, dtype=float)

    if reference.shape != calculated.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference.shape}, "
            f"calculated={calculated.shape}"
        )

    if not np.all(np.isfinite(reference)):
        raise ValueError(f"Non-finite reference values: {reference}")

    if not np.all(np.isfinite(calculated)):
        raise ValueError(f"Non-finite calculated values: {calculated}")

    return float(np.max(np.abs(reference - calculated)))


def validate_model_results(results, reference):
    """Compare run_model() results against Part 1 Excel reference."""

    reg = results["regression"]
    one = results["one_shot"]
    fm = results["fama_macbeth"]

    calculated = {
        CATEGORIES[0]: np.concatenate(
            [
                reg["alpha"].ravel(),
                reg["beta"].ravel(),
            ]
        ),
        CATEGORIES[1]: np.concatenate(
            [
                reg["se_alpha"].ravel(),
                reg["se_beta"].ravel(),
            ]
        ),
        CATEGORIES[2]: np.concatenate(
            [
                reg["t_alpha"].ravel(),
                reg["t_beta"].ravel(),
            ]
        ),
        CATEGORIES[3]: np.array(
            [
                results["grs_stat"],
                results["grs_pvalue"],
            ]
        ),
        CATEGORIES[4]: np.concatenate(
            [
                one["gamma"].ravel(),
                one["se_gamma"].ravel(),
            ]
        ),
        CATEGORIES[5]: np.concatenate(
            [
                fm["avg_gamma"].ravel(),
                fm["se_gamma"].ravel(),
                fm["t_gamma"].ravel(),
            ]
        ),
    }

    differences = {}

    for category in CATEGORIES:
        try:
            differences[category] = max_absolute_difference(
                reference[category],
                calculated[category],
            )
        except ValueError as error:
            raise ValueError(f"Validation failed for '{category}': {error}") from error

    return differences


def validate_class_benchmark(capm_results, ff3_results, reference):
    """Compare CAPM and FF3 outputs against class Table A2."""

    comparisons = {
        "GRS statistic and p-value": [],
        "GRS critical value": [],
        "One-shot gamma and standard error": [],
        "Fama-MacBeth gamma": [],
        "Fama-MacBeth standard error": [],
        "Fama-MacBeth t-statistic": [],
        "Cross-sectional R-squared": [],
        "Factor means": [],
    }

    for name, results in [
        ("CAPM", capm_results),
        ("FF3", ff3_results),
    ]:
        ref = reference[name]
        fm = results["fama_macbeth"]

        comparisons["GRS statistic and p-value"].append(
            max_absolute_difference(
                [ref["grs_stat"], ref["grs_pvalue"]],
                [results["grs_stat"], results["grs_pvalue"]],
            )
        )

        comparisons["GRS critical value"].append(
            max_absolute_difference(
                [ref["grs_fcrit"]],
                [results["grs_fcrit"]],
            )
        )

        comparisons["Cross-sectional R-squared"].append(
            max_absolute_difference(
                [ref["r_squared"]],
                [results["one_shot"]["r_squared"]],
            )
        )

        comparisons["Fama-MacBeth gamma"].append(
            max_absolute_difference(
                ref["fm_gamma"],
                fm["avg_gamma"][1:],
            )
        )

        comparisons["Fama-MacBeth standard error"].append(
            max_absolute_difference(
                ref["fm_se"],
                fm["se_gamma"][1:],
            )
        )

        comparisons["Fama-MacBeth t-statistic"].append(
            max_absolute_difference(
                ref["fm_t"],
                fm["t_gamma"][1:],
            )
        )

        if name == "CAPM":
            one = results["one_shot"]

            comparisons["One-shot gamma and standard error"].append(
                max_absolute_difference(
                    [ref["one_shot_gamma"], ref["one_shot_se"]],
                    [one["gamma"][1], one["se_gamma"][1]],
                )
            )

        if name == "FF3":
            factor_means = results["factor_summary"]["Mean"].to_numpy()

            comparisons["Factor means"].append(
                max_absolute_difference(
                    ref["factor_means"],
                    factor_means,
                )
            )

    return {
        category: max(values) if values else None
        for category, values in comparisons.items()
    }
