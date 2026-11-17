import warnings
from datetime import datetime
import pandas as pd
import numpy as np


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
