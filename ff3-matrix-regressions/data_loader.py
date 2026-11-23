from pathlib import Path
import numpy as np
import pandas as pd
from pandas_datareader import data as web

import validation as v

CATEGORIES = [
    "Regression coefficients",
    "Standard errors",
    "t-statistics",
    "GRS F statistic and p-value",
    "One-shot gammas and standard errors",
    "Fama-MacBeth gammas, standard errors, t-statistics",
]

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "data" / "asset_pricing_class_data.xlsx"

DATE_COL = "Month"
RF_COL = "RF"
PORTFOLIO_COLS = [f"ME{i} BM{j}" for i in range(1, 6) for j in range(1, 6)]


def clean_data(data):
    """Validate, clean, and sort monthly asset-pricing data."""
    data = data.copy()
    data.columns = data.columns.str.strip()

    required = [DATE_COL, RF_COL] + PORTFOLIO_COLS
    v.validate_data_schema(data, required)

    data = data.dropna(subset=[DATE_COL]).copy()

    # Standardize dates to integer YYYYMM
    data[DATE_COL] = data[DATE_COL].apply(
        lambda x: v.validate_date(
            str(int(x))
            if isinstance(x, (int, float, np.integer, np.floating))
            else str(x)
        )
    )

    # Convert financial columns to numeric
    value_cols = data.columns.drop(DATE_COL)

    data[value_cols] = data[value_cols].apply(pd.to_numeric, errors="coerce")

    # Validate missing observations
    v.validate_missing_values(data)

    # Ensure one observation per month
    if data[DATE_COL].duplicated().any():
        raise ValueError("Duplicate months found in dataset.")

    return data.sort_values(DATE_COL).reset_index(drop=True)


def load_excel_data(file_path=EXCEL_PATH):
    data = pd.read_excel(
        file_path,
        sheet_name="Data",
        header=1,
        usecols="A:D,F,H:AF",
    )

    # Keep only rows with valid YYYYMM dates
    months = pd.to_numeric(data["Month"], errors="coerce")

    valid_months = (
        months.notna() & months.between(190001, 209912) & months.mod(100).between(1, 12)
    )

    data = data.loc[valid_months].copy()
    data["Month"] = months.loc[valid_months].astype(int)

    return clean_data(data)


def load_french_data():
    """Download monthly FF3, momentum, and 25 MB-BM portfolios."""

    # Fama-French 3 factors
    factors = web.DataReader(
        "F-F_Research_Data_Factors", "famafrench", start="1926-07-01"
    )[0]

    # Momentum factor
    momentum = web.DataReader("F-F_Momentum_Factor", "famafrench", start="1926-07-01")[
        0
    ]

    # Monthly value-weighted portfolios
    portfolios = web.DataReader("25_Portfolios_5x5", "famafrench", start="1926-07-01")[
        0
    ]

    # Standardize factor names
    factors = factors.rename(columns={"Mkt-RF": "RMRF"})
    momentum.columns = ["MOM"]

    # Validate portfolio count
    if portfolios.shape[1] != 25:
        raise ValueError(f"Expected 25 portfolios, found {portfolios.shape[1]}")

    portfolios = portfolios.copy()
    portfolios.columns = PORTFOLIO_COLS

    # Join all datasets on their monthly PeriodIndex
    data = factors.join(momentum, how="inner")
    data = data.join(portfolios, how="inner")

    # Convert dates to YYYYMM
    data[DATE_COL] = data.index.year * 100 + data.index.month

    # Replace French missing-value codes
    data = data.replace([-99.99, -999], np.nan)

    return clean_data(data.reset_index(drop=True))


def load_data(source="french"):
    if source == "french":
        return load_french_data()
    elif source == "excel":
        return load_excel_data()
    raise ValueError("Invalid data source. Choose French or Excel")


def read_part1_reference(table_values):
    """Read Part 1 FF3 reference from the Tables sheet."""

    rows = table_values

    # Table 2: FF3 time-series regression, rows 27–51
    portfolio_rows = rows[26:51]

    coef = np.array(
        [[r[j] for r in portfolio_rows] for j in [6, 8, 10, 12]],
        dtype=float,
    )

    tstat = np.array(
        [[r[j] for r in portfolio_rows] for j in [7, 9, 11, 13]],
        dtype=float,
    )

    se = np.abs(
        np.divide(
            coef,
            tstat,
            out=np.full_like(coef, np.nan),
            where=tstat != 0,
        )
    )

    # Table 4: FF3 one-shot and Fama-MacBeth
    one_rows = rows[65:69]
    fm_rows = rows[69:73]

    return {
        CATEGORIES[0]: coef.ravel(),
        CATEGORIES[1]: se.ravel(),
        CATEGORIES[2]: tstat.ravel(),
        # Table 3: FF3 GRS, Excel row 56
        CATEGORIES[3]: np.array([rows[55][1], rows[55][5]], dtype=float),
        CATEGORIES[4]: np.array(
            [r[3] for r in one_rows] + [r[4] for r in one_rows],
            dtype=float,
        ),
        CATEGORIES[5]: np.array(
            [r[3] for r in fm_rows] + [r[4] for r in fm_rows] + [r[5] for r in fm_rows],
            dtype=float,
        ),
    }


def read_class_reference():
    """Rounded reference values from class Table A2."""

    return {
        "CAPM": {
            "grs_stat": 2.3322,
            "grs_fcrit": 1.5391,
            "grs_pvalue": 0.0004,
            "one_shot_gamma": -0.3465,
            "one_shot_se": 0.3301,
            "fm_gamma": [-0.3465],
            "fm_se": [0.4627],
            "fm_t": [-0.7489],
            "r_squared": 0.046,
        },
        "FF3": {
            "grs_stat": 1.5282,
            "grs_fcrit": 1.5393,
            "grs_pvalue": 0.0529,
            "fm_gamma": [
                -0.1018,
                0.2068,
                0.5066,
            ],
            "fm_se": [
                0.4291,
                0.1544,
                0.1382,
            ],
            "fm_t": [
                -0.2372,
                1.339,
                3.665,
            ],
            "r_squared": 0.790,
            "factor_means": [
                0.4215,
                0.2696,
                0.4600,
            ],
        },
    }


def load_part1_data(file_path):
    """Load the exact monthly observations from the Part 1 workbook."""

    factors = pd.read_excel(
        file_path,
        sheet_name="RAW F-F_Research_Data_Factors",
    )

    portfolios = pd.read_excel(
        file_path,
        sheet_name="RAW 25 Portfolio 5x5",
    )

    factors = factors.rename(columns={"Mkt-RF": "RMRF"})

    portfolios.columns = [DATE_COL] + PORTFOLIO_COLS

    data = factors.merge(
        portfolios,
        on=DATE_COL,
        how="inner",
        validate="one_to_one",
    )

    return clean_data(data)
