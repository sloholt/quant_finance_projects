from pathlib import Path
import numpy as np
import pandas as pd
from pandas_datareader import data as web

import validation as v

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
    data = pd.read_excel(file_path, sheet_name="Data")
    return clean_data(data)


def load_french_data():
    """Download monthly FF3 factors and 25 MB-BM portfolios"""
    # faactor returns
    factors = web.DataReader(
        "F-F_Research_Data_Factors", "famafrench", start="1926-07-01"
    )[0]

    # monthly value-weighted portfolio returns
    portfolios = web.DataReader("25_Portfolios_5x5", "famafrench", start="1926-07-01")[
        0
    ]

    factors = factors.rename(columns={"Mkt-RF": "RMRF"})
    # Explicitly map size and book-to-market labels
    # rather than relying on column position.
    size_labels = ["SMALL", "ME2", "ME3", "ME4", "BIG"]
    bm_labels = ["LoBM", "BM2", "BM3", "BM4", "HiBM"]

    expected = [f"{size} {bm}" for size in size_labels for bm in bm_labels]

    missing = [col for col in expected if col not in portfolios.columns]
    if missing:
        raise ValueError(f"Unexpected French portfolio columns: {missing}")

    portfolios = portfolios[expected].copy()
    portfolios.columns = PORTFOLIO_COLS

    data = factors.join(portfolios, how="inner")

    # Convert PeriodIndex to YYYYMM
    data[DATE_COL] = data.index.year * 100 + data.index.month

    data = data.replace([-99.99, -999], np.nan)

    return clean_data(data.reset_index(drop=True))


def load_data(source="french"):
    if source == "french":
        return load_french_data
    elif source == "excel":
        return load_excel_data()
    raise ValueError("Invalid data source. Choose French or Excel")
