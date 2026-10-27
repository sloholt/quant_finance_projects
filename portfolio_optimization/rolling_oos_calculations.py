import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

# Config
DATA_FILE = "sector_etf_returns__96_mos_.xlsx"
SHEET_NAME = "Returns"
DATE_COL = "Month End"
RF_COL = "Risk-Free"
MKT_COL = "Market"
TICKERS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
ASSET_COLS = TICKERS

START = "2018-09-30"
END = "2026-08-31"
N_MONTHS = 96
N_INSAMPLE = 60
N_SCORE = 3
N_TRIALS = 12

# COUNT = returns.shape[0]
CLIENT_WEIGHTS = {t: 1 / 11 for t in TICKERS}
SEED = 42
rng = np.random.default_rng(SEED)
LAMBDA = 0.5


def load_data():
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    df.columns = df.columns.str.strip()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)

    # normalize dates to months
    df.index = df.index.to_period("M")
    df = df.loc[START:END].copy()

    # Check & fix units if needed
    ret_cols = ASSET_COLS + [MKT_COL]
    if df[ret_cols].abs().max().max() > 1.0:
        print("Returns look like percent -> dividing by 100")
        df[ret_cols] = df[ret_cols] / 100
    if df[RF_COL].abs().max() > 0.05:
        print("Risk-free looks like percent -> dividing by 100")
        df[RF_COL] = df[RF_COL] / 100

    # Arrays
    R = df[ASSET_COLS].to_numpy()  # asset returns
    RF = df[RF_COL].to_numpy()  # monthly risk free
    MKT = df[MKT_COL].to_numpy()
    dates = df.index

    # Client weight checks
    unknown = set(CLIENT_WEIGHTS) - set(ASSET_COLS)
    assert not unknown, f"Client weights reference unknown assets: {unknown}"
    w_client = np.array([CLIENT_WEIGHTS.get(a, 0.0) for a in ASSET_COLS])
    assert np.isclose(
        w_client.sum(), 1.0
    ), f"Client weights sum to {w_client.sum():.6f}"

    return R, RF, MKT, dates, w_client


def rolling_windows(n_insample=N_INSAMPLE, n_score=N_SCORE, n_trials=N_TRIALS):
    """
    Trial k: in-sample rows [3k, 3k+60), scored rows [3k+60, 3k+63)
    """
    windows = []
    for k in range(n_trials):
        start = n_score * k
        in_slice = slice(start, start + n_insample)
        out_slice = slice(start + n_insample, start + n_insample + n_score)
        windows.append((in_slice, out_slice))
    return windows


def rf_annual(rf_in):
    return 12 * rf_in.mean()


def mean_raw(R_in):
    return 12 * R_in.mean(axis=0)


def capm_betas(R_in, rf_in, mkt_in):
    """beta_i = Cov(r_i - rf, r_m - rf) / Var(r_m - rf)."""
    ex = R_in - rf_in[:, None]
    mex = mkt_in - rf_in
    ex_c = ex - ex.mean(axis=0)
    mex_c = mex - mex.mean()
    return (ex_c * mex_c[:, None]).sum(axis=0) / (mex_c**2).sum()


def mean_capm(R_in, rf_in, mkt_in, lam=LAMBDA):
    beta = capm_betas(R_in, rf_in, mkt_in)
    mu_capm = rf_in.mean() + beta * (mkt_in - rf_in).mean()  # monthly
    mu_star = (1 - lam) * R_in.mean(axis=0) + lam * mu_capm  # monthly
    return 12 * mu_star, beta


if __name__ == "__main__":
    # Load & check data
    R, RF, MKT, dates, w_client = load_data()
    assert (
        len(dates) == N_MONTHS
    ), f"Expected {N_MONTHS} months, got {len(dates)}"  # <- new line
    N = R.shape[1]

    print(f"Period: {dates[0]} to {dates[-1]} ({len(dates)} months)")
    print(f"Assets ({N}): {ASSET_COLS}")
    print(f"Scored months: {dates[N_INSAMPLE]} to {dates[-1]}")
    print(f"Mean monthly RF: {RF.mean():.5f}  (annual ~ {12 * RF.mean():.4f})")

    # Part 1 consistency check: should match 12.90% / 16.03%
    r_c = R @ w_client
    print(
        f"Client full-sample: return {12 * r_c.mean():.4f}, "
        f"std {np.sqrt(12) * r_c.std(ddof=1):.4f}"
    )

    # Rolling windows
    windows = rolling_windows()
    scored_rows = np.concatenate([np.arange(out.start, out.stop) for _, out in windows])
    assert np.array_equal(
        scored_rows, np.arange(N_INSAMPLE, N_MONTHS)
    ), "Scored slices do not cover rows 60-95 exactly once"
    assert all(ins.stop == out.start for ins, out in windows)

    for k, (ins, out) in enumerate(windows):
        print(
            f"Trial {k:2d}: in-sample {dates[ins.start]} to {dates[ins.stop - 1]}, "
            f"scored {dates[out.start]} to {dates[out.stop - 1]}"
        )
