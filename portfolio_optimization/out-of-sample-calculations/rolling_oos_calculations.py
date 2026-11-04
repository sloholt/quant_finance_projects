"""
Part 2: Rolling Out-of-Sample Test

Question: if we only use *past* data to build a portfolio, does it beat the
client in months it has never seen?

Plan: 12 times, use 60 months to pick weights, then score those weights on the
next 3 months. Stitch the 12 three-month pieces together to get 36
out-of-sample months (Sep 2023 to Aug 2026).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# Part 1 lives in a hyphenated folder, so add it to the path to import from it
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "in-sample-calculations")
)
from calculations import (  # noqa: E402
    CLIENT_FILE,
    check_budget_constraint,
    get_client_weights,
    load_client_weights,
    save_client_weights,
)

# Step 1: Settings and data
DATA_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "sector_etf_returns__96_mos_.xlsx"
)
returns = pd.read_excel(
    DATA_FILE, sheet_name="Returns", index_col="Month End", parse_dates=True
)

TICKERS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]

# Raise error for missing tickers:
missing = set(TICKERS) - set(returns.columns)
assert not missing, f"Tickers not found in data: {missing}"

R = returns[TICKERS]
MKT = returns["Market"]
RF = returns["Risk-Free"]

WINDOW = 60
HOLD = 3
TRIALS = 12
LAM = 0.5
CAP = 0.20

# The 12 portfolios = every combination of these three choices
MEANS = ["Raw", "CAPM"]
COVS = ["Sample", "LW"]
RULES = ["Unconstrained", "Long-only", "Cap 20%"]
NAMES = [f"{m} | {c} | {r}" for m in MEANS for c in COVS for r in RULES]


# Step 2: Estimate inputs from ONE 60-month window
# Everything here uses only the 60 months passed in, so there is no look-ahead.
def estimate(R_win, mkt_win, rf_win):
    rf_m = rf_win.mean()

    # Means, version 1: plain historical average
    mu_raw = R_win.mean() * 12

    # Means, version 2: shrink toward CAPM
    #   beta_i = Cov(ETF excess, market excess) / Var(market excess)
    #   CAPM mean = rf + beta * (average market excess return)
    #   final = 50% historical + 50% CAPM
    etf_ex = R_win.sub(rf_win, axis=0)
    mkt_ex = mkt_win - rf_win
    beta = etf_ex.apply(lambda col: col.cov(mkt_ex)) / mkt_ex.var()
    mu_capm = (1 - LAM) * R_win.mean() + LAM * (rf_m + beta * mkt_ex.mean())
    mu_capm = mu_capm * 12

    # Covariance, version 1: normal sample covariance
    cov_sample = R_win.cov() * 12
    # Covariance, version 2: Ledoit-Wolf (shrinks the noisy sample covariance toward a simple target)
    cov_lw = LedoitWolf().fit(R_win).covariance_ * 12

    return {
        "mu": {"Raw": mu_raw.to_numpy(), "CAPM": mu_capm.to_numpy()},
        "cov": {"Sample": cov_sample.to_numpy(), "LW": cov_lw},
        "rf": rf_m * 12,
    }


# Step 3: Find the max-Sharpe weights
def sharpe(w, mu, cov, rf):
    return (w @ mu - rf) / np.sqrt(w @ cov @ w)


def best_weights(mu, cov, rf, rule):
    n_assets = len(mu)

    if rule == "Unconstrained":  # Part 1 ORP formula
        z = np.linalg.inv(cov) @ (mu - rf)
        return z / z.sum()

    max_w = 1.0 if rule == "Long-only" else CAP
    result = minimize(
        lambda w: -sharpe(w, mu, cov, rf),
        x0=np.ones(n_assets) / n_assets,
        method="SLSQP",
        bounds=[(0, max_w)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result.x


# Step 4: The rolling loop
# Trial 1 estimates on months 1-60 and scores 61-63. Trial 2 estimates on 4-63
# and scores 64-66. ... Trial 12 scores 94-96.
def rolling_test():
    weights = {name: [] for name in NAMES}
    oos = {name: [] for name in NAMES}
    in_sample = {name: [] for name in NAMES}
    cond_rows = []
    solver_ok = []
    rng = np.random.default_rng(42)

    for k in range(TRIALS):
        start = HOLD * k
        end = start + WINDOW

        est = estimate(R.iloc[start:end], MKT.iloc[start:end], RF.iloc[start:end])

        R_next = R.iloc[end : end + HOLD].to_numpy()

        cond_rows.append(
            {
                "Window": f"{R.index[start]:%b %Y} – {R.index[end - 1]:%b %Y}",
                "Scored": f"{R.index[end]:%b %Y} – {R.index[end + HOLD - 1]:%b %Y}",
                "κ Sample": np.linalg.cond(est["cov"]["Sample"]),
                "κ LW": np.linalg.cond(est["cov"]["LW"]),
            }
        )

        for m in MEANS:
            for c in COVS:
                for r in RULES:
                    name = f"{m} | {c} | {r}"
                    mu, cov, rf = est["mu"][m], est["cov"][c], est["rf"]
                    w = best_weights(mu, cov, rf, r)
                    weights[name].append(w)
                    in_sample[name].append(sharpe(w, mu, cov, rf))
                    oos[name].extend(R_next @ w)

                    # Solver check: no random valid portfolio should beat the optimizer
                    if r != "Unconstrained":
                        W_rand = rng.dirichlet(np.ones(len(TICKERS)), 20000)
                        if r == "Cap 20%":
                            W_rand = W_rand[W_rand.max(axis=1) <= CAP]
                        sr_rand = (W_rand @ mu - rf) / np.sqrt(
                            np.einsum("ij,jk,ik->i", W_rand, cov, W_rand)
                        )
                        solver_ok.append(sharpe(w, mu, cov, rf) >= sr_rand.max())

    print(
        f"Solver check: optimizer beat every random portfolio in {sum(solver_ok)} of {len(solver_ok)} cases"
    )
    print("Each portfolio now has", len(oos[NAMES[0]]), "out-of-sample months")
    return weights, oos, in_sample, cond_rows


# Step 5: Table 2 — score everything on the 36 unseen months
def score(r, rf_36):
    r = np.array(r)
    er = r.mean() * 12
    sd = r.std(ddof=1) * np.sqrt(12)
    return er, sd, (er - rf_36) / sd, np.prod(1 + r) - 1


def make_table2(weights, oos, client_36, w_client, rf_36):
    table2 = {}
    for name in NAMES:
        W = np.array(weights[name])
        er, sd, sr, cum = score(oos[name], rf_36)
        table2[name] = {
            "Return": er,
            "Std dev": sd,
            "Sharpe": sr,
            "Cumulative": cum,
            "Gross exposure": np.abs(W).sum(axis=1).mean(),
            "Largest weight": W.max(),
            "Turnover/qtr": (0.5 * np.abs(np.diff(W, axis=0)).sum(axis=1)).mean(),
        }
    er, sd, sr, cum = score(client_36, rf_36)
    table2["Given client"] = {
        "Return": er,
        "Std dev": sd,
        "Sharpe": sr,
        "Cumulative": cum,
        "Gross exposure": np.abs(w_client).sum(),
        "Largest weight": w_client.max(),
        "Turnover/qtr": 0.0,
    }
    return pd.DataFrame(table2).T


# Step 5b: In-sample vs out-of-sample Sharpe
# In-sample = Sharpe on the 60 months used to build the weights (averaged over
# the 12 windows). Out-of-sample = Sharpe on the 36 unseen months.
def make_compare(in_sample, table2):
    compare = pd.DataFrame(
        {
            "In-sample Sharpe (avg of 12 windows)": {
                n: np.mean(in_sample[n]) for n in NAMES
            },
            "Out-of-sample Sharpe (36 months)": table2.loc[NAMES, "Sharpe"],
        }
    )
    compare["Change"] = compare.iloc[:, 1] - compare.iloc[:, 0]
    return compare


# Step 6: Table 3 — compare with 10,000 random clients
def simulate_clients(R_36, rf_36, n_sims=10000, seed=42):
    sim_w = np.random.default_rng(seed).dirichlet(
        np.ones(len(TICKERS)), size=n_sims
    )  # 10,000 random portfolios
    sim_r = R_36 @ sim_w.T
    sim_er = sim_r.mean(axis=0) * 12
    sim_sd = sim_r.std(axis=0, ddof=1) * np.sqrt(12)
    sim_sr = (sim_er - rf_36) / sim_sd
    return sim_r, sim_er, sim_sd, sim_sr


def make_table3(sim_er, sim_sd, sim_sr, table2):
    table3a = pd.DataFrame(
        {
            f"{p}th": [np.percentile(x, p) for x in (sim_er, sim_sd, sim_sr)]
            for p in [5, 25, 50, 75, 95]
        },
        index=["Return", "Std dev", "Sharpe"],
    )
    table3b = pd.DataFrame(
        {
            "Sharpe": table2["Sharpe"],
            "Beats % of random clients": [
                (sim_sr < s).mean() for s in table2["Sharpe"]
            ],
        }
    )
    return table3a, table3b


# Step 7: Table 4 — condition numbers (how sensitive the weights are to estimation errors)
def make_table4(cond_rows):
    table4 = pd.DataFrame(cond_rows, index=range(1, TRIALS + 1))
    table4["Ratio"] = table4["κ Sample"] / table4["κ LW"]
    return table4


# Step 8: Charts
RULE_COLORS = {"Unconstrained": "#D55E00", "Long-only": "#0072B2", "Cap 20%": "#009E73"}
CLIENT_COLOR = "#1B1B1B"
BAND_COLOR = "#DCE4EC"
MEDIAN_COLOR = "#7F8FA0"
HIST_COLOR = "#B9CCE0"
TICKER_COLORS = [
    "#1F4E79",
    "#2E86C1",
    "#85C1E9",
    "#117A65",
    "#48C9B0",
    "#D4AC0D",
    "#F5B041",
    "#CA6F1E",
    "#A93226",
    "#7D3C98",
    "#95A5A6",
]

# Chart 2 line styles: color family = rule, shade = expected-return input
# (Raw dark, CAPM light), line style = covariance estimator (LW solid, Sample dashed)
LINE_COLORS = {
    ("Raw", "Unconstrained"): "#B34700",
    ("CAPM", "Unconstrained"): "#F39C5A",
    ("Raw", "Long-only"): "#0B3C7A",
    ("CAPM", "Long-only"): "#5DA5E0",
    ("Raw", "Cap 20%"): "#0B6B4F",
    ("CAPM", "Cap 20%"): "#5CC9A0",
}


def grow(r):
    return np.concatenate([[1], np.cumprod(1 + np.array(r))])


def plot_growth(oos, client_36, sim_r):
    # Chart 1: growth of $1
    dates = [R.index[WINDOW - 1]] + list(R.index[WINDOW:])
    sim_grow = np.vstack([np.ones(sim_r.shape[1]), np.cumprod(1 + sim_r, axis=0)])
    plt.figure(figsize=(11, 6))
    plt.fill_between(
        dates,
        np.percentile(sim_grow, 5, axis=1),
        np.percentile(sim_grow, 95, axis=1),
        color=BAND_COLOR,
        label="Random clients, 5th–95th pct",
    )
    plt.plot(
        dates,
        np.median(sim_grow, axis=1),
        color=MEDIAN_COLOR,
        lw=1.5,
        ls="--",
        label="Random clients, median",
    )
    plt.plot(dates, grow(client_36), color=CLIENT_COLOR, lw=2.5, label="Given client")
    for r in RULES:
        plt.plot(
            dates,
            grow(oos[f"Raw | LW | {r}"]),
            color=RULE_COLORS[r],
            lw=2,
            label=f"Raw | LW | {r}",
        )
    plt.title("Growth of $1, out of sample")
    plt.ylabel("Value of $1")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.show()


def plot_sharpe_hist(sim_sr, table2):
    # Chart 2: histogram of random clients' Sharpe ratios
    plt.figure(figsize=(12, 6))
    plt.hist(
        sim_sr,
        bins=60,
        color="#D5DDE6",
        edgecolor="white",
        linewidth=0.5,
        label=f"{len(sim_sr):,} random clients",
    )
    for name in NAMES:
        inp, est, rule = name.split(" | ")
        plt.axvline(
            table2.loc[name, "Sharpe"],
            label=name,
            color=LINE_COLORS[(inp, rule)],
            lw=2 if est == "LW" else 1.6,
            linestyle="-" if est == "LW" else (0, (4, 2)),
        )
    plt.axvline(
        table2.loc["Given client", "Sharpe"],
        color="#1B1B1B",
        lw=3,
        label="Given client",
    )
    plt.title("Out-of-sample Sharpe ratios")
    plt.xlabel("Sharpe ratio")
    plt.ylabel("Count")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_weights(weights, w_client, pick="Raw | LW | Long-only"):
    # Chart 3: stacked weights by quarter (client first)
    bars = pd.DataFrame(
        [w_client] + weights[pick],
        columns=TICKERS,
        index=["Client"]
        + [f"{R.index[WINDOW + HOLD * k]:%b %Y}" for k in range(TRIALS)],
    )
    bars.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        width=0.8,
        color=TICKER_COLORS[: len(TICKERS)],
        edgecolor="white",
        linewidth=0.4,
    )
    plt.title(f"Weights by quarter: {pick}")
    plt.ylabel("Weight")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Client's portfolio: use the weights last entered in Part 1
    if CLIENT_FILE.exists():
        client_weights = load_client_weights()
        print(f"Loaded client weights from {CLIENT_FILE}")
    else:
        print("No saved client weights found, enter them now:")
        client_weights = get_client_weights()
        save_client_weights(client_weights)
    check_budget_constraint(client_weights)
    w_client = np.array([client_weights.get(t, 0.0) for t in TICKERS])

    # Step 1: Settings and data
    print(len(NAMES), "portfolios:", *NAMES, sep="\n  ")

    # Steps 2-4: estimate, optimize, and roll through the 12 trials
    weights, oos, in_sample, cond_rows = rolling_test()

    # Step 5: Table 2
    R_36 = R.iloc[WINDOW:].to_numpy()
    rf_36 = RF.iloc[WINDOW:].mean() * 12
    client_36 = R_36 @ w_client

    table2 = make_table2(weights, oos, client_36, w_client, rf_36)
    print(f"\nTable 2: Out-of-sample results (36 months, annualized, rf = {rf_36:.4%})")
    print(table2.round(4).to_string())

    # Step 5b: In-sample vs out-of-sample Sharpe
    compare = make_compare(in_sample, table2)
    print("\nIn-sample vs out-of-sample Sharpe ratios")
    print(compare.round(2).to_string())

    # Step 6: Table 3
    sim_r, sim_er, sim_sd, sim_sr = simulate_clients(R_36, rf_36)
    table3a, table3b = make_table3(sim_er, sim_sd, sim_sr, table2)
    print("\nTable 3a: Random clients (percentiles)")
    print(table3a.round(4).to_string())
    print("\nTable 3b: Share of random clients beaten on Sharpe")
    print(
        table3b.to_string(
            formatters={
                "Sharpe": "{:.4f}".format,
                "Beats % of random clients": "{:.1%}".format,
            }
        )
    )

    # Step 7: Table 4
    table4 = make_table4(cond_rows)
    print("\nTable 4: Condition numbers by window")
    print(table4.round(2).to_string())

    # Step 8: Charts
    plot_growth(oos, client_36, sim_r)
    plot_sharpe_hist(sim_sr, table2)
    plot_weights(weights, w_client)
