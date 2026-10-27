"""
- Global minimum variance portfolio
- Efficient portfolio
- Optimal risky portfolio
- Efficient Fronter
- Capital Allocation Line & Fund Separation Theorem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE = "sector_etf_returns__96_mos_.xlsx"
returns = pd.read_excel(
    FILE, sheet_name="Returns", index_col="Month End", parse_dates=True
)

etf_returns = returns.drop(columns=["Market", "Risk-Free"])

TICKERS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
COUNT = returns.shape[0]
ONE_VEC = np.ones((len(TICKERS), 1))  # Shape (11, 1)
n = len(TICKERS)

# Raise error for missing tickers:
missing = set(TICKERS) - set(etf_returns.columns)
assert not missing, f"Tickers not found in data: {missing}"

# Annualized mean-return vector:
mu = etf_returns.mean() * 12
MU = mu.reindex(TICKERS).to_numpy()

# Covariance Matrix:
cov_mat = etf_returns.cov() * 12
COV = cov_mat.loc[TICKERS, TICKERS].to_numpy()

# Annual Risk-Free Rate:
rf_rate = returns["Risk-Free"].mean() * 12


# User Input
def get_client_weights(tickers=TICKERS, seed=42, tol=1e-6):
    while True:
        raw = {}
        print("Enter a weight for each ticker as a decimal")
        print(
            "(Press Enter at the first prompt for a random portfolio; "
            "after that, blank = 0.)"
        )

        random_portfolio = False
        for i, ticker in enumerate(tickers):
            while True:
                entry = input(f"  {ticker:<5}: ").strip().rstrip("%")
                if entry == "":
                    if i == 0:
                        random_portfolio = True
                    else:
                        raw[ticker] = 0.0
                    break
                try:
                    value = float(entry)
                except ValueError:
                    print("    Please enter a number.")
                    continue
                if value < 0:
                    print("    Weights can't be negative.")
                    continue
                raw[ticker] = value
                break
            if random_portfolio:
                break

        if random_portfolio:
            rng = np.random.default_rng(seed)
            draw = rng.dirichlet(np.ones(len(tickers)))
            weights = dict(zip(tickers, draw))
            print(f"\nRandom long-only portfolio (Dirichlet, seed={seed}):")
        else:
            total = sum(raw.values())
            if total == 0:
                print("\nAll weights are zero. Let's try again.\n")
                continue
            if abs(total - 1) > tol:
                print(f"\nWeights summed to {total:.4f}; normalizing to 1.")
            weights = {t: w / total for t, w in raw.items()}
            print("\nWeights:")

        for ticker, w in weights.items():
            print(f"  {ticker:<5} {w:7.2%}")
        print(f"  {'Total':<5} {sum(weights.values()):7.2%}")

        confirm = input("\nUse these weights? [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            return weights
        print()


def check_budget_constraint(portfolio, tol=1e-6):
    W = (
        pd.Series(portfolio, dtype=float)
        .reindex(TICKERS)
        .fillna(0.0)
        .to_numpy()
        .reshape(-1, 1)
    )  # (n, 1)

    budget = (W.T @ ONE_VEC).item()  # Wᵀ1
    satisfied = np.isclose(budget, 1.0, atol=tol, rtol=0)

    print(f"Sum of weights: {budget:.7f}  ->  {'OK' if satisfied else 'VIOLATED'}")
    return satisfied


def get_portfolio_metrics(portfolio):
    # Align to TICKERS, then convert to column vectors / matrix
    W = (
        pd.Series(portfolio, dtype=float)
        .reindex(TICKERS)
        .fillna(0.0)
        .to_numpy()
        .reshape(-1, 1)
    )  # (n, 1)
    MU = mu.reindex(TICKERS).to_numpy().reshape(-1, 1)  # (n, 1)
    COV = cov_mat.loc[TICKERS, TICKERS].to_numpy()  # (n, n)

    exp_ret = (W.T @ MU).item()  # E[R_p] = WᵀMU
    variance = (W.T @ COV @ W).item()  # σ²_p  = WᵀΣW
    std = np.sqrt(variance)
    sharpe = (exp_ret - rf_rate) / std

    print(f"\nExpected return:    {exp_ret:.7f}")
    print(f"Variance:{variance:.7f}")
    print(f"Standard deviation: {std:.7f}")
    print(f"Sharpe ratio:       {sharpe:.7f}  (rf = {rf_rate:.7f})")

    return {
        "expected_return": exp_ret,
        "variance": variance,
        "std_dev": std,
        "sharpe_ratio": sharpe,
    }


def get_gmv():
    # Closed-form solution:
    cov_inv = np.linalg.inv(COV)
    W_closed = ((cov_inv @ ONE_VEC) / (ONE_VEC.T @ cov_inv @ ONE_VEC).item()).ravel()

    # Matrix solution:
    A = np.block([[2 * COV, ONE_VEC], [ONE_VEC.T, np.zeros((1, 1))]])

    # construct the b vector for the form Ax=b
    b = np.zeros(n + 1)  # num portfolio weights + 1 for the lagrange
    b[-1] = 1.0

    # solve for x = A^(-1)b
    x = np.linalg.inv(A) @ b
    W_matrix = x[:n]

    compare_methods = pd.DataFrame(
        {"Closed Form": W_closed, "Matrix Form": W_matrix}, index=TICKERS
    )
    print("\nGMV portfolio weights:")
    print(compare_methods.to_string(float_format=lambda w: f"{w:.6f}"))

    return {
        "closed_form": dict(zip(TICKERS, W_closed)),
        "matrix_solution": dict(zip(TICKERS, W_matrix)),
        "A": A,
        "b": b,
        "x": x,
    }


def get_efficient_portfolio(target):
    A = np.block(
        [
            [2 * COV, ONE_VEC.reshape(-1, 1), MU.reshape(-1, 1)],
            [ONE_VEC.T, np.zeros((1, 2))],
            [MU.reshape(1, -1), np.zeros((1, 2))],
        ]
    )

    # constructing b vector
    b = np.zeros(n + 2)
    b[-2] = 1.0
    b[-1] = target

    # solving for x
    x = np.linalg.solve(A, b)
    W = x[:n]

    return dict(zip(TICKERS, W))


def get_orp(rf=rf_rate):
    # mu - rf1
    excess_returns = MU.reshape(-1, 1) - rf * ONE_VEC

    cov_inv = np.linalg.inv(COV)
    numerator = cov_inv @ excess_returns
    denom = (ONE_VEC.T @ numerator).item()

    # zero division check:
    if np.isclose(denom, 0.0, atol=1e-12, rtol=0):
        raise ValueError("Cannot normalize tangency weights: denominator is near zero.")

    W = (numerator / denom).ravel()
    return dict(zip(TICKERS, W))


def check_orp():
    orp_weights = get_orp(rf_rate)
    print("\nOptimal risky portfolio weights:")
    for ticker, weight in orp_weights.items():
        print(f"  {ticker:<5} {weight:.6f}")

    budget_ok = check_budget_constraint(orp_weights)
    orp_metrics = get_portfolio_metrics(orp_weights)
    exceeds_rf = orp_metrics["expected_return"] > rf_rate

    print(f"\nRisk-free rate: {rf_rate:.7f}")
    print(f"Expected return exceeds risk-free rate: {exceeds_rf}")

    if not exceeds_rf:
        print(
            "The calculated portfolio does not satisfy the "
            "positive-excess-return tangency condition."
        )
    return {
        "weights": orp_weights,
        "metrics": orp_metrics,
        "budget_ok": budget_ok,
        "exceeds_rf": exceeds_rf,
    }


def efficient_frontier():
    # gmv = get_gmv()
    W_gmv = np.array([gmv["closed_form"][t] for t in TICKERS])

    target = client_metrics["expected_return"]
    ep_weights = get_efficient_portfolio(target)
    W_ep = np.array([ep_weights[t] for t in TICKERS])

    orp_weights = get_orp(rf_rate)
    W_orp = np.array([orp_weights[t] for t in TICKERS])

    mean_returns = MU.ravel()
    ret_gmv = (W_gmv @ mean_returns).item()
    ret_ep = (W_ep @ mean_returns).item()
    ret_orp = (W_orp @ mean_returns).item()

    # If EP equals GMV, they cannot generate the whole frontier
    if np.isclose(ret_gmv, ret_ep, atol=1e-12, rtol=0):
        raise ValueError(
            "The client's target return equals the GMV return. "
            "Choose a different target to generate the frontier."
        )

    a_orp = (ret_orp - ret_ep) / (ret_gmv - ret_ep)

    # Range includes a=0, a=1, and a_orp, with a margin
    step = 0.05
    lower = min(0.0, a_orp) - 0.10
    upper = max(1.0, a_orp) + 0.10

    start = int(np.floor(lower / step))
    stop = int(np.ceil(upper / step))
    a_values = np.arange(start, stop + 1) * step

    # Include the exact special values in addition to the grid
    a_values = np.unique(np.concatenate([a_values, [0.0, 1.0, a_orp]]))

    rows = []

    for a in a_values:
        W = a * W_gmv + (1 - a) * W_ep
        expected_return = (W @ mean_returns).item()
        variance = (W @ COV @ W).item()
        std_dev = np.sqrt(variance)

        rows.append(
            {
                "a": a,
                "expected_return": expected_return,
                "std_dev": std_dev,
            }
        )
    frontier = pd.DataFrame(rows)

    # Verify ORP combination
    W_reproduced = a_orp * W_gmv + (1 - a_orp) * W_ep
    weights_match = np.allclose(W_reproduced, W_orp, atol=1e-10, rtol=0)

    print("\nFrontier sweep:")
    print(frontier.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print(f"\na that reproduces the ORP: {a_orp:.10f}")
    print(f"ORP weights match: {weights_match}")
    print(
        "Maximum absolute weight difference:",
        f"{np.max(np.abs(W_reproduced - W_orp)):.3e}",
    )

    return {
        "frontier": frontier,
        "a_orp": a_orp,
        "orp_weights_match": weights_match,
        "reproduced_orp_weights": dict(zip(TICKERS, W_reproduced)),
    }


"""Chat Prompt: Given this function [...] give the python code that implements 
    def plot_frontier() that creates a plot for the efficient frontier calculated. 
    
    Now, update the function to include the CAL on the same plot 
"""


def plot_portfolios():
    frontier = frontier_results["frontier"]
    gmv_return = closed_metrics["expected_return"]

    # Separate the efficient and inefficient branches
    upper = frontier[frontier["expected_return"] >= gmv_return - 1e-12].sort_values(
        "expected_return"
    )

    lower = frontier[frontier["expected_return"] < gmv_return].sort_values(
        "expected_return"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        upper["std_dev"],
        upper["expected_return"],
        color="navy",
        label="Efficient frontier",
    )

    ax.plot(
        lower["std_dev"],
        lower["expected_return"],
        color="gray",
        linestyle="--",
        label="Inefficient branch",
    )

    # Mark the portfolios calculated in earlier steps
    portfolios = [
        ("Client", client_metrics, "gray", "o"),
        ("GMV", closed_metrics, "blue", "s"),
        ("EP", efficient_metrics, "green", "^"),
        ("ORP", orp["metrics"], "red", "*"),
    ]

    for label, metrics, color, marker in portfolios:
        risk = metrics["std_dev"]
        expected_return = metrics["expected_return"]

        ax.scatter(
            risk,
            expected_return,
            color=color,
            marker=marker,
            s=130 if label == "ORP" else 70,
            label=label,
            zorder=5,
        )

        ax.annotate(
            label,
            (risk, expected_return),
            xytext=(7, 7),
            textcoords="offset points",
        )

    ax.set(
        title="Portfolio Frontier",
        xlabel="Annualized standard deviation",
        ylabel="Annualized expected return",
    )

    ax.grid(alpha=0.3)

    ax.plot(
        cal_results["std_dev"],
        cal_results["expected_return"],
        color="darkorange",
        label="Capital allocation line",
    )

    ax.scatter(
        0,
        rf_rate,
        color="black",
        marker="D",
        label="Risk-free asset",
        zorder=5,
    )

    ax.legend()
    fig.tight_layout()
    plt.show()

    return fig, ax


def capital_allocation_line():
    orp_return = orp["metrics"]["expected_return"]
    orp_std = orp["metrics"]["std_dev"]

    y_values = np.arange(31) * 0.05

    rows = []
    for y in y_values:
        expected_return = rf_rate + y * (orp_return - rf_rate)

        std_dev = abs(y) * orp_std

        rows.append(
            {
                "y": y,
                "risk_free_weight": 1 - y,
                "expected_return": expected_return,
                "std_dev": std_dev,
            }
        )
    cal = pd.DataFrame(rows)
    print("\nCapital allocation line:")
    print(cal.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    return cal


def compare_cal_portfolios():
    client_return = client_metrics["expected_return"]
    client_std = client_metrics["std_dev"]

    orp_return = orp["metrics"]["expected_return"]
    orp_std = orp["metrics"]["std_dev"]
    orp_weights = orp["weights"]

    excess_return = orp_return - rf_rate

    if excess_return <= 0 or orp_std <= 0:
        raise ValueError(
            "The ORP must have positive excess return "
            "and positive standard deviation."
        )

    # Two choices of exposure to the ORP
    allocations = {
        "Same expected return": ((client_return - rf_rate) / excess_return),
        "Same standard deviation": client_std / orp_std,
    }
    results = {}

    for name, y in allocations.items():
        # Total-portfolio allocations
        rf_weight = 1 - y
        etf_weights = {ticker: y * orp_weights[ticker] for ticker in TICKERS}

        # Metrics including the risk-free asset
        expected_return = rf_rate + y * excess_return
        std_dev = abs(y) * orp_std

        sharpe = (expected_return - rf_rate) / std_dev if std_dev > 0 else np.nan

        budget = rf_weight + sum(etf_weights.values())

        print(f"\n{name}:")
        print(f"y (weight in ORP): {y:.7f}")
        print(f"Weight in risk-free asset: {rf_weight:.7f}")

        print("\nETF weights:")
        for ticker, weight in etf_weights.items():
            print(f"  {ticker:<5} {weight:.7f}")

        print(f"\nTotal weight including risk-free: {budget:.7f}")
        print(f"Expected return:    {expected_return:.7f}")
        print(f"Standard deviation: {std_dev:.7f}")
        print(f"Sharpe ratio:       {sharpe:.7f}")

        if y > 1:
            print(f"Borrows {y - 1:.2%} of initial capital " f"at rf = {rf_rate:.7f}.")

        if name == "Same expected return":
            improvement = client_std - std_dev

            print(
                f"\nAt the same expected return "
                f"({client_return:.2%}), standard deviation "
                f"changes from {client_std:.2%} to {std_dev:.2%}."
            )
            print(f"Risk reduction: {improvement * 100:.4f} " "percentage points.")

        else:
            improvement = expected_return - client_return

            print(
                f"\nAt the same standard deviation "
                f"({client_std:.2%}), expected return "
                f"changes from {client_return:.2%} "
                f"to {expected_return:.2%}."
            )
            print(f"Return increase: {improvement * 100:.4f} " "percentage points.")

        results[name] = {
            "y": y,
            "risk_free_weight": rf_weight,
            "etf_weights": etf_weights,
            "expected_return": expected_return,
            "std_dev": std_dev,
            "sharpe_ratio": sharpe,
            "improvement": improvement,
        }

    return results


if __name__ == "__main__":
    """
    For testing: equal weighted portfolios 0.0909
    """
    # Step 2: Client's portfolio
    client_weights = get_client_weights()
    check_budget_constraint(client_weights)
    print("Client Portfolio:")
    client_metrics = get_portfolio_metrics(client_weights)

    # Step 3: GMV portfolio
    gmv = get_gmv()

    # Closed-form weights
    print("\nClosed-form GMV:")
    check_budget_constraint(gmv["closed_form"])
    closed_metrics = get_portfolio_metrics(gmv["closed_form"])

    # Matrix-solution weights
    print("\nMatrix-solution GMV:")
    check_budget_constraint(gmv["matrix_solution"])
    matrix_metrics = get_portfolio_metrics(gmv["matrix_solution"])

    # Step 4: Efficient Portfolio
    target = client_metrics["expected_return"]
    efficient_weights = get_efficient_portfolio(target)

    print("\nPortfolio at the client's expected return:")
    for ticker, weight in efficient_weights.items():
        print(f"  {ticker:<5} {weight:.6f}")

    check_budget_constraint(efficient_weights)
    efficient_metrics = get_portfolio_metrics(efficient_weights)

    print(
        "\nTarget return achieved:",
        np.isclose(efficient_metrics["expected_return"], target, atol=1e-10, rtol=1e-8),
    )

    # Step 5: Optimal Risky Portfolio
    orp = check_orp()

    # Step 6: Efficient Frontier
    frontier_results = efficient_frontier()

    # Step 7: Capital Allocation Line
    cal_results = capital_allocation_line()

    # Step 8: CAL by Fund Separation
    cal_comparisons = compare_cal_portfolios()

    # Step 9: Plot the portfolios with the frontier and CAL
    fig, ax = plot_portfolios()
