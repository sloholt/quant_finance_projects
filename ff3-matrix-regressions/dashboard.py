"""Interactive CAPM / Fama-French dashboard (Plotly Dash).

Run:  python dashboard.py   then open http://127.0.0.1:8050

Data comes from data_loader.load_data (Kenneth French's library by default).
Set the environment variable DATA_SOURCE=excel to use the class Excel file instead.

Inputs are checked with the same validation module the command-line script uses, and
all estimation is done by asset_pricing_analysis.run_model, so the numbers here match
the script for the same date range and factors. Note that run_model also writes the
Excel report to results/ on every run.
"""

import calendar
import os
import re
import warnings

import numpy as np
import plotly.graph_objects as go
from dash import Dash, ALL, Input, Output, State, ctx, dcc, html, no_update

import validation as v
from asset_pricing_analysis import available_factors, run_model
from data_loader import DATE_COL, PORTFOLIO_COLS, RF_COL, load_data

# =====================================
# DATA & CONSTANTS
# =====================================

DATA_SOURCE = os.environ.get("DATA_SOURCE", "french").strip().lower()
DATA = load_data(DATA_SOURCE)
FACTORS = available_factors(DATA)
FIRST, LAST = int(DATA[DATE_COL].min()), int(DATA[DATE_COL].max())
YEARS = list(range(FIRST // 100, LAST // 100 + 1))
FF3 = ["RMRF", "SMB", "HML"]
N_PORT = len(PORTFOLIO_COLS)
T_CRIT = 1.96
LOW_POWER_MONTHS = 60  # validation.validate_sample_size warns below this
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Labels used in chips (setup panel)
CHIP_LABELS = {"RMRF": "Market excess return", "SMB": "Size (SMB)", "HML": "Value (HML)"}
CHIP_WIDTHS = {"RMRF": "183px", "SMB": "109px", "HML": "117px"}  # chip widths in the Figma frame
BETA_SUB = {"RMRF": "ₘ", "SMB": "ₛ", "HML": "ₕ"}

# Design tokens (mirrors assets/dashboard.css)
INK, MUTED, BAR = "#182b36", "#62727e", "#3e6385"
GRID, AXIS = "#edf0f3", "#cfd8de"


def chip_label(f):
    return CHIP_LABELS.get(f, f)


def factor_phrase(f):
    """Factor name as used mid-sentence, e.g. 'value (HML)'."""
    return {"RMRF": "the market (RMRF)", "SMB": "size (SMB)", "HML": "value (HML)"}.get(f, f)


def tidy(err):
    """Validation messages are built from concatenated sentences; add the missing spaces."""
    return re.sub(r"([.)])(?=[A-Z])", r" ", str(err)).strip()


def yyyymm(year, month):
    return int(year) * 100 + int(month)


def month_label(ym):
    return f"{calendar.month_abbr[ym % 100]} {ym // 100}"


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


# =====================================
# MODEL SELECTION LOGIC
# =====================================


def factor_model_name(factors):
    if factors == ["RMRF"]:
        return "CAPM"
    if sorted(factors) == sorted(FF3):
        return "Fama–French"
    return "Custom model"


def factor_model_desc(factors):
    if sorted(factors) == sorted(FF3):
        return "Three-factor model"
    return f"{len(factors)}-factor model · {' · '.join(factors)}"


def requested_models(models, factors):
    """Turn the two model toggles + chosen factors into [(name, factor_list)]."""
    out = []
    if "capm" in models:
        out.append(("CAPM", ["RMRF"]))
    if "multi" in models and factors and not ("capm" in models and factors == ["RMRF"]):
        out.append((factor_model_name(factors), list(factors)))
    return out


# =====================================
# INPUT VALIDATION (validation.py)
# =====================================


def validate(sy, sm, ey, em, factors, models):
    """Run the script's validation checks on the setup inputs.

    Returns (sample_df or None, specs, message, level) where level is
    "ok", "warn" (valid but < 60 months, low GRS power) or "error".
    """
    try:
        if None in (sy, sm, ey, em):
            raise ValueError("Choose a start and end month.")
        start = v.validate_date(f"{int(sy):04d}{int(sm):02d}")
        end = v.validate_date(f"{int(ey):04d}{int(em):02d}")

        specs = requested_models(models, factors)
        if not specs:
            raise ValueError("Select at least one model.")
        specs = [(name, v.validate_factors(fs, FACTORS)) for name, fs in specs]
        k = max(len(fs) for _, fs in specs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v.validate_date_range(start, end, N=N_PORT, K=k)
            v.validate_data_range(start, end, DATA[DATE_COL])
            sample = DATA[(DATA[DATE_COL] >= start) & (DATA[DATE_COL] <= end)]
            T = len(sample)
            v.validate_sample_size(T, N_PORT, k)
    except ValueError as err:
        return None, [], tidy(err), "error"

    msg = f"{T} monthly observations"
    if any(issubclass(w.category, UserWarning) for w in caught):
        return (sample.reset_index(drop=True), specs,
                f"{msg} · under {LOW_POWER_MONTHS}, so the GRS test has low power", "warn")
    return sample.reset_index(drop=True), specs, msg, "ok"


# =====================================
# ANALYSIS (wraps run_model)
# =====================================


def _list(x):
    return np.asarray(x, dtype=float).tolist()


def analyse(sample, specs):
    """Run every requested model and return a JSON-serialisable result bundle."""
    T = len(sample)
    models = []
    portfolio_summary = None

    for name, factors in specs:
        res = run_model(sample, PORTFOLIO_COLS, factors)
        reg, fm, os_, cons = res["regression"], res["fama_macbeth"], res["one_shot"], res["consistency"]
        alpha, t_alpha = np.asarray(reg["alpha"]), np.asarray(reg["t_alpha"])
        terms = ["Intercept"] + factors

        if portfolio_summary is None:  # identical for every model on the same sample
            ps = res["portfolio_summary"]
            portfolio_summary = {
                col: _list(ps[src]) for col, src in
                [("mean", "Mean"), ("var", "Variance"), ("std", "Std Dev"), ("sharpe", "Sharpe Ratio")]
            }

        models.append(
            {
                "name": name,
                "factors": factors,
                "factor_summary": {
                    f: {"mean": float(r["Mean"]), "var": float(r["Variance"]),
                        "std": float(r["Std Dev"]), "sharpe": float(r["Sharpe Ratio"])}
                    for f, r in res["factor_summary"].iterrows()
                },
                "alpha": _list(alpha),
                "t_alpha": _list(t_alpha),
                "beta": _list(reg["beta"]),
                "t_beta": _list(reg["t_beta"]),
                "grs": float(res["grs_stat"]),
                "p": float(res["grs_pvalue"]),
                "fcrit": float(res["grs_fcrit"]),
                "df": [N_PORT, T - N_PORT - len(factors)],
                "n_sig": int((np.abs(t_alpha) > T_CRIT).sum()),
                "mean_abs_alpha": float(np.abs(alpha).mean()),
                "one_shot": {"terms": terms, "gamma": _list(os_["gamma"]), "se": _list(os_["se_gamma"]),
                             "t": _list(os_["t_gamma"]), "r2": float(os_["r_squared"])},
                "fm": {"terms": terms, "gamma": _list(fm["avg_gamma"]), "se": _list(fm["se_gamma"]),
                       "t": _list(fm["t_gamma"])},
                "consistency": {
                    "factor_means": _list(cons["factor_means"]),
                    "os_premia": _list(cons["one_shot_premia"]),
                    "os_diff": _list(cons["one_shot_diff"]),
                    "fm_premia": _list(cons["fm_premia"]),
                    "fm_diff": _list(cons["fm_diff"]),
                    "os_intercept": float(cons["one_shot_intercept"]),
                    "fm_intercept": float(cons["fm_intercept"]),
                    "fm_intercept_t": float(cons["fm_intercept_t"]),
                },
            }
        )

    excess = sample[PORTFOLIO_COLS].to_numpy(float) - sample[[RF_COL]].to_numpy(float)
    return {
        "start": int(sample[DATE_COL].iloc[0]),
        "end": int(sample[DATE_COL].iloc[-1]),
        "T": T,
        "factors": list(dict.fromkeys(f for _, fs in specs for f in fs)),
        "models": models,
        "portfolio_summary": portfolio_summary,
        "avg_excess": _list(excess.mean(axis=0)),
    }


# =====================================
# SMALL UI BUILDERS
# =====================================


def chip(text, cls="chip"):
    return html.Span(text, className=cls)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_plain(x, digits=4):
    return f"{x:.{digits}f}".replace("-", "−")


def is_sig(t):
    return abs(t) > T_CRIT


def metric(label, value, sub, value_cls=""):
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className=f"metric-value {value_cls}"),
            html.Div(sub, className="metric-sub"),
        ],
        className="metric",
    )


def verdict_card(m):
    reject = m["p"] < 0.05
    k = len(m["factors"])
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(m["name"], className="verdict-name"),
                            chip(f"{plural(k, 'factor')} · {' '.join(m['factors'])}"),
                        ],
                        className="verdict-model",
                    ),
                    html.Span(
                        "✕  Reject null hypothesis" if reject else "✓  Fail to reject the null",
                        className="pill pill-red" if reject else "pill pill-green",
                    ),
                ],
                className="verdict-top",
            ),
            html.Div(className="divider-soft"),
            html.Div(
                [
                    metric("GRS F-stat", f"{m['grs']:.2f}", f"F({m['df'][0]}, {m['df'][1]})"),
                    metric("p-value", fmt_p(m["p"]), "vs. 5% threshold",
                           "text-red" if reject else "text-green"),
                    metric("Significant α", f"{m['n_sig']} of {N_PORT}", f"|t| > {T_CRIT}"),
                    metric("Mean |α|", f"{m['mean_abs_alpha']:.2f}%", "per month"),
                ],
                className="metrics",
            ),
        ],
        className="card verdict",
    )


def join_words(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def takeaway_text(models):
    if len(models) == 1:
        m = models[0]
        verb = "rejects" if m["p"] < 0.05 else "does not reject"
        return (
            f"The GRS test {verb} zero alphas for {m['name']} at 5% (p = {fmt_p(m['p'])}). "
            f"{m['n_sig']} of {N_PORT} alphas are individually significant; "
            f"average mispricing is {m['mean_abs_alpha']:.2f}% a month."
        )
    a, b = models[0], models[1]
    added = [f for f in b["factors"] if f not in a["factors"]]
    a_abs, b_abs = a["mean_abs_alpha"], b["mean_abs_alpha"]
    change = "cuts" if b_abs < a_abs else "raises"
    if set(a["factors"]) <= set(b["factors"]) and added:
        lead = f"Adding {join_words(factor_phrase(f) for f in added)}"
    else:
        lead = f"Switching from {a['name']} to {b['name']}"
    ra, rb = a["p"] < 0.05, b["p"] < 0.05
    if ra and rb:
        grs = "The GRS test rejects both models at 5%."
    elif ra:
        grs = f"The GRS test rejects {a['name']} at 5% but not {b['name']}."
    elif rb:
        grs = f"The GRS test rejects {b['name']} at 5% but not {a['name']}."
    else:
        grs = "The GRS test rejects neither model at 5%."
    return f"{lead} {change} average mispricing from {a_abs:.2f}% to {b_abs:.2f}% a month. {grs}"


# --- Detail cards & tables ---


def model_subtitle(m, suffix):
    """e.g. 'Fama–French 3 factors' / 'Fama–French 3-factor model'."""
    k = len(m["factors"])
    if suffix == "model":
        return f"{m['name']} {k}-factor model"
    return f"{m['name']} {plural(k, 'factor')}"


def detail_card(title, subtitle, body, footer=None, cls=""):
    children = [
        html.Div([html.Div(title, className="dc-title"), html.Div(subtitle, className="dc-sub")],
                 className="dc-header"),
        body,
    ]
    if footer:
        children.append(html.Div(footer, className="dc-foot"))
    return html.Div(children, className=f"card dcard {cls}".strip())


def cell(value, cls=""):
    """A table cell: plain text, or (text, extra_class) for highlighted values."""
    if isinstance(value, tuple):
        value, extra = value
        cls = f"{cls} {extra}"
    return html.Div(value, className=f"t-cell {cls}".strip())


def table(header, rows, first_width):
    def row(cells, cls):
        return html.Div(
            [html.Div(cells[0], className="t-cell t-first", style={"width": first_width})]
            + [cell(c, "t-num") for c in cells[1:]],
            className=cls,
        )

    return html.Div([row(header, "t-row t-head")] + [row(r, "t-row") for r in rows],
                    className="t-table")


def t_cell(t, sig_cls="text-green"):
    """t-statistic, emphasised when significant at 5%."""
    return (fmt_plain(t), f"{sig_cls} strong") if is_sig(t) else fmt_plain(t)


def factor_summary_card(m):
    rows = [
        [f, fmt_plain(s["mean"]), fmt_plain(s["var"]), fmt_plain(s["std"]), fmt_plain(s["sharpe"])]
        for f, s in m["factor_summary"].items()
    ]
    return detail_card(
        "Factor summary", model_subtitle(m, "factors"),
        table(["Factor", "Mean", "Variance", "Std dev", "Sharpe"], rows, "44px"),
        cls="flex-1",
    )


def gamma_rows(res):
    return [
        [term, fmt_plain(g), fmt_plain(se), t_cell(t)]
        for term, g, se, t in zip(res["terms"], res["gamma"], res["se"], res["t"])
    ]


def one_shot_card(m):
    os_ = m["one_shot"]
    return detail_card(
        "One-shot cross-sectional", f"Single regression across {N_PORT} portfolios",
        table(["Term", "γ", "SE", "t"], gamma_rows(os_), "60px"),
        footer=f"Cross-sectional R² = {fmt_plain(os_['r2'])}",
        cls="flex-1",
    )


def fm_card(m):
    fm = m["fm"]
    return detail_card(
        "Fama–MacBeth", "Average of period-by-period γ",
        table(["Term", "γ", "SE", "t"], gamma_rows(fm), "60px"),
        footer=f"Intercept t-statistic = {fmt_plain(fm['t'][0])}",
        cls="flex-1",
    )


def grs_card(m):
    rows = [
        ["GRS F-statistic", fmt_plain(m["grs"])],
        ["F-critical (5%)", fmt_plain(m["fcrit"])],
        ["p-value", fmt_plain(m["p"])],
        ["Reject H₀ (5%)", str(m["p"] < 0.05)],
    ]
    return detail_card("GRS joint alpha test", model_subtitle(m, "model"),
                       table(["Statistic", "Value"], rows, "130px"), cls="fixed-328")


def consistency_card(m):
    c = m["consistency"]
    rows = [
        [f, fmt_plain(c["factor_means"][i]), fmt_plain(c["os_premia"][i]), fmt_plain(c["os_diff"][i]),
         fmt_plain(c["fm_premia"][i]), fmt_plain(c["fm_diff"][i])]
        for i, f in enumerate(m["factors"])
    ]
    footer = (
        f"Pricing intercept: one-shot {fmt_plain(c['os_intercept'])} · "
        f"Fama–MacBeth {fmt_plain(c['fm_intercept'])} (t = {fmt_plain(c['fm_intercept_t'])})"
    )
    return detail_card(
        "Model consistency checks", "Estimated γ vs. factor mean (diff = γ − mean)",
        table(["Factor", "Factor mean", "One-shot γ", "One-shot diff", "FM γ", "FM diff"], rows, "52px"),
        footer=footer, cls="flex-1",
    )


def ts_card(m):
    factors = m["factors"]
    header = ["Portfolio", "α", "t(α)"]
    for f in factors:
        header += [f"β {f}", "t"]
    rows = []
    for i, p in enumerate(PORTFOLIO_COLS):
        r = [p, fmt_plain(m["alpha"][i]), t_cell(m["t_alpha"][i], "text-red")]
        for k in range(len(factors)):
            r += [fmt_plain(m["beta"][k][i]), fmt_plain(m["t_beta"][k][i])]
        rows.append(r)
    eq = " + ".join(f"β{BETA_SUB.get(f, '')} {f}" for f in factors)
    return detail_card(
        f"Time-series regressions · {m['name']} ({N_PORT})",
        f"rₐ − rᶠ = α + {eq}  ·  |t| > {T_CRIT} highlighted in α",
        table(header, rows, "120px"),
    )


def portfolio_card(bundle):
    ps = bundle["portfolio_summary"]
    rows = [
        [p, fmt_plain(ps["mean"][i]), fmt_plain(ps["var"][i]), fmt_plain(ps["std"][i]),
         fmt_plain(ps["sharpe"][i])]
        for i, p in enumerate(PORTFOLIO_COLS)
    ]
    return detail_card(
        "Portfolio raw returns", f"{N_PORT} size × book-to-market portfolios",
        table(["Portfolio", "Mean", "Variance", "Std dev", "Sharpe"], rows, "50px"),
        cls="portfolio-card",
    )


# --- Plots ---


def base_layout(fig, height=240):
    fig.update_layout(
        height=height,
        margin=dict(l=34, r=4, t=6, b=6),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(family="Inter, sans-serif", size=9, color=MUTED),
        hoverlabel=dict(font_family="Inter"),
    )
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=AXIS, zerolinewidth=1,
                     tickfont=dict(size=9), ticks="", showline=False)
    return fig


def returns_figure(bundle):
    """Average excess return of each portfolio, grouped by size quintile."""
    sizes = [p.split()[0] for p in PORTFOLIO_COLS]
    bms = [p.split()[1] for p in PORTFOLIO_COLS]
    fig = go.Figure(
        go.Bar(
            x=[sizes, bms],
            y=bundle["avg_excess"],
            marker=dict(color=BAR, cornerradius=2),
            customdata=PORTFOLIO_COLS,
            hovertemplate="%{customdata}<br>%{y:.3f}% per month<extra></extra>",
        )
    )
    base_layout(fig)
    fig.update_layout(bargap=0.36, margin=dict(b=30))
    fig.update_xaxes(showgrid=False, showline=False, ticks="", tickfont=dict(size=8, color=MUTED),
                     showdividers=False)
    fig.update_yaxes(tickformat=".1f", nticks=6)
    return fig


def sml_figure(m, bundle):
    """CAPM beta vs. average excess return, with the one-shot cross-sectional fit."""
    betas = np.asarray(m["beta"][0])
    rets = np.asarray(bundle["avg_excess"])
    g0, g1 = m["one_shot"]["gamma"]
    xs = np.array([betas.min(), betas.max()])
    pad = 0.05 * (xs[1] - xs[0] or 1)
    xs = xs + [-pad, pad]
    fig = go.Figure(
        [
            go.Scatter(x=xs, y=g0 + g1 * xs, mode="lines", line=dict(color=AXIS, width=1.5, dash="dash"),
                       hovertemplate="One-shot fit: γ₀ + γ₁β<extra></extra>"),
            go.Scatter(x=betas, y=rets, mode="markers", marker=dict(color=BAR, size=7),
                       customdata=PORTFOLIO_COLS,
                       hovertemplate="%{customdata}<br>β = %{x:.2f}<br>%{y:.3f}% per month<extra></extra>"),
        ]
    )
    base_layout(fig)
    fig.update_layout(margin=dict(b=28))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(size=9),
                     title=dict(text="CAPM β", font=dict(size=9)), title_standoff=4)
    fig.update_yaxes(tickformat=".1f")
    return fig


def graph(fig):
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, className="plot")


def sml_card(bundle, model_names):
    capm = next((m for m in bundle["models"] if m["name"] == "CAPM"), None)
    if capm is None:
        runs = join_words(model_names)
        body = html.Div(
            [
                html.Div("No CAPM results loaded", className="empty-title"),
                html.Div(
                    f"The SML plot uses CAPM betas. This output only contains the {runs} run. "
                    "Select the CAPM model and run again to populate it.",
                    className="empty-text",
                ),
            ],
            className="empty-state",
        )
    else:
        body = graph(sml_figure(capm, bundle))
    return detail_card("Security Market Line",
                       "CAPM beta vs. average monthly return (%)",
                       body, cls="fixed-328")


def returns_card(bundle):
    return detail_card(
        "Average monthly portfolio returns",
        f"Average excess return (%) · {N_PORT} size × book-to-market portfolios",
        graph(returns_figure(bundle)),
        cls="flex-1",
    )


def detailed_results(bundle, idx):
    m = bundle["models"][idx]
    names = [x["name"] for x in bundle["models"]]
    return [
        html.Div([factor_summary_card(m), one_shot_card(m), fm_card(m)], className="d-row"),
        html.Div([sml_card(bundle, names), returns_card(bundle)], className="d-row"),
        html.Div([grs_card(m), consistency_card(m)], className="d-row"),
        # Time-series regressions for every model run, not just the selected one
        *[ts_card(x) for x in bundle["models"]],
    ]


# =====================================
# LAYOUT
# =====================================

MONTH_OPTIONS = [{"label": calendar.month_name[i], "value": i} for i in range(1, 13)]
YEAR_OPTIONS = [{"label": str(y), "value": y} for y in YEARS]


def select(id_, options, value, cls):
    return dcc.Dropdown(id=id_, options=options, value=value, clearable=False,
                        searchable=False, className=f"select {cls}")


def model_card(key, title, desc):
    return html.Button(
        [
            html.Div("✓", className="checkbox"),
            html.Div([html.Div(title, id=f"{key}-title", className="model-title"),
                      html.Div(desc, id=f"{key}-desc", className="model-desc")]),
        ],
        id=f"card-{key}",
        n_clicks=0,
        className="model-card selected",
    )


setup_panel = html.Div(
    [
        html.Div("Analysis setup", className="setup-title"),
        html.Div("Define your model inputs.", className="muted-12"),
        html.Div(className="divider"),
        html.Div("01   DATE RANGE", className="step"),
        html.Div("Start date", className="field-label"),
        html.Div(
            [select("start-month", MONTH_OPTIONS, FIRST % 100, "select-month"),
             select("start-year", YEAR_OPTIONS, FIRST // 100, "select-year")],
            className="date-row",
        ),
        html.Div("End date", className="field-label"),
        html.Div(
            [select("end-month", MONTH_OPTIONS, LAST % 100, "select-month"),
             select("end-year", YEAR_OPTIONS, LAST // 100, "select-year")],
            className="date-row",
        ),
        html.Div(id="obs-note", className="hint"),
        html.Div(className="divider"),
        html.Div("02   FACTORS", className="step"),
        dcc.Dropdown(id="factor-add", placeholder="Choose factors", clearable=False,
                     searchable=False, className="select select-full"),
        html.Div(id="factor-chips", className="chips"),
        html.Div("Market factor is required for CAPM.", className="hint"),
        html.Div(className="divider"),
        html.Div("03   MODELS", className="step"),
        model_card("capm", "CAPM", "Market risk"),
        model_card("multi", "Fama–French", "Three-factor model"),
        html.Button("Run analysis", id="run", n_clicks=0, className="run-btn"),
        html.Div(
            f"Monthly returns · USD · Risk-free adjusted · "
            f"{'Ken French library' if DATA_SOURCE == 'french' else 'Class Excel data'}",
            className="muted-10 footnote",
        ),
    ],
    className="setup",
)

results_panel = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Analysis results", className="results-title"),
                        html.Div(id="selection-chips", className="selections"),
                    ],
                    className="results-titles",
                ),
                html.Div(id="status"),
            ],
            className="results-header",
        ),
        dcc.Loading(
            html.Div(id="results-body", className="results-body"),
            type="dot",
            color="#087f67",
            # Only show the spinner for a full re-run, not for model switches
            target_components={"results-body": "children"},
        ),
    ],
    className="results",
)

app = Dash(
    __name__,
    title="CAPM vs Fama–French",
    # Result cards (model switcher, detail body) are created by the run callback
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"
    ],
)

app.layout = html.Div(
    [
        dcc.Store(id="factors", data=[f for f in FF3 if f in FACTORS]),
        dcc.Store(id="models", data=["capm", "multi"]),
        dcc.Store(id="bundle"),
        html.Div(
            [
                html.Header(
                    [
                        html.Div("November 2026", className="eyebrow"),
                        html.H1("FIN 394 - Project 2", className="page-title"),
                        html.Div("Comparing CAPM and Fama–French models.",
                                 className="page-subtitle"),
                    ],
                    className="page-header",
                ),
                html.Div(
                    [
                        html.Div([setup_panel, html.Div(id="portfolio-summary")], className="side"),
                        results_panel,
                    ],
                    className="columns",
                ),
            ],
            className="page",
        ),
    ],
    className="app",
)


# =====================================
# CALLBACKS
# =====================================


@app.callback(
    Output("factors", "data"),
    Output("factor-add", "value"),
    Input("factor-add", "value"),
    Input({"type": "chip-remove", "factor": ALL}, "n_clicks"),
    State("factors", "data"),
    prevent_initial_call=True,
)
def edit_factors(added, _removes, factors):
    trig = ctx.triggered_id
    if trig == "factor-add":
        if added and added not in factors:
            # Keep the data file's column order so model labels are stable
            factors = [f for f in FACTORS if f in factors or f == added]
        return factors, None
    if isinstance(trig, dict) and ctx.triggered[0]["value"]:
        return [f for f in factors if f != trig["factor"]], no_update
    return no_update, no_update


@app.callback(
    Output("factor-chips", "children"),
    Output("factor-add", "options"),
    Input("factors", "data"),
)
def render_chips(factors):
    chips = [
        html.Button(
            [html.Span(chip_label(f)), html.Span("×", className="chip-x")],
            id={"type": "chip-remove", "factor": f},
            className="factor-chip",
            title=f"Remove {f}",
            style={"minWidth": CHIP_WIDTHS[f]} if f in CHIP_WIDTHS else None,
        )
        for f in factors
    ]
    options = [{"label": chip_label(f), "value": f} for f in FACTORS if f not in factors]
    return chips, options


@app.callback(
    Output("models", "data"),
    Input("card-capm", "n_clicks"),
    Input("card-multi", "n_clicks"),
    State("models", "data"),
    prevent_initial_call=True,
)
def toggle_model(_a, _b, models):
    key = ctx.triggered_id.replace("card-", "")
    return [m for m in models if m != key] if key in models else models + [key]


@app.callback(
    Output("card-capm", "className"),
    Output("card-multi", "className"),
    Output("multi-title", "children"),
    Output("multi-desc", "children"),
    Output("obs-note", "children"),
    Output("obs-note", "className"),
    Output("run", "disabled"),
    Input("models", "data"),
    Input("factors", "data"),
    Input("start-month", "value"),
    Input("start-year", "value"),
    Input("end-month", "value"),
    Input("end-year", "value"),
)
def update_setup(models, factors, sm, sy, em, ey):
    def card_cls(key):
        return "model-card selected" if key in models else "model-card"

    if factors:
        title, desc = factor_model_name(factors), factor_model_desc(factors)
        if title == "CAPM":
            desc = "Same as CAPM — add SMB or HML"
    else:
        title, desc = "Factor model", "Choose at least one factor"

    _, _, msg, level = validate(sy, sm, ey, em, factors, models)
    hint_cls = {"ok": "hint", "warn": "hint warn", "error": "hint error"}[level]
    return card_cls("capm"), card_cls("multi"), title, desc, msg, hint_cls, level == "error"


@app.callback(
    Output("bundle", "data"),
    Output("results-body", "children"),
    Output("selection-chips", "children"),
    Output("status", "children"),
    Output("portfolio-summary", "children"),
    Input("run", "n_clicks"),
    State("start-month", "value"),
    State("start-year", "value"),
    State("end-month", "value"),
    State("end-year", "value"),
    State("factors", "data"),
    State("models", "data"),
)
def run_analysis(_n, sm, sy, em, ey, factors, models):
    def fail(msg):
        return None, no_update, no_update, html.Div(msg, className="status status-error"), no_update

    sample, specs, msg, level = validate(sy, sm, ey, em, factors, models)
    if level == "error":
        return fail(msg)

    try:
        bundle = analyse(sample, specs)
    except (ValueError, np.linalg.LinAlgError) as err:
        # run_model re-validates the data and checks matrix conditioning
        return fail(f"Analysis failed: {tidy(err)}")

    ms = bundle["models"]
    selections = [
        html.Span("Your selections", className="muted-10 strong"),
        chip(f"{month_label(bundle['start'])} – {month_label(bundle['end'])} · "
             f"{bundle['T']} months"),
        chip(" · ".join(bundle["factors"])),
        chip(" + ".join(m["name"] for m in ms)),
        chip(f"{N_PORT} Size × B/M portfolios"),
    ]
    status_children = [html.Span("✓  Complete", className="strong"),
                       html.Span(f"{len(ms)} model{'s' if len(ms) > 1 else ''} tested")]
    if level == "warn":
        status_children.append(html.Span(f"Low power: T < {LOW_POWER_MONTHS}"))
    status = html.Div(status_children, className="status")

    default_idx = len(ms) - 1  # show the multi-factor model's details first
    body = [
        html.Div([verdict_card(m) for m in ms], className="verdicts"),
        html.Div(
            [html.Span("Takeaway", className="takeaway-label"),
             html.Span(takeaway_text(ms), className="takeaway-text")],
            className="takeaway",
        ),
        html.Div(
            [
                html.Div("Detailed results", className="section-title"),
                dcc.RadioItems(
                    id="detail-model",
                    options=[{"label": m["name"], "value": i} for i, m in enumerate(ms)],
                    value=default_idx,
                    inline=True,
                    className="segmented",
                    style=None if len(ms) > 1 else {"display": "none"},
                ),
            ],
            className="section-head",
        ),
        html.Div(detailed_results(bundle, default_idx), id="detail-body", className="details"),
    ]
    return bundle, body, selections, status, portfolio_card(bundle)


@app.callback(
    Output("detail-body", "children"),
    Input("detail-model", "value"),
    State("bundle", "data"),
    prevent_initial_call=True,
)
def switch_model(idx, bundle):
    if not bundle or idx is None:
        return no_update
    return detailed_results(bundle, idx)


if __name__ == "__main__":
    # Hot reload on, but hide Dash's floating dev-tools bar so it doesn't cover the layout
    app.run(debug=True, dev_tools_ui=False)
