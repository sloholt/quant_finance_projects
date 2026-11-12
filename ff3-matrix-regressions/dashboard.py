"""Interactive CAPM / Fama-French dashboard (Plotly Dash).

Run:  python dashboard.py   then open http://127.0.0.1:8050

All estimation is done by asset_pricing_analysis.run_model, so the numbers here
match the command-line script for the same date range and factors.
"""

import calendar

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from asset_pricing_analysis import (
    DATE_COL,
    PORTFOLIO_COLS,
    available_factors,
    load_data,
    min_obs_required,
    run_model,
)

# =====================================
# DATA & CONSTANTS
# =====================================

DATA = load_data()
FACTORS = available_factors(DATA)
FIRST, LAST = int(DATA[DATE_COL].iloc[0]), int(DATA[DATE_COL].iloc[-1])
YEARS = list(range(FIRST // 100, LAST // 100 + 1))
FF3 = ["RMRF", "SMB", "HML"]
N_PORT = len(PORTFOLIO_COLS)
T_CRIT = 1.96
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Labels used in chips (setup panel) and in tables / charts
CHIP_LABELS = {"RMRF": "Market excess return", "SMB": "Size (SMB)", "HML": "Value (HML)"}
CHIP_WIDTHS = {"RMRF": "183px", "SMB": "109px", "HML": "117px"}  # chip widths in the Figma frame
FACTOR_LABELS = {"RMRF": "Market (RMRF)", "SMB": "Size (SMB)", "HML": "Value (HML)"}

# Design tokens (mirrors assets/dashboard.css)
INK, MUTED, GREEN, RED = "#182b36", "#62727e", "#087f67", "#b3412f"
SCALE_NEG, SCALE_MID, SCALE_POS = (61, 100, 176), (243, 245, 247), (196, 97, 43)
CAPM_COLOR, OTHER_COLORS = "#6b87bf", ["#087f67", "#c4612b", "#8a6bbf"]


def model_colors(models):
    """CAPM is always blue; other models take green, then the remaining colors."""
    others = iter(OTHER_COLORS)
    return [CAPM_COLOR if m["name"] == "CAPM" else next(others) for m in models]


def chip_label(f):
    return CHIP_LABELS.get(f, f)


def factor_label(f):
    return FACTOR_LABELS.get(f, f)


def factor_phrase(f):
    """Factor name as used mid-sentence, e.g. 'value (HML)'."""
    return {"RMRF": "the market (RMRF)", "SMB": "size (SMB)", "HML": "value (HML)"}.get(f, f)


def yyyymm(year, month):
    return int(year) * 100 + int(month)


def month_label(ym):
    return f"{calendar.month_abbr[ym % 100]} {ym // 100}"


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
    n = len(factors)
    return f"{n}-factor model · {' · '.join(factors)}"


def requested_models(models, factors):
    """Turn the two model toggles + chosen factors into [(name, factor_list)]."""
    out = []
    if "capm" in models:
        out.append(("CAPM", ["RMRF"]))
    if "multi" in models and factors and not ("capm" in models and factors == ["RMRF"]):
        out.append((factor_model_name(factors), list(factors)))
    return out


def validate(sy, sm, ey, em, factors, models):
    """Return (sample_df or None, message, is_error)."""
    start, end = yyyymm(sy, sm), yyyymm(ey, em)
    if start > end:
        return None, "Start date must be on or before the end date.", True
    if start < FIRST or end > LAST:
        return None, f"Data runs {month_label(FIRST)} – {month_label(LAST)}.", True
    sample = DATA[(DATA[DATE_COL] >= start) & (DATA[DATE_COL] <= end)]
    T = len(sample)
    specs = requested_models(models, factors)
    if not specs:
        return None, f"{T} monthly observations · select at least one model.", True
    k = max(len(f) for _, f in specs)
    need = min_obs_required(N_PORT, k)
    if T < need:
        return None, f"{T} monthly observations · GRS needs at least {need}.", True
    return sample.reset_index(drop=True), f"{T} monthly observations", False


# =====================================
# ANALYSIS (wraps run_model)
# =====================================


def analyse(sample, specs):
    """Run every requested model and return a JSON-serialisable result bundle."""
    T = len(sample)
    models = []
    summary = {}
    for name, factors in specs:
        res = run_model(sample, PORTFOLIO_COLS, factors)
        # Factor summary straight from the script (same sample => same stats per factor)
        for f, row in res["factor_summary"].iterrows():
            summary[f] = {
                "mean": float(row["Mean"]),
                "var": float(row["Variance"]),
                "std": float(row["Std Dev"]),
                "sharpe": float(row["Sharpe Ratio"]),
            }
        reg, fm = res["regression"], res["fama_macbeth"]
        alpha, t_alpha = reg["alpha"], reg["t_alpha"]
        models.append(
            {
                "name": name,
                "factors": factors,
                "alpha": alpha.tolist(),
                "t_alpha": t_alpha.tolist(),
                "beta": np.asarray(reg["beta"]).tolist(),
                "t_beta": np.asarray(reg["t_beta"]).tolist(),
                "grs": float(res["grs_stat"]),
                "p": float(res["grs_pvalue"]),
                "fcrit": float(res["grs_fcrit"]),
                "df": [N_PORT, T - N_PORT - len(factors)],
                "n_sig": int((np.abs(t_alpha) > T_CRIT).sum()),
                "mean_abs_alpha": float(np.abs(alpha).mean()),
                "fm": {
                    "terms": ["Intercept"] + factors,
                    "avg": np.asarray(fm["avg_gamma"]).tolist(),
                    "se": np.asarray(fm["se_gamma"]).tolist(),
                    "t": np.asarray(fm["t_gamma"]).tolist(),
                },
            }
        )

    used = list(dict.fromkeys(f for _, fs in specs for f in fs))
    summary = {f: summary[f] for f in used}
    return {
        "start": int(sample[DATE_COL].iloc[0]),
        "end": int(sample[DATE_COL].iloc[-1]),
        "T": T,
        "factors": used,
        "models": models,
        "summary": summary,
    }


# =====================================
# SMALL UI BUILDERS
# =====================================


def chip(text, cls="chip"):
    return html.Span(text, className=cls)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_num(x, digits=2):
    s = f"{x:+.{digits}f}" if digits else f"{x:+.0f}"
    return s.replace("-", "−")


def fmt_plain(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


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
                            chip(f"{k} factor{'s' if k > 1 else ''} · {' '.join(m['factors'])}"),
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
                    metric("F Critical-Value", f"{m['fcrit']:.2f}", "5% level"),
                    metric(
                        "p-value",
                        fmt_p(m["p"]),
                        "vs. 5% threshold",
                        "text-red" if reject else "text-green",
                    ),
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
    return (
        f"{lead} {change} average mispricing from {a_abs:.2f}% to {b_abs:.2f}% a month. {grs}"
    )


# --- Alpha heatmap ---


def cell_color(v, scale):
    t = max(-1.0, min(1.0, v / scale))
    end = SCALE_NEG if t < 0 else SCALE_POS
    rgb = [round(m + abs(t) * (e - m)) for m, e in zip(SCALE_MID, end)]
    return "#{:02x}{:02x}{:02x}".format(*rgb), abs(t) >= 0.7


def heatmap_grid(m, scale):
    alpha = np.asarray(m["alpha"]).reshape(5, 5)
    tstat = np.asarray(m["t_alpha"]).reshape(5, 5)
    col_labels = ["Low", "2", "3", "4", "High"]
    row_labels = ["Small", "2", "3", "4", "Big"]

    rows = [
        html.Div(
            [html.Div(className="hm-spacer-axis"), html.Span("Book-to-market  →   Growth … Value")],
            className="hm-axis",
        ),
        html.Div(
            [html.Div(className="hm-rowlabel")]
            + [html.Div(c, className="hm-collabel") for c in col_labels],
            className="hm-row",
        ),
    ]
    for i, rl in enumerate(row_labels):
        cells = []
        for j in range(5):
            v, t = alpha[i, j], tstat[i, j]
            bg, light = cell_color(v, scale)
            cls = "hm-cell" + (" sig" if abs(t) > T_CRIT else "") + (" light" if light else "")
            cells.append(
                html.Div(
                    fmt_num(v),
                    className=cls,
                    style={"backgroundColor": bg},
                    title=f"{PORTFOLIO_COLS[i * 5 + j]}: α = {v:.3f}%, t = {t:.2f}",
                )
            )
        rows.append(html.Div([html.Div(rl, className="hm-rowlabel")] + cells, className="hm-row"))
    rows.append(html.Div("↑ Size: Small → Big (rows)", className="hm-axis"))

    return html.Div(
        [
            html.Div(
                [
                    html.Span(m["name"], className="hm-title"),
                    html.Span(f"{m['n_sig']} of {N_PORT} significant", className="muted-11"),
                ],
                className="hm-label",
            ),
            html.Div(rows, className="hm-grid"),
        ],
        className="hm-block",
    )


def heatmap_card(models):
    max_abs = max(np.abs(m["alpha"]).max() for m in models)
    scale = max(0.1, np.ceil(max_abs * 10) / 10)
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Where does each model misprice?", className="card-title"),
                            html.Div(
                                "Time-series alpha by portfolio, % per month · "
                                "25 size × book-to-market portfolios",
                                className="muted-11",
                            ),
                        ],
                        className="titles",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(f"−{scale:.1f}%", className="muted-10"),
                                    html.Div(className="scale-bar"),
                                    html.Span(f"+{scale:.1f}%", className="muted-10"),
                                ],
                                className="legend-row",
                            ),
                            html.Div(
                                [
                                    html.Div(className="sig-key"),
                                    html.Span(
                                        "Outlined = significant (|t| > 1.96)", className="muted-10"
                                    ),
                                ],
                                className="legend-row",
                            ),
                        ],
                        className="hm-legend",
                    ),
                ],
                className="heading-row",
            ),
            html.Div([heatmap_grid(m, scale) for m in models], className="hm-wrap"),
        ],
        className="card section",
    )


# --- Fama-MacBeth risk premia chart ---


def premia_figure(models):
    terms = ["Intercept"] + list(dict.fromkeys(f for m in models for f in m["factors"]))
    labels = ["Intercept γ₀" if t == "Intercept" else factor_label(t) for t in terms]
    fig = go.Figure()
    annotations = []
    n = len(models)
    width = 0.8 / max(n, 1)
    colors = model_colors(models)

    for idx, m in enumerate(models):
        lookup = dict(zip(m["fm"]["terms"], zip(m["fm"]["avg"], m["fm"]["se"], m["fm"]["t"])))
        y = [lookup[t][0] if t in lookup else None for t in terms]
        err = [T_CRIT * lookup[t][1] if t in lookup else None for t in terms]
        fig.add_bar(
            name=m["name"],
            x=labels,
            y=y,
            error_y={"type": "data", "array": err, "color": INK, "thickness": 1.5, "width": 5},
            marker_color=colors[idx],
            customdata=[[lookup[t][2] if t in lookup else None] for t in terms],
            hovertemplate="%{x}<br>γ = %{y:.3f}%<br>t = %{customdata[0]:.2f}<extra>"
            + m["name"]
            + "</extra>",
        )
        offset = (idx - (n - 1) / 2) * width
        for k, t in enumerate(terms):
            if t not in lookup:
                annotations.append(
                    dict(x=k + offset, y=0, text="n/a", showarrow=False, yshift=10,
                         font=dict(size=9, color="#b6c2cc"))
                )
            elif t != "Intercept" and abs(lookup[t][2]) > T_CRIT:
                g, se, tv = lookup[t]
                top = g + T_CRIT * se if g >= 0 else g - T_CRIT * se
                annotations.append(
                    dict(x=k + offset, y=top, text=f"t = {tv:.1f}", showarrow=False,
                         yshift=10 if g >= 0 else -10,
                         font=dict(size=9, color=GREEN, family="Inter", weight=600))
                )

    fig.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.12,
        height=250,
        margin=dict(l=40, r=8, t=12, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(family="Inter, sans-serif", size=10, color=MUTED),
        annotations=annotations,
        hoverlabel=dict(font_family="Inter"),
    )
    fig.update_xaxes(showgrid=False, showline=False, ticks="", tickfont=dict(size=10))
    fig.update_yaxes(
        ticksuffix="%",
        gridcolor="#edf0f3",
        zeroline=True,
        zerolinecolor="#b6c2cc",
        zerolinewidth=1,
        tickfont=dict(size=10),
        tickformat="+.1~f",
    )
    return fig


def premia_caption(models):
    priced = {}
    intercept_sig = False
    for m in models:
        for term, g, t in zip(m["fm"]["terms"], m["fm"]["avg"], m["fm"]["t"]):
            if abs(t) > T_CRIT:
                if term == "Intercept":
                    intercept_sig = True
                else:
                    priced[term] = g
    if priced:
        names = join_words(factor_phrase(f) for f in priced)
        direction = "above" if all(g > 0 for g in priced.values()) else "different from"
        verb = "earns a premium" if len(priced) == 1 else "earn premia"
        first = f"Only {names} {verb} clearly {direction} zero."
        first = first[0].upper() + first[1:]
    else:
        first = "No factor earns a premium clearly different from zero."
    second = (
        " A significant intercept means betas leave some average return unexplained."
        if intercept_sig
        else " The intercept is not significant, so betas account for average returns."
    )
    return first + second


def premia_card(models):
    legend = [
        html.Div(
            [
                html.Div(className="legend-swatch", style={"backgroundColor": c}),
                html.Span(m["name"], className="muted-10"),
            ],
            className="legend-item",
        )
        for m, c in zip(models, model_colors(models))
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Which factors earn a premium?", className="card-title"),
                    html.Div(
                        "Average cross-sectional γ, % per month · whiskers = ±1.96 SE",
                        className="muted-11",
                    ),
                    html.Div(legend, className="legend"),
                ],
                className="titles titles-8",
            ),
            dcc.Graph(figure=premia_figure(models), config={"displayModeBar": False},
                      className="premia-graph"),
            html.Div(premia_caption(models), className="muted-11"),
        ],
        className="card section half",
    )


# --- Detailed results tables ---


def table(header, rows, first_col_width="150px", numeric_from=1):
    def row(cells, cls):
        return html.Div(
            [
                html.Div(c, className="t-cell t-first" if i == 0 else "t-cell t-num",
                         style={"width": first_col_width} if i == 0 else None)
                for i, c in enumerate(cells)
            ],
            className=cls,
        )

    return html.Div(
        [row(header, "t-row t-head")] + [row(r, "t-row") for r in rows], className="t-table"
    )


def factor_summary_table(bundle):
    rows = [
        [factor_label(f), fmt_plain(s["mean"]), f"{s['var']:.3g}".replace("-", "−"),
         fmt_plain(s["std"]), fmt_plain(s["sharpe"])]
        for f, s in bundle["summary"].items()
    ]
    return html.Div(
        table(["Factor", "Mean", "Variance", "Std dev", "Sharpe"], rows),
        className="tab-body",
    )


def ts_table(m):
    header = ["Portfolio", "α", "t(α)"] + [f"β {f}" for f in m["factors"]]
    rows = []
    for i, p in enumerate(PORTFOLIO_COLS):
        rows.append(
            [p, fmt_plain(m["alpha"][i], 3), fmt_plain(m["t_alpha"][i])]
            + [fmt_plain(m["beta"][k][i]) for k in range(len(m["factors"]))]
        )
    return html.Div(table(header, rows, "90px"), className="t-scroll")


def fm_table(models):
    rows = []
    for m in models:
        for term, g, se, t in zip(m["fm"]["terms"], m["fm"]["avg"], m["fm"]["se"], m["fm"]["t"]):
            rows.append(
                [m["name"], "γ₀ Intercept" if term == "Intercept" else factor_label(term),
                 fmt_plain(g, 3), fmt_plain(se, 3), fmt_plain(t)]
            )
    return html.Div(
        [
            table(["Model", "Term", "Avg γ", "SE", "t"], rows, "110px"),
            html.Div("|t| > 1.96 means the factor is priced at the 5% level.",
                     className="muted-11"),
        ],
        className="tab-body",
    )


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
        html.Div("Monthly returns · USD · Risk-free adjusted", className="muted-10 footnote"),
    ],
    className="setup",
)

hypothesis_panel = html.Div(
    [
        html.Div("Null hypothesis", className="hypothesis-title"),
        html.Div(f"H₀: α₁ = α₂ = … = α{str(N_PORT).translate(SUBSCRIPT)} = 0",
                 className="hypothesis-eq"),
        html.Div(
            f"All {N_PORT} portfolio alphas are jointly zero, so the model fully explains "
            "average excess returns. The GRS test rejects H₀ at 5% when the F-stat exceeds "
            "the F critical-value (p < 0.05).",
            className="hypothesis-text",
        ),
    ],
    className="hypothesis",
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
            color=GREEN,
            # Only show the spinner for a full re-run, not for tab switches
            target_components={"results-body": "children"},
        ),
    ],
    className="results",
)

app = Dash(
    __name__,
    title="CAPM vs Fama–French",
    # Result cards (tabs, model switcher) are created by the run callback
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
                    [html.Div([setup_panel, hypothesis_panel], className="side"), results_panel],
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

    _, msg, err = validate(sy, sm, ey, em, factors, models)
    return (card_cls("capm"), card_cls("multi"), title, desc, msg,
            "hint error" if err else "hint", err)


@app.callback(
    Output("bundle", "data"),
    Output("results-body", "children"),
    Output("selection-chips", "children"),
    Output("status", "children"),
    Input("run", "n_clicks"),
    State("start-month", "value"),
    State("start-year", "value"),
    State("end-month", "value"),
    State("end-year", "value"),
    State("factors", "data"),
    State("models", "data"),
)
def run_analysis(_n, sm, sy, em, ey, factors, models):
    sample, msg, err = validate(sy, sm, ey, em, factors, models)
    if err:
        status = html.Div(msg, className="status status-error")
        return None, no_update, no_update, status

    specs = requested_models(models, factors)
    bundle = analyse(sample, specs)
    ms = bundle["models"]

    selections = [
        html.Span("Your selections", className="muted-10 strong"),
        chip(f"{month_label(bundle['start'])} – {month_label(bundle['end'])} · "
             f"{bundle['T']} months"),
        chip(" · ".join(bundle["factors"])),
        chip(" + ".join(m["name"] for m in ms)),
        chip(f"{N_PORT} Size × B/M portfolios"),
    ]
    status = html.Div(
        [html.Span("✓  Complete", className="strong"),
         html.Span(f"{len(ms)} model{'s' if len(ms) > 1 else ''} tested")],
        className="status",
    )

    body = [
        html.Div([verdict_card(m) for m in ms], className="verdicts"),
        html.Div(
            [html.Span("Takeaway", className="takeaway-label"),
             html.Span(takeaway_text(ms), className="takeaway-text")],
            className="takeaway",
        ),
        heatmap_card(ms),
        html.Div(
            [
                premia_card(ms),
                html.Div(
                    [
                        html.Div("Detailed results", className="card-title"),
                        dcc.Tabs(
                            id="detail-tab",
                            value="summary",
                            className="tabs",
                            children=[
                                dcc.Tab(label="Factor summary", value="summary",
                                        className="tab", selected_className="tab-active"),
                                dcc.Tab(label=f"Time-series regressions ({N_PORT})", value="ts",
                                        className="tab", selected_className="tab-active"),
                                dcc.Tab(label="Fama–MacBeth", value="fm",
                                        className="tab", selected_className="tab-active"),
                            ],
                        ),
                        dcc.RadioItems(
                            id="ts-model",
                            options=[{"label": m["name"], "value": i} for i, m in enumerate(ms)],
                            value=0,
                            inline=True,
                            className="segmented",
                        ),
                        html.Div(id="detail-body"),
                    ],
                    className="card section half",
                ),
            ],
            className="bottom-row",
        ),
    ]
    return bundle, body, selections, status


@app.callback(
    Output("detail-body", "children"),
    Output("ts-model", "style"),
    Input("detail-tab", "value"),
    Input("ts-model", "value"),
    Input("bundle", "data"),
)
def render_detail(tab, model_idx, bundle):
    if not bundle:
        return no_update, no_update
    hidden, shown = {"display": "none"}, {}
    if tab == "ts":
        m = bundle["models"][model_idx or 0]
        return ts_table(m), shown if len(bundle["models"]) > 1 else hidden
    if tab == "fm":
        return fm_table(bundle["models"]), hidden
    return factor_summary_table(bundle), hidden


if __name__ == "__main__":
    # Hot reload on, but hide Dash's floating dev-tools bar so it doesn't cover the layout
    app.run(debug=True, dev_tools_ui=False)
