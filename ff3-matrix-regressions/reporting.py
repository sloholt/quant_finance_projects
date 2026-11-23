from pathlib import Path
import numpy as np
from openpyxl import Workbook
import matplotlib.pyplot as plt
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


def format_workbook(ws):
    """Format a comparison worksheet with portfolios as columns."""

    navy = "17365D"
    light_blue = "DCEAF7"
    white = "FFFFFF"
    green = "E2F0D9"

    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "B4"

    ws.column_dimensions["A"].width = 34

    for col in range(2, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col)].width = 16

    # Main title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ws.max_column)

    title = ws.cell(1, 1)
    title.font = Font(size=16, bold=True, color=white)
    title.fill = PatternFill("solid", fgColor=navy)
    title.alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 32

    # Portfolio column headers
    for cell in ws[3]:
        cell.font = Font(bold=True, color=white)
        cell.fill = PatternFill("solid", fgColor=navy)
        cell.alignment = Alignment(horizontal="center")

    ws.row_dimensions[3].height = 25

    # Data and section formatting
    for row in ws.iter_rows(min_row=4):

        first_cell = row[0]
        is_section = first_cell.value is not None and first_cell.font.bold

        for cell in row:

            if is_section:
                cell.fill = PatternFill("solid", fgColor=navy)
                cell.font = Font(bold=True, color=white)

            else:
                if cell.row % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor=light_blue)

                if isinstance(cell.value, (int, float, np.number)):
                    cell.number_format = "0.000000"
                    cell.alignment = Alignment(horizontal="right")

                    if "p-value" in str(first_cell.value).lower() and cell.value < 0.05:
                        cell.fill = PatternFill("solid", fgColor=green)


def export_results_to_excel(model_results, factors, portfolio, output_path=None):
    """
    Export model results with portfolios as columns
    and statistics as rows on one worksheet.
    """

    factor_set = set(factors)

    if factor_set == {"RMRF"}:
        model_name = "CAPM"
    elif factor_set == {"RMRF", "SMB", "HML"}:
        model_name = "FF3"
    elif factor_set == {"RMRF", "SMB", "HML", "MOM"}:
        model_name = "Carhart4"
    else:
        model_name = "_".join(sorted(factors))

    if output_path is None:
        output_path = f"results/{model_name}_Asset_Pricing_Report.xlsx"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = model_name

    ws["A1"] = f"{model_name} ASSET PRICING ANALYSIS"

    # Portfolio column headers
    ws.cell(3, 1, "Statistic")

    for col_idx, name in enumerate(portfolio, 2):
        ws.cell(3, col_idx, name)

    row = 4
    N = len(portfolio)

    # Portfolio Summary
    ws.cell(row, 1, "PORTFOLIO RAW RETURN SUMMARY STATISTICS")
    ws.cell(row, 1).font = Font(bold=True)
    row += 1

    summary = model_results["portfolio_summary"]

    for metric in summary.columns:
        ws.cell(row, 1, metric)

        for i, name in enumerate(portfolio):
            ws.cell(row, i + 2, float(summary.loc[name, metric]))

        row += 1

    row += 2

    # Time-Series Regression
    ws.cell(row, 1, "TIME-SERIES REGRESSION")
    ws.cell(row, 1).font = Font(bold=True)
    row += 1

    regression = model_results["regression"]

    # Alpha and alpha t-statistics
    alphas = np.asarray(regression["alpha"]).reshape(-1)
    t_alphas = np.asarray(regression["t_alpha"]).reshape(-1)

    # Betas and beta t-statistics
    # Shape: (K factors, N portfolios)
    betas = np.asarray(regression["beta"])
    t_betas = np.asarray(regression["t_beta"])

    # Alpha
    ws.cell(row, 1, "Alpha")
    for i in range(N):
        ws.cell(row, i + 2, float(alphas[i]))
    row += 1

    # Factor betas
    for j, factor in enumerate(factors):
        ws.cell(row, 1, f"Beta_{factor}")

        for i in range(N):
            ws.cell(row, i + 2, float(betas[j, i]))

        row += 1

    # Alpha t-statistics
    ws.cell(row, 1, "t_Alpha")
    for i in range(N):
        ws.cell(row, i + 2, float(t_alphas[i]))
    row += 1

    # Factor beta t-statistics
    for j, factor in enumerate(factors):
        ws.cell(row, 1, f"t_Beta_{factor}")

        for i in range(N):
            ws.cell(row, i + 2, float(t_betas[j, i]))

        row += 1

    row += 2

    # GRS Test
    ws.cell(row, 1, "GRS TEST (MODEL-WIDE)")
    ws.cell(row, 1).font = Font(bold=True)
    row += 1

    grs_results = {
        "GRS F-Statistic": model_results["grs_stat"],
        "GRS P-Value": model_results["grs_pvalue"],
        "GRS Critical Value": model_results["grs_fcrit"],
        "Reject Null at 5%": "Yes" if model_results["grs_pvalue"] < 0.05 else "No",
    }

    for metric, value in grs_results.items():
        ws.cell(row, 1, metric)
        ws.cell(row, 2, value)
        row += 1

    # One-Shot Cross-Sectional
    row += 2

    ws.cell(row, 1, "ONE-SHOT CROSS-SECTIONAL REGRESSION")
    ws.cell(row, 1).font = Font(bold=True)
    row += 1

    one_shot = model_results["one_shot"]

    # Intercept and factor risk premia
    gamma_labels = ["Intercept"] + factors

    for label, value in zip(gamma_labels, one_shot["gamma"]):
        ws.cell(row, 1, f"Gamma_{label}")
        ws.cell(row, 2, float(value))
        row += 1

    # Model R-squared
    ws.cell(row, 1, "Cross-Sectional R²")
    ws.cell(row, 2, float(one_shot["r_squared"]))
    row += 2

    # Portfolio-level fitted returns and pricing errors
    ws.cell(row, 1, "FITTED RETURNS AND PRICING ERRORS")
    ws.cell(row, 1).font = Font(bold=True)
    row += 1

    for metric, values in {
        "Fitted Excess Return": one_shot["fitted"],
        "Pricing Error (Residual)": one_shot["residuals"],
    }.items():
        ws.cell(row, 1, metric)

        for i, value in enumerate(values):
            ws.cell(row, i + 2, float(value))

        row += 1

    # Model Consistency Checks
    row += 2

    ws.cell(row, 1, "MODEL CONSISTENCY CHECKS")
    ws.cell(row, 1).font = Font(bold=True)
    row += 1

    consistency = model_results["consistency"]

    for i, factor in enumerate(factors):
        metrics = {
            f"{factor} Factor Mean": consistency["factor_means"][i],
            f"{factor} One-Shot Gamma": consistency["one_shot_premia"][i],
            f"{factor} One-Shot Difference": consistency["one_shot_diff"][i],
            f"{factor} FM Gamma": consistency["fm_premia"][i],
            f"{factor} FM Difference": consistency["fm_diff"][i],
        }

        for label, value in metrics.items():
            ws.cell(row, 1, label)
            ws.cell(row, 2, float(value))
            row += 1

    row += 1

    intercept_metrics = {
        "One-Shot Pricing Intercept": consistency["one_shot_intercept"],
        "FM Pricing Intercept": consistency["fm_intercept"],
        "FM Intercept t-statistic": consistency["fm_intercept_t"],
    }

    for label, value in intercept_metrics.items():
        ws.cell(row, 1, label)
        ws.cell(row, 2, float(value))
        row += 1

    format_workbook(ws)
    wb.save(output_path)

    print(f"Excel report updated: {output_path.resolve()}")


def plot_portfolio_returns(rawrets, portfolio_cols):
    """
    Replicate Excel exploratory data analysis chart:
    Average monthly raw returns for 25 size/BM portfolios.
    """
    # Calculate expected returns (equivalent to EXPRET)
    expected_returns = np.asarray(rawrets.mean(axis=0)).reshape(-1)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(portfolio_cols, expected_returns, color="#3E6385")

    ax.set_title("Average Monthly Portfolio Returns")
    ax.set_xlabel("Portfolio")
    ax.set_ylabel("Average Monthly Return (%)")

    ax.set_xticks(range(len(portfolio_cols)))
    ax.set_xticklabels(portfolio_cols, rotation=90, fontsize=8)

    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


def plot_sml(betas, expected_returns, portfolio_cols):
    """
    Plot the Security Market Line (SML) analysis:
    CAPM betas vs. average monthly portfolio returns.
    """
    betas = np.asarray(betas, dtype=float).reshape(-1)
    expected_returns = np.asarray(expected_returns, dtype=float).reshape(-1)

    if len(betas) != len(expected_returns) or len(betas) != len(portfolio_cols):
        raise ValueError(
            "Betas, returns, and portfolio labels must have matching lengths."
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Scatter plot
    ax.scatter(betas, expected_returns, color="#3E6385", s=60)

    ax.set_title("Security Market Line (SML) Analysis")
    ax.set_xlabel("Portfolio Beta")
    ax.set_ylabel("Average Monthly Return (%)")

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


def generate_summary(results, model_name):
    """Generate deterministic model interpretation."""

    p_value = results["grs_pvalue"]
    f_stat = results["grs_stat"]

    if p_value < 0.05:
        conclusion = (
            "The GRS test rejects the hypothesis that "
            "all portfolio alphas are jointly zero."
        )
    else:
        conclusion = (
            "The GRS test fails to reject the hypothesis "
            "that all portfolio alphas are jointly zero."
        )

    return (
        f"\n{model_name} Summary:\n"
        f"GRS F-statistic: {f_stat:.4f}\n"
        f"GRS p-value: {p_value:.4g}\n"
        f"{conclusion}"
    )


def generate_summary(results, model_name):
    """Generate deterministic model interpretation."""

    p_value = results["grs_pvalue"]
    f_stat = results["grs_stat"]

    if p_value < 0.05:
        conclusion = (
            "The GRS test rejects the hypothesis that "
            "all portfolio alphas are jointly zero."
        )
    else:
        conclusion = (
            "The GRS test fails to reject the hypothesis "
            "that all portfolio alphas are jointly zero."
        )

    return (
        f"\n{model_name} Summary:\n"
        f"GRS F-statistic: {f_stat:.4f}\n"
        f"GRS p-value: {p_value:.4g}\n"
        f"{conclusion}"
    )


def display_results(model_name, results, output_tables):
    """Display all statistical results for a model."""

    tables = output_tables[model_name]

    print(f"\n{'=' * 60}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'=' * 60}")

    print("\nFactor Summary Statistics:")
    print(tables["Factor Summary"].round(4).to_string())

    print("\nPortfolio Raw Return Summary:")
    print(results["portfolio_summary"].round(6).to_string())

    for name in [
        "Regression",
        "GRS",
        "One-Shot Cross-Sectional",
        "Fama-MacBeth",
        "Consistency",
    ]:
        print(f"\n{name}:")
        print(tables[name].round(4).to_string())

    print(f"\nCross-sectional R²: " f"{results['one_shot']['r_squared']:.4f}")

    consistency = results["consistency"]

    print(f"\nOne-Shot Pricing Intercept: " f"{consistency['one_shot_intercept']:.6f}")
    print(f"Fama-MacBeth Pricing Intercept: " f"{consistency['fm_intercept']:.6f}")
    print(
        f"Fama-MacBeth Intercept t-statistic: " f"{consistency['fm_intercept_t']:.4f}"
    )


def export_validation_table(
    part1_diffs,
    class_diffs,
    output_path="results/Table_6_Validation.xlsx",
):
    """Export both validation comparisons to Table 6."""

    from pathlib import Path
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Table 6 Validation"

    ws.merge_cells("A1:C2")
    ws["A1"] = "Table 6: Numerical Validation"

    headers = [
        "Statistic",
        "Part 1 vs Python (30 years)",
        "Class Table A2 vs Python (196302–199301)",
    ]

    for col, header in enumerate(headers, 1):
        ws.cell(4, col, header)

    row_mapping = [
        (
            "Regression coefficients",
            "Regression coefficients",
            None,
        ),
        (
            "Standard errors",
            "Standard errors",
            "Fama-MacBeth standard error",
        ),
        (
            "t-statistics",
            "t-statistics",
            "Fama-MacBeth t-statistic",
        ),
        (
            "GRS F-statistic and p-value",
            "GRS F statistic and p-value",
            "GRS statistic and p-value",
        ),
        (
            "GRS critical value",
            None,
            "GRS critical value",
        ),
        (
            "One-shot gammas and standard errors",
            "One-shot gammas and standard errors",
            "One-shot gamma and standard error",
        ),
        (
            "Fama-MacBeth gammas",
            "Fama-MacBeth gammas, standard errors, t-statistics",
            "Fama-MacBeth gamma",
        ),
        (
            "Cross-sectional R-squared",
            None,
            "Cross-sectional R-squared",
        ),
        (
            "Factor means",
            None,
            "Factor means",
        ),
    ]

    tolerance = 1e-6

    for row, (label, part1_key, class_key) in enumerate(row_mapping, 5):
        ws.cell(row, 1, label)

        for col, diffs, key in [
            (2, part1_diffs, part1_key),
            (3, class_diffs, class_key),
        ]:
            value = diffs.get(key) if key else None
            cell = ws.cell(row, col)

            if value is None:
                cell.value = "N/A"
            else:
                value = abs(float(value))
                cell.value = value
                cell.number_format = "0.000000000E+00"

                cell.fill = PatternFill(
                    "solid",
                    fgColor=("E2F0D9" if value < tolerance else "FCE8E6"),
                )

    last_row = 4 + len(row_mapping)

    ws.cell(last_row + 2, 1, "Required tolerance")
    ws.cell(last_row + 2, 2, tolerance)
    ws.cell(last_row + 2, 2).number_format = "0.0E+00"

    ws.merge_cells(
        start_row=last_row + 4,
        start_column=1,
        end_row=last_row + 4,
        end_column=3,
    )

    note = ws.cell(last_row + 4, 1)
    note.value = (
        "Class Table A2 values are rounded; differences against "
        "displayed values do not establish full-precision agreement. "
        "Investigate material discrepancies before attributing them "
        "to library or data revisions."
    )
    note.alignment = Alignment(
        wrap_text=True,
        vertical="center",
    )
    ws.row_dimensions[last_row + 4].height = 42

    navy = "17365D"

    for row in [1, 4]:
        for cell in ws[row][:3]:
            cell.fill = PatternFill(
                "solid",
                fgColor=navy,
            )
            cell.font = Font(
                bold=True,
                color="FFFFFF",
            )
            cell.alignment = Alignment(
                wrap_text=True,
                vertical="center",
            )

    ws["A1"].font = Font(
        size=15,
        bold=True,
        color="FFFFFF",
    )

    ws.column_dimensions["A"].width = 45
    ws.column_dimensions["B"].width = 32
    ws.column_dimensions["C"].width = 43

    ws.row_dimensions[4].height = 42
    ws.freeze_panes = "B5"
    ws.sheet_view.showGridLines = False

    wb.save(output_path)

    print(f"\nValidation table saved: {output_path.resolve()}")

    return output_path
