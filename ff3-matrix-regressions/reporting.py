from pathlib import Path
import numpy as np
from openpyxl import Workbook
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

    # Identify model
    if factors == ["RMRF"]:
        model_name = "CAPM"
    elif set(factors) == {"RMRF", "SMB", "HML"}:
        model_name = "FF3"
    else:
        model_name = f"{len(factors)}_Factor"

    if output_path is None:
        output_path = f"results/{model_name}_Asset_Pricing_Report.xlsx"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = model_name

    # Workbook title
    ws["A1"] = f"{model_name} ASSET PRICING ANALYSIS"

    # Portfolio column headers
    ws.cell(3, 1, "Statistic")

    for col_idx, name in enumerate(portfolio, 2):
        ws.cell(3, col_idx, name)

    row = 4
    N = len(portfolio)

    # -------------------------
    # Portfolio Summary
    # -------------------------
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

    # -------------------------
    # Time-Series Regression
    # -------------------------
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

    # -------------------------
    # GRS Test
    # -------------------------
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

    # -------------------------
    # One-Shot Cross-Sectional Regression
    # -------------------------
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

        # -------------------------
    # Model Consistency Checks
    # -------------------------
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

    # Format and save
    format_workbook(ws)
    wb.save(output_path)

    print(f"Excel report updated: {output_path.resolve()}")
