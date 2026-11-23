# Asset Pricing Analysis

This project tests CAPM, Fama–French three-factor, and other selected factor models (including momentum) on 25 size/book-to-market portfolios. It uses NumPy matrix calculations for time-series regressions, the Gibbons–Ross–Shanken (GRS) test, one-shot cross-sectional regressions, and Fama–MacBeth estimation. Results are displayed in the terminal, exported to Excel, and available through an optional dashboard.

## Setup and running

From the project root, using Python 3.10+:  

```bash
pip install -r requirements.txt
```

To launch the interactive dashboard:  

```bash
python dashboard.py
```
Open `http://127.0.0.1:8050`. The dashboard uses the same calculations and expects `assets/dashboard.css` for its styling.  

To view the command-line only model:  
```bash
python asset_pricing_analysis.py
```

The script downloads monthly data from [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), then prompts for factor combinations and a start/end month (`YYYYMM`). Available factors are `RMRF`, `SMB`, `HML`, and `MOM`. Results are written to `results/`. No machine-specific absolute paths are required.  

To run automated tests:

```bash
python -m unittest discover -s tests -v
```

## Architecture and design notes

| File | Purpose |
|---|---|
| `data_loader.py` | Download, combine, and clean factor and portfolio data; support Excel inputs. |
| `validation.py` | Check dates, factors, missing values, sample sizes, and matrix conditioning. |
| `asset_pricing_analysis.py` | Run regressions, GRS tests, cross-sectional analysis, and model comparisons. |
| `reporting.py` | Print results, create plots, and export Excel reports. |
| `dashboard.py` | Interactive interface using the same analysis functions. |

**Data flow:** Load data → validate inputs → calculate excess returns → estimate factor models → test joint alphas and risk premia → report results.

I initially built the calculations step by step from the in-class CAPM Excel workbook, using matrix algebra to reproduce the workbook's results. After studying Fama–French, I refactored the CAPM-specific code into reusable functions that accept different factor combinations while preserving the original Excel calculation sequence. I separated loading, validation, estimation, and reporting so the code would be easier to test and the dashboard could reuse the same results. I used NumPy rather than a regression package to keep the calculations transparent and directly comparable to Excel.

**Next improvements:** Make dashboard exports optional rather than automatic and extend potential portfolio options and risk factors for increased flexibility and customization. 

## Hypotheses

For the GRS joint-alpha test:

- **H₀:** All 25 portfolio alphas are zero; the selected factor model has no joint pricing errors.
- **H₁:** At least one portfolio alpha is nonzero.

At the 5% significance level, a p-value below 0.05 leads to rejection of H₀.

## Contribution statement

- **Analysis and implementation:** Sloane — [Specify who implemented data processing, model calculations, and integration].
- **Testing and validation:** Sloane — [Specify who wrote and ran tests and checked results against Excel].
- **Reporting and dashboard:** Sloane — [Specify who implemented exports, plots, and the dashboard].
- **AI assistance:** See Appendix II of report 
