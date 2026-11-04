# Portfolio Optimization

## Overview

This project builds and tests mean-variance (Markowitz) portfolios for 11 SPDR
sector ETFs (XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLK, XLB, XLRE, XLU) using 96
months of returns (Sep 2018 – Aug 2026). It compares the optimized portfolios
against a client's existing portfolio.

- **Part 1 – In-sample MVO:** uses all 96 months to find the global minimum
  variance portfolio, the efficient portfolio at the client's return, the
  optimal risky (max-Sharpe) portfolio, the efficient frontier, and the capital
  allocation line.
- **Part 2 – Rolling out-of-sample test:** asks whether portfolios built only
  from *past* data beat the client on months they have never seen. 12 times, it
  estimates on 60 months and holds the weights for the next 3, giving 36 unseen
  months (Sep 2023 – Aug 2026). It tests 12 portfolio variants: raw vs.
  CAPM-shrunk means, sample vs. Ledoit-Wolf covariance, and unconstrained vs.
  long-only vs. 20% cap. It then compares them with the client and with 10,000
  random portfolios.

**Assumed knowledge:** you should be comfortable with basic portfolio theory
(expected return, variance, Sharpe ratio, efficient frontier) and know what
CAPM beta is. Ledoit-Wolf shrinkage and condition numbers are explained briefly
in the notebook.

### Project layout

```
portfolio_optimization/
├── combined_calculations.ipynb      # Part 1 + Part 2 together, with explanations
├── in_sample_mvo.xlsx               # Excel version of the in-sample work
├── data/
│   ├── sector_etf_returns__96_mos_.xlsx   # monthly returns (sheet "Returns")
│   └── client_weights.json                # last client weights entered in Part 1
├── in-sample-calculations/
│   ├── calculations.py              # Part 1 script
│   └── *.png                        # saved Part 1 charts
└── out-of-sample-calculations/
    ├── rolling_oos_calculations.py  # Part 2 script
    └── *.png                        # saved Part 2 charts
```

### TODOs

- Add `__pycache__/` to `.gitignore` and decide whether `data/client_weights.json`
  should be committed.
- Add a `requirements.txt`.
- Have the scripts save their charts automatically (right now they only call
  `plt.show()`; the PNGs were saved by hand).
- Make the notebook read and write `data/client_weights.json` like the scripts do.

### Gotchas

- **Part 1 asks for input.** `calculations.py` prompts for a weight for each
  ticker. Pressing **Enter** at the first prompt gives a random long-only
  portfolio (Dirichlet, seed 42). This is the client used for the results in
  the notebook.
- **Part 2 reuses Part 1's client.** Every time you enter weights in Part 1 they
  are saved to `data/client_weights.json`, and Part 2 loads that file. If the
  file doesn't exist, Part 2 asks for the weights itself. To change the client,
  run Part 1 again (or delete the file).
- **Folder names have hyphens.** `in-sample-calculations` can't be imported as a
  normal package, so the Part 2 script adds it to `sys.path` to import from
  `calculations.py`. Keep the folder structure as it is.
- **The notebook loads data differently.** It was written for Google Colab
  (`drive.mount(...)`) and reads the Excel file by its bare name. Put
  `sector_etf_returns__96_mos_.xlsx` in the notebook's working directory, or
  edit `FILE`. Skip the `drive.mount` cell when running locally.
- **Unconstrained portfolios are extreme.** Some take gross exposure above 10x
  and individual weights above 800%. This is expected, and it is part of what
  the out-of-sample test shows.
- **Part 2 takes a little while.** It runs SLSQP optimizations and 20,000-draw
  random solver checks for every window.

## Setup / Installation

Requires Python 3.10+ (tested on 3.13).

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl jupyter
```

`openpyxl` is needed for pandas to read the `.xlsx` data file.

## Running

You can run the scripts from any folder, because they find the data file
relative to their own location.

**Part 1 (in-sample):**

```bash
python in-sample-calculations/calculations.py
```

Enter the client's weights when prompted, or press Enter for the random
portfolio. Then confirm. The script prints each portfolio's metrics, shows the
efficient frontier / CAL charts, and saves the weights for Part 2.

**Part 2 (rolling out-of-sample):**

```bash
python out-of-sample-calculations/rolling_oos_calculations.py
```

It prints:
- Table 2: out-of-sample results
- in-sample vs. out-of-sample Sharpe
- Table 3: comparison with random clients
- Table 4: condition numbers

Then it shows three charts: growth of $1, the Sharpe-ratio histogram, and
weights by quarter.

**Notebook:**

```bash
jupyter notebook combined_calculations.ipynb
```

Run the cells top to bottom. Part 1 must run before Part 2, because Part 2 uses
`returns`, `TICKERS` and `client_weights` from Part 1.
