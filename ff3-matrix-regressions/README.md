# CAPM Tests on 25 Size/Book-to-Market Portfolios
 
This script tests the CAPM on the 25 Fama-French portfolios sorted on size (ME) and book-to-market (BM). It uses matrix algebra in NumPy instead of a regression package, and it reproduces an Excel implementation of the same steps.
 
It runs three things:
 
1. **Time-series regressions.** Each portfolio's excess return is regressed on the market excess return to estimate its alpha and beta.
2. **GRS F-test.** Tests whether all 25 alphas are jointly zero.
3. **Fama-MacBeth regressions.** Tests whether beta is priced in the cross-section of returns.
## Requirements
 
```bash
pip install numpy pandas scipy openpyxl
```

## Data  

ME1 holds the smallest stocks and ME5 the largest. BM1 is the lowest book-to-market (growth) and BM5 the highest (value). Column names are stripped of surrounding spaces before use.

## Usage 

```bash 
python capm_analysis.py
```
Everything prints to the console. The script doesn't currently write any files.

## Method 
**Notation:** T = months, N = portfolios, K = factors (K = 1 for the CAPM).  

| Symbol | Shape | Meaning |
|---|---|---|
| R | T × N | Portfolio excess returns |
| F | T × K | Factor returns (market excess return) |
| X = [1, F] | T × (K+1) | Regressors with an intercept column |
| B | (K+1) × N | Coefficients: row 0 = alphas, row 1 = betas |
| E | T × N | Residuals |  

| Step | Function | Formula |
|---|---|---|
| 1. Excess returns | `excess_returns` | R = RAWRET − rf·1' |
| 2. Summary stats (raw returns) | `summary_stats` | mean, var (ddof=1), sd; Sharpe = (mean − mean(rf)) / sd |
| 3. Time-series OLS | `time_series_regression` | B = (X'X)⁻¹X'R, E = R − XB, SE = √(diag((X'X)⁻¹) · e'e/(T−K−1)), t = B/SE |
| 4. Residual covariance | `resid_covmat` | Σ = Ec'Ec / (T−1), where Ec = E − mean(E) |
| 5. GRS test | `grs_test` | (T−N−K)/N · α'Σ⁻¹α / (1 + mean(RMRF)²/var(RMRF)) ~ F(N, T−N−K) |
| 6. Fama-MacBeth | `get_gamma`, `capm_cross_sectional` | γ_t = monthly slope of R_t on β; γ̄ = mean(γ_t), SE = std(γ_t)/√T |
 
The GRS test rejects H₀ (all alphas = 0) when F is greater than the 5% critical value, which is evidence against the CAPM. In the Fama-MacBeth step, a |t(γ̄)| below about 2 means beta isn't priced.