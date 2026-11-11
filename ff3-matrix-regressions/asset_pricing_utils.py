import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f, linregress

OUTPUT_DIR_NAME = "."
FILE_NAME = "data/asset_pricing_class_data.xlsx"

# current script location
script_directory = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(script_directory, OUTPUT_DIR_NAME)

file_path = os.path.join(output_dir, FILE_NAME)
# create directory if missing
os.makedirs(output_dir, exist_ok=True)

RISK_FACTORS = ["RMRF", "SMB", "HML"]

# --- Helper Functions ---


def as_2d(a):
    """Turns a 1D vector into a 2D array to simplify transposing later"""
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def add_constant(F):
    """Adds a column of 1s to the beginning of a matrix to include the intercept later
    X = [1, F]  (T x K+1)."""
    F = as_2d(F)
    return np.column_stack([np.ones(F.shape[0]), F])


def excess_returns(rawrets, rf):
    """ret = Rawret - rf*1
    Note: Use rf*1 because multiplying rf by 1' expands each month's risk-free rate across
    all asset columns to match the dimensions of rawret.
    """
    return as_2d(rawrets) - as_2d(rf)


def summary_stats(vec, rfvec=None):
    """Mean, sample variance, std dev, and Sharpe ratio of the inputted vector"""
    vec = as_2d(vec)
    mean = vec.mean(axis=0)
    var = vec.var(axis=0, ddof=1)
    sd = np.sqrt(var)
    rfrate = 0 if rfvec is None else np.asarray(rfvec).mean()

    return {
        "mean": mean,
        "var": var,
        "std": sd,
        "rf rate": rfrate,
        "sr": (mean - rfrate) / sd,
    }


def resid_covmat(E, num_factors):
    """Calculate residuals, demeaned residuals, and residual covariance matrix.
    E = R - XB                      Residuals (T x N)
    Ec = E - mean(E)                Demeaned residuals (T x N)
    Sigma = Ec' Ec / (T-1)          Residual covariance matrix (N x N)

    """
    # Dimensionality checks:
    E = as_2d(E)

    T = E.shape[0]

    emat = E - E.mean(axis=0)
    ecovmat = emat.T @ emat / (T - num_factors - 1)

    return {"resids": E, "residemat": emat, "residcovmat": ecovmat}


# --- Time Series Regressions ----


def time_series_regression(R, F):
    """Run N time-series OLS regressions simultaneously.

    B    = (X'X)^-1 X'R           Coefficient matrix (K+1 x N)
    E    = R - XB                  Residual matrix (T x N)
    s2   = sum(e_i^2)/(T-K-1)     Residual variance for each asset
    SE   = sqrt(diag((X'X)^-1) outer s2)

    """
    # Ensure inputs are 2D arrays
    R, F = as_2d(R), as_2d(F)

    # Number of observations (T) and factors (K)
    T, K = F.shape

    # Add intercept column to factor matrix
    X = add_constant(F)

    # Estimate OLS coefficients
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    XtX_inv = np.linalg.inv(X.T @ X)

    E = R - X @ B

    # Estimate residual variance for each asset
    s2 = (E**2).sum(axis=0) / (T - K - 1)

    # Calculate standard errors and t-statistics
    SE = np.sqrt(np.outer(np.diag(XtX_inv), s2))
    tstat = B / SE
    # Return regression results as a dictionary
    return {
        "alpha": B[0],
        "beta": B[1:],
        "se_alpha": SE[0],
        "se_beta": SE[1:],
        "t_alpha": tstat[0],
        "t_beta": tstat[1:],
        "resids": E,
    }


# --- Extraction Functions ---


def get_alphas(regression_results):
    """Return the CAPM alpha estimates as a row vector"""
    return regression_results["alpha"].reshape(1, -1)


def get_betas(regression_results):
    """Return factor betas as a K x N matrix."""
    return regression_results["beta"]


def get_gamma(R, betas):
    """Calculate cross-sectional risk premia for each time period."""
    R = as_2d(R)
    betas = as_2d(betas)
    X = add_constant(betas.T)
    gamma = np.linalg.lstsq(X, R.T, rcond=None)[0].T
    return gamma
