import unittest
import numpy as np

from asset_pricing_analysis import (
    time_series_regression,
    get_betas,
    get_gamma,
    cross_sectional,
)


class TestSimulation(unittest.TestCase):

    def test_risk_premium_recovery(self):
        rng = np.random.default_rng(42)

        T = 5000
        N = 25
        K = 2

        # Known true monthly factor risk premiums (%)
        true_premia = np.array([0.6, 0.3])

        # Simulate factor returns with known means
        F = rng.normal(loc=true_premia, scale=[4.0, 2.0], size=(T, K))

        # Generate sufficiently varied portfolio betas
        true_betas = rng.uniform(0.5, 1.5, size=(K, N))

        # Simulate idiosyncratic return noise
        noise = rng.normal(0, 0.5, size=(T, N))

        # Generate excess returns from the known model
        R = F @ true_betas + noise

        # Estimate time-series betas
        regression = time_series_regression(R, F)
        estimated_betas = get_betas(regression)

        # Estimate Fama-MacBeth risk premiums
        gamma = get_gamma(R, estimated_betas)
        fm_results = cross_sectional(gamma)

        estimated_premia = fm_results["avg_gamma"][1:]

        # Verify beta recovery
        np.testing.assert_allclose(estimated_betas, true_betas, atol=0.05, rtol=0)

        # Verify risk premium recovery
        np.testing.assert_allclose(estimated_premia, true_premia, atol=0.10, rtol=0)

        print("\nTrue risk premiums:", true_premia)
        print("Estimated risk premiums:", estimated_premia)


if __name__ == "__main__":
    unittest.main()
