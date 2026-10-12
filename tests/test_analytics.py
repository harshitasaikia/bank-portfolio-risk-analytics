import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analytics import prepare_returns, performance, calculate_risk
from statistics import NormalDist


class AnalyticsTests(unittest.TestCase):
    def test_weight_labels_and_missing_prices(self):
        prices = pd.DataFrame({'B': [100, 110, 121, 133.1, 146.41], 'A': [100, 90, np.nan, 81, 72.9]})
        returns, weights = prepare_returns(prices, {'A': 0.75, 'B': 0.25})
        self.assertEqual(list(returns.index), [1, 4])
        np.testing.assert_allclose(returns.dot(weights), [-0.05, -0.05])

    def test_drawdown_includes_initial_capital_and_exact_log(self):
        returns = pd.DataFrame({'A': [-0.1, 0.05], 'B': [0.0, -0.05]})
        stats, series = performance(returns, pd.Series({'A': 0.5, 'B': 0.5}), pd.Series([0.01, -0.01]), 0)
        self.assertAlmostEqual(stats.max_drawdown, 0.05)
        np.testing.assert_allclose(series.log_return, np.log1p(series.simple_return))
        self.assertAlmostEqual(stats.sortino_ratio, -np.sqrt(126))

    def test_covariance_and_monte_carlo_convergence(self):
        rng = np.random.default_rng(123)
        a = rng.normal(0.001, 0.02, 5000)
        returns = pd.DataFrame({'A': a, 'B': 0.8*a + rng.normal(0, 0.005, len(a))})
        weights = {'B': 0.3, 'A': 0.7}
        risk, losses = calculate_risk(returns, weights, simulations=300_000)
        portfolio = returns.dot(pd.Series(weights))
        analytical = risk[(risk.method == 'variance_covariance') & (risk.confidence == 0.99)].iloc[0]
        expected = 1_000_000*(NormalDist().inv_cdf(0.99)*portfolio.std() - portfolio.mean())
        self.assertAlmostEqual(analytical.var_dollars, expected, places=7)
        simulated = risk[(risk.method == 'monte_carlo') & (risk.confidence == 0.99)].iloc[0]
        self.assertLess(abs(simulated.var_dollars / expected - 1), 0.02)
        self.assertTrue((risk.expected_shortfall_dollars >= risk.var_dollars).all())
        _, repeat = calculate_risk(returns, weights, simulations=300_000)
        np.testing.assert_array_equal(losses, repeat)

    def test_historical_quantile_and_fractional_tail(self):
        returns = pd.DataFrame({'A': [-0.04, -0.03, -0.02, -0.01, 0]})
        risk, _ = calculate_risk(returns, {'A': 1}, portfolio_value=100, confidences=[0.7], simulations=1000)
        row = risk[risk.method == 'historical'].iloc[0]
        self.assertAlmostEqual(row.var_dollars, 2.8)
        self.assertAlmostEqual(row.expected_shortfall_dollars, (4 + 0.5*3)/1.5)

    def test_invalid_inputs_and_zero_risk(self):
        prices = pd.DataFrame({'A': [100, 100, 100]})
        with self.assertRaises(ValueError):
            prepare_returns(prices, {'A': 0.5})
        returns, weights = prepare_returns(prices, {'A': 1})
        risk, _ = calculate_risk(returns, weights, simulations=1000)
        np.testing.assert_allclose(risk.var_dollars, 0)
        stats, _ = performance(returns, weights, pd.Series([0, 0, 0]), 0)
        self.assertTrue(np.isnan(stats.sharpe_ratio))
        self.assertTrue(np.isnan(stats.beta))


if __name__ == '__main__':
    unittest.main()
