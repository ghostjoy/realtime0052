from __future__ import annotations

import unittest
import warnings

import pandas as pd

from backtest.metrics import compute_metrics_from_equity, compute_yearly_returns


class BacktestMetricsTests(unittest.TestCase):
    def test_compute_yearly_returns_skips_missing_years_without_futurewarning(self):
        equity = pd.Series(
            [100.0, 120.0],
            index=pd.to_datetime(["2024-12-31", "2026-12-31"], utc=True),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yearly = compute_yearly_returns(equity)

        self.assertEqual(set(yearly.keys()), {"2026"})
        self.assertAlmostEqual(yearly["2026"], 0.2)
        self.assertFalse(any(issubclass(w.category, FutureWarning) for w in caught))

    def test_compute_metrics_from_equity_handles_internal_nan_without_futurewarning(self):
        equity_curve = pd.DataFrame(
            {"equity": [100_000.0, None, 102_000.0, 103_500.0]},
            index=pd.date_range("2026-03-17", periods=4, freq="B", tz="UTC"),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics, drawdown, yearly_returns = compute_metrics_from_equity(
                equity_curve=equity_curve,
                initial_capital=100_000.0,
                trade_pnls=[1_000.0, -500.0],
            )

        self.assertAlmostEqual(metrics.total_return, 0.035)
        self.assertEqual(len(drawdown), 4)
        self.assertEqual(yearly_returns, {})
        self.assertFalse(any(issubclass(w.category, FutureWarning) for w in caught))
