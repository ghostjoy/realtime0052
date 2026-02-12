from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backtest.comparison import build_buy_hold_equity, interval_return


def _bars(start: str, closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="B", tz="UTC")
    return pd.DataFrame({"close": closes}, index=idx)


class ComparisonTests(unittest.TestCase):
    def test_build_buy_hold_equity_single(self):
        bars_by_symbol = {"A": _bars("2026-01-01", [100, 110, 121])}
        idx = pd.date_range("2026-01-01", periods=3, freq="B", tz="UTC")
        eq = build_buy_hold_equity(bars_by_symbol, idx, initial_capital=1_000_000)
        self.assertAlmostEqual(float(eq.iloc[0]), 1_000_000.0, places=3)
        self.assertAlmostEqual(float(eq.iloc[-1]), 1_210_000.0, places=3)

    def test_build_buy_hold_equity_equal_weight_portfolio(self):
        bars_by_symbol = {
            "A": _bars("2026-01-01", [100, 110]),
            "B": _bars("2026-01-01", [200, 180]),
        }
        idx = pd.date_range("2026-01-01", periods=2, freq="B", tz="UTC")
        eq = build_buy_hold_equity(bars_by_symbol, idx, initial_capital=1_000_000)
        # A +10%, B -10% => portfolio 0%
        self.assertAlmostEqual(float(eq.iloc[0]), 1_000_000.0, places=3)
        self.assertAlmostEqual(float(eq.iloc[-1]), 1_000_000.0, places=3)

    def test_interval_return(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="B", tz="UTC")
        series = pd.Series([100.0, 105.0, 110.0, 108.0, 115.0], index=idx)
        out = interval_return(series, start_date=date(2026, 1, 1), end_date=date(2026, 1, 7))
        self.assertTrue(out["ok"])
        self.assertAlmostEqual(float(out["return"]), 0.15, places=6)


if __name__ == "__main__":
    unittest.main()
