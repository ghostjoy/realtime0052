from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backtest.comparison import (
    build_buy_hold_equity,
    build_dca_benchmark_equity,
    build_dca_contribution_plan,
    build_dca_equity,
    dca_summary_metrics,
    interval_return,
)


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

    def test_build_dca_contribution_plan(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-02", tz="UTC"),
                pd.Timestamp("2026-01-05", tz="UTC"),
                pd.Timestamp("2026-02-02", tz="UTC"),
                pd.Timestamp("2026-03-02", tz="UTC"),
            ]
        )
        plan = build_dca_contribution_plan(idx)
        self.assertEqual(pd.Timestamp(plan["initial_date"]), pd.Timestamp("2026-01-02", tz="UTC"))
        monthly_dates = list(plan["monthly_dates"])
        self.assertEqual(
            monthly_dates,
            [pd.Timestamp("2026-02-02", tz="UTC"), pd.Timestamp("2026-03-02", tz="UTC")],
        )

    def test_build_dca_equity_single_lump_sum_only(self):
        bars_by_symbol = {"A": _bars("2026-01-01", [100, 110, 121])}
        idx = pd.date_range("2026-01-01", periods=3, freq="B", tz="UTC")
        eq = build_dca_equity(
            bars_by_symbol=bars_by_symbol,
            target_index=idx,
            initial_lump_sum=1_000_000.0,
            monthly_contribution=0.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        self.assertAlmostEqual(float(eq.iloc[0]), 1_000_000.0, places=3)
        self.assertAlmostEqual(float(eq.iloc[-1]), 1_210_000.0, places=3)

    def test_build_dca_equity_single_with_monthly_contribution(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-02", tz="UTC"),
                pd.Timestamp("2026-01-05", tz="UTC"),
                pd.Timestamp("2026-02-02", tz="UTC"),
                pd.Timestamp("2026-02-03", tz="UTC"),
            ]
        )
        bars_by_symbol = {"A": pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0]}, index=idx)}
        eq = build_dca_equity(
            bars_by_symbol=bars_by_symbol,
            target_index=idx,
            initial_lump_sum=1_000.0,
            monthly_contribution=100.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        self.assertAlmostEqual(float(eq.iloc[-1]), 1_100.0, places=3)

    def test_build_dca_equity_equal_weight_portfolio(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-02", tz="UTC"),
                pd.Timestamp("2026-02-02", tz="UTC"),
            ]
        )
        bars_by_symbol = {
            "A": pd.DataFrame({"close": [100.0, 110.0]}, index=idx),
            "B": pd.DataFrame({"close": [100.0, 90.0]}, index=idx),
        }
        eq = build_dca_equity(
            bars_by_symbol=bars_by_symbol,
            target_index=idx,
            initial_lump_sum=1_000.0,
            monthly_contribution=100.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        self.assertAlmostEqual(float(eq.iloc[-1]), 1_100.0, places=3)

    def test_build_dca_equity_applies_fee_and_slippage(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="UTC")])
        bars_by_symbol = {"A": pd.DataFrame({"close": [100.0]}, index=idx)}
        eq = build_dca_equity(
            bars_by_symbol=bars_by_symbol,
            target_index=idx,
            initial_lump_sum=1_000.0,
            monthly_contribution=0.0,
            fee_rate=0.001,
            slippage_rate=0.01,
        )
        self.assertLess(float(eq.iloc[-1]), 1_000.0)

    def test_build_dca_equity_handles_missing_close_as_cash(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-02", tz="UTC"),
                pd.Timestamp("2026-02-02", tz="UTC"),
            ]
        )
        bars_by_symbol = {
            "A": pd.DataFrame({"close": [100.0, 100.0]}, index=idx),
            "B": pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2026-02-02", tz="UTC")]),
        }
        eq = build_dca_equity(
            bars_by_symbol=bars_by_symbol,
            target_index=idx,
            initial_lump_sum=1_000.0,
            monthly_contribution=0.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        self.assertAlmostEqual(float(eq.iloc[-1]), 1_000.0, places=3)

    def test_build_dca_benchmark_equity_same_schedule(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-02", tz="UTC"),
                pd.Timestamp("2026-02-02", tz="UTC"),
            ]
        )
        benchmark_close = pd.Series([100.0, 100.0], index=idx)
        eq = build_dca_benchmark_equity(
            benchmark_close=benchmark_close,
            target_index=idx,
            initial_lump_sum=1_000.0,
            monthly_contribution=100.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        self.assertAlmostEqual(float(eq.iloc[-1]), 1_100.0, places=3)

    def test_dca_summary_metrics_and_excess_sign(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-02", tz="UTC"),
                pd.Timestamp("2026-02-02", tz="UTC"),
            ]
        )
        dca_equity = pd.Series([1_000.0, 1_200.0], index=idx)
        bench_equity = pd.Series([1_000.0, 1_100.0], index=idx)
        dca_m = dca_summary_metrics(dca_equity, total_contribution=1_000.0)
        bench_m = dca_summary_metrics(bench_equity, total_contribution=1_000.0)
        self.assertAlmostEqual(float(dca_m["total_return"]), 0.2, places=6)
        self.assertAlmostEqual(float(bench_m["total_return"]), 0.1, places=6)
        excess_pct = (float(dca_m["total_return"]) - float(bench_m["total_return"])) * 100.0
        self.assertGreater(excess_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
