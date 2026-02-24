from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.rotation import run_tw_etf_rotation_backtest


def _make_bars(n: int = 320, drift: float = 0.0008, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    noise = rng.normal(0.0, 0.0003, n)
    daily_ret = drift + noise
    close = 100.0 * np.cumprod(1.0 + daily_ret)
    open_ = close * (1.0 - 0.0005)
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class RotationBacktestTests(unittest.TestCase):
    def test_market_filter_off_stays_in_cash(self):
        bars_map = {
            "A": _make_bars(drift=0.0010, seed=1),
            "B": _make_bars(drift=0.0009, seed=2),
            "C": _make_bars(drift=0.0008, seed=3),
        }
        benchmark = _make_bars(drift=-0.0010, seed=99)
        result = run_tw_etf_rotation_backtest(
            bars_by_symbol=bars_map,
            benchmark_bars=benchmark,
            top_n=3,
            initial_capital=1_000_000.0,
        )
        self.assertTrue(all(len(rec.selected_symbols) == 0 for rec in result.rebalance_records))
        self.assertAlmostEqual(float(result.equity_curve["equity"].iloc[0]), 1_000_000.0, places=6)
        self.assertAlmostEqual(float(result.equity_curve["equity"].iloc[-1]), 1_000_000.0, places=6)

    def test_ranking_selects_top_n_by_momentum(self):
        bars_map = {
            "S1": _make_bars(drift=0.0016, seed=11),
            "S2": _make_bars(drift=0.0013, seed=12),
            "S3": _make_bars(drift=0.0011, seed=13),
            "S4": _make_bars(drift=0.0004, seed=14),
            "S5": _make_bars(drift=-0.0002, seed=15),
            "S6": _make_bars(drift=-0.0003, seed=16),
        }
        benchmark = _make_bars(drift=0.0010, seed=101)
        result = run_tw_etf_rotation_backtest(
            bars_by_symbol=bars_map,
            benchmark_bars=benchmark,
            top_n=3,
            initial_capital=1_000_000.0,
        )
        first_pick = next(
            (
                rec
                for rec in result.rebalance_records
                if rec.market_filter_on and rec.selected_symbols
            ),
            None,
        )
        self.assertIsNotNone(first_pick)
        assert first_pick is not None
        self.assertEqual(first_pick.selected_symbols, ["S1", "S2", "S3"])
        self.assertTrue(all(abs(w - (1.0 / 3.0)) < 1e-9 for w in first_pick.weights.values()))

    def test_rebalance_executes_on_effective_date_open(self):
        bars_map = {
            "S1": _make_bars(drift=0.0016, seed=21),
            "S2": _make_bars(drift=0.0012, seed=22),
            "S3": _make_bars(drift=0.0010, seed=23),
        }
        benchmark = _make_bars(drift=0.0009, seed=102)
        result = run_tw_etf_rotation_backtest(
            bars_by_symbol=bars_map,
            benchmark_bars=benchmark,
            top_n=2,
            initial_capital=1_000_000.0,
        )
        first_pick = next(
            (
                rec
                for rec in result.rebalance_records
                if rec.market_filter_on and rec.selected_symbols
            ),
            None,
        )
        self.assertIsNotNone(first_pick)
        assert first_pick is not None
        self.assertFalse(result.trades.empty)
        trades = result.trades.copy()
        trades["date"] = pd.to_datetime(trades["date"], utc=True)
        first_trade_date = pd.Timestamp(trades["date"].min())
        self.assertEqual(first_trade_date, first_pick.effective_date)

    def test_requires_minimum_bars(self):
        bars_map = {
            "A": _make_bars(n=100, drift=0.0010, seed=31),
            "B": _make_bars(n=100, drift=0.0008, seed=32),
            "C": _make_bars(n=100, drift=0.0006, seed=33),
        }
        benchmark = _make_bars(n=100, drift=0.0008, seed=111)
        with self.assertRaises(ValueError):
            run_tw_etf_rotation_backtest(
                bars_by_symbol=bars_map,
                benchmark_bars=benchmark,
                top_n=3,
                initial_capital=1_000_000.0,
            )


if __name__ == "__main__":
    unittest.main()
