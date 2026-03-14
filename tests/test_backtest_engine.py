from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, CostModel


class BacktestEngineTests(unittest.TestCase):
    def _sample_bars(self, n: int = 80) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        trend = np.linspace(100.0, 140.0, n)
        noise = np.sin(np.linspace(0, 12, n))
        close = trend + noise
        open_ = close * 0.998
        high = close * 1.005
        low = close * 0.995
        volume = np.full(n, 1_000_000)
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )

    def test_run_backtest_with_custom_strategy(self):
        bars = self._sample_bars()

        def custom_signal(frame: pd.DataFrame):
            signal = pd.Series(0, index=frame.index, dtype=int)
            signal.iloc[10:35] = 1
            signal.iloc[45:70] = 1
            return signal

        with (
            patch("backtest.engine.STRATEGIES", {"test_strategy": custom_signal}),
            patch("backtest.engine.get_strategy_min_bars", return_value=2),
        ):
            result = BacktestEngine().run(
                bars=bars,
                strategy_name="test_strategy",
                strategy_params={},
                cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
                initial_capital=100_000.0,
            )

        self.assertGreaterEqual(len(result.trades), 2)
        self.assertEqual(len(result.signals), len(bars))
        self.assertFalse(result.equity_curve.empty)
        self.assertIsInstance(result.yearly_returns, dict)

    def test_requires_minimum_bars(self):
        bars = self._sample_bars(n=20)
        with self.assertRaises(ValueError):
            BacktestEngine().run(bars=bars, strategy_name="sma_cross")

    def test_buy_hold_allows_short_sample(self):
        bars = self._sample_bars(n=5)
        result = BacktestEngine().run(
            bars=bars,
            strategy_name="buy_hold",
            strategy_params={},
            cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
            initial_capital=100_000.0,
        )
        self.assertFalse(result.equity_curve.empty)
        self.assertEqual(len(result.signals), len(bars))
        self.assertGreaterEqual(len(result.trades), 1)

    def test_buy_hold_requires_two_bars(self):
        bars = self._sample_bars(n=1)
        with self.assertRaises(ValueError):
            BacktestEngine().run(bars=bars, strategy_name="buy_hold")

    def test_signal_filter_only_gates_entry(self):
        bars = self._sample_bars(n=20)

        def custom_signal(frame: pd.DataFrame):
            signal = pd.Series(0, index=frame.index, dtype=int)
            signal.iloc[2:10] = 1
            return signal

        gate = pd.Series(0, index=bars.index, dtype=int)
        gate.iloc[4:] = 1
        with (
            patch("backtest.engine.STRATEGIES", {"test_strategy": custom_signal}),
            patch("backtest.engine.get_strategy_min_bars", return_value=2),
        ):
            result = BacktestEngine().run(
                bars=bars,
                strategy_name="test_strategy",
                strategy_params={},
                cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
                initial_capital=100_000.0,
                signal_filter=gate,
            )

        self.assertEqual(int(result.signals.iloc[2]), 0)
        self.assertEqual(int(result.signals.iloc[4]), 1)
        self.assertEqual(int(result.signals.iloc[9]), 1)
        self.assertEqual(int(result.signals.iloc[10]), 0)

    def test_daily_k_strategies_can_run(self):
        bars = self._sample_bars(n=220)
        for strategy_name, params in [
            ("sma_trend_filter", {"fast": 20.0, "slow": 60.0, "trend": 120.0}),
            ("donchian_breakout", {"entry_n": 55.0, "exit_n": 20.0, "trend": 120.0}),
        ]:
            result = BacktestEngine().run(
                bars=bars,
                strategy_name=strategy_name,
                strategy_params=params,
                cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
                initial_capital=100_000.0,
            )
            self.assertFalse(result.equity_curve.empty)
            self.assertEqual(len(result.signals), len(bars))

    def test_daily_k_strategy_requires_longer_sample(self):
        bars = self._sample_bars(n=80)
        with self.assertRaises(ValueError):
            BacktestEngine().run(
                bars=bars,
                strategy_name="sma_trend_filter",
                strategy_params={"fast": 20.0, "slow": 60.0, "trend": 120.0},
            )


if __name__ == "__main__":
    unittest.main()
