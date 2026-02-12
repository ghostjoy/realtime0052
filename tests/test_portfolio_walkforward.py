from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.engine import CostModel
from backtest.portfolio import run_portfolio_backtest
from backtest.walkforward import walk_forward_portfolio, walk_forward_single


def _make_bars(seed: int = 1, n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    drift = np.linspace(80.0, 140.0, n)
    noise = rng.normal(0, 1.0, n).cumsum() * 0.2
    close = drift + noise
    open_ = close * (1.0 - 0.001)
    high = close * (1.0 + 0.005)
    low = close * (1.0 - 0.005)
    volume = np.full(n, 1_500_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class PortfolioWalkForwardTests(unittest.TestCase):
    def test_portfolio_backtest_returns_component_metrics(self):
        bars_map = {"AAA": _make_bars(seed=1), "BBB": _make_bars(seed=2)}
        result = run_portfolio_backtest(
            bars_by_symbol=bars_map,
            strategy_name="sma_cross",
            strategy_params={"fast": 10.0, "slow": 30.0},
            cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
            initial_capital=200_000.0,
        )
        self.assertEqual(set(result.component_results.keys()), {"AAA", "BBB"})
        self.assertFalse(result.equity_curve.empty)
        self.assertGreater(len(result.signals.columns), 1)

    def test_walk_forward_single_returns_split_and_params(self):
        bars = _make_bars(seed=11)
        result = walk_forward_single(
            bars=bars,
            strategy_name="ema_cross",
            cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
            train_ratio=0.7,
            objective="sharpe",
            initial_capital=100_000.0,
        )
        self.assertIsInstance(result.best_params, dict)
        self.assertGreater(result.candidates, 0)
        self.assertFalse(result.train_result.equity_curve.empty)
        self.assertFalse(result.test_result.equity_curve.empty)

    def test_walk_forward_portfolio_runs(self):
        bars_map = {"AAA": _make_bars(seed=5), "BBB": _make_bars(seed=6)}
        result = walk_forward_portfolio(
            bars_by_symbol=bars_map,
            strategy_name="sma_cross",
            cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
            train_ratio=0.65,
            objective="cagr",
            initial_capital=200_000.0,
        )
        self.assertEqual(set(result.symbol_results.keys()), {"AAA", "BBB"})
        self.assertFalse(result.train_portfolio.equity_curve.empty)
        self.assertFalse(result.test_portfolio.equity_curve.empty)


if __name__ == "__main__":
    unittest.main()
