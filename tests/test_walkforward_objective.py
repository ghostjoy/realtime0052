from __future__ import annotations

import unittest
from dataclasses import dataclass

from backtest.walkforward import _score, required_walkforward_bars


@dataclass
class _Metrics:
    cagr: float
    total_return: float
    max_drawdown: float
    sharpe: float


@dataclass
class _Result:
    metrics: _Metrics


class WalkForwardObjectiveTests(unittest.TestCase):
    def test_mdd_prefers_smaller_drawdown(self):
        better = _Result(metrics=_Metrics(cagr=0.0, total_return=0.0, max_drawdown=-0.10, sharpe=0.0))
        worse = _Result(metrics=_Metrics(cagr=0.0, total_return=0.0, max_drawdown=-0.30, sharpe=0.0))
        self.assertGreater(_score(better, "mdd"), _score(worse, "mdd"))

    def test_required_walkforward_bars_increases_for_daily_k_strategy(self):
        base = required_walkforward_bars("sma_cross", train_ratio=0.7)
        daily_k = required_walkforward_bars("sma_trend_filter", train_ratio=0.7)
        self.assertGreater(daily_k, base)


if __name__ == "__main__":
    unittest.main()
