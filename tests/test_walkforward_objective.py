from __future__ import annotations

import unittest
from dataclasses import dataclass

from backtest.walkforward import _score


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


if __name__ == "__main__":
    unittest.main()
