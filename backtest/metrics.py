from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

import numpy as np
import pandas as pd

from backtest.types import BacktestMetrics


def compute_yearly_returns(equity: pd.Series) -> dict[str, float]:
    yearly = equity.resample("YE").last().pct_change().dropna()
    return {str(idx.year): float(val) for idx, val in yearly.items()}


def compute_metrics_from_equity(
    equity_curve: pd.DataFrame,
    initial_capital: float,
    trade_pnls: Iterable[float],
) -> tuple[BacktestMetrics, pd.Series, dict[str, float]]:
    returns = equity_curve["equity"].pct_change().fillna(0.0)
    running_max = equity_curve["equity"].cummax()
    drawdown = equity_curve["equity"] / running_max - 1.0
    years = max((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25, 1 / 365.25)
    total_return = equity_curve["equity"].iloc[-1] / initial_capital - 1.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    sharpe = (returns.mean() / returns.std() * sqrt(252.0)) if returns.std() > 0 else 0.0

    pnl_values = list(trade_pnls)
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]
    win_rate = (len(wins) / len(pnl_values)) if pnl_values else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    metrics = BacktestMetrics(
        total_return=float(total_return),
        cagr=float(cagr),
        max_drawdown=float(drawdown.min()),
        sharpe=float(sharpe),
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        trades=len(pnl_values),
    )
    return metrics, drawdown, compute_yearly_returns(equity_curve["equity"])
