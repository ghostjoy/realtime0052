from __future__ import annotations

import pandas as pd

from backtest.engine import CostModel, run_backtest
from backtest.metrics import compute_metrics_from_equity
from backtest.types import BacktestResult, PortfolioBacktestResult


def _aggregate_component_equity(
    component_results: dict[str, BacktestResult],
    per_symbol_capital: float,
) -> pd.DataFrame:
    union_index = None
    for result in component_results.values():
        if union_index is None:
            union_index = result.equity_curve.index
        else:
            union_index = union_index.union(result.equity_curve.index)
    if union_index is None or len(union_index) == 0:
        return pd.DataFrame(columns=["equity"])
    union_index = union_index.sort_values()

    total = pd.Series(0.0, index=union_index, dtype=float)
    for result in component_results.values():
        series = (
            result.equity_curve["equity"].reindex(union_index).ffill().fillna(per_symbol_capital)
        )
        total = total + series
    return pd.DataFrame({"equity": total})


def portfolio_from_components(
    component_results: dict[str, BacktestResult],
    initial_capital: float,
    per_symbol_capital: float,
) -> PortfolioBacktestResult:
    if not component_results:
        raise ValueError("component_results cannot be empty")

    equity_curve = _aggregate_component_equity(component_results, per_symbol_capital)
    metrics, drawdown, yearly_returns = compute_metrics_from_equity(
        equity_curve=equity_curve,
        initial_capital=initial_capital,
        trade_pnls=[trade.pnl for result in component_results.values() for trade in result.trades],
    )

    trades_rows = []
    for symbol, result in component_results.items():
        for trade in result.trades:
            trades_rows.append(
                {
                    "symbol": symbol,
                    "entry_date": trade.entry_date,
                    "entry_price": trade.entry_price,
                    "exit_date": trade.exit_date,
                    "exit_price": trade.exit_price,
                    "qty": trade.qty,
                    "fee": trade.fee,
                    "tax": trade.tax,
                    "slippage": trade.slippage,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                }
            )
    trades = pd.DataFrame(trades_rows)

    signals = pd.DataFrame(
        {
            symbol: result.signals.reindex(equity_curve.index).ffill().fillna(0).astype(int)
            for symbol, result in component_results.items()
        },
        index=equity_curve.index,
    )

    return PortfolioBacktestResult(
        equity_curve=equity_curve,
        metrics=metrics,
        drawdown_series=drawdown,
        yearly_returns=yearly_returns,
        trades=trades,
        signals=signals,
        component_results=component_results,
    )


def run_portfolio_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_name: str,
    strategy_params: dict[str, float] | None = None,
    cost_model: CostModel | None = None,
    initial_capital: float = 1_000_000.0,
) -> PortfolioBacktestResult:
    if not bars_by_symbol:
        raise ValueError("bars_by_symbol cannot be empty")

    valid = {
        symbol: bars
        for symbol, bars in bars_by_symbol.items()
        if bars is not None and not bars.empty
    }
    if not valid:
        raise ValueError("all symbols have empty bars")

    per_symbol_capital = initial_capital / float(len(valid))
    component_results: dict[str, BacktestResult] = {}
    for symbol, bars in valid.items():
        component_results[symbol] = run_backtest(
            bars=bars,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            cost_model=cost_model,
            initial_capital=per_symbol_capital,
        )

    return portfolio_from_components(
        component_results=component_results,
        initial_capital=initial_capital,
        per_symbol_capital=per_symbol_capital,
    )
