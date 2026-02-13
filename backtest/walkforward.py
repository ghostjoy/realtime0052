from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import pandas as pd

from backtest.engine import CostModel, run_backtest
from backtest.portfolio import portfolio_from_components
from backtest.strategy_library import get_strategy_min_bars
from backtest.types import PortfolioWalkForwardResult, WalkForwardResult


def strategy_param_grid(strategy_name: str) -> list[Dict[str, float]]:
    if strategy_name in {"sma_cross", "ema_cross"}:
        fast_values = [5.0, 10.0, 15.0, 20.0]
        slow_values = [20.0, 30.0, 50.0, 80.0, 120.0]
        return [{"fast": f, "slow": s} for f in fast_values for s in slow_values if f < s]
    if strategy_name == "sma_trend_filter":
        fast_values = [10.0, 20.0, 30.0]
        slow_values = [50.0, 80.0, 120.0]
        trend_values = [120.0, 160.0, 200.0]
        return [
            {"fast": f, "slow": s, "trend": t}
            for f in fast_values
            for s in slow_values
            for t in trend_values
            if f < s < t
        ]
    if strategy_name == "donchian_breakout":
        entry_values = [20.0, 40.0, 55.0, 80.0]
        exit_values = [10.0, 15.0, 20.0, 30.0]
        trend_values = [100.0, 120.0, 160.0]
        return [
            {"entry_n": e, "exit_n": x, "trend": t}
            for e in entry_values
            for x in exit_values
            for t in trend_values
            if x < e
        ]
    if strategy_name == "rsi_reversion":
        buy_values = [20.0, 25.0, 30.0, 35.0]
        sell_values = [50.0, 55.0, 60.0, 65.0, 70.0]
        return [{"buy_below": b, "sell_above": s} for b in buy_values for s in sell_values if b < s]
    return [{}]


def _score(result, objective: str) -> float:
    if objective == "cagr":
        return result.metrics.cagr
    if objective == "total_return":
        return result.metrics.total_return
    if objective == "mdd":
        return result.metrics.max_drawdown
    return result.metrics.sharpe


def required_walkforward_bars(strategy_name: str, train_ratio: float = 0.7) -> int:
    ratio = float(train_ratio)
    ratio = min(max(ratio, 0.4), 0.9)
    strategy_min = max(2, int(get_strategy_min_bars(strategy_name)))
    req_train = int(math.ceil(strategy_min / ratio))
    req_test = int(math.ceil(max(strategy_min - 1, 1) / max(1.0 - ratio, 0.01)))
    return max(80, req_train, req_test)


def walk_forward_single(
    bars: pd.DataFrame,
    strategy_name: str,
    cost_model: Optional[CostModel] = None,
    train_ratio: float = 0.7,
    objective: str = "sharpe",
    initial_capital: float = 1_000_000.0,
    custom_grid: Optional[Iterable[Dict[str, float]]] = None,
) -> WalkForwardResult:
    if train_ratio <= 0.4 or train_ratio >= 0.9:
        raise ValueError("train_ratio must be between 0.4 and 0.9")
    required_bars = required_walkforward_bars(strategy_name, train_ratio)
    if len(bars) < required_bars:
        raise ValueError(f"not enough bars for walk-forward ({strategy_name} need >= {required_bars})")

    frame = bars.sort_index().dropna(subset=["open", "high", "low", "close"], how="any")
    split_idx = int(len(frame) * train_ratio)
    strategy_min = max(2, int(get_strategy_min_bars(strategy_name)))
    if split_idx < 40 or len(frame) - split_idx < 30:
        raise ValueError("invalid train/test split for the available sample")
    if split_idx < strategy_min or (len(frame) - split_idx + 1) < strategy_min:
        raise ValueError(
            "invalid train/test split for strategy minimum sample "
            f"(train >= {strategy_min}, test >= {strategy_min})"
        )

    train = frame.iloc[:split_idx]
    test = frame.iloc[split_idx - 1 :]
    split_date = pd.Timestamp(frame.index[split_idx])

    candidates = list(custom_grid) if custom_grid is not None else strategy_param_grid(strategy_name)
    best_params: Dict[str, float] = {}
    best_train = None
    best_score = float("-inf")

    for params in candidates:
        try:
            train_result = run_backtest(
                bars=train,
                strategy_name=strategy_name,
                strategy_params=params,
                cost_model=cost_model,
                initial_capital=initial_capital,
            )
        except Exception:
            continue
        sc = _score(train_result, objective)
        if sc > best_score:
            best_score = sc
            best_params = dict(params)
            best_train = train_result

    if best_train is None:
        raise ValueError("walk-forward failed: no valid parameter candidate")

    test_result = run_backtest(
        bars=test,
        strategy_name=strategy_name,
        strategy_params=best_params,
        cost_model=cost_model,
        initial_capital=initial_capital,
    )

    return WalkForwardResult(
        split_date=split_date,
        best_params=best_params,
        train_result=best_train,
        test_result=test_result,
        objective=objective,
        candidates=len(candidates),
    )


def walk_forward_portfolio(
    bars_by_symbol: Dict[str, pd.DataFrame],
    strategy_name: str,
    cost_model: Optional[CostModel] = None,
    train_ratio: float = 0.7,
    objective: str = "sharpe",
    initial_capital: float = 1_000_000.0,
) -> PortfolioWalkForwardResult:
    if not bars_by_symbol:
        raise ValueError("bars_by_symbol cannot be empty")

    valid = {symbol: bars for symbol, bars in bars_by_symbol.items() if bars is not None and not bars.empty}
    if not valid:
        raise ValueError("all symbols have empty bars")

    capital_per_symbol = initial_capital / float(len(valid))
    symbol_results: Dict[str, WalkForwardResult] = {}
    split_date = None
    for symbol, bars in valid.items():
        wf = walk_forward_single(
            bars=bars,
            strategy_name=strategy_name,
            cost_model=cost_model,
            train_ratio=train_ratio,
            objective=objective,
            initial_capital=capital_per_symbol,
        )
        symbol_results[symbol] = wf
        if split_date is None:
            split_date = wf.split_date
        elif wf.split_date < split_date:
            split_date = wf.split_date

    train_component = {symbol: wf.train_result for symbol, wf in symbol_results.items()}
    test_component = {symbol: wf.test_result for symbol, wf in symbol_results.items()}
    train_portfolio = portfolio_from_components(
        component_results=train_component,
        initial_capital=initial_capital,
        per_symbol_capital=capital_per_symbol,
    )
    test_portfolio = portfolio_from_components(
        component_results=test_component,
        initial_capital=initial_capital,
        per_symbol_capital=capital_per_symbol,
    )

    return PortfolioWalkForwardResult(
        split_date=split_date or pd.Timestamp.utcnow(),
        symbol_results=symbol_results,
        train_portfolio=train_portfolio,
        test_portfolio=test_portfolio,
    )
