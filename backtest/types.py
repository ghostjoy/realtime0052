from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    qty: float
    fee: float
    tax: float
    slippage: float
    pnl: float
    pnl_pct: float


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    avg_win: float
    avg_loss: float
    trades: int


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: List[Trade]
    metrics: BacktestMetrics
    drawdown_series: pd.Series
    yearly_returns: Dict[str, float]
    signals: pd.Series


@dataclass(frozen=True)
class PortfolioBacktestResult:
    equity_curve: pd.DataFrame
    metrics: BacktestMetrics
    drawdown_series: pd.Series
    yearly_returns: Dict[str, float]
    trades: pd.DataFrame
    signals: pd.DataFrame
    component_results: Dict[str, BacktestResult]


@dataclass(frozen=True)
class WalkForwardResult:
    split_date: pd.Timestamp
    best_params: Dict[str, float]
    train_result: BacktestResult
    test_result: BacktestResult
    objective: str
    candidates: int


@dataclass(frozen=True)
class PortfolioWalkForwardResult:
    split_date: pd.Timestamp
    symbol_results: Dict[str, WalkForwardResult]
    train_portfolio: PortfolioBacktestResult
    test_portfolio: PortfolioBacktestResult


@dataclass(frozen=True)
class RotationRebalanceRecord:
    signal_date: pd.Timestamp
    effective_date: pd.Timestamp
    market_filter_on: bool
    selected_symbols: List[str]
    weights: Dict[str, float]
    scores: Dict[str, float]


@dataclass(frozen=True)
class RotationBacktestResult:
    equity_curve: pd.DataFrame
    metrics: BacktestMetrics
    drawdown_series: pd.Series
    yearly_returns: Dict[str, float]
    weights: pd.DataFrame
    trades: pd.DataFrame
    rebalance_records: List[RotationRebalanceRecord]
