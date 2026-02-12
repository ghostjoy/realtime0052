from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest.metrics import compute_metrics_from_equity
from backtest.strategy_library import STRATEGIES
from backtest.types import BacktestResult, Trade


@dataclass(frozen=True)
class CostModel:
    fee_rate: float = 0.001425
    sell_tax_rate: float = 0.003
    slippage_rate: float = 0.0005


class BacktestEngine:
    def run(
        self,
        bars: pd.DataFrame,
        strategy_name: str,
        strategy_params: Optional[Dict[str, float]] = None,
        cost_model: Optional[CostModel] = None,
        initial_capital: float = 1_000_000.0,
    ) -> BacktestResult:
        if strategy_name not in STRATEGIES:
            raise ValueError(f"unsupported strategy: {strategy_name}")
        min_bars = 2 if strategy_name == "buy_hold" else 40
        if len(bars) < min_bars:
            raise ValueError(f"not enough bars for backtest ({strategy_name} need >= {min_bars})")
        if not {"open", "high", "low", "close"}.issubset(set(bars.columns)):
            raise ValueError("bars must include open/high/low/close")

        strategy_params = strategy_params or {}
        cost_model = cost_model or CostModel()
        frame = bars.sort_index().copy()
        frame = frame.dropna(subset=["open", "high", "low", "close"], how="any")

        signal_fn = STRATEGIES[strategy_name]
        signals = signal_fn(frame, **strategy_params).astype(int).reindex(frame.index).fillna(0)
        signals = signals.clip(lower=0, upper=1)

        cash = float(initial_capital)
        qty = 0.0
        entry_price = 0.0
        entry_notional = 0.0
        entry_date = None
        entry_fee = 0.0
        entry_slip = 0.0
        trades: list[Trade] = []
        equity_rows = []

        for i, dt in enumerate(frame.index):
            open_px = float(frame["open"].iloc[i])
            close_px = float(frame["close"].iloc[i])
            target = int(signals.iloc[i - 1]) if i > 0 else 0

            if target == 1 and qty == 0 and open_px > 0:
                px = open_px * (1.0 + cost_model.slippage_rate)
                entry_slip = open_px * cost_model.slippage_rate
                qty = cash / (px * (1.0 + cost_model.fee_rate)) if px > 0 else 0.0
                entry_notional = qty * px
                fee = entry_notional * cost_model.fee_rate
                cash -= qty * px + fee
                entry_fee = fee
                entry_price = px
                entry_date = pd.Timestamp(dt)
            elif target == 0 and qty > 0:
                px = open_px * (1.0 - cost_model.slippage_rate)
                gross = qty * px
                fee = gross * cost_model.fee_rate
                tax = gross * cost_model.sell_tax_rate
                cash += gross - fee - tax
                pnl = (gross - fee - tax) - entry_notional - entry_fee
                denom = entry_notional if entry_notional > 0 else np.nan
                pnl_pct = pnl / denom if denom and not np.isnan(denom) else 0.0
                trades.append(
                    Trade(
                        entry_date=entry_date or pd.Timestamp(dt),
                        entry_price=entry_price,
                        exit_date=pd.Timestamp(dt),
                        exit_price=px,
                        qty=qty,
                        fee=entry_fee + fee,
                        tax=tax,
                        slippage=entry_slip + (open_px * cost_model.slippage_rate),
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                )
                qty = 0.0
                entry_price = 0.0
                entry_notional = 0.0
                entry_date = None
                entry_fee = 0.0
                entry_slip = 0.0

            equity = cash + qty * close_px
            equity_rows.append({"date": pd.Timestamp(dt), "equity": equity, "position": qty})

        if qty > 0:
            dt = frame.index[-1]
            close_px = float(frame["close"].iloc[-1])
            px = close_px * (1.0 - cost_model.slippage_rate)
            gross = qty * px
            fee = gross * cost_model.fee_rate
            tax = gross * cost_model.sell_tax_rate
            cash += gross - fee - tax
            pnl = (gross - fee - tax) - entry_notional - entry_fee
            denom = entry_notional if entry_notional > 0 else np.nan
            pnl_pct = pnl / denom if denom and not np.isnan(denom) else 0.0
            trades.append(
                Trade(
                    entry_date=entry_date or pd.Timestamp(dt),
                    entry_price=entry_price,
                    exit_date=pd.Timestamp(dt),
                    exit_price=px,
                    qty=qty,
                    fee=entry_fee + fee,
                    tax=tax,
                    slippage=entry_slip + (close_px * cost_model.slippage_rate),
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            )
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["position"] = 0.0

        equity_curve = pd.DataFrame(equity_rows).set_index("date")
        metrics, drawdown, yearly_returns = compute_metrics_from_equity(
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            trade_pnls=[t.pnl for t in trades],
        )
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            drawdown_series=drawdown,
            yearly_returns=yearly_returns,
            signals=signals,
        )


def run_backtest(
    bars: pd.DataFrame,
    strategy_name: str,
    strategy_params: Optional[Dict[str, float]] = None,
    cost_model: Optional[CostModel] = None,
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    engine = BacktestEngine()
    return engine.run(
        bars=bars,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        cost_model=cost_model,
        initial_capital=initial_capital,
    )
