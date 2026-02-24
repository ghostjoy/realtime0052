from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from backtest.engine import CostModel
from backtest.metrics import compute_metrics_from_equity
from backtest.types import RotationBacktestResult, RotationRebalanceRecord

ROTATION_DEFAULT_UNIVERSE = ["0050", "0052", "00935", "0056", "00878", "00919"]
ROTATION_MIN_BARS = 130


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close"]
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return pd.DataFrame(columns=required + ["volume"])
    frame = bars.copy().sort_index()
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    if "adj close" in frame.columns and "adj_close" not in frame.columns:
        frame = frame.rename(columns={"adj close": "adj_close"})
    if "close" not in frame.columns:
        return pd.DataFrame(columns=required + ["volume"])

    close = pd.to_numeric(frame["close"], errors="coerce")
    for col in ["open", "high", "low"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(close)
        else:
            frame[col] = close
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    else:
        frame["volume"] = 0.0
    frame["close"] = close
    frame = frame.dropna(subset=required, how="any")
    return frame[required + ["volume"]]


def _coerce_symbols(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, bars in bars_by_symbol.items():
        sym = str(symbol or "").strip().upper()
        if not sym:
            continue
        frame = _normalize_bars(bars)
        if frame.empty:
            continue
        out[sym] = frame
    return out


def _union_index(frames: Iterable[pd.DataFrame]) -> pd.DatetimeIndex:
    idx: pd.DatetimeIndex | None = None
    for frame in frames:
        if frame.empty:
            continue
        if idx is None:
            idx = pd.DatetimeIndex(frame.index)
        else:
            idx = idx.union(pd.DatetimeIndex(frame.index))
    if idx is None:
        return pd.DatetimeIndex([])
    return idx.sort_values()


def _monthly_first_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if len(index) == 0:
        return pd.DatetimeIndex([])
    marker = pd.Series(index=index, data=1)
    if marker.index.tz is not None:
        periods = marker.index.tz_localize(None).to_period("M")
    else:
        periods = marker.index.to_period("M")
    return marker.groupby(periods).head(1).index


def run_tw_etf_rotation_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    *,
    top_n: int = 3,
    initial_capital: float = 1_000_000.0,
    cost_model: CostModel | None = None,
    momentum_w20: float = 0.2,
    momentum_w60: float = 0.5,
    momentum_w120: float = 0.3,
    require_positive_momentum: bool = True,
    require_above_sma60: bool = True,
) -> RotationBacktestResult:
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    symbols_bars = _coerce_symbols(bars_by_symbol)
    if not symbols_bars:
        raise ValueError("bars_by_symbol cannot be empty")

    benchmark = _normalize_bars(benchmark_bars)
    if len(benchmark) < 60:
        raise ValueError("benchmark bars are insufficient (need >= 60)")

    min_symbol_bars = min(len(frame) for frame in symbols_bars.values())
    if min_symbol_bars < ROTATION_MIN_BARS:
        raise ValueError(f"not enough bars for rotation backtest (need >= {ROTATION_MIN_BARS})")

    all_index = _union_index([benchmark, *symbols_bars.values()])
    if len(all_index) < ROTATION_MIN_BARS:
        raise ValueError(
            f"not enough aligned bars for rotation backtest (need >= {ROTATION_MIN_BARS})"
        )

    symbols = sorted(symbols_bars.keys())
    close_df = pd.DataFrame(index=all_index, columns=symbols, dtype=float)
    open_df = pd.DataFrame(index=all_index, columns=symbols, dtype=float)
    for symbol in symbols:
        frame = symbols_bars[symbol]
        close_df[symbol] = pd.to_numeric(frame["close"], errors="coerce").reindex(all_index).ffill()
        open_s = pd.to_numeric(frame["open"], errors="coerce").reindex(all_index)
        open_df[symbol] = open_s
    open_df = open_df.fillna(close_df)

    benchmark_close = pd.to_numeric(benchmark["close"], errors="coerce").reindex(all_index).ffill()
    benchmark_sma60 = benchmark_close.rolling(window=60, min_periods=60).mean()
    market_filter = benchmark_close > benchmark_sma60
    signal_dates = _monthly_first_trading_days(benchmark_close.dropna().index)
    if len(signal_dates) == 0:
        raise ValueError("no monthly signal dates available")

    ret20 = close_df / close_df.shift(20) - 1.0
    ret60 = close_df / close_df.shift(60) - 1.0
    ret120 = close_df / close_df.shift(120) - 1.0
    score_df = momentum_w20 * ret20 + momentum_w60 * ret60 + momentum_w120 * ret120
    symbol_sma60 = close_df.rolling(window=60, min_periods=60).mean()

    schedule: dict[pd.Timestamp, dict[str, float]] = {}
    rebalance_records: list[RotationRebalanceRecord] = []
    for signal_date in signal_dates:
        next_candidates = all_index[all_index > signal_date]
        if len(next_candidates) == 0:
            continue
        effective_date = pd.Timestamp(next_candidates[0])
        market_on = bool(market_filter.get(signal_date, False))

        chosen: list[tuple[str, float]] = []
        if market_on and signal_date in score_df.index:
            row_score = score_df.loc[signal_date]
            row_close = close_df.loc[signal_date]
            row_sma60 = symbol_sma60.loc[signal_date]
            for symbol in symbols:
                score = pd.to_numeric(row_score.get(symbol), errors="coerce")
                close = pd.to_numeric(row_close.get(symbol), errors="coerce")
                sma60 = pd.to_numeric(row_sma60.get(symbol), errors="coerce")
                if pd.isna(score) or pd.isna(close):
                    continue
                if require_positive_momentum and float(score) <= 0.0:
                    continue
                if require_above_sma60:
                    if pd.isna(sma60) or float(close) <= float(sma60):
                        continue
                chosen.append((symbol, float(score)))
            chosen.sort(key=lambda item: item[1], reverse=True)
            chosen = chosen[: int(top_n)]

        if chosen:
            w = 1.0 / float(len(chosen))
            target_weights = {symbol: w for symbol, _ in chosen}
            target_scores = {symbol: score for symbol, score in chosen}
        else:
            target_weights = {}
            target_scores = {}

        schedule[effective_date] = target_weights
        rebalance_records.append(
            RotationRebalanceRecord(
                signal_date=pd.Timestamp(signal_date),
                effective_date=effective_date,
                market_filter_on=market_on,
                selected_symbols=list(target_weights.keys()),
                weights=target_weights,
                scores=target_scores,
            )
        )

    cost = cost_model or CostModel()
    fee_rate = float(cost.fee_rate)
    sell_tax_rate = float(cost.sell_tax_rate)
    slippage_rate = float(cost.slippage_rate)

    cash = float(initial_capital)
    position_qty = dict.fromkeys(symbols, 0.0)
    position_cost = dict.fromkeys(symbols, 0.0)
    trade_rows: list[dict[str, object]] = []
    realized_pnls: list[float] = []
    equity_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    for dt in all_index:
        open_px = pd.to_numeric(open_df.loc[dt], errors="coerce")
        close_px = pd.to_numeric(close_df.loc[dt], errors="coerce")

        if dt in schedule:
            target_weights = schedule[dt]
            mark_open = open_px.fillna(close_px)
            equity_at_open = cash
            for symbol in symbols:
                px = pd.to_numeric(mark_open.get(symbol), errors="coerce")
                if pd.isna(px) or float(px) <= 0:
                    continue
                equity_at_open += position_qty[symbol] * float(px)

            current_value = {}
            desired_value = {}
            for symbol in symbols:
                px = pd.to_numeric(mark_open.get(symbol), errors="coerce")
                if pd.isna(px) or float(px) <= 0:
                    current_value[symbol] = 0.0
                else:
                    current_value[symbol] = position_qty[symbol] * float(px)
                desired_value[symbol] = float(target_weights.get(symbol, 0.0)) * equity_at_open

            for symbol in symbols:
                qty = position_qty[symbol]
                if qty <= 0.0:
                    continue
                px_raw = pd.to_numeric(mark_open.get(symbol), errors="coerce")
                if pd.isna(px_raw) or float(px_raw) <= 0:
                    continue
                current_val = current_value[symbol]
                sell_value = max(0.0, current_val - desired_value[symbol])
                if sell_value <= 1e-9:
                    continue
                px_exec = float(px_raw) * (1.0 - slippage_rate)
                if px_exec <= 0:
                    continue
                qty_sell = min(qty, sell_value / px_exec)
                if qty_sell <= 1e-12:
                    continue

                gross = qty_sell * px_exec
                fee = gross * fee_rate
                tax = gross * sell_tax_rate
                net = gross - fee - tax
                slip_cost = qty_sell * float(px_raw) * slippage_rate

                prev_qty = qty
                prev_cost = position_cost[symbol]
                cost_reduced = prev_cost * (qty_sell / prev_qty) if prev_qty > 0 else 0.0
                pnl = net - cost_reduced
                realized_pnls.append(float(pnl))

                cash += net
                position_qty[symbol] = prev_qty - qty_sell
                if position_qty[symbol] <= 1e-10:
                    position_qty[symbol] = 0.0
                    position_cost[symbol] = 0.0
                else:
                    position_cost[symbol] = max(0.0, prev_cost - cost_reduced)

                trade_rows.append(
                    {
                        "date": pd.Timestamp(dt),
                        "symbol": symbol,
                        "side": "SELL",
                        "qty": float(qty_sell),
                        "price": float(px_exec),
                        "notional": float(gross),
                        "fee": float(fee),
                        "tax": float(tax),
                        "slippage": float(slip_cost),
                        "pnl": float(pnl),
                        "target_weight": float(target_weights.get(symbol, 0.0)),
                    }
                )

            equity_at_open = cash
            for symbol in symbols:
                px = pd.to_numeric(mark_open.get(symbol), errors="coerce")
                if pd.isna(px) or float(px) <= 0:
                    continue
                equity_at_open += position_qty[symbol] * float(px)

            buy_plan: list[tuple[str, float, float]] = []
            for symbol in symbols:
                px_raw = pd.to_numeric(mark_open.get(symbol), errors="coerce")
                if pd.isna(px_raw) or float(px_raw) <= 0:
                    continue
                desired_val = float(target_weights.get(symbol, 0.0)) * equity_at_open
                current_val = position_qty[symbol] * float(px_raw)
                need = max(0.0, desired_val - current_val)
                if need > 1e-9:
                    buy_plan.append((symbol, float(px_raw), need))

            total_cash_need = sum(need * (1.0 + fee_rate) for _, _, need in buy_plan)
            scale = min(1.0, cash / total_cash_need) if total_cash_need > 0 else 0.0
            for symbol, px_raw, need in buy_plan:
                notional = need * scale
                if notional <= 1e-9:
                    continue
                px_exec = px_raw * (1.0 + slippage_rate)
                if px_exec <= 0:
                    continue
                qty_buy = notional / px_exec
                fee = notional * fee_rate
                cash_cost = notional + fee
                if cash_cost > cash and cash > 0:
                    notional = cash / (1.0 + fee_rate)
                    fee = notional * fee_rate
                    cash_cost = notional + fee
                    qty_buy = notional / px_exec if px_exec > 0 else 0.0
                if qty_buy <= 1e-12 or cash_cost <= 0:
                    continue

                slip_cost = qty_buy * px_raw * slippage_rate
                cash -= cash_cost
                position_qty[symbol] += qty_buy
                position_cost[symbol] += cash_cost
                trade_rows.append(
                    {
                        "date": pd.Timestamp(dt),
                        "symbol": symbol,
                        "side": "BUY",
                        "qty": float(qty_buy),
                        "price": float(px_exec),
                        "notional": float(notional),
                        "fee": float(fee),
                        "tax": 0.0,
                        "slippage": float(slip_cost),
                        "pnl": np.nan,
                        "target_weight": float(target_weights.get(symbol, 0.0)),
                    }
                )

        equity = cash
        value_by_symbol: dict[str, float] = {}
        for symbol in symbols:
            px = pd.to_numeric(close_px.get(symbol), errors="coerce")
            if pd.isna(px) or float(px) <= 0:
                value = 0.0
            else:
                value = position_qty[symbol] * float(px)
            value_by_symbol[symbol] = value
            equity += value

        equity_rows.append({"date": pd.Timestamp(dt), "equity": float(equity)})
        row = {"date": pd.Timestamp(dt), "cash": float(cash)}
        if equity > 0:
            row["cash_weight"] = float(cash / equity)
            for symbol in symbols:
                row[symbol] = float(value_by_symbol[symbol] / equity)
        else:
            row["cash_weight"] = 1.0
            for symbol in symbols:
                row[symbol] = 0.0
        weight_rows.append(row)

    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    weights = pd.DataFrame(weight_rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        trades = trades.sort_values(["date", "symbol", "side"]).reset_index(drop=True)

    metrics, drawdown, yearly_returns = compute_metrics_from_equity(
        equity_curve=equity_curve,
        initial_capital=float(initial_capital),
        trade_pnls=realized_pnls,
    )

    return RotationBacktestResult(
        equity_curve=equity_curve,
        metrics=metrics,
        drawdown_series=drawdown,
        yearly_returns=yearly_returns,
        weights=weights,
        trades=trades,
        rebalance_records=rebalance_records,
    )
