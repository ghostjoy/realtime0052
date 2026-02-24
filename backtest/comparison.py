from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _normalize_target_index(target_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(target_index).sort_values()
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        return idx
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def build_dca_contribution_plan(target_index: pd.DatetimeIndex) -> dict[str, object]:
    idx = _normalize_target_index(target_index)
    if len(idx) == 0:
        return {"initial_date": None, "monthly_dates": []}

    initial_date = pd.Timestamp(idx[0])
    month_first_map: dict[tuple[int, int], pd.Timestamp] = {}
    for ts in idx:
        ts_key = (int(ts.year), int(ts.month))
        if ts_key not in month_first_map:
            month_first_map[ts_key] = pd.Timestamp(ts)

    month_keys_sorted = sorted(month_first_map.keys())
    monthly_dates = [month_first_map[key] for key in month_keys_sorted]
    monthly_dates = [ts for ts in monthly_dates if ts > initial_date]
    return {
        "initial_date": initial_date,
        "monthly_dates": monthly_dates,
    }


def _normalize_close_series(bars: pd.DataFrame) -> pd.Series:
    if bars is None or bars.empty or "close" not in bars.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").dropna()
    if close.empty:
        return pd.Series(dtype=float)
    close.index = pd.to_datetime(close.index, utc=True, errors="coerce")
    close = close[~close.index.isna()].sort_index()
    return close


def build_dca_equity(
    bars_by_symbol: dict[str, pd.DataFrame],
    target_index: pd.DatetimeIndex,
    initial_lump_sum: float,
    monthly_contribution: float,
    *,
    fee_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    slippage_rate: float = 0.0,
    contribution_day_rule: str = "month_first_trading_day_close",
    allocation_mode: str = "equal_weight",
) -> pd.Series:
    _ = sell_tax_rate
    if contribution_day_rule != "month_first_trading_day_close":
        raise ValueError(f"unsupported contribution_day_rule: {contribution_day_rule}")
    if allocation_mode != "equal_weight":
        raise ValueError(f"unsupported allocation_mode: {allocation_mode}")

    idx = _normalize_target_index(target_index)
    if len(idx) == 0:
        return pd.Series(dtype=float)

    symbols = [str(sym).strip().upper() for sym in bars_by_symbol if str(sym).strip()]
    symbols = list(dict.fromkeys(symbols))
    symbol_count = len(symbols)

    price_exact: dict[str, pd.Series] = {}
    price_valuation: dict[str, pd.Series] = {}
    for symbol in symbols:
        close = _normalize_close_series(bars_by_symbol.get(symbol, pd.DataFrame()))
        if close.empty:
            price_exact[symbol] = pd.Series(np.nan, index=idx, dtype=float)
            price_valuation[symbol] = pd.Series(np.nan, index=idx, dtype=float)
            continue
        exact = close.reindex(idx)
        valuation = close.reindex(idx).ffill()
        price_exact[symbol] = exact
        price_valuation[symbol] = valuation

    plan = build_dca_contribution_plan(idx)
    initial_date = plan.get("initial_date")
    initial_ts = pd.Timestamp(initial_date) if initial_date is not None else None
    monthly_dates_raw = plan.get("monthly_dates", [])
    monthly_dates = list(monthly_dates_raw) if isinstance(monthly_dates_raw, list) else []
    monthly_set = {pd.Timestamp(ts) for ts in monthly_dates}

    fee_rate = max(0.0, float(fee_rate))
    slippage_rate = max(0.0, float(slippage_rate))
    initial_lump_sum = max(0.0, float(initial_lump_sum))
    monthly_contribution = max(0.0, float(monthly_contribution))
    qty_by_symbol = dict.fromkeys(symbols, 0.0)
    cash = 0.0
    equity_values: list[float] = []

    for dt in idx:
        contribution_amount = 0.0
        if initial_ts is not None and dt == initial_ts:
            contribution_amount += initial_lump_sum
        if dt in monthly_set:
            contribution_amount += monthly_contribution
        if contribution_amount > 0:
            cash += contribution_amount

            if symbol_count > 0:
                per_symbol_budget = contribution_amount / float(symbol_count)
                for symbol in symbols:
                    budget = float(per_symbol_budget)
                    if budget <= 0:
                        continue
                    px = _safe_float(price_exact[symbol].get(dt))
                    if px is None or px <= 0:
                        continue
                    execution_px = px * (1.0 + slippage_rate)
                    if not np.isfinite(execution_px) or execution_px <= 0:
                        continue
                    denom = execution_px * (1.0 + fee_rate)
                    if denom <= 0:
                        continue
                    qty_buy = budget / denom
                    if not np.isfinite(qty_buy) or qty_buy <= 0:
                        continue
                    notional = qty_buy * execution_px
                    fee = notional * fee_rate
                    total_cost = notional + fee
                    if total_cost > cash:
                        total_cost = cash
                        qty_buy = total_cost / denom if denom > 0 else 0.0
                        if qty_buy <= 0:
                            continue
                    cash -= total_cost
                    qty_by_symbol[symbol] = float(qty_by_symbol[symbol] + qty_buy)

        total_equity = float(cash)
        for symbol in symbols:
            val_px = _safe_float(price_valuation[symbol].get(dt))
            if val_px is None or val_px <= 0:
                continue
            total_equity += float(qty_by_symbol[symbol]) * float(val_px)
        equity_values.append(total_equity)

    return pd.Series(equity_values, index=idx, dtype=float)


def build_dca_benchmark_equity(
    benchmark_close: pd.Series,
    target_index: pd.DatetimeIndex,
    initial_lump_sum: float,
    monthly_contribution: float,
    *,
    fee_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    slippage_rate: float = 0.0,
    contribution_day_rule: str = "month_first_trading_day_close",
) -> pd.Series:
    close = pd.to_numeric(benchmark_close, errors="coerce").dropna()
    if close.empty:
        return pd.Series(dtype=float, index=_normalize_target_index(target_index))
    close.index = pd.to_datetime(close.index, utc=True, errors="coerce")
    close = close[~close.index.isna()].sort_index()
    bars = pd.DataFrame({"close": close})
    return build_dca_equity(
        bars_by_symbol={"BENCHMARK": bars},
        target_index=target_index,
        initial_lump_sum=initial_lump_sum,
        monthly_contribution=monthly_contribution,
        fee_rate=fee_rate,
        sell_tax_rate=sell_tax_rate,
        slippage_rate=slippage_rate,
        contribution_day_rule=contribution_day_rule,
        allocation_mode="equal_weight",
    )


def dca_summary_metrics(equity: pd.Series, *, total_contribution: float) -> dict[str, float]:
    total_contribution = max(0.0, float(total_contribution))
    series = (
        pd.to_numeric(equity, errors="coerce").dropna()
        if isinstance(equity, pd.Series)
        else pd.Series(dtype=float)
    )
    if series.empty:
        return {
            "end_value": np.nan,
            "total_contribution": total_contribution,
            "pnl": np.nan,
            "total_return": np.nan,
        }
    end_value = float(series.iloc[-1])
    pnl = float(end_value - total_contribution)
    total_return = float(pnl / total_contribution) if total_contribution > 0 else np.nan
    return {
        "end_value": end_value,
        "total_contribution": total_contribution,
        "pnl": pnl,
        "total_return": total_return,
    }


def build_buy_hold_equity(
    bars_by_symbol: dict[str, pd.DataFrame],
    target_index: pd.DatetimeIndex,
    initial_capital: float,
) -> pd.Series:
    if len(target_index) == 0:
        return pd.Series(dtype=float)
    if not bars_by_symbol:
        return pd.Series(dtype=float, index=target_index)

    per_symbol = float(initial_capital) / float(len(bars_by_symbol))
    out = pd.Series(0.0, index=target_index, dtype=float)

    for bars in bars_by_symbol.values():
        if bars is None or bars.empty or "close" not in bars.columns:
            out = out + per_symbol
            continue
        close = pd.to_numeric(bars["close"], errors="coerce").dropna().sort_index()
        if close.empty:
            out = out + per_symbol
            continue

        aligned = close.reindex(target_index).ffill()
        non_na = aligned.dropna()
        if non_na.empty:
            out = out + per_symbol
            continue
        base = float(non_na.iloc[0])
        if not np.isfinite(base) or base <= 0:
            out = out + per_symbol
            continue
        norm = aligned / base
        norm = norm.fillna(1.0)
        out = out + per_symbol * norm

    return out


def _safe_float(value: object) -> float | None:
    try:
        fv = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if np.isnan(fv) or np.isinf(fv):
        return None
    return fv


def interval_return(
    equity: pd.Series,
    start_date: date,
    end_date: date,
) -> dict:
    if equity is None or equity.empty:
        return {"ok": False, "reason": "no_data"}

    index = pd.DatetimeIndex(equity.index).sort_values()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    if end_ts < start_ts:
        return {"ok": False, "reason": "invalid_range"}

    start_candidates = index[index >= start_ts]
    end_candidates = index[index <= end_ts]
    if len(start_candidates) == 0 or len(end_candidates) == 0:
        return {"ok": False, "reason": "out_of_range"}

    start_used = pd.Timestamp(start_candidates[0])
    end_used = pd.Timestamp(end_candidates[-1])
    if end_used < start_used:
        return {"ok": False, "reason": "no_overlap"}

    start_v = float(equity.loc[start_used])
    end_v = float(equity.loc[end_used])
    if not np.isfinite(start_v) or start_v == 0:
        return {"ok": False, "reason": "bad_start_value"}

    return {
        "ok": True,
        "start_used": start_used,
        "end_used": end_used,
        "start_value": start_v,
        "end_value": end_v,
        "return": end_v / start_v - 1.0,
    }
