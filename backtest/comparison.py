from __future__ import annotations

from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd


def build_buy_hold_equity(
    bars_by_symbol: Dict[str, pd.DataFrame],
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
