from __future__ import annotations

from datetime import date

import pandas as pd

StartMode = str  # "沿用回測起始日期" | "指定日期" | "指定第幾根K"


def resolve_start_index(
    bars: pd.DataFrame,
    mode: StartMode,
    start_date: date | None = None,
    start_k: int | None = None,
) -> tuple[int, pd.Timestamp]:
    if bars is None or bars.empty:
        raise ValueError("bars is empty")

    frame = bars.sort_index()
    if mode == "沿用回測起始日期":
        idx = 0
    elif mode == "指定日期":
        if start_date is None:
            raise ValueError("start_date is required when mode=指定日期")
        dt = pd.Timestamp(start_date)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        pos = int(frame.index.searchsorted(dt, side="left"))
        if pos >= len(frame):
            raise ValueError("指定日期晚於可用資料最後日期")
        idx = pos
    elif mode == "指定第幾根K":
        if start_k is None:
            raise ValueError("start_k is required when mode=指定第幾根K")
        idx = max(0, min(int(start_k), len(frame) - 1))
    else:
        raise ValueError(f"unknown start mode: {mode}")

    return idx, pd.Timestamp(frame.index[idx])


def apply_start_to_bars_map(
    bars_by_symbol: dict[str, pd.DataFrame],
    mode: StartMode,
    start_date: date | None = None,
    start_k: int | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    trimmed: dict[str, pd.DataFrame] = {}
    rows = []

    for symbol, bars in bars_by_symbol.items():
        try:
            idx, dt = resolve_start_index(
                bars=bars,
                mode=mode,
                start_date=start_date,
                start_k=start_k,
            )
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "status": "SKIP",
                    "reason": str(exc),
                    "start_k": None,
                    "start_date": None,
                    "rows_after_start": 0,
                }
            )
            continue

        sliced = bars.sort_index().iloc[idx:].copy()
        if sliced.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "status": "SKIP",
                    "reason": "投入起點後無資料",
                    "start_k": idx,
                    "start_date": dt.strftime("%Y-%m-%d"),
                    "rows_after_start": 0,
                }
            )
            continue

        trimmed[symbol] = sliced
        rows.append(
            {
                "symbol": symbol,
                "status": "OK",
                "reason": "",
                "start_k": idx,
                "start_date": dt.strftime("%Y-%m-%d"),
                "rows_after_start": len(sliced),
            }
        )

    return trimmed, pd.DataFrame(rows)
