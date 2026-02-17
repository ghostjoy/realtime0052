from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

from market_data_types import SyncPlanResult

if TYPE_CHECKING:
    from storage import HistoryStore


def normalize_symbols(symbols: list[str]) -> list[str]:
    ordered: list[str] = []
    for symbol in symbols:
        text = str(symbol or "").strip().upper()
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def bars_need_backfill(bars: pd.DataFrame, *, start: datetime, end: datetime) -> bool:
    if bars is None or bars.empty:
        return True
    idx = pd.to_datetime(bars.index, utc=True, errors="coerce").dropna()
    if idx.empty:
        return True
    first_ts = pd.Timestamp(idx.min()).to_pydatetime().replace(tzinfo=timezone.utc)
    last_ts = pd.Timestamp(idx.max()).to_pydatetime().replace(tzinfo=timezone.utc)
    return first_ts > start or last_ts < end


def sync_symbols_history(
    *,
    store: HistoryStore,
    market: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
    parallel: bool = True,
    max_workers: int = 6,
) -> tuple[dict[str, object], list[str]]:
    ordered_symbols = normalize_symbols(symbols)
    reports: dict[str, object] = {}
    issues: list[str] = []
    if not ordered_symbols:
        return reports, issues

    def _run_one(symbol: str) -> object:
        return store.sync_symbol_history(symbol=symbol, market=market, start=start, end=end)

    use_parallel = parallel and len(ordered_symbols) > 1
    if not use_parallel:
        for symbol in ordered_symbols:
            try:
                report = _run_one(symbol)
                reports[symbol] = report
                err = str(getattr(report, "error", "") or "").strip()
                if err:
                    issues.append(f"{symbol}: {err}")
            except Exception as exc:
                reports[symbol] = None
                issues.append(f"{symbol}: {exc}")
        return reports, issues

    workers = max(1, min(int(max_workers), len(ordered_symbols)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, symbol): symbol for symbol in ordered_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                report = future.result()
                reports[symbol] = report
                err = str(getattr(report, "error", "") or "").strip()
                if err:
                    issues.append(f"{symbol}: {err}")
            except Exception as exc:
                reports[symbol] = None
                issues.append(f"{symbol}: {exc}")
    return reports, issues


def sync_symbols_if_needed(
    *,
    store: HistoryStore,
    market: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
    parallel: bool = True,
    max_workers: int = 6,
    mode: str = "backfill",
    min_rows: int = 0,
) -> tuple[dict[str, object], SyncPlanResult]:
    ordered = normalize_symbols(symbols)
    mode_token = str(mode or "backfill").strip().lower()

    if mode_token == "all":
        need_sync = list(ordered)
    elif mode_token == "backfill":
        need_sync = []
        for symbol in ordered:
            bars = store.load_daily_bars(symbol=symbol, market=market, start=start, end=end)
            if bars_need_backfill(bars, start=start, end=end):
                need_sync.append(symbol)
    elif mode_token == "min_rows":
        required_rows = max(1, int(min_rows))
        need_sync = []
        for symbol in ordered:
            bars = store.load_daily_bars(symbol=symbol, market=market, start=start, end=end)
            if not isinstance(bars, pd.DataFrame) or len(bars) < required_rows:
                need_sync.append(symbol)
    else:
        raise ValueError(f"Unsupported sync mode: {mode}")

    skipped = [sym for sym in ordered if sym not in need_sync]
    source_chain = [f"mode:{mode_token}", f"need:{len(need_sync)}", f"skip:{len(skipped)}"]
    if not need_sync:
        return {}, SyncPlanResult(synced_symbols=[], skipped_symbols=skipped, issues=[], source_chain=source_chain)

    reports, issues = sync_symbols_history(
        store=store,
        market=market,
        symbols=need_sync,
        start=start,
        end=end,
        parallel=parallel,
        max_workers=max_workers,
    )
    return reports, SyncPlanResult(
        synced_symbols=need_sync,
        skipped_symbols=skipped,
        issues=issues,
        source_chain=source_chain,
    )


__all__ = [
    "normalize_symbols",
    "bars_need_backfill",
    "sync_symbols_history",
    "sync_symbols_if_needed",
]
