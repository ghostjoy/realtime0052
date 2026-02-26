from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

from market_data_types import SyncPlanResult

if TYPE_CHECKING:
    from storage import HistoryStore


def _is_retryable_duckdb_handle_conflict(message: object) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    if "unique file handle conflict" in text:
        return True
    if ("transactioncontext error" in text) and ("conflict on tuple deletion" in text):
        return True
    if ("transactioncontext error" in text) and ("conflict on tuple update" in text):
        return True
    return ("cannot attach" in text) and ("market_history" in text)


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


def _coerce_utc_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_sync_last_success_date(
    store: HistoryStore, *, symbol: str, market: str
) -> datetime | None:
    loader = getattr(store, "load_sync_state", None)
    if not callable(loader):
        return None
    try:
        state = loader(symbol=symbol, market=market)
    except TypeError:
        try:
            state = loader(symbol, market)
        except Exception:
            return None
    except Exception:
        return None
    if isinstance(state, dict):
        return _coerce_utc_datetime(state.get("last_success_date"))
    return None


def _load_daily_coverage(
    store: HistoryStore,
    *,
    symbol: str,
    market: str,
    start: datetime,
    end: datetime,
) -> tuple[int, datetime | None, datetime | None] | None:
    loader = getattr(store, "load_daily_coverage", None)
    if not callable(loader):
        return None
    try:
        payload = loader(symbol=symbol, market=market, start=start, end=end)
    except TypeError:
        try:
            payload = loader(symbol, market, start, end)
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    row_count = max(0, int(payload.get("row_count") or 0))
    first_date = _coerce_utc_datetime(payload.get("first_date"))
    last_date = _coerce_utc_datetime(payload.get("last_date"))
    return row_count, first_date, last_date


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
    issues_by_symbol: dict[str, str] = {}
    if not ordered_symbols:
        return reports, []

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
                    issues_by_symbol[symbol] = f"{symbol}: {err}"
            except Exception as exc:
                reports[symbol] = None
                issues_by_symbol[symbol] = f"{symbol}: {exc}"
        issues = [
            issues_by_symbol[symbol] for symbol in ordered_symbols if symbol in issues_by_symbol
        ]
        return reports, issues

    workers = max(1, min(int(max_workers), len(ordered_symbols)))
    retry_symbols: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, symbol): symbol for symbol in ordered_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                report = future.result()
                reports[symbol] = report
                err = str(getattr(report, "error", "") or "").strip()
                if err:
                    issues_by_symbol[symbol] = f"{symbol}: {err}"
                    if _is_retryable_duckdb_handle_conflict(err):
                        retry_symbols.add(symbol)
            except Exception as exc:
                reports[symbol] = None
                issues_by_symbol[symbol] = f"{symbol}: {exc}"
                if _is_retryable_duckdb_handle_conflict(exc):
                    retry_symbols.add(symbol)

    for symbol in ordered_symbols:
        if symbol not in retry_symbols:
            continue
        try:
            report = _run_one(symbol)
            reports[symbol] = report
            err = str(getattr(report, "error", "") or "").strip()
            if err:
                issues_by_symbol[symbol] = f"{symbol}: {err}"
            else:
                issues_by_symbol.pop(symbol, None)
        except Exception as exc:
            reports[symbol] = None
            issues_by_symbol[symbol] = f"{symbol}: {exc}"

    issues = [issues_by_symbol[symbol] for symbol in ordered_symbols if symbol in issues_by_symbol]
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
            last_success = _load_sync_last_success_date(store, symbol=symbol, market=market)
            # Fast-path: metadata already tells us the requested end is not covered.
            if last_success is not None and last_success < end:
                need_sync.append(symbol)
                continue

            coverage = _load_daily_coverage(
                store,
                symbol=symbol,
                market=market,
                start=start,
                end=end,
            )
            if coverage is not None:
                row_count, first_date, last_date = coverage
                if (
                    row_count <= 0
                    or first_date is None
                    or last_date is None
                    or first_date > start
                    or last_date < end
                ):
                    need_sync.append(symbol)
                continue

            bars = store.load_daily_bars(symbol=symbol, market=market, start=start, end=end)
            if bars_need_backfill(bars, start=start, end=end):
                need_sync.append(symbol)
    elif mode_token == "min_rows":
        required_rows = max(1, int(min_rows))
        need_sync = []
        for symbol in ordered:
            coverage = _load_daily_coverage(
                store,
                symbol=symbol,
                market=market,
                start=start,
                end=end,
            )
            if coverage is not None:
                row_count, _, _ = coverage
                if row_count < required_rows:
                    need_sync.append(symbol)
                continue
            bars = store.load_daily_bars(symbol=symbol, market=market, start=start, end=end)
            if not isinstance(bars, pd.DataFrame) or len(bars) < required_rows:
                need_sync.append(symbol)
    else:
        raise ValueError(f"Unsupported sync mode: {mode}")

    skipped = [sym for sym in ordered if sym not in need_sync]
    source_chain = [f"mode:{mode_token}", f"need:{len(need_sync)}", f"skip:{len(skipped)}"]
    if not need_sync:
        return {}, SyncPlanResult(
            synced_symbols=[], skipped_symbols=skipped, issues=[], source_chain=source_chain
        )

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
