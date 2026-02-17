from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from backtest import apply_split_adjustment
from market_data_types import BenchmarkLoadResult

if TYPE_CHECKING:
    from storage import HistoryStore


_BASE_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=_BASE_OHLCV_COLUMNS)


def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return _empty_bars()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        renamed: list[str] = []
        for col in out.columns:
            parts = [str(part).strip().lower() for part in col if str(part).strip()]
            candidate = ""
            for item in reversed(parts):
                if item in {"open", "high", "low", "close", "adj close", "adj_close", "volume", "price"}:
                    candidate = item
                    break
            renamed.append(candidate or (parts[-1] if parts else ""))
        out.columns = renamed
    else:
        out.columns = [str(col).strip().lower() for col in out.columns]

    if "adj close" in out.columns and "adj_close" not in out.columns:
        out = out.rename(columns={"adj close": "adj_close"})
    if "price" in out.columns and "close" not in out.columns:
        out["close"] = out["price"]
    if "close" not in out.columns:
        return _empty_bars()

    close = pd.to_numeric(out["close"], errors="coerce")
    for col in ["open", "high", "low"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(close)
        else:
            out[col] = close
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0
    out["close"] = close
    out = out[_BASE_OHLCV_COLUMNS].dropna(subset=["open", "high", "low", "close"], how="any")
    return out.sort_index()


def benchmark_candidates_tw(choice: str, *, allow_twii_fallback: bool = False) -> list[str]:
    twii_candidates = ["^TWII", "0050", "006208"] if bool(allow_twii_fallback) else ["^TWII"]
    mapping = {
        "twii": twii_candidates,
        "0050": ["0050"],
        "006208": ["006208"],
    }
    raw = mapping.get(str(choice or "").strip().lower(), twii_candidates)
    ordered: list[str] = []
    for symbol in raw:
        text = str(symbol or "").strip().upper()
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def load_tw_benchmark_bars(
    *,
    store: HistoryStore,
    choice: str,
    start_dt: datetime,
    end_dt: datetime,
    sync_first: bool,
    allow_twii_fallback: bool = False,
    min_rows: int = 2,
) -> BenchmarkLoadResult:
    sync_issues: list[str] = []
    source_chain: list[str] = []
    candidates = benchmark_candidates_tw(choice, allow_twii_fallback=allow_twii_fallback)
    required_rows = max(1, int(min_rows))
    for benchmark_symbol in candidates:
        if sync_first:
            report = store.sync_symbol_history(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
            if report.error:
                sync_issues.append(f"{benchmark_symbol}: {report.error}")

        bench_bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt))
        if bench_bars.empty and not sync_first:
            report = store.sync_symbol_history(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
            if report.error:
                sync_issues.append(f"{benchmark_symbol}: {report.error}")
            bench_bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt))
        if bench_bars.empty:
            source_chain.append(f"{benchmark_symbol}:empty")
            continue

        bench_bars, _ = apply_split_adjustment(
            bars=bench_bars,
            symbol=benchmark_symbol,
            market="TW",
            use_known=True,
            use_auto_detect=True,
        )
        if len(bench_bars) < required_rows:
            source_chain.append(f"{benchmark_symbol}:insufficient_rows({len(bench_bars)})")
            continue

        source_chain.append(f"{benchmark_symbol}:ok")
        close = pd.to_numeric(bench_bars["close"], errors="coerce").dropna().sort_index()
        return BenchmarkLoadResult(
            bars=bench_bars,
            close=close,
            symbol_used=benchmark_symbol,
            sync_issues=sync_issues,
            source_chain=source_chain,
            candidates=candidates,
        )

    return BenchmarkLoadResult(
        bars=_empty_bars(),
        close=pd.Series(dtype=float),
        symbol_used="",
        sync_issues=sync_issues,
        source_chain=source_chain,
        candidates=candidates,
    )


def load_tw_benchmark_close(
    *,
    store: HistoryStore,
    choice: str,
    start_dt: datetime,
    end_dt: datetime,
    sync_first: bool,
    allow_twii_fallback: bool = False,
) -> BenchmarkLoadResult:
    result = load_tw_benchmark_bars(
        store=store,
        choice=choice,
        start_dt=start_dt,
        end_dt=end_dt,
        sync_first=sync_first,
        allow_twii_fallback=allow_twii_fallback,
        min_rows=2,
    )
    if len(result.close) >= 2:
        return result

    return BenchmarkLoadResult(
        bars=result.bars,
        close=pd.Series(dtype=float),
        symbol_used=result.symbol_used if len(result.bars) >= 2 else "",
        sync_issues=list(result.sync_issues),
        source_chain=list(result.source_chain),
        candidates=list(result.candidates),
    )


__all__ = [
    "benchmark_candidates_tw",
    "load_tw_benchmark_bars",
    "load_tw_benchmark_close",
]
