from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional, Protocol

import pandas as pd

from backtest import ROTATION_MIN_BARS, apply_split_adjustment
from services.sync_orchestrator import sync_symbols_history


class HistoryStoreLike(Protocol):
    def load_daily_bars(self, symbol: str, market: str, start: datetime, end: datetime) -> pd.DataFrame:
        ...

    def sync_symbol_history(self, symbol: str, market: str, start: datetime, end: datetime):
        ...


@dataclass(frozen=True)
class RotationBarsPreparationResult:
    bars_by_symbol: dict[str, pd.DataFrame]
    skipped_symbols: list[str]
    sync_issues: list[str]


def prepare_rotation_bars(
    *,
    store: HistoryStoreLike,
    symbols: list[str],
    start_dt: datetime,
    end_dt: datetime,
    sync_before_run: bool,
    parallel_sync: bool,
    normalize_ohlcv_frame: Callable[[pd.DataFrame], pd.DataFrame],
    min_required: int = ROTATION_MIN_BARS,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> RotationBarsPreparationResult:
    symbol_sync_issues: list[str] = []
    if sync_before_run and symbols:
        _, symbol_sync_issues = sync_symbols_history(
            store=store,
            market="TW",
            symbols=symbols,
            start=start_dt,
            end=end_dt,
            parallel=parallel_sync,
        )

    bars_cache: dict[str, pd.DataFrame] = {}
    symbols_need_sync: list[str] = []
    for symbol in symbols:
        bars_local = store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
        bars_local = normalize_ohlcv_frame(bars_local)
        bars_cache[symbol] = bars_local
        if len(bars_local) < int(min_required):
            symbols_need_sync.append(symbol)

    if symbols_need_sync and not sync_before_run:
        _, lazy_sync_issues = sync_symbols_history(
            store=store,
            market="TW",
            symbols=symbols_need_sync,
            start=start_dt,
            end=end_dt,
            parallel=parallel_sync,
        )
        symbol_sync_issues.extend(lazy_sync_issues)
        for symbol in symbols_need_sync:
            bars_local = store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            bars_cache[symbol] = normalize_ohlcv_frame(bars_local)

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    skipped_symbols: list[str] = []
    total = max(len(symbols), 1)
    for idx, symbol in enumerate(symbols):
        bars = bars_cache.get(symbol, pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
        if bars.empty:
            skipped_symbols.append(symbol)
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue
        bars, _ = apply_split_adjustment(
            bars=bars,
            symbol=symbol,
            market="TW",
            use_known=True,
            use_auto_detect=True,
        )
        if len(bars) < int(min_required):
            skipped_symbols.append(symbol)
        else:
            bars_by_symbol[symbol] = bars
        if progress_callback:
            progress_callback((idx + 1) / total)

    return RotationBarsPreparationResult(
        bars_by_symbol=bars_by_symbol,
        skipped_symbols=skipped_symbols,
        sync_issues=symbol_sync_issues,
    )


def build_rotation_holding_rank(
    *,
    weights_df: Optional[pd.DataFrame],
    selected_symbol_lists: list[list[str]],
    universe_symbols: list[str],
    name_map: dict[str, str],
) -> list[dict[str, object]]:
    selected_counts = {sym: 0 for sym in universe_symbols}
    for symbols in selected_symbol_lists:
        for sym in symbols:
            if sym in selected_counts:
                selected_counts[sym] += 1

    total_days = int(len(weights_df)) if isinstance(weights_df, pd.DataFrame) and not weights_df.empty else 0
    total_signals = max(1, len(selected_symbol_lists))
    rows: list[dict[str, object]] = []
    for sym in universe_symbols:
        hold_days = 0
        if total_days > 0 and weights_df is not None and sym in weights_df.columns:
            hold_days = int((pd.to_numeric(weights_df[sym], errors="coerce").fillna(0.0) > 1e-10).sum())
        selected_months = int(selected_counts.get(sym, 0))
        if hold_days <= 0 and selected_months <= 0:
            continue
        hold_ratio_pct = (hold_days / total_days * 100.0) if total_days > 0 else 0.0
        selected_ratio_pct = selected_months / total_signals * 100.0
        rows.append(
            {
                "symbol": sym,
                "name": name_map.get(sym, sym),
                "hold_days": hold_days,
                "hold_ratio_pct": hold_ratio_pct,
                "selected_months": selected_months,
                "selected_ratio_pct": selected_ratio_pct,
            }
        )
    rows.sort(
        key=lambda r: (
            int(r.get("hold_days", 0)),
            int(r.get("selected_months", 0)),
            str(r.get("symbol", "")),
        ),
        reverse=True,
    )
    return rows


def build_rotation_payload(
    *,
    run_key: str,
    benchmark_symbol: str,
    universe_symbols: list[str],
    bars_by_symbol: dict[str, pd.DataFrame],
    skipped_symbols: list[str],
    start_date: date,
    end_date: date,
    top_n: int,
    initial_capital: float,
    metrics: dict[str, object],
    equity_series: pd.Series,
    benchmark_equity: pd.Series,
    buy_hold_equity: pd.Series,
    rebalance_records: list[dict[str, object]],
    trades_df: pd.DataFrame,
    holding_rank: list[dict[str, object]],
) -> dict[str, object]:
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    return {
        "run_key": run_key,
        "generated_at": now_iso,
        "strategy": "tw_etf_rotation_v1",
        "benchmark_symbol": benchmark_symbol,
        "universe_symbols": list(universe_symbols),
        "used_symbols": sorted(list(bars_by_symbol.keys())),
        "skipped_symbols": sorted(skipped_symbols),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "top_n": int(top_n),
        "initial_capital": float(initial_capital),
        "metrics": metrics,
        "equity_curve": [
            {"date": pd.Timestamp(idx).isoformat(), "equity": float(val)}
            for idx, val in equity_series.items()
        ],
        "benchmark_curve": [
            {"date": pd.Timestamp(idx).isoformat(), "equity": float(val)}
            for idx, val in benchmark_equity.dropna().items()
        ],
        "buy_hold_curve": [
            {"date": pd.Timestamp(idx).isoformat(), "equity": float(val)}
            for idx, val in buy_hold_equity.dropna().items()
        ],
        "rebalance_records": rebalance_records,
        "trades": trades_df.to_dict("records"),
        "holding_rank": holding_rank,
    }


__all__ = [
    "RotationBarsPreparationResult",
    "build_rotation_holding_rank",
    "build_rotation_payload",
    "prepare_rotation_bars",
]
