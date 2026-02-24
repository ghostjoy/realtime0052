from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import pandas as pd

from backtest import CostModel, apply_split_adjustment, run_backtest
from services.sync_orchestrator import sync_symbols_history


class HistoryStoreLike(Protocol):
    def load_daily_bars(
        self, symbol: str, market: str, start: datetime, end: datetime
    ) -> pd.DataFrame: ...

    def sync_symbol_history(self, symbol: str, market: str, start: datetime, end: datetime): ...


@dataclass(frozen=True)
class HeatmapRunInput:
    page_key: str
    start_date: date
    end_date: date
    benchmark_choice: str
    strategy: str
    strategy_params: dict[str, float]
    fee_rate: float
    sell_tax: float
    slippage: float
    selected_symbols: list[str]


@dataclass(frozen=True)
class HeatmapBarsPreparationResult:
    bars_cache: dict[str, pd.DataFrame]
    sync_issues: list[str]


def build_heatmap_run_key(config: HeatmapRunInput) -> str:
    strategy_token = json.dumps(
        config.strategy_params if isinstance(config.strategy_params, dict) else {}, sort_keys=True
    )
    symbols_token = ",".join(config.selected_symbols)
    return (
        f"{config.page_key}_heatmap:{config.start_date}:{config.end_date}:{config.benchmark_choice}:{config.strategy}:{strategy_token}:"
        f"{config.fee_rate}:{config.sell_tax}:{config.slippage}:{symbols_token}"
    )


def prepare_heatmap_bars(
    *,
    store: HistoryStoreLike,
    symbols: list[str],
    start_dt: datetime,
    end_dt: datetime,
    min_required: int,
    sync_before_run: bool,
    parallel_sync: bool,
    lazy_sync_on_insufficient: bool = True,
    normalize_ohlcv_frame: Callable[[pd.DataFrame], pd.DataFrame],
) -> HeatmapBarsPreparationResult:
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

    if symbols_need_sync and (not sync_before_run) and bool(lazy_sync_on_insufficient):
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
            bars_local = store.load_daily_bars(
                symbol=symbol, market="TW", start=start_dt, end=end_dt
            )
            bars_cache[symbol] = normalize_ohlcv_frame(bars_local)

    return HeatmapBarsPreparationResult(bars_cache=bars_cache, sync_issues=symbol_sync_issues)


def compute_heatmap_rows(
    *,
    run_symbols: list[str],
    bars_cache: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    strategy: str,
    strategy_params: dict[str, float],
    cost_model: CostModel,
    name_map: dict[str, str],
    min_required: int,
    progress_callback: Callable[[float], None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = max(len(run_symbols), 1)
    for idx, symbol in enumerate(run_symbols):
        bars = bars_cache.get(
            symbol, pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        )
        if len(bars) < int(min_required):
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
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        try:
            bt = run_backtest(
                bars=bars,
                strategy_name=strategy,
                strategy_params=strategy_params,
                cost_model=cost_model,
                initial_capital=1_000_000.0,
            )
        except Exception:
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        strategy_curve = pd.to_numeric(bt.equity_curve["equity"], errors="coerce").dropna()
        if len(strategy_curve) < 2:
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        comp = pd.concat(
            [
                strategy_curve.rename("strategy"),
                benchmark_close.reindex(strategy_curve.index).ffill().rename("benchmark"),
            ],
            axis=1,
        ).dropna()
        if len(comp) < 2:
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        strategy_ret = float(comp["strategy"].iloc[-1] / comp["strategy"].iloc[0] - 1.0)
        benchmark_ret = float(comp["benchmark"].iloc[-1] / comp["benchmark"].iloc[0] - 1.0)
        excess_pct = (strategy_ret - benchmark_ret) * 100.0
        rows.append(
            {
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "strategy_return_pct": strategy_ret * 100.0,
                "benchmark_return_pct": benchmark_ret * 100.0,
                "excess_pct": excess_pct,
                "status": "WIN" if excess_pct > 0 else ("LOSE" if excess_pct < 0 else "TIE"),
                "bars": int(len(comp)),
            }
        )
        if progress_callback:
            progress_callback((idx + 1) / total)
    return rows


__all__ = [
    "HeatmapBarsPreparationResult",
    "HeatmapRunInput",
    "build_heatmap_run_key",
    "compute_heatmap_rows",
    "prepare_heatmap_bars",
]
