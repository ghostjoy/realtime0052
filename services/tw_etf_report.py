from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from indicators import add_indicators
from services.benchmark_loader import load_tw_benchmark_close
from services.chart_export import export_backtest_chart_artifact
from services.heatmap_runner import compute_heatmap_rows, prepare_heatmap_bars
from services.tw_etf_constituent_sync import (
    DEFAULT_TW_ETF_CONSTITUENT_LOG_DIR,
    TW_ETF_CONSTITUENTS_DATASET_KEY,
    load_latest_tw_etf_constituent_snapshot,
    sync_tw_etf_constituent_snapshots,
)
from services.tw_etf_report_logging import build_sync_log_entry, write_sync_log_files
from services.tw_etf_super_export import (
    _call_app_quiet,
    _load_app_module,
    _sync_tw_etf_super_export_sources,
    build_tw_etf_all_types_main_export_frame,
    build_tw_etf_super_export_table,
    resolve_tw_etf_super_export_periods,
)
from services.sync_orchestrator import sync_symbols_if_needed
from storage import HistoryStore
from utils import normalize_ohlcv_frame


DEFAULT_TW_ETF_REPORT_ROOT = Path("artifacts") / "reports"
DEFAULT_TW_ETF_REPORT_LOG_DIR = Path("artifacts") / "logs"
REPORT_HEATMAP_EXCESS_COLORSCALE = [
    [0.00, "rgba(239,68,68,0.78)"],
    [0.24, "rgba(248,113,113,0.58)"],
    [0.50, "rgba(148,163,184,0.18)"],
    [0.76, "rgba(52,211,153,0.58)"],
    [1.00, "rgba(5,150,105,0.78)"],
]
REPORT_HEATMAP_TEXT_COLOR = "#0F172A"
REPORT_HEATMAP_FONT_FAMILY = "Noto Sans TC, Segoe UI, sans-serif"
_INDICATOR_COLUMNS = [
    "sma_5",
    "sma_20",
    "sma_60",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "vwap",
    "atr_14",
    "obv",
    "stoch_k",
    "stoch_d",
    "mfi_14",
]


def export_tw_etf_report_artifact(
    *,
    store: HistoryStore,
    symbol: str,
    out: str | None = None,
    ytd_start: str | None = None,
    ytd_end: str | None = None,
    compare_start: str | None = None,
    compare_end: str | None = None,
    backtest_start: str | None = None,
    backtest_end: str | None = None,
    daily_lookback_days: int = 14,
    force: bool = False,
    sync_constituents: bool = True,
    log_dir: str | Path | None = None,
) -> dict[str, object]:
    etf_code = str(symbol or "").strip().upper()
    if not etf_code:
        raise ValueError("symbol is required")

    app_module = _load_app_module()
    periods = resolve_tw_etf_super_export_periods(
        ytd_start=ytd_start,
        ytd_end=ytd_end,
        compare_start=compare_start,
        compare_end=compare_end,
    )
    target_trade_token = (
        str(_call_app_quiet(app_module._resolve_latest_tw_trade_day_token, periods["ytd_end"])).strip()
        or periods["ytd_end"]
    )
    report_dir = resolve_tw_etf_report_output_dir(
        symbol=etf_code,
        trade_token=target_trade_token,
        out=out,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    external_log_dir = Path(log_dir).expanduser() if str(log_dir or "").strip() else DEFAULT_TW_ETF_REPORT_LOG_DIR
    external_log_dir.mkdir(parents=True, exist_ok=True)

    constituent_sync_summary: dict[str, object] = {}
    if bool(sync_constituents):
        constituent_sync_summary = sync_tw_etf_constituent_snapshots(
            store=store,
            symbols=[etf_code],
            force=bool(force),
            max_workers=4,
            full_refresh=False,
            log_dir=log_dir or DEFAULT_TW_ETF_CONSTITUENT_LOG_DIR,
        )

    refresh_summary = _sync_tw_etf_super_export_sources(
        store=store,
        app_module=app_module,
        daily_lookback_days=max(1, int(daily_lookback_days)),
        force=bool(force),
        target_trade_date=target_trade_token,
    )
    overview_df, daily_market_df, margin_df, mis_df, three_df, aum_track_df, main_meta = _build_single_etf_frames(
        app_module=app_module,
        store=store,
        symbol=etf_code,
        periods=periods,
        target_trade_token=target_trade_token,
        daily_lookback_days=max(1, int(daily_lookback_days)),
    )
    if overview_df.empty:
        raise RuntimeError(f"{etf_code}: no overview row found in TW ETF export universe")

    constituent_rows, constituent_source, constituent_issue = _resolve_single_etf_constituents(
        store=store,
        app_module=app_module,
        symbol=etf_code,
        force=bool(force),
    )
    constituents_df = pd.DataFrame(constituent_rows)

    range_start_dt, range_end_dt = _resolve_backtest_window(
        backtest_start=backtest_start,
        backtest_end=backtest_end,
    )
    sync_symbols_if_needed(
        store=store,
        market="TW",
        symbols=[etf_code],
        start=range_start_dt,
        end=range_end_dt,
        parallel=False,
        max_workers=1,
        mode="backfill",
        min_rows=120,
    )
    bars = normalize_ohlcv_frame(
        store.load_daily_bars(symbol=etf_code, market="TW", start=range_start_dt, end=range_end_dt)
    ).sort_index()
    indicator_timeseries_df, indicator_snapshot_df, indicator_summary_md = _build_indicator_outputs(
        symbol=etf_code,
        bars=bars,
    )

    issues_rows: list[dict[str, object]] = []
    for issue in list(refresh_summary.get("issues", [])) if isinstance(refresh_summary, dict) else []:
        issues_rows.append({"section": "refresh", "issue": str(issue)})
    if constituent_issue:
        issues_rows.append({"section": "constituents", "issue": constituent_issue})

    backtest_chart_path = report_dir / f"{etf_code}_backtest.png"
    try:
        export_backtest_chart_artifact(
            store=store,
            symbols=[etf_code],
            layout="single",
            market="TW",
            start=range_start_dt,
            end=range_end_dt,
            strategy="buy_hold",
            benchmark_choice="auto",
            initial_capital=1_000_000.0,
            fee_rate=None,
            sell_tax=None,
            slippage=None,
            sync_before_run=True,
            use_split_adjustment=True,
            use_total_return_adjustment=True,
            theme="soft-gray",
            width=1600,
            height=900,
            scale=2,
            out=str(backtest_chart_path),
            out_dir=None,
        )
    except Exception as exc:
        issues_rows.append({"section": "backtest_chart", "issue": str(exc)})

    aum_chart_path = report_dir / f"{etf_code}_aum_track.png"
    try:
        _write_aum_chart(
            frame=aum_track_df,
            output_path=aum_chart_path,
            symbol=etf_code,
        )
    except Exception as exc:
        issues_rows.append({"section": "aum_chart", "issue": str(exc)})

    heatmap_chart_path = report_dir / f"{etf_code}_constituent_heatmap.png"
    try:
        if not _write_constituent_heatmap(
            store=store,
            symbol=etf_code,
            constituent_rows=constituent_rows,
            output_path=heatmap_chart_path,
            start_dt=range_start_dt,
            end_dt=range_end_dt,
        ):
            issues_rows.append(
                {
                    "section": "constituent_heatmap",
                    "issue": "constituent heatmap skipped because TW constituent rows were unavailable",
                }
            )
    except Exception as exc:
        issues_rows.append({"section": "constituent_heatmap", "issue": str(exc)})

    dataset_entries = _build_report_sync_log_entries(
        requested_trade_date=target_trade_token,
        refresh_summary=refresh_summary,
        constituent_sync_summary=constituent_sync_summary,
        symbol=etf_code,
        constituent_rows=constituent_rows,
        constituent_source=constituent_source,
        constituent_issue=constituent_issue,
        overview_df=overview_df,
        daily_market_df=daily_market_df,
        margin_df=margin_df,
        mis_df=mis_df,
        three_df=three_df,
        aum_track_df=aum_track_df,
    )
    bundle_log_paths = write_sync_log_files(
        log_dir=report_dir,
        file_stem=f"{etf_code}_sync_log",
        title=f"{etf_code} TW ETF Report Sync Log",
        entries=dataset_entries,
        meta={
            "symbol": etf_code,
            "trade_date_anchor": target_trade_token,
            "sync_constituents": bool(sync_constituents),
        },
    )
    external_log_paths = write_sync_log_files(
        log_dir=external_log_dir,
        file_stem=f"tw_etf_report_sync_{etf_code}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        title=f"{etf_code} TW ETF Report Sync Log",
        entries=dataset_entries,
        meta={
            "symbol": etf_code,
            "trade_date_anchor": target_trade_token,
            "report_dir": str(report_dir),
        },
    )

    _write_csv(report_dir / f"{etf_code}_overview.csv", overview_df)
    _write_csv(report_dir / f"{etf_code}_daily_market.csv", daily_market_df)
    _write_csv(report_dir / f"{etf_code}_margin.csv", margin_df)
    _write_csv(report_dir / f"{etf_code}_mis.csv", mis_df)
    _write_csv(report_dir / f"{etf_code}_three_investors.csv", three_df)
    _write_csv(report_dir / f"{etf_code}_aum_track.csv", aum_track_df)
    _write_csv(report_dir / f"{etf_code}_constituents.csv", constituents_df)
    _write_csv(report_dir / f"{etf_code}_indicators_snapshot.csv", indicator_snapshot_df)
    _write_csv(report_dir / f"{etf_code}_indicators_timeseries.csv", indicator_timeseries_df)
    _write_csv(report_dir / f"{etf_code}_issues.csv", pd.DataFrame(issues_rows))
    (report_dir / f"{etf_code}_indicators_summary.md").write_text(
        indicator_summary_md,
        encoding="utf-8",
    )
    summary_md = _build_report_summary_markdown(
        symbol=etf_code,
        trade_date_anchor=target_trade_token,
        overview_df=overview_df,
        daily_market_df=daily_market_df,
        margin_df=margin_df,
        mis_df=mis_df,
        three_df=three_df,
        aum_track_df=aum_track_df,
        constituents_df=constituents_df,
        indicator_snapshot_df=indicator_snapshot_df,
        issues_rows=issues_rows,
        dataset_entries=dataset_entries,
        report_dir=report_dir,
        bundle_log_paths=bundle_log_paths,
    )
    (report_dir / f"{etf_code}_summary.md").write_text(summary_md, encoding="utf-8")

    output_files = sorted(
        [
            path.name
            for path in report_dir.iterdir()
            if path.is_file()
        ]
    )
    return {
        "symbol": etf_code,
        "report_dir": str(report_dir),
        "trade_date_anchor": target_trade_token,
        "ytd_start": periods["ytd_start"],
        "ytd_end": periods["ytd_end"],
        "compare_start": periods["compare_start"],
        "compare_end": periods["compare_end"],
        "backtest_start": range_start_dt.date().isoformat(),
        "backtest_end": range_end_dt.date().isoformat(),
        "file_count": len(output_files),
        "files": output_files,
        "issues": [str(row.get("issue") or "").strip() for row in issues_rows if str(row.get("issue") or "").strip()],
        "bundle_log_json_path": bundle_log_paths["json_path"],
        "bundle_log_markdown_path": bundle_log_paths["markdown_path"],
        "external_log_json_path": external_log_paths["json_path"],
        "external_log_markdown_path": external_log_paths["markdown_path"],
        "constituent_sync_summary": constituent_sync_summary,
    }


def export_tw_etf_constituent_heatmap_artifact(
    *,
    store: HistoryStore,
    symbol: str,
    out: str | None = None,
    backtest_start: str | None = None,
    backtest_end: str | None = None,
    ytd_end: str | None = None,
    force: bool = False,
    sync_constituents: bool = True,
    log_dir: str | Path | None = None,
) -> dict[str, object]:
    etf_code = str(symbol or "").strip().upper()
    if not etf_code:
        raise ValueError("symbol is required")

    app_module = _load_app_module()
    periods = resolve_tw_etf_super_export_periods(ytd_end=ytd_end)
    target_trade_token = (
        str(_call_app_quiet(app_module._resolve_latest_tw_trade_day_token, periods["ytd_end"])).strip()
        or periods["ytd_end"]
    )
    if bool(sync_constituents):
        sync_tw_etf_constituent_snapshots(
            store=store,
            symbols=[etf_code],
            force=bool(force),
            max_workers=4,
            full_refresh=False,
            log_dir=log_dir or DEFAULT_TW_ETF_CONSTITUENT_LOG_DIR,
        )
    range_start_dt, range_end_dt = _resolve_backtest_window(
        backtest_start=backtest_start,
        backtest_end=backtest_end,
    )
    output_path = resolve_tw_etf_constituent_heatmap_output_path(
        symbol=etf_code,
        trade_token=target_trade_token,
        out=out,
    )
    issues: list[str] = []
    universe_id = f"TW:{etf_code}"
    cached_rows_df = (
        _load_matching_cached_constituent_heatmap_rows(
            store=store,
            universe_id=universe_id,
            start_dt=range_start_dt,
            end_dt=range_end_dt,
        )
        if not bool(sync_constituents)
        else pd.DataFrame()
    )
    if not cached_rows_df.empty:
        _write_constituent_heatmap_rows(
            symbol=etf_code,
            rows_df=cached_rows_df,
            output_path=output_path,
        )
        issues.append("reused heatmap_runs cache")
        return {
            "symbol": etf_code,
            "trade_date_anchor": target_trade_token,
            "output_path": str(output_path),
            "backtest_start": range_start_dt.date().isoformat(),
            "backtest_end": range_end_dt.date().isoformat(),
            "constituent_count": int(len(cached_rows_df)),
            "constituent_source": "heatmap_runs_cache",
            "issues": issues,
        }

    constituent_rows, constituent_source, constituent_issue = _resolve_single_etf_constituents(
        store=store,
        app_module=app_module,
        symbol=etf_code,
        force=bool(force),
    )
    if constituent_issue:
        issues.append(str(constituent_issue))

    rows_df, benchmark_symbol = _compute_constituent_heatmap_rows(
        store=store,
        constituent_rows=constituent_rows,
        start_dt=range_start_dt,
        end_dt=range_end_dt,
    )
    if rows_df.empty:
        raise RuntimeError("constituent heatmap skipped because TW constituent rows were unavailable")
    _write_constituent_heatmap_rows(
        symbol=etf_code,
        rows_df=rows_df,
        output_path=output_path,
    )
    payload = _build_constituent_heatmap_cache_payload(
        rows_df=rows_df,
        benchmark_symbol=benchmark_symbol,
        start_dt=range_start_dt,
        end_dt=range_end_dt,
        universe_count=len(constituent_rows),
    )
    store.save_heatmap_run(universe_id=universe_id, payload=payload)
    return {
        "symbol": etf_code,
        "trade_date_anchor": target_trade_token,
        "output_path": str(output_path),
        "backtest_start": range_start_dt.date().isoformat(),
        "backtest_end": range_end_dt.date().isoformat(),
        "constituent_count": int(len(rows_df)),
        "constituent_source": constituent_source,
        "issues": issues,
    }


def resolve_tw_etf_report_output_dir(
    *,
    symbol: str,
    trade_token: str,
    out: str | None = None,
) -> Path:
    dir_name = f"tw_etf_report_{str(symbol or '').strip().upper()}_{str(trade_token or '').strip() or 'latest'}"
    if not str(out or "").strip():
        return DEFAULT_TW_ETF_REPORT_ROOT / dir_name
    candidate = Path(str(out).strip()).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate / dir_name
    if str(candidate).endswith("/"):
        return candidate / dir_name
    if candidate.suffix:
        return candidate
    return candidate / dir_name


def resolve_tw_etf_constituent_heatmap_output_path(
    *,
    symbol: str,
    trade_token: str,
    out: str | None = None,
) -> Path:
    code = str(symbol or "").strip().upper()
    token = str(trade_token or "").strip() or "latest"
    file_name = f"{code}_constituent_heatmap_{token}.png"
    if not str(out or "").strip():
        return resolve_tw_etf_report_output_dir(symbol=code, trade_token=token, out=None) / f"{code}_constituent_heatmap.png"
    candidate = Path(str(out).strip()).expanduser()
    if candidate.suffix.lower() == ".png":
        return candidate
    if candidate.exists() and candidate.is_dir():
        return candidate / file_name
    if str(candidate).endswith("/"):
        return candidate / file_name
    if candidate.suffix:
        return candidate
    return candidate / file_name


def _build_single_etf_frames(
    *,
    app_module,
    store: HistoryStore,
    symbol: str,
    periods: dict[str, str],
    target_trade_token: str,
    daily_lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    build_performance_table = app_module._build_tw_etf_all_types_performance_table
    build_aum_export_frame = getattr(app_module, "_build_tw_etf_aum_export_frame", None)
    build_daily_market_overview = app_module._build_tw_etf_daily_market_overview
    build_margin_overview = app_module._build_tw_etf_margin_overview
    build_mis_overview = app_module._build_tw_etf_mis_overview
    build_three_investors_overview = app_module._build_tw_etf_three_investors_overview
    table_df, main_meta = _call_app_quiet(
        build_performance_table,
        ytd_start_yyyymmdd=periods["ytd_start"],
        ytd_end_yyyymmdd=periods["ytd_end"],
        compare_start_yyyymmdd=periods["compare_start"],
        compare_end_yyyymmdd=periods["compare_end"],
    )
    daily_market_df, _ = _call_app_quiet(
        build_daily_market_overview,
        lookback_days=max(30, int(daily_lookback_days)),
        top_n=0,
        target_trade_date=target_trade_token,
    )
    margin_df, _ = _call_app_quiet(
        build_margin_overview,
        top_n=0,
        target_trade_date=target_trade_token,
    )
    mis_df, _ = _call_app_quiet(
        build_mis_overview,
        top_n=0,
        target_trade_date=target_trade_token,
    )
    three_df, _ = _call_app_quiet(
        build_three_investors_overview,
        etf_codes=(symbol,),
        lookback_days=max(1, int(daily_lookback_days)),
        top_n=0,
        target_trade_date=target_trade_token,
    )
    aum_history_df = store.load_tw_etf_aum_history(etf_codes=(symbol,), keep_days=0)
    aum_track_df = _prepare_aum_track_frame(aum_history_df)
    aum_summary_frame = (
        _call_app_quiet(build_aum_export_frame, aum_history_df)
        if callable(build_aum_export_frame)
        else pd.DataFrame()
    )
    main_frame = build_tw_etf_all_types_main_export_frame(table_df=table_df, meta=main_meta)
    overview_df = build_tw_etf_super_export_table(
        main_frame=main_frame,
        aum_frame=aum_summary_frame,
        daily_market_frame=daily_market_df,
        margin_frame=margin_df,
        mis_frame=mis_df,
        three_investors_frame=three_df,
    )
    return (
        _filter_symbol_frame(overview_df, symbol),
        _filter_symbol_frame(daily_market_df, symbol),
        _filter_symbol_frame(margin_df, symbol),
        _filter_symbol_frame(mis_df, symbol),
        _filter_symbol_frame(three_df, symbol),
        aum_track_df,
        dict(main_meta or {}),
    )


def _prepare_aum_track_frame(history_df: pd.DataFrame) -> pd.DataFrame:
    frame = history_df.copy() if isinstance(history_df, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=["etf_code", "etf_name", "trade_date", "aum_billion", "aum_change_billion", "aum_change_pct"])
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    work["aum_billion"] = pd.to_numeric(work.get("aum_billion"), errors="coerce")
    work["aum_change_billion"] = work["aum_billion"].diff()
    prev = work["aum_billion"].shift(1)
    work["aum_change_pct"] = (work["aum_billion"] / prev - 1.0) * 100.0
    work["trade_date"] = work["trade_date"].dt.strftime("%Y-%m-%d")
    return work


def _resolve_single_etf_constituents(
    *,
    store: HistoryStore,
    app_module,
    symbol: str,
    force: bool,
) -> tuple[list[dict[str, object]], str, str]:
    snapshot = load_latest_tw_etf_constituent_snapshot(store=store, symbol=symbol)
    if isinstance(snapshot, dict) and list(snapshot.get("rows") or []):
        return (
            list(snapshot.get("rows") or []),
            str(snapshot.get("source") or "").strip(),
            str(snapshot.get("issue") or "").strip(),
        )
    service = app_module._market_service()
    rows, source, issue = _call_app_quiet(
        app_module._load_etf_constituents_rows,
        service=service,
        etf_code=symbol,
        force_refresh_constituents=bool(force),
    )
    return list(rows or []), str(source or "").strip(), str(issue or "").strip()


def _resolve_backtest_window(
    *,
    backtest_start: str | None,
    backtest_end: str | None,
) -> tuple[datetime, datetime]:
    end_dt = (
        pd.Timestamp(backtest_end).to_pydatetime().replace(tzinfo=timezone.utc)
        if str(backtest_end or "").strip()
        else datetime.now(tz=timezone.utc)
    )
    start_dt = (
        pd.Timestamp(backtest_start).to_pydatetime().replace(tzinfo=timezone.utc)
        if str(backtest_start or "").strip()
        else datetime(2000, 1, 1, tzinfo=timezone.utc)
    )
    if start_dt >= end_dt:
        raise ValueError("backtest_start must be earlier than backtest_end")
    return start_dt, end_dt


def _load_matching_cached_constituent_heatmap_rows(
    *,
    store: HistoryStore,
    universe_id: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    cached_run = store.load_latest_heatmap_run(universe_id)
    payload = cached_run.payload if cached_run else {}
    if not isinstance(payload, dict):
        return pd.DataFrame()
    if str(payload.get("strategy") or "").strip() != "buy_hold":
        return pd.DataFrame()
    if str(payload.get("benchmark_symbol") or "").strip().upper() != "^TWII":
        return pd.DataFrame()
    if str(payload.get("start_date") or "").strip() != start_dt.date().isoformat():
        return pd.DataFrame()
    if str(payload.get("end_date") or "").strip() != end_dt.date().isoformat():
        return pd.DataFrame()
    selected_count = payload.get("selected_count")
    universe_count = payload.get("universe_count")
    if selected_count is not None and universe_count is not None:
        if int(selected_count or 0) != int(universe_count or 0):
            return pd.DataFrame()
    rows_df = pd.DataFrame(payload.get("rows") or [])
    if rows_df.empty:
        return pd.DataFrame()
    return rows_df


def _build_constituent_heatmap_cache_payload(
    *,
    rows_df: pd.DataFrame,
    benchmark_symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    universe_count: int,
) -> dict[str, object]:
    clean_df = rows_df.copy()
    clean_df = clean_df.where(pd.notna(clean_df), None)
    return {
        "rows": clean_df.to_dict(orient="records"),
        "benchmark_symbol": str(benchmark_symbol or "").strip().upper(),
        "selected_count": int(max(0, universe_count)),
        "universe_count": int(max(0, universe_count)),
        "start_date": start_dt.date().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "strategy": "buy_hold",
        "strategy_label": "Buy & Hold",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _filter_symbol_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "代碼" not in frame.columns:
        return pd.DataFrame(columns=getattr(frame, "columns", []))
    code_series = frame["代碼"].astype(str).str.strip().str.upper()
    return frame.loc[code_series == str(symbol).strip().upper()].reset_index(drop=True)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    data = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    safe = data.astype("object").where(pd.notna(data), None)
    safe.to_csv(path, index=False, encoding="utf-8-sig")


def _build_indicator_outputs(
    *,
    symbol: str,
    bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        empty_snapshot = pd.DataFrame([{"symbol": symbol, "status": "no_bars"}])
        return pd.DataFrame(), empty_snapshot, f"# {symbol} 指標摘要\n\n- 無可用日 K 資料。\n"
    work = bars.copy()
    work = add_indicators(work)
    work = work.reset_index().rename(columns={"index": "date"})
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    timeseries_cols = [
        col
        for col in ["date", "open", "high", "low", "close", "volume", *_INDICATOR_COLUMNS]
        if col in work.columns
    ]
    timeseries_df = work[timeseries_cols].copy()
    last_row = timeseries_df.iloc[-1].to_dict()
    close_val = _safe_float(last_row.get("close"))
    rsi_val = _safe_float(last_row.get("rsi_14"))
    macd_val = _safe_float(last_row.get("macd"))
    macd_signal_val = _safe_float(last_row.get("macd_signal"))
    k_val = _safe_float(last_row.get("stoch_k"))
    d_val = _safe_float(last_row.get("stoch_d"))
    bb_upper = _safe_float(last_row.get("bb_upper"))
    bb_lower = _safe_float(last_row.get("bb_lower"))
    atr_val = _safe_float(last_row.get("atr_14"))
    snapshot_row = {
        "symbol": symbol,
        **last_row,
        "rsi_state": _describe_rsi_state(rsi_val),
        "macd_state": _describe_macd_state(macd_val, macd_signal_val),
        "kd_state": _describe_kd_state(k_val, d_val),
        "bb_position": _describe_bollinger_position(close_val, bb_lower, bb_upper),
        "atr_pct": ((atr_val / close_val) * 100.0) if atr_val is not None and close_val not in {None, 0} else None,
    }
    snapshot_df = pd.DataFrame([snapshot_row])
    summary_lines = [
        f"# {symbol} 指標摘要",
        "",
        f"- 日期：`{str(last_row.get('date') or '')}`",
        f"- RSI(14)：`{_fmt_num(rsi_val)}`，{snapshot_row['rsi_state']}",
        f"- MACD：`{_fmt_num(macd_val)}` / Signal `{'{:.4f}'.format(macd_signal_val) if macd_signal_val is not None else 'n/a'}`，{snapshot_row['macd_state']}",
        f"- KD：K `{'{:.2f}'.format(k_val) if k_val is not None else 'n/a'}` / D `{'{:.2f}'.format(d_val) if d_val is not None else 'n/a'}`，{snapshot_row['kd_state']}",
        f"- 布林位置：{snapshot_row['bb_position']}",
        f"- ATR/Close：`{_fmt_pct(snapshot_row.get('atr_pct'))}`",
    ]
    return timeseries_df, snapshot_df, "\n".join(summary_lines).strip() + "\n"


def _write_aum_chart(*, frame: pd.DataFrame, output_path: Path, symbol: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError("no AUM history available")
    x = pd.to_datetime(frame["trade_date"], errors="coerce")
    y = pd.to_numeric(frame["aum_billion"], errors="coerce")
    work = pd.DataFrame({"trade_date": x, "aum_billion": y}).dropna()
    if work.empty:
        raise RuntimeError("AUM history is empty after normalization")
    latest_value = float(work["aum_billion"].iloc[-1])
    fig = go.Figure(
        data=[
            go.Scatter(
                x=work["trade_date"],
                y=work["aum_billion"],
                mode="lines+markers",
                name="基金規模(億)",
                line=dict(color="#1D4ED8", width=2.4),
                marker=dict(size=6, color="#1D4ED8"),
            )
        ]
    )
    fig.update_layout(
        title=f"{symbol} 基金規模追蹤",
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", size=14),
        margin=dict(l=40, r=30, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.18)"),
        yaxis=dict(title="億元", showgrid=True, gridcolor="rgba(148,163,184,0.18)"),
        annotations=[
            dict(
                x=1.0,
                y=1.12,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                text=f"最新規模：{latest_value:,.2f} 億",
                font=dict(size=14, color="#0F172A"),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(15,23,42,0.18)",
                borderwidth=1,
                borderpad=6,
            )
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", width=1600, height=700, scale=2)


def _build_constituent_heatmap_figure(
    *,
    symbol: str,
    rows_df: pd.DataFrame,
) -> tuple[go.Figure, int, int]:
    if not isinstance(rows_df, pd.DataFrame) or rows_df.empty:
        raise RuntimeError("no constituent heatmap rows available")

    frame = rows_df.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame["name"] = frame["name"].astype(str).str.strip()
    frame["excess_pct"] = pd.to_numeric(frame["excess_pct"], errors="coerce")
    if "strategy_return_pct" not in frame.columns and "asset_return_pct" in frame.columns:
        frame["strategy_return_pct"] = frame["asset_return_pct"]
    frame["strategy_return_pct"] = pd.to_numeric(frame["strategy_return_pct"], errors="coerce")
    frame["benchmark_return_pct"] = pd.to_numeric(frame["benchmark_return_pct"], errors="coerce")
    frame = frame.dropna(subset=["excess_pct", "strategy_return_pct", "benchmark_return_pct"])
    if frame.empty:
        raise RuntimeError("constituent heatmap rows became empty after normalization")
    frame = frame.sort_values("excess_pct", ascending=False).reset_index(drop=True)

    tiles_per_row = 8
    tile_rows = int(math.ceil(len(frame) / tiles_per_row))
    z = np.full((tile_rows, tiles_per_row), np.nan)
    text = np.full((tile_rows, tiles_per_row), "", dtype=object)
    custom = np.empty((tile_rows, tiles_per_row, 5), dtype=object)
    custom[:, :, :] = None

    for i, row in frame.iterrows():
        r = i // tiles_per_row
        c = i % tiles_per_row
        excess_pct = float(row["excess_pct"])
        strategy_return_pct = float(row["strategy_return_pct"])
        benchmark_return_pct = float(row["benchmark_return_pct"])
        label = str(row["symbol"]).strip()
        name_text = str(row.get("name", "")).strip()
        weight_val = pd.to_numeric(row.get("weight_pct"), errors="coerce")
        weight_text = f"{float(weight_val):.2f}%" if pd.notna(weight_val) else "—"
        z[r, c] = excess_pct
        if name_text:
            text[r, c] = f"<b>{label}</b><br>{name_text}<br>權重 {weight_text}<br>{excess_pct:+.2f}%"
        else:
            text[r, c] = f"<b>{label}</b><br>權重 {weight_text}<br>{excess_pct:+.2f}%"
        custom[r, c] = [
            strategy_return_pct,
            benchmark_return_pct,
            str(row.get("status") or ""),
            name_text,
            weight_text,
        ]

    finite = np.abs(frame["excess_pct"].to_numpy(dtype=float))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        max_abs = 1.0
    else:
        raw_max = float(np.nanmax(finite))
        p85 = float(np.nanpercentile(finite, 85))
        capped = min(raw_max, p85 * 1.35)
        max_abs = max(0.01, max(0.15, capped))

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                text=text,
                texttemplate="%{text}",
                textfont=dict(
                    size=12,
                    color=REPORT_HEATMAP_TEXT_COLOR,
                    family=REPORT_HEATMAP_FONT_FAMILY,
                ),
                customdata=custom,
                zmin=-max_abs,
                zmax=max_abs,
                zmid=0.0,
                colorscale=REPORT_HEATMAP_EXCESS_COLORSCALE,
                xgap=6,
                ygap=6,
                hoverongaps=False,
                colorbar=dict(title="相對大盤 %", thickness=14, len=0.8),
                hovertemplate=(
                    "公司：%{customdata[3]}<br>"
                    "策略報酬：%{customdata[0]:+.2f}%<br>"
                    "大盤報酬：%{customdata[1]:+.2f}%<br>"
                    "超額：%{z:+.2f}%<br>"
                    "狀態：%{customdata[2]}<br>"
                    "權重：%{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")

    width = 1280
    height = max(280, 90 * tile_rows)
    fig.update_layout(
        title=f"{symbol} 成分股熱力圖（相對大盤）",
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=REPORT_HEATMAP_TEXT_COLOR, size=14, family=REPORT_HEATMAP_FONT_FAMILY),
        margin=dict(l=10, r=10, t=52, b=10),
        width=width,
        height=height,
    )
    return fig, width, height


def _compute_constituent_heatmap_rows(
    *,
    store: HistoryStore,
    constituent_rows: list[dict[str, object]],
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[pd.DataFrame, str]:
    tw_rows = []
    for row in constituent_rows:
        if not isinstance(row, dict):
            continue
        market = str(row.get("market") or "").strip().upper()
        code = str(row.get("tw_code") or "").strip().upper()
        if market == "TW" and code:
            tw_rows.append(row)
    weight_map: dict[str, float] = {}
    for row in tw_rows:
        code = str(row.get("tw_code") or "").strip().upper()
        if not code:
            continue
        weight_val = pd.to_numeric(row.get("weight_pct"), errors="coerce")
        if pd.isna(weight_val):
            continue
        existing = weight_map.get(code)
        if existing is None or float(weight_val) > float(existing):
            weight_map[code] = float(weight_val)
    run_symbols = sorted({str(row.get("tw_code") or "").strip().upper() for row in tw_rows if str(row.get("tw_code") or "").strip()})
    if not run_symbols:
        return pd.DataFrame(), ""
    benchmark_result = load_tw_benchmark_close(
        store=store,
        choice="twii",
        start_dt=start_dt,
        end_dt=end_dt,
        sync_first=False,
        allow_twii_fallback=False,
    )
    benchmark_close = benchmark_result.close
    if len(benchmark_close) < 2:
        return pd.DataFrame(), ""
    prepared = prepare_heatmap_bars(
        store=store,
        symbols=run_symbols,
        start_dt=start_dt,
        end_dt=end_dt,
        min_required=20,
        sync_before_run=False,
        parallel_sync=True,
        lazy_sync_on_insufficient=True,
        normalize_ohlcv_frame=normalize_ohlcv_frame,
    )
    name_map = {
        str(row.get("tw_code") or "").strip().upper(): str(row.get("name") or "").strip()
        for row in tw_rows
    }
    from backtest import CostModel

    rows = compute_heatmap_rows(
        run_symbols=run_symbols,
        bars_cache=prepared.bars_cache,
        benchmark_close=benchmark_close,
        strategy="buy_hold",
        strategy_params={},
        cost_model=CostModel(fee_rate=0.001425, sell_tax_rate=0.003, slippage_rate=0.0),
        name_map=name_map,
        min_required=20,
        progress_callback=None,
        max_workers=1,
    )
    if not rows:
        return pd.DataFrame(), ""
    for item in rows:
        if not isinstance(item, dict):
            continue
        symbol_token = str(item.get("symbol") or "").strip().upper()
        item["weight_pct"] = weight_map.get(symbol_token)
    return pd.DataFrame(rows), str(benchmark_result.symbol_used or "").strip().upper()


def _write_constituent_heatmap_rows(
    *,
    symbol: str,
    rows_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, width, height = _build_constituent_heatmap_figure(symbol=symbol, rows_df=rows_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", width=width, height=height, scale=2)
 

def _write_constituent_heatmap(
    *,
    store: HistoryStore,
    symbol: str,
    constituent_rows: list[dict[str, object]],
    output_path: Path,
    start_dt: datetime,
    end_dt: datetime,
) -> bool:
    rows_df, _ = _compute_constituent_heatmap_rows(
        store=store,
        constituent_rows=constituent_rows,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    if rows_df.empty:
        return False
    _write_constituent_heatmap_rows(
        symbol=symbol,
        rows_df=rows_df,
        output_path=output_path,
    )
    return True


def _build_report_sync_log_entries(
    *,
    requested_trade_date: str,
    refresh_summary: dict[str, object],
    constituent_sync_summary: dict[str, object],
    symbol: str,
    constituent_rows: list[dict[str, object]],
    constituent_source: str,
    constituent_issue: str,
    overview_df: pd.DataFrame,
    daily_market_df: pd.DataFrame,
    margin_df: pd.DataFrame,
    mis_df: pd.DataFrame,
    three_df: pd.DataFrame,
    aum_track_df: pd.DataFrame,
) -> list[dict[str, object]]:
    requested_iso = _normalize_trade_date(requested_trade_date)
    entries: list[dict[str, object]] = []
    main_summary = dict(refresh_summary.get("main") or {})
    entries.append(
        build_sync_log_entry(
            dataset_name="main_snapshot",
            status=_map_main_status(main_summary),
            requested_trade_date=requested_iso,
            used_trade_date=_normalize_trade_date(main_summary.get("used_trade_date")),
            row_count_after=len(overview_df),
            source="twse_snapshot",
        )
    )
    entries.append(
        _build_dataset_entry_from_sync_summary(
            dataset_name="daily_market",
            requested_iso=requested_iso,
            summary=dict(refresh_summary.get("daily_market") or {}),
            row_count_after=len(daily_market_df),
            source="twse_etf_daily_market",
        )
    )
    entries.append(
        _build_dataset_entry_from_sync_summary(
            dataset_name="margin",
            requested_iso=requested_iso,
            summary=dict(refresh_summary.get("margin") or {}),
            row_count_after=len(margin_df),
            source="twse_margin",
        )
    )
    entries.append(
        _build_dataset_entry_from_sync_summary(
            dataset_name="mis",
            requested_iso=requested_iso,
            summary=dict(refresh_summary.get("mis") or {}),
            row_count_after=len(mis_df),
            source="twse_mis",
        )
    )
    three_summary = dict(refresh_summary.get("three_investors") or {})
    entries.append(
        build_sync_log_entry(
            dataset_name="three_investors",
            status=_map_generic_status(three_summary.get("status"), fallback_row_count=len(three_df)),
            requested_trade_date=requested_iso,
            used_trade_date=_normalize_trade_date(three_summary.get("used_trade_date")),
            row_count_after=len(three_df),
            updated_rows=int(three_summary.get("row_count") or 0),
            source="twse_t86",
        )
    )
    aum_used_date = ""
    if not aum_track_df.empty and "trade_date" in aum_track_df.columns:
        aum_used_date = str(aum_track_df["trade_date"].iloc[-1]).strip()
    entries.append(
        build_sync_log_entry(
            dataset_name="aum_track",
            status=_map_aum_status(
                summary=dict(refresh_summary.get("aum_track") or {}),
                requested_iso=requested_iso,
                used_iso=aum_used_date,
                row_count=len(aum_track_df),
            ),
            requested_trade_date=requested_iso,
            used_trade_date=aum_used_date,
            row_count_after=len(aum_track_df),
            updated_rows=int(dict(refresh_summary.get("aum_track") or {}).get("updated") or 0),
            source="twse_etf_aum",
        )
    )
    constituent_entry = None
    for entry in list(constituent_sync_summary.get("entries") or []):
        if str(entry.get("dataset_name") or "").strip() == f"constituents:{symbol}":
            constituent_entry = dict(entry)
            break
    entries.append(
        build_sync_log_entry(
            dataset_name="constituents",
            status=str((constituent_entry or {}).get("status") or ("updated" if constituent_rows else "missing")),
            requested_trade_date=requested_iso,
            used_trade_date=str((constituent_entry or {}).get("used_trade_date") or datetime.now(tz=timezone.utc).date().isoformat()),
            row_count_after=len(constituent_rows),
            updated_rows=int((constituent_entry or {}).get("updated_rows") or (len(constituent_rows) if constituent_rows else 0)),
            source=constituent_source,
            notes=constituent_issue,
            error=str((constituent_entry or {}).get("error") or ""),
        )
    )
    return entries


def _build_dataset_entry_from_sync_summary(
    *,
    dataset_name: str,
    requested_iso: str,
    summary: dict[str, object],
    row_count_after: int,
    source: str,
) -> dict[str, object]:
    status = "unchanged"
    if int(summary.get("saved_rows") or 0) > 0 or int(summary.get("synced_days") or 0) > 0:
        status = "updated"
    elif list(summary.get("issues") or []):
        status = "error"
    latest_date = _normalize_trade_date(
        summary.get("available_date") or summary.get("latest_date") or summary.get("end_date")
    )
    if latest_date and requested_iso and latest_date < requested_iso and status not in {"error", "updated"}:
        status = "delayed"
    return build_sync_log_entry(
        dataset_name=dataset_name,
        status=status,
        requested_trade_date=requested_iso,
        used_trade_date=latest_date,
        row_count_after=row_count_after,
        updated_rows=int(summary.get("saved_rows") or 0),
        source=source,
        notes="; ".join(str(item) for item in list(summary.get("issues") or [])[:5]),
    )


def _build_report_summary_markdown(
    *,
    symbol: str,
    trade_date_anchor: str,
    overview_df: pd.DataFrame,
    daily_market_df: pd.DataFrame,
    margin_df: pd.DataFrame,
    mis_df: pd.DataFrame,
    three_df: pd.DataFrame,
    aum_track_df: pd.DataFrame,
    constituents_df: pd.DataFrame,
    indicator_snapshot_df: pd.DataFrame,
    issues_rows: list[dict[str, object]],
    dataset_entries: list[dict[str, object]],
    report_dir: Path,
    bundle_log_paths: dict[str, str],
) -> str:
    overview_row = overview_df.iloc[0].to_dict() if isinstance(overview_df, pd.DataFrame) and not overview_df.empty else {}
    lines = [
        f"# {symbol} 單檔報表包",
        "",
        f"- trade_date_anchor: `{trade_date_anchor}`",
        f"- report_dir: `{report_dir}`",
        "",
        "## Overview",
        "",
    ]
    if overview_row:
        for key in ["ETF", "類型", "YTD績效(%)", "ETF規模(億)", "收盤", "今日漲幅"]:
            if key in overview_row:
                lines.append(f"- {key}: `{overview_row.get(key)}`")
    lines.extend(
        [
            "",
            "## Dataset Status",
            "",
        ]
    )
    for entry in dataset_entries:
        lines.append(
            f"- {entry.get('dataset_name')}: `{entry.get('status')}` "
            f"(used=`{entry.get('used_trade_date') or ''}` rows={entry.get('row_count_after') or 0})"
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- daily_market rows: `{len(daily_market_df)}`",
            f"- margin rows: `{len(margin_df)}`",
            f"- mis rows: `{len(mis_df)}`",
            f"- three_investors rows: `{len(three_df)}`",
            f"- aum_track rows: `{len(aum_track_df)}`",
            f"- constituents rows: `{len(constituents_df)}`",
            f"- indicators snapshot rows: `{len(indicator_snapshot_df)}`",
            "",
            "## Files",
            "",
        ]
    )
    for path in sorted([item.name for item in report_dir.iterdir() if item.is_file()]):
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Logs",
            "",
            f"- json: `{Path(bundle_log_paths['json_path']).name}`",
            f"- markdown: `{Path(bundle_log_paths['markdown_path']).name}`",
        ]
    )
    if issues_rows:
        lines.extend(["", "## Issues", ""])
        for row in issues_rows:
            lines.append(f"- {row.get('section')}: {row.get('issue')}")
    return "\n".join(lines).strip() + "\n"


def _normalize_trade_date(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _map_main_status(summary: dict[str, object]) -> str:
    token = str(summary.get("status") or "").strip().lower()
    if token == "synced":
        return "updated"
    if token in {"fallback", "error"}:
        return token
    return "unchanged"


def _map_generic_status(status: object, *, fallback_row_count: int) -> str:
    token = str(status or "").strip().lower()
    if token == "synced":
        return "updated"
    if token in {"fallback", "error"}:
        return token
    return "updated" if int(fallback_row_count) > 0 else "missing"


def _map_aum_status(*, summary: dict[str, object], requested_iso: str, used_iso: str, row_count: int) -> str:
    token = str(summary.get("status") or "").strip().lower()
    if token == "error":
        return "error"
    if token == "empty" and row_count <= 0:
        return "missing"
    if not used_iso:
        return "missing" if row_count <= 0 else "updated"
    if requested_iso and used_iso and used_iso < requested_iso:
        return "delayed"
    if token == "synced":
        return "updated"
    return "unchanged" if row_count > 0 else "missing"


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _describe_rsi_state(value: float | None) -> str:
    if value is None:
        return "RSI 不足"
    if value >= 70.0:
        return "RSI 過熱"
    if value <= 30.0:
        return "RSI 偏弱"
    return "RSI 中性"


def _describe_macd_state(macd: float | None, signal: float | None) -> str:
    if macd is None or signal is None:
        return "MACD 不足"
    return "MACD 多方" if macd >= signal else "MACD 空方"


def _describe_kd_state(k_value: float | None, d_value: float | None) -> str:
    if k_value is None or d_value is None:
        return "KD 不足"
    return "K 在 D 上方" if k_value >= d_value else "K 在 D 下方"


def _describe_bollinger_position(
    close_value: float | None,
    lower: float | None,
    upper: float | None,
) -> str:
    if close_value is None or lower is None or upper is None:
        return "布林不足"
    if close_value >= upper:
        return "接近上緣"
    if close_value <= lower:
        return "接近下緣"
    return "位於通道內"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _fmt_pct(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
    except Exception:
        pass
    return f"{float(value):.2f}%"
