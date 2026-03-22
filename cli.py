"""Command-line interface for data sync/bootstrap/backtest operations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import click
import pandas as pd

from backtest import CostModel
from backtest import apply_split_adjustment as apply_split_adjustment_core
from config_loader import cfg_or_env_str, get_config_source
from di import get_history_store
from services.backtest_runner import (
    BacktestExecutionInput,
    default_cost_params,
    execute_backtest_run,
    load_and_prepare_symbol_bars,
    parse_symbols,
)
from services.bootstrap_loader import run_market_data_bootstrap
from services.chart_export import (
    DEFAULT_CHART_THEME,
    SUPPORTED_CHART_LAYOUTS,
    SUPPORTED_CHART_STRATEGIES,
    SUPPORTED_CHART_THEMES,
    export_backtest_chart_artifact,
)
from services.etf_briefing import (
    DEFAULT_ETF_BRIEFING_NEWS_WINDOW_DAYS,
    DEFAULT_ETF_BRIEFING_SINGLE_SYMBOL,
    DEFAULT_ETF_BRIEFING_START,
    DEFAULT_ETF_BRIEFING_THEME,
    export_etf_briefing_artifact,
)
from services.sync_orchestrator import normalize_symbols, sync_symbols_if_needed
from services.tw_etf_constituent_sync import sync_tw_etf_constituent_snapshots
from services.tw_etf_daily_sync import sync_twse_etf_daily_market
from services.tw_etf_mis_sync import sync_twse_etf_mis_daily
from services.tw_etf_report import (
    export_tw_etf_constituent_heatmap_artifact,
    export_tw_etf_report_artifact,
)
from services.tw_etf_super_export import export_tw_etf_super_table_artifact


def _resolve_store():
    return get_history_store()


def _normalize_market(market: str) -> str:
    token = str(market or "TW").strip().upper()
    if token in {"TW", "OTC"}:
        return "TW"
    if token in {"US", "USA"}:
        return "US"
    return "TW"


def _infer_market_from_symbol(symbol: str) -> str:
    token = str(symbol or "").strip().upper()
    if token.isdigit() and len(token) in {4, 5, 6}:
        return "TW"
    return "US"


def _parse_iso_datetime(value: str | None, *, default: datetime) -> datetime:
    if not value:
        return default
    parsed = datetime.fromisoformat(str(value).strip())
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _apply_total_return_adjustment_cli(
    bars: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return bars, {"applied": False, "reason": "empty"}
    if "adj_close" not in bars.columns or "close" not in bars.columns:
        return bars, {"applied": False, "reason": "no_adj_close"}

    close = pd.to_numeric(bars["close"], errors="coerce")
    adj_close = pd.to_numeric(bars["adj_close"], errors="coerce")
    valid = close.gt(0) & adj_close.gt(0)
    coverage_pct = float(valid.mean() * 100.0) if len(valid) else 0.0
    if coverage_pct < 60.0:
        return bars, {"applied": False, "reason": "coverage_low", "coverage_pct": coverage_pct}

    out = bars.copy()
    factor = (adj_close / close).where(valid)
    for col in ["open", "high", "low", "close"]:
        if col in out.columns:
            base = pd.to_numeric(out[col], errors="coerce")
            adjusted = (base * factor).where(valid)
            out[col] = adjusted.where(valid, base)
    out["close"] = adj_close.where(valid, out["close"])
    return out, {"applied": True, "coverage_pct": coverage_pct}


@click.group()
def cli() -> None:
    """Realtime0052 CLI."""


@cli.command()
@click.option("--symbols", "symbols_text", help="Comma-separated symbols")
@click.option("--market", "market_text", default="TW", help="Market: TW or US")
@click.option("--days", default=30, type=int, show_default=True, help="Days to sync")
@click.option(
    "--mode",
    type=click.Choice(["backfill", "min_rows", "all"], case_sensitive=False),
    default="backfill",
    show_default=True,
    help="backfill=補區間, min_rows=補到最少資料列數, all=強制全跑",
)
@click.option("--min-rows", default=120, type=int, show_default=True, help="Min rows target for min_rows mode")
@click.option("--max-workers", default=6, type=int, show_default=True, help="Parallel workers")
def sync(
    symbols_text: str | None,
    market_text: str,
    days: int,
    mode: str,
    min_rows: int,
    max_workers: int,
) -> None:
    """Sync historical bars for symbols.

    Examples:
      uv run realtime0052 sync --help
      uv run realtime0052 sync --market TW --symbols 0050,0052 --days 60
      uv run realtime0052 sync --market US --symbols SPY,QQQ --days 120 --mode min_rows --min-rows 250
    """
    store = _resolve_store()
    market = _normalize_market(market_text)
    symbols = parse_symbols(symbols_text or "")
    if not symbols:
        symbols = ["0050", "0052"] if market == "TW" else ["SPY", "QQQ"]
    symbols = normalize_symbols(symbols)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=max(1, int(days)))

    reports, plan = sync_symbols_if_needed(
        store=store,
        market=market,
        symbols=symbols,
        start=start,
        end=end,
        parallel=True,
        max_workers=max(1, int(max_workers)),
        mode=str(mode).lower(),
        min_rows=max(1, int(min_rows)),
    )

    click.echo(f"market={market} symbols={','.join(symbols)}")
    click.echo(
        f"synced={len(plan.synced_symbols)} skipped={len(plan.skipped_symbols)} issues={len(plan.issues)}"
    )
    for sym in plan.synced_symbols:
        report = reports.get(sym)
        rows = int(getattr(report, "rows_upserted", 0) or 0)
        src = str(getattr(report, "source", "unknown") or "unknown")
        err = str(getattr(report, "error", "") or "").strip()
        click.echo(f"- {sym}: rows={rows} source={src}{' error=' + err if err else ''}")
    if plan.issues:
        for issue in plan.issues[:10]:
            click.echo(f"! {issue}")


@cli.command()
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option("--lookback-days", default=7, type=int, show_default=True, help="Fallback lookback window when start/end not provided")
@click.option("--force", is_flag=True, default=False, help="Re-fetch covered dates")
def sync_twse_etf_daily(
    start: str | None,
    end: str | None,
    lookback_days: int,
    force: bool,
) -> None:
    """Sync TWSE ETF official daily market snapshot.

    Examples:
      uv run realtime0052 sync-twse-etf-daily --help
      uv run realtime0052 sync-twse-etf-daily --lookback-days 5
      uv run realtime0052 sync-twse-etf-daily --start 2026-03-01 --end 2026-03-20 --force
    """
    store = _resolve_store()
    summary = sync_twse_etf_daily_market(
        store=store,
        start=start,
        end=end,
        lookback_days=max(0, int(lookback_days)),
        force=bool(force),
    )
    click.echo(
        "twse_etf_daily "
        f"start={summary.get('start_date')} end={summary.get('end_date')} "
        f"latest={summary.get('latest_date') or 'n/a'}"
    )
    click.echo(
        f"requested_days={int(summary.get('requested_days') or 0)} "
        f"synced_days={int(summary.get('synced_days') or 0)} "
        f"skipped_days={int(summary.get('skipped_days') or 0)} "
        f"empty_days={int(summary.get('empty_days') or 0)} "
        f"saved_rows={int(summary.get('saved_rows') or 0)}"
    )
    issues = summary.get("issues") if isinstance(summary, dict) else []
    if isinstance(issues, list):
        for issue in issues[:10]:
            click.echo(f"! {issue}")


@cli.command()
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option("--force", is_flag=True, default=False, help="Re-fetch covered latest date")
def sync_twse_etf_mis(
    start: str | None,
    end: str | None,
    force: bool,
) -> None:
    """Sync TWSE MIS ETF indicator snapshot.

    Examples:
      uv run realtime0052 sync-twse-etf-mis --help
      uv run realtime0052 sync-twse-etf-mis
      uv run realtime0052 sync-twse-etf-mis --start 2026-03-01 --end 2026-03-20
    """
    store = _resolve_store()
    summary = sync_twse_etf_mis_daily(
        store=store,
        start=start,
        end=end,
        force=bool(force),
    )
    click.echo(
        "twse_etf_mis "
        f"start={summary.get('start_date') or 'n/a'} end={summary.get('end_date') or 'n/a'} "
        f"latest={summary.get('latest_date') or 'n/a'}"
    )
    click.echo(
        f"requested_days={int(summary.get('requested_days') or 0)} "
        f"synced_days={int(summary.get('synced_days') or 0)} "
        f"skipped_days={int(summary.get('skipped_days') or 0)} "
        f"empty_days={int(summary.get('empty_days') or 0)} "
        f"saved_rows={int(summary.get('saved_rows') or 0)}"
    )
    issues = summary.get("issues") if isinstance(summary, dict) else []
    if isinstance(issues, list):
        for issue in issues[:10]:
            click.echo(f"! {issue}")


@cli.command()
@click.option("--out", help="Output CSV path. Defaults to ./tw_etf_super_export_<trade_day>.csv")
@click.option("--ytd-start", help="YTD start date (YYYY-MM-DD or YYYYMMDD)")
@click.option("--ytd-end", help="YTD end date (YYYY-MM-DD or YYYYMMDD)")
@click.option("--compare-start", help="Compare period start date (YYYY-MM-DD or YYYYMMDD)")
@click.option("--compare-end", help="Compare period end date (YYYY-MM-DD or YYYYMMDD)")
@click.option("--daily-lookback-days", default=14, type=int, show_default=True)
@click.option(
    "--force", is_flag=True, default=False, help="Force re-fetch for covered official datasets"
)
def export_tw_etf_super_table(
    out: str | None,
    ytd_start: str | None,
    ytd_end: str | None,
    compare_start: str | None,
    compare_end: str | None,
    daily_lookback_days: int,
    force: bool,
) -> None:
    """Sync sources and export the TW ETF super table CSV.

    Examples:
      uv run realtime0052 export-tw-etf-super-table --help
      uv run realtime0052 export-tw-etf-super-table --out ./tw_etf_super_export_latest.csv
    """
    store = _resolve_store()
    result = export_tw_etf_super_table_artifact(
        store=store,
        out=out,
        ytd_start=ytd_start,
        ytd_end=ytd_end,
        compare_start=compare_start,
        compare_end=compare_end,
        force=bool(force),
        daily_lookback_days=max(1, int(daily_lookback_days)),
    )
    click.echo(
        "tw_etf_super_export "
        f"ytd={result.get('ytd_start')}->{result.get('ytd_end')} "
        f"compare={result.get('compare_start')}->{result.get('compare_end')} "
        f"trade_date={result.get('trade_date_anchor')}"
    )
    click.echo(
        f"path={result.get('output_path')} "
        f"rows={int(result.get('row_count') or 0)} "
        f"cols={int(result.get('column_count') or 0)} "
        f"run_id={result.get('run_id') or 'n/a'}"
    )
    refresh_summary = result.get("refresh_summary") if isinstance(result, dict) else {}
    if isinstance(refresh_summary, dict):
        main_summary = refresh_summary.get("main", {})
        daily_summary = refresh_summary.get("daily_market", {})
        margin_summary = refresh_summary.get("margin", {})
        mis_summary = refresh_summary.get("mis", {})
        three_summary = refresh_summary.get("three_investors", {})
        aum_summary = refresh_summary.get("aum_track", {})
        main_used_trade_date = str(main_summary.get("used_trade_date") or "").strip()
        three_used_trade_date = str(three_summary.get("used_trade_date") or "").strip()
        aum_trade_date = str(aum_summary.get("trade_date") or "").strip()
        click.echo(
            "refresh "
            f"main={str(main_summary.get('status') or 'unknown')}"
            f"{f'({main_used_trade_date})' if main_used_trade_date else ''} "
            f"daily=synced:{int(daily_summary.get('synced_days') or 0)}/saved:{int(daily_summary.get('saved_rows') or 0)} "
            f"margin=synced:{int(margin_summary.get('synced_days') or 0)}/saved:{int(margin_summary.get('saved_rows') or 0)} "
            f"mis=synced:{int(mis_summary.get('synced_days') or 0)}/saved:{int(mis_summary.get('saved_rows') or 0)} "
            f"three_investors={str(three_summary.get('status') or 'unknown')}"
            f"{f'({three_used_trade_date})' if three_used_trade_date else ''}"
            f"/rows:{int(three_summary.get('row_count') or 0)} "
            f"aum_track={str(aum_summary.get('status') or 'unknown')}"
            f"{f'({aum_trade_date})' if aum_trade_date else ''}"
            f"/updated:{int(aum_summary.get('updated') or 0)}"
        )
    issues = result.get("issues") if isinstance(result, dict) else []
    if isinstance(issues, list):
        for issue in issues[:10]:
            click.echo(f"! {issue}")


@cli.command()
@click.option("--symbols", help="Comma-separated TW ETF symbols. Defaults to all TW ETFs in the export universe.")
@click.option("--force", is_flag=True, default=False, help="Force remote refresh before comparing snapshots")
@click.option("--max-workers", default=4, type=int, show_default=True, help="Parallel workers for constituent fetch")
@click.option(
    "--full-refresh/--incremental",
    default=False,
    show_default=True,
    help="full-refresh=force remote fetch for every ETF, incremental=only fetch current snapshots and compare with latest stored snapshot",
)
@click.option("--log-dir", help="Directory for JSON/Markdown sync logs")
def sync_tw_etf_constituents(
    symbols: str | None,
    force: bool,
    max_workers: int,
    full_refresh: bool,
    log_dir: str | None,
) -> None:
    """Sync TW ETF constituent snapshots into DuckDB market_snapshots.

    This command can run independently without starting Streamlit first.

    Examples:
      uv run realtime0052 sync-tw-etf-constituents --help
      uv run realtime0052 sync-tw-etf-constituents
      uv run realtime0052 sync-tw-etf-constituents --symbols 0050,0052 --full-refresh
    """
    store = _resolve_store()
    symbol_list = normalize_symbols(parse_symbols(symbols or ""))
    result = sync_tw_etf_constituent_snapshots(
        store=store,
        symbols=symbol_list or None,
        force=bool(force),
        max_workers=max(1, int(max_workers)),
        full_refresh=bool(full_refresh),
        log_dir=log_dir,
    )
    click.echo(
        "sync_tw_etf_constituents "
        f"etf_count={int(result.get('etf_count') or 0)} "
        f"updated={int(result.get('updated_count') or 0)} "
        f"unchanged={int(result.get('unchanged_count') or 0)} "
        f"missing={int(result.get('missing_count') or 0)} "
        f"errors={int(result.get('error_count') or 0)}"
    )
    click.echo(
        f"log_json={result.get('json_log_path')} "
        f"log_md={result.get('markdown_log_path')}"
    )
    for issue in list(result.get("issues") or [])[:10]:
        click.echo(f"! {issue}")


@cli.command()
@click.option("--symbol", required=True, help="Single TW ETF symbol, for example 0052")
@click.option("--out", help="Output root directory. A report folder named tw_etf_report_<symbol>_<trade_day> will be created under it.")
@click.option(
    "--sync-constituents/--no-sync-constituents",
    default=True,
    show_default=True,
    help="Refresh the requested ETF constituent snapshot before exporting; disable to reuse DuckDB/local fallback",
)
@click.option("--ytd-start", help="YTD start date (YYYY-MM-DD or YYYYMMDD)")
@click.option("--ytd-end", help="YTD end date (YYYY-MM-DD or YYYYMMDD)")
@click.option("--compare-start", help="Compare period start date (YYYY-MM-DD or YYYYMMDD)")
@click.option("--compare-end", help="Compare period end date (YYYY-MM-DD or YYYYMMDD)")
@click.option("--backtest-start", help="Backtest/chart/indicator start date (YYYY-MM-DD)")
@click.option("--backtest-end", help="Backtest/chart/indicator end date (YYYY-MM-DD)")
@click.option("--daily-lookback-days", default=14, type=int, show_default=True)
@click.option("--force", is_flag=True, default=False, help="Force re-fetch for covered official datasets")
@click.option("--log-dir", help="Directory for external JSON/Markdown sync logs")
@click.option("--heatmap-only", is_flag=True, default=False, help="Only export the constituent heatmap PNG")
def export_tw_etf_report(
    symbol: str,
    out: str | None,
    sync_constituents: bool,
    ytd_start: str | None,
    ytd_end: str | None,
    compare_start: str | None,
    compare_end: str | None,
    backtest_start: str | None,
    backtest_end: str | None,
    daily_lookback_days: int,
    force: bool,
    log_dir: str | None,
    heatmap_only: bool,
) -> None:
    """Export a single TW ETF report bundle with tables, charts, indicators, and logs.

    This command can run independently without starting Streamlit first.

    Examples:
      uv run realtime0052 export-tw-etf-report --help
      uv run realtime0052 export-tw-etf-report --symbol 0052 --out ./reports/
      uv run realtime0052 export-tw-etf-report --symbol 0052 --backtest-start 2023-01-01 --backtest-end 2026-03-21
      uv run realtime0052 export-tw-etf-report --symbol 0052 --heatmap-only --out ./0052_constituent_heatmap.png
    """
    symbols = normalize_symbols(parse_symbols(symbol))
    if len(symbols) != 1:
        raise click.ClickException("export-tw-etf-report requires exactly one TW ETF symbol")
    store = _resolve_store()
    if bool(heatmap_only):
        result = export_tw_etf_constituent_heatmap_artifact(
            store=store,
            symbol=symbols[0],
            out=out,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            ytd_end=ytd_end,
            force=bool(force),
            sync_constituents=bool(sync_constituents),
            log_dir=log_dir,
        )
        click.echo(
            "export_tw_etf_constituent_heatmap "
            f"symbol={result.get('symbol')} "
            f"trade_date={result.get('trade_date_anchor')} "
            f"backtest={result.get('backtest_start')}->{result.get('backtest_end')}"
        )
        click.echo(
            f"path={result.get('output_path')} "
            f"constituents={int(result.get('constituent_count') or 0)}"
        )
        for issue in list(result.get("issues") or [])[:10]:
            click.echo(f"! {issue}")
        return
    result = export_tw_etf_report_artifact(
        store=store,
        symbol=symbols[0],
        out=out,
        ytd_start=ytd_start,
        ytd_end=ytd_end,
        compare_start=compare_start,
        compare_end=compare_end,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        daily_lookback_days=max(1, int(daily_lookback_days)),
        force=bool(force),
        sync_constituents=bool(sync_constituents),
        log_dir=log_dir,
    )
    click.echo(
        "export_tw_etf_report "
        f"symbol={result.get('symbol')} "
        f"trade_date={result.get('trade_date_anchor')} "
        f"backtest={result.get('backtest_start')}->{result.get('backtest_end')}"
    )
    click.echo(
        f"report_dir={result.get('report_dir')} "
        f"files={int(result.get('file_count') or 0)}"
    )
    click.echo(
        f"bundle_log_json={result.get('bundle_log_json_path')} "
        f"bundle_log_md={result.get('bundle_log_markdown_path')}"
    )
    for issue in list(result.get("issues") or [])[:10]:
        click.echo(f"! {issue}")


@cli.command()
@click.option("--out-root", help="Output root directory. Defaults to ~/Downloads/etf")
@click.option("--start", "briefing_start", default=DEFAULT_ETF_BRIEFING_START, show_default=True, help="Chart/report start date (YYYY-MM-DD)")
@click.option("--end", "briefing_end", help="Chart/report end date (YYYY-MM-DD). Defaults to today.")
@click.option("--single-symbol", default=DEFAULT_ETF_BRIEFING_SINGLE_SYMBOL, show_default=True, help="Single ETF symbol for the focused single-chart section")
@click.option(
    "--theme",
    type=click.Choice(list(SUPPORTED_CHART_THEMES), case_sensitive=False),
    default=DEFAULT_ETF_BRIEFING_THEME,
    show_default=True,
    help="Chart theme for generated PNG files",
)
@click.option("--news-window-days", default=DEFAULT_ETF_BRIEFING_NEWS_WINDOW_DAYS, type=int, show_default=True, help="Lookback window for public geopolitics news collection")
@click.option(
    "--include-extra-splits/--no-include-extra-splits",
    default=True,
    show_default=True,
    help="Also export split chart packs for tech ETFs and active ETFs",
)
def export_etf_briefing(
    out_root: str | None,
    briefing_start: str,
    briefing_end: str | None,
    single_symbol: str,
    theme: str,
    news_window_days: int,
    include_extra_splits: bool,
) -> None:
    """Export a one-shot ETF briefing bundle with CSVs, charts, HTML, and FB copy.

    This command can run independently without starting Streamlit first.

    Examples:
      uv run realtime0052 export-etf-briefing --help
      uv run realtime0052 export-etf-briefing
      uv run realtime0052 export-etf-briefing --out-root ~/Downloads/etf --start 2023-01-01 --single-symbol 00935
    """
    store = _resolve_store()
    result = export_etf_briefing_artifact(
        store=store,
        out_root=out_root,
        start=briefing_start,
        end=briefing_end,
        single_symbol=single_symbol,
        theme=theme,
        news_window_days=max(1, int(news_window_days)),
        include_extra_splits=bool(include_extra_splits),
    )
    click.echo(
        "export_etf_briefing "
        f"dir={result.get('briefing_dir')} "
        f"super_csv={result.get('super_export_csv')}"
    )
    click.echo(
        f"report_html={result.get('report_html')} "
        f"fb_text={result.get('fb_text')} "
        f"news_sources={result.get('news_sources')}"
    )
    for issue in list(result.get("issues") or [])[:10]:
        click.echo(f"! {issue}")


@cli.command()
@click.option("--symbol", required=True, help="Single symbol or comma-separated symbols")
@click.option("--market", default="auto", help="TW / US / auto")
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option("--strategy", default="buy_hold", show_default=True)
@click.option("--initial-capital", default=1_000_000.0, type=float, show_default=True)
@click.option("--fee-rate", type=float, default=None)
@click.option("--sell-tax", type=float, default=None)
@click.option("--slippage", type=float, default=None)
@click.option("--no-split-adjustment", is_flag=True, default=False)
@click.option("--no-total-return-adjustment", is_flag=True, default=False)
def backtest(
    symbol: str,
    market: str,
    start: str | None,
    end: str | None,
    strategy: str,
    initial_capital: float,
    fee_rate: float | None,
    sell_tax: float | None,
    slippage: float | None,
    no_split_adjustment: bool,
    no_total_return_adjustment: bool,
) -> None:
    """Run backtest and print core metrics.

    Examples:
      uv run realtime0052 backtest --help
      uv run realtime0052 backtest --symbol 0050 --market TW --strategy buy_hold
    """
    store = _resolve_store()
    symbols = normalize_symbols(parse_symbols(symbol))
    if not symbols:
        raise click.ClickException("No valid symbol provided")

    market_token = str(market or "auto").strip().upper()
    market_code = (
        _infer_market_from_symbol(symbols[0])
        if market_token == "AUTO"
        else _normalize_market(market_token)
    )

    end_dt = _parse_iso_datetime(end, default=datetime.now(tz=timezone.utc))
    start_dt = _parse_iso_datetime(start, default=end_dt - timedelta(days=365 * 5))

    sync_symbols_if_needed(
        store=store,
        market=market_code,
        symbols=symbols,
        start=start_dt,
        end=end_dt,
        parallel=True,
        max_workers=4,
        mode="backfill",
        min_rows=120,
    )

    prepared = load_and_prepare_symbol_bars(
        store=store,
        market_code=market_code,
        symbols=symbols,
        start=start_dt,
        end=end_dt,
        use_total_return_adjustment=not no_total_return_adjustment,
        use_split_adjustment=not no_split_adjustment,
        auto_detect_split=True,
        apply_total_return_adjustment=_apply_total_return_adjustment_cli,
    )
    bars_by_symbol = prepared.bars_by_symbol
    if not bars_by_symbol:
        raise click.ClickException("No bars available after sync/preparation")

    fee_default, tax_default, slip_default = default_cost_params(market_code, symbols)
    cost_model = CostModel(
        fee_rate=float(fee_default if fee_rate is None else fee_rate),
        sell_tax_rate=float(tax_default if sell_tax is None else sell_tax),
        slippage_rate=float(slip_default if slippage is None else slippage),
    )

    payload = execute_backtest_run(
        bars_by_symbol=bars_by_symbol,
        config=BacktestExecutionInput(
            mode="單一標的" if len(bars_by_symbol) == 1 else "投組(多標的)",
            strategy=str(strategy).strip(),
            strategy_params={},
            enable_walk_forward=False,
            train_ratio=0.7,
            objective="sharpe",
            initial_capital=float(initial_capital),
            cost_model=cost_model,
        ),
    )

    result = payload.get("result")
    metrics = getattr(result, "metrics", None)
    if metrics is None:
        raise click.ClickException("Backtest completed but metrics are unavailable")

    click.echo(
        f"mode={payload.get('mode')} symbols={','.join(bars_by_symbol.keys())} strategy={strategy}"
    )
    click.echo(f"total_return={metrics.total_return * 100.0:.2f}%")
    click.echo(f"cagr={metrics.cagr * 100.0:.2f}%")
    click.echo(f"mdd={metrics.max_drawdown * 100.0:.2f}%")
    click.echo(f"sharpe={metrics.sharpe:.3f}")
    click.echo(f"trades={int(metrics.trades)}")


@cli.command()
@click.option("--symbols", required=True, help="Comma-separated symbols")
@click.option(
    "--layout",
    type=click.Choice(list(SUPPORTED_CHART_LAYOUTS), case_sensitive=False),
    default="single",
    show_default=True,
    help="single=單一標的單張圖, combined=多標的同圖, split=逐檔各輸出一張",
)
@click.option("--market", default="auto", show_default=True, help="auto / TW / US")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--strategy",
    type=click.Choice(list(SUPPORTED_CHART_STRATEGIES), case_sensitive=False),
    default="buy_hold",
    show_default=True,
    help="Backtest strategy used to build the chart",
)
@click.option("--benchmark", default="auto", show_default=True, help="Benchmark choice per market")
@click.option("--initial-capital", default=1_000_000.0, type=float, show_default=True)
@click.option("--fee-rate", type=float, default=None, help="Fee rate override")
@click.option("--sell-tax", type=float, default=None, help="Sell tax override")
@click.option("--slippage", type=float, default=None, help="Slippage override")
@click.option(
    "--sync-before-run/--no-sync-before-run",
    default=True,
    show_default=True,
    help="Sync missing history before chart export",
)
@click.option(
    "--theme",
    type=click.Choice(list(SUPPORTED_CHART_THEMES), case_sensitive=False),
    default=DEFAULT_CHART_THEME,
    show_default=True,
    help="Chart theme",
)
@click.option("--width", default=1600, type=int, show_default=True, help="PNG width")
@click.option("--height", default=900, type=int, show_default=True, help="PNG height")
@click.option("--scale", default=2, type=int, show_default=True, help="PNG scale")
@click.option("--out", help="Output PNG path for single/combined layouts")
@click.option("--out-dir", help="Output directory for generated PNG files")
@click.option("--fast", type=int, default=None, help="Strategy param: fast")
@click.option("--slow", type=int, default=None, help="Strategy param: slow")
@click.option("--trend", type=int, default=None, help="Strategy param: trend")
@click.option("--entry-n", type=int, default=None, help="Strategy param: entry_n")
@click.option("--exit-n", type=int, default=None, help="Strategy param: exit_n")
@click.option(
    "--annotate-extrema",
    is_flag=True,
    default=False,
    help="Single/split only: add highest/lowest price annotations like the reference chart",
)
@click.option(
    "--show-signals",
    is_flag=True,
    default=False,
    help="Single/split only: draw buy/sell signal markers on the price panel",
)
@click.option(
    "--show-fills",
    is_flag=True,
    default=False,
    help="Single/split only: draw buy/sell fill markers on the equity panel",
)
@click.option(
    "--show-trade-path",
    is_flag=True,
    default=False,
    help="Single/split only: connect each trade entry/exit on the equity panel",
)
@click.option(
    "--show-end-marker",
    is_flag=True,
    default=False,
    help="Add a vertical end marker near the right edge of the chart",
)
@click.option(
    "--reference-annotations",
    is_flag=True,
    default=False,
    help="Single/split only: enable extrema, signals, fills, trade path, and end marker together",
)
@click.option(
    "--include-ew-portfolio/--no-include-ew-portfolio",
    default=False,
    show_default=True,
    help="Combined only: include the EW portfolio strategy line on the comparison chart",
)
@click.option("--no-split-adjustment", is_flag=True, default=False)
@click.option("--no-total-return-adjustment", is_flag=True, default=False)
def chart_backtest(
    symbols: str,
    layout: str,
    market: str,
    start: str,
    end: str,
    strategy: str,
    benchmark: str,
    initial_capital: float,
    fee_rate: float | None,
    sell_tax: float | None,
    slippage: float | None,
    sync_before_run: bool,
    theme: str,
    width: int,
    height: int,
    scale: int,
    out: str | None,
    out_dir: str | None,
    fast: int | None,
    slow: int | None,
    trend: int | None,
    entry_n: int | None,
    exit_n: int | None,
    annotate_extrema: bool,
    show_signals: bool,
    show_fills: bool,
    show_trade_path: bool,
    show_end_marker: bool,
    reference_annotations: bool,
    include_ew_portfolio: bool,
    no_split_adjustment: bool,
    no_total_return_adjustment: bool,
) -> None:
    """Export backtest charts as PNG for AI workflows.

    Examples:
      uv run realtime0052 chart-backtest --help
      uv run realtime0052 chart-backtest --symbols 0050 --layout single --start 2024-01-01 --end 2026-03-20
      uv run realtime0052 chart-backtest --symbols 0050,0052,006208 --layout combined --start 2024-01-01 --end 2026-03-20
      uv run realtime0052 chart-backtest --symbols 0050 --layout single --reference-annotations --start 2024-01-01 --end 2026-03-20
      uv run realtime0052 chart-backtest --symbols 0050,SPY --layout split --market auto --start 2024-01-01 --end 2026-03-20 --out-dir ./artifacts/charts
    """
    store = _resolve_store()
    symbol_list = normalize_symbols(parse_symbols(symbols))
    if not symbol_list:
        raise click.ClickException("No valid symbols provided")

    start_dt = _parse_iso_datetime(start, default=datetime.now(tz=timezone.utc) - timedelta(days=365))
    end_dt = _parse_iso_datetime(end, default=datetime.now(tz=timezone.utc))
    if start_dt >= end_dt:
        raise click.ClickException("start must be earlier than end")

    try:
        result = export_backtest_chart_artifact(
            store=store,
            symbols=symbol_list,
            layout=str(layout).lower(),
            market=market,
            start=start_dt,
            end=end_dt,
            strategy=str(strategy).lower(),
            benchmark_choice=benchmark,
            initial_capital=float(initial_capital),
            fee_rate=fee_rate,
            sell_tax=sell_tax,
            slippage=slippage,
            sync_before_run=bool(sync_before_run),
            use_split_adjustment=not no_split_adjustment,
            use_total_return_adjustment=not no_total_return_adjustment,
            theme=str(theme).lower(),
            width=max(320, int(width)),
            height=max(240, int(height)),
            scale=max(1, int(scale)),
            out=out,
            out_dir=out_dir,
            fast=fast,
            slow=slow,
            trend=trend,
            entry_n=entry_n,
            exit_n=exit_n,
            annotate_extrema=bool(annotate_extrema),
            show_signals=bool(show_signals),
            show_fills=bool(show_fills),
            show_trade_path=bool(show_trade_path),
            show_end_marker=bool(show_end_marker),
            reference_annotations=bool(reference_annotations),
            include_ew_portfolio=bool(include_ew_portfolio),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        lowered = str(exc).lower()
        if "kaleido" in lowered:
            raise click.ClickException(
                "PNG export requires the kaleido package. Run `uv sync` to install updated dependencies."
            ) from exc
        raise

    click.echo(
        "chart_backtest "
        f"layout={result.get('layout')} "
        f"format={result.get('format')} "
        f"requested={int(result.get('requested_count') or 0)} "
        f"exported={int(result.get('exported_count') or 0)}"
    )
    for item in result.get("items", []):
        if not isinstance(item, dict):
            continue
        click.echo(f"- {item.get('symbol')}: {item.get('path')}")
    for issue in result.get("issues", [])[:10]:
        click.echo(f"! {issue}")
    if int(result.get("exported_count") or 0) <= 0:
        raise click.ClickException("No chart exported")


@cli.command()
@click.option(
    "--scope",
    type=click.Choice(["tw", "us", "both"], case_sensitive=False),
    default="both",
    show_default=True,
)
@click.option("--years", default=5, type=int, show_default=True)
@click.option("--max-workers", default=6, type=int, show_default=True)
@click.option(
    "--sync-mode",
    type=click.Choice(["min_rows", "backfill", "all"], case_sensitive=False),
    default="min_rows",
    show_default=True,
)
@click.option("--min-rows", default=900, type=int, show_default=True)
def bootstrap(scope: str, years: int, max_workers: int, sync_mode: str, min_rows: int) -> None:
    """Bootstrap local market data.

    Examples:
      uv run realtime0052 bootstrap --help
      uv run realtime0052 bootstrap --scope tw --years 3
      uv run realtime0052 bootstrap --scope both --years 5 --sync-mode min_rows --min-rows 900
    """
    store = _resolve_store()
    summary = run_market_data_bootstrap(
        store=store,
        scope=str(scope).lower(),
        years=max(1, int(years)),
        parallel=True,
        max_workers=max(1, int(max_workers)),
        sync_mode=str(sync_mode).lower(),
        min_rows=max(1, int(min_rows)),
    )

    click.echo(
        " ".join(
            [
                f"scope={summary.get('scope')}",
                f"years={summary.get('years')}",
                f"metadata_upserted={summary.get('metadata_upserted')}",
                f"total_symbols={summary.get('total_symbols')}",
                f"synced_success={summary.get('synced_success')}",
                f"failed={summary.get('failed_symbols')}",
                f"issues={summary.get('issue_count')}",
            ]
        )
    )
    issues = summary.get("issues", []) if isinstance(summary.get("issues"), list) else []
    for issue in issues[:10]:
        click.echo(f"! {issue}")


@cli.command()
def info() -> None:
    """Show basic runtime configuration.

    Examples:
      uv run realtime0052 info
    """
    click.echo("=== Realtime0052 Info ===")
    click.echo(f"Config source: {get_config_source()}")
    click.echo(
        f"Storage backend: {cfg_or_env_str('features.storage_backend', 'REALTIME0052_STORAGE_BACKEND', 'duckdb')}"
    )
    click.echo(
        f"DuckDB path: {cfg_or_env_str('storage.duckdb.db_path', 'REALTIME0052_DUCKDB_PATH', 'Not set')}"
    )
    click.echo(
        f"Parquet root: {cfg_or_env_str('storage.duckdb.parquet_root', 'REALTIME0052_PARQUET_ROOT', 'Not set')}"
    )


@cli.command()
def serve() -> None:
    """Start the Streamlit app locally.

    Examples:
      uv run realtime0052 serve
    """
    import subprocess
    import sys

    click.echo("Starting Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=False)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
