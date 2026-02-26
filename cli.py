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
from services.sync_orchestrator import normalize_symbols, sync_symbols_if_needed


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
)
@click.option("--min-rows", default=120, type=int, show_default=True)
@click.option("--max-workers", default=6, type=int, show_default=True)
def sync(
    symbols_text: str | None,
    market_text: str,
    days: int,
    mode: str,
    min_rows: int,
    max_workers: int,
) -> None:
    """Sync historical bars for symbols."""
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
    """Run backtest and print metrics."""
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
    """Bootstrap local market data."""
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
    """Show basic runtime configuration."""
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
    """Start Streamlit app."""
    import subprocess
    import sys

    click.echo("Starting Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=False)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
