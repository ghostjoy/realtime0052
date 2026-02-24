from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from services.sync_orchestrator import normalize_symbols, sync_symbols_if_needed

if TYPE_CHECKING:
    from storage import HistoryStore

DEFAULT_US_BOOTSTRAP_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "BRK-B",
    "AVGO",
    "AMD",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "SOXX",
    "SMH",
]


def _looks_like_tw_symbol(symbol: str) -> bool:
    token = str(symbol or "").strip().upper()
    return bool(re.fullmatch(r"\d{4,6}[A-Z]?", token))


def _normalize_scope(scope: str) -> str:
    token = str(scope or "both").strip().lower()
    if token in {"tw", "taiwan"}:
        return "tw"
    if token in {"us", "usa"}:
        return "us"
    return "both"


def _parse_us_symbols(us_symbols: list[str] | None = None) -> list[str]:
    if us_symbols is None:
        return normalize_symbols(DEFAULT_US_BOOTSTRAP_SYMBOLS)
    return normalize_symbols([str(item or "") for item in us_symbols])


def fetch_tw_symbol_metadata(timeout_sec: int = 12) -> tuple[list[dict[str, object]], list[str]]:
    import requests

    metadata_map: dict[str, dict[str, object]] = {}
    issues: list[str] = []

    def _upsert(
        symbol: str,
        *,
        name: str = "",
        exchange: str = "",
        industry: str = "",
        source: str = "",
    ):
        code = str(symbol or "").strip().upper()
        if not _looks_like_tw_symbol(code):
            return
        row = metadata_map.setdefault(
            code,
            {
                "symbol": code,
                "market": "TW",
                "name": "",
                "exchange": "",
                "industry": "",
                "asset_type": "",
                "currency": "TWD",
                "source": "",
            },
        )
        if str(name or "").strip():
            row["name"] = str(name).strip()
        if str(exchange or "").strip():
            row["exchange"] = str(exchange).strip().upper()
        if str(industry or "").strip():
            row["industry"] = str(industry).strip()
        if str(source or "").strip():
            row["source"] = str(source).strip()

    try:
        resp = requests.get(
            "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=timeout_sec
        )
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                _upsert(
                    str(row.get("Code", "")),
                    name=str(row.get("Name", "")),
                    exchange="TW",
                    source="twse_stock_day_all",
                )
    except Exception as exc:
        issues.append(f"twse_daily:{exc}")

    try:
        resp = requests.get(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                _upsert(
                    str(row.get("SecuritiesCompanyCode", "")),
                    name=str(row.get("CompanyName", "")),
                    exchange="OTC",
                    source="tpex_mainboard_daily_close_quotes",
                )
    except Exception as exc:
        issues.append(f"tpex_daily:{exc}")

    try:
        resp = requests.get(
            "https://openapi.twse.com.tw/v1/opendata/t187ap03_L", timeout=timeout_sec
        )
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                _upsert(
                    str(row.get("公司代號", "") or row.get("Code", "")),
                    name=str(
                        row.get("公司簡稱", "") or row.get("公司名稱", "") or row.get("Name", "")
                    ),
                    exchange="TW",
                    industry=str(row.get("產業別", "")),
                    source="twse_t187ap03_l",
                )
    except Exception as exc:
        issues.append(f"twse_profile:{exc}")

    try:
        resp = requests.get(
            "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", timeout=timeout_sec
        )
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                _upsert(
                    str(row.get("SecuritiesCompanyCode", "") or row.get("公司代號", "")),
                    name=str(
                        row.get("CompanyName", "")
                        or row.get("公司簡稱", "")
                        or row.get("SecuritiesCompanyName", "")
                    ),
                    exchange="OTC",
                    industry=str(row.get("SecuritiesIndustryCode", "")),
                    source="tpex_mopsfin_t187ap03_o",
                )
    except Exception as exc:
        issues.append(f"tpex_profile:{exc}")

    rows = [metadata_map[key] for key in sorted(metadata_map)]
    return rows, issues


def run_market_data_bootstrap(
    *,
    store: HistoryStore,
    scope: str = "both",
    years: int = 5,
    parallel: bool = True,
    max_workers: int = 6,
    tw_limit: int | None = None,
    us_symbols: list[str] | None = None,
    sync_mode: str = "min_rows",
    min_rows: int | None = None,
) -> dict[str, object]:
    scope_token = _normalize_scope(scope)
    years_token = max(1, int(years))
    workers = max(1, int(max_workers))

    start = datetime(datetime.now(tz=timezone.utc).year - years_token, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(tz=timezone.utc)

    use_tw = scope_token in {"tw", "both"}
    use_us = scope_token in {"us", "both"}

    tw_rows: list[dict[str, object]] = []
    us_rows: list[dict[str, object]] = []
    issues: list[str] = []

    if use_tw:
        tw_rows, tw_issues = fetch_tw_symbol_metadata(timeout_sec=12)
        issues.extend(tw_issues)
        if tw_limit is not None:
            tw_rows = tw_rows[: max(1, int(tw_limit))]

    if use_us:
        us_list = _parse_us_symbols(us_symbols)
        us_rows = [
            {
                "symbol": symbol,
                "market": "US",
                "name": symbol,
                "exchange": "US",
                "industry": "",
                "asset_type": "",
                "currency": "USD",
                "source": "bootstrap_default_us",
            }
            for symbol in us_list
        ]

    upsert_count = store.upsert_symbol_metadata([*tw_rows, *us_rows])
    tw_symbols = normalize_symbols([str(row.get("symbol", "")) for row in tw_rows])
    us_symbols_final = normalize_symbols([str(row.get("symbol", "")) for row in us_rows])

    params = {
        "scope": scope_token,
        "years": years_token,
        "parallel": bool(parallel),
        "max_workers": workers,
        "tw_limit": None if tw_limit is None else int(tw_limit),
        "sync_mode": str(sync_mode or "min_rows"),
    }
    run_id = store.start_bootstrap_run(scope=f"manual:{scope_token}", params=params)

    total_symbols = len(tw_symbols) + len(us_symbols_final)
    synced_success = 0
    failed_symbols = 0
    skipped_symbols = 0

    try:
        row_threshold = max(120, years_token * 180) if min_rows is None else max(1, int(min_rows))

        def _sync_market(market: str, symbols: list[str]):
            nonlocal synced_success, failed_symbols, skipped_symbols
            if not symbols:
                return
            reports, plan = sync_symbols_if_needed(
                store=store,
                market=market,
                symbols=symbols,
                start=start,
                end=end,
                parallel=parallel,
                max_workers=workers,
                mode=sync_mode,
                min_rows=row_threshold,
            )
            skipped_symbols += len(plan.skipped_symbols)
            for symbol in plan.synced_symbols:
                report = reports.get(symbol)
                err = str(getattr(report, "error", "") or "").strip()
                if report is not None and not err:
                    synced_success += 1
                elif err:
                    issues.append(f"{market}:{symbol}:{err}")
            failed_symbols += len(plan.issues)
            issues.extend([f"{market}:{item}" for item in plan.issues])

        if use_tw:
            _sync_market("TW", tw_symbols)
        if use_us:
            _sync_market("US", us_symbols_final)

        status = "completed"
        if failed_symbols > 0:
            status = "partial_failed" if synced_success > 0 else "failed"

        summary = {
            "run_id": run_id,
            "scope": scope_token,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tw_symbols": len(tw_symbols),
            "us_symbols": len(us_symbols_final),
            "total_symbols": total_symbols,
            "synced_success": synced_success,
            "skipped_symbols": skipped_symbols,
            "failed_symbols": failed_symbols,
            "metadata_rows_upserted": upsert_count,
            "issues": issues,
        }
        store.finish_bootstrap_run(
            run_id,
            status=status,
            total_symbols=total_symbols,
            synced_symbols=synced_success,
            failed_symbols=failed_symbols,
            summary=summary,
            error=None,
        )
        summary["status"] = status
        return summary
    except Exception as exc:
        error_text = str(exc)
        store.finish_bootstrap_run(
            run_id,
            status="failed",
            total_symbols=total_symbols,
            synced_symbols=synced_success,
            failed_symbols=max(failed_symbols, 1),
            summary={
                "run_id": run_id,
                "scope": scope_token,
                "issues": issues,
            },
            error=error_text,
        )
        raise


def run_incremental_refresh(
    *,
    store: HistoryStore,
    years: int = 5,
    parallel: bool = True,
    max_workers: int = 4,
    tw_limit: int = 180,
    us_limit: int = 80,
) -> dict[str, object]:
    years_token = max(1, int(years))
    start = datetime(datetime.now(tz=timezone.utc).year - years_token, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(tz=timezone.utc)
    tw_symbols = store.list_symbols("TW", limit=max(1, int(tw_limit)))
    us_symbols = store.list_symbols("US", limit=max(1, int(us_limit)))
    if not tw_symbols and not us_symbols:
        return {
            "run_id": "",
            "scope": "daily_incremental",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tw_symbols": 0,
            "us_symbols": 0,
            "total_symbols": 0,
            "synced_success": 0,
            "skipped_symbols": 0,
            "failed_symbols": 0,
            "issues": [],
            "status": "skipped_no_symbols",
        }

    params = {
        "years": years_token,
        "parallel": bool(parallel),
        "max_workers": int(max_workers),
        "tw_limit": int(tw_limit),
        "us_limit": int(us_limit),
        "sync_mode": "backfill",
    }
    run_id = store.start_bootstrap_run(scope="daily_incremental", params=params)

    total_symbols = len(tw_symbols) + len(us_symbols)
    synced_success = 0
    failed_symbols = 0
    skipped_symbols = 0
    issues: list[str] = []
    workers = max(1, int(max_workers))

    try:

        def _sync_market(market: str, symbols: list[str]):
            nonlocal synced_success, failed_symbols, skipped_symbols
            if not symbols:
                return
            reports, plan = sync_symbols_if_needed(
                store=store,
                market=market,
                symbols=symbols,
                start=start,
                end=end,
                parallel=parallel,
                max_workers=workers,
                mode="backfill",
            )
            skipped_symbols += len(plan.skipped_symbols)
            for symbol in plan.synced_symbols:
                report = reports.get(symbol)
                err = str(getattr(report, "error", "") or "").strip()
                if report is not None and not err:
                    synced_success += 1
                elif err:
                    issues.append(f"{market}:{symbol}:{err}")
            failed_symbols += len(plan.issues)
            issues.extend([f"{market}:{item}" for item in plan.issues])

        _sync_market("TW", tw_symbols)
        _sync_market("US", us_symbols)

        status = "completed"
        if failed_symbols > 0:
            status = "partial_failed" if synced_success > 0 else "failed"

        summary = {
            "run_id": run_id,
            "scope": "daily_incremental",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tw_symbols": len(tw_symbols),
            "us_symbols": len(us_symbols),
            "total_symbols": total_symbols,
            "synced_success": synced_success,
            "skipped_symbols": skipped_symbols,
            "failed_symbols": failed_symbols,
            "issues": issues,
            "status": status,
        }
        store.finish_bootstrap_run(
            run_id,
            status=status,
            total_symbols=total_symbols,
            synced_symbols=synced_success,
            failed_symbols=failed_symbols,
            summary=summary,
            error=None,
        )
        return summary
    except Exception as exc:
        store.finish_bootstrap_run(
            run_id,
            status="failed",
            total_symbols=total_symbols,
            synced_symbols=synced_success,
            failed_symbols=max(failed_symbols, 1),
            summary={"run_id": run_id, "scope": "daily_incremental", "issues": issues},
            error=str(exc),
        )
        raise


__all__ = [
    "DEFAULT_US_BOOTSTRAP_SYMBOLS",
    "fetch_tw_symbol_metadata",
    "run_market_data_bootstrap",
    "run_incremental_refresh",
]
