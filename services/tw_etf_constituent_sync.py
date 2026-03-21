from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from services.market_data_service import MarketDataService
from services.tw_etf_report_logging import build_sync_log_entry, write_sync_log_files
from services.tw_etf_super_export import _call_app_quiet, _load_app_module
from storage import HistoryStore


TW_ETF_CONSTITUENTS_DATASET_KEY = "tw_etf_constituents"
DEFAULT_TW_ETF_CONSTITUENT_LOG_DIR = Path("artifacts") / "logs"


def sync_tw_etf_constituent_snapshots(
    *,
    store: HistoryStore,
    symbols: list[str] | None = None,
    force: bool = False,
    max_workers: int = 4,
    full_refresh: bool = False,
    log_dir: str | Path | None = None,
) -> dict[str, object]:
    app_module = _load_app_module()
    etf_codes = _resolve_target_etf_codes(app_module=app_module, symbols=symbols)
    if not etf_codes:
        raise RuntimeError("no TW ETF symbols resolved for constituent sync")

    generated_at = datetime.now(tz=timezone.utc)
    log_entries: list[dict[str, object]] = []
    issues: list[str] = []
    saved_count = 0

    def _fetch_one(etf_code: str) -> dict[str, object]:
        service = MarketDataService()
        rows, source, issue = _call_app_quiet(
            app_module._load_etf_constituents_rows,
            service=service,
            etf_code=etf_code,
            force_refresh_constituents=bool(force or full_refresh),
        )
        normalized_rows = _normalize_constituent_rows(rows)
        return {
            "symbol": etf_code,
            "rows": normalized_rows,
            "row_count": len(normalized_rows),
            "source": str(source or "").strip() or "unknown",
            "issue": str(issue or "").strip(),
        }

    workers = max(1, min(int(max_workers), len(etf_codes)))
    results: list[dict[str, object]] = []
    if workers <= 1 or len(etf_codes) <= 1:
        for code in etf_codes:
            try:
                results.append(_fetch_one(code))
            except Exception as exc:
                results.append(
                    {
                        "symbol": code,
                        "rows": [],
                        "row_count": 0,
                        "source": "error",
                        "issue": str(exc),
                        "error": str(exc),
                    }
                )
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tw-etf-constituents") as executor:
            futures = {executor.submit(_fetch_one, code): code for code in etf_codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "symbol": code,
                            "rows": [],
                            "row_count": 0,
                            "source": "error",
                            "issue": str(exc),
                            "error": str(exc),
                        }
                    )

    results.sort(key=lambda item: str(item.get("symbol") or ""))
    for result in results:
        symbol = str(result.get("symbol") or "").strip().upper()
        rows = list(result.get("rows") or [])
        source = str(result.get("source") or "").strip() or "unknown"
        issue = str(result.get("issue") or "").strip()
        error = str(result.get("error") or "").strip()
        latest = store.load_latest_market_snapshot(
            dataset_key=TW_ETF_CONSTITUENTS_DATASET_KEY,
            market="TW",
            symbol=symbol,
            interval="constituents",
        )
        previous_payload = latest.get("payload") if isinstance(latest, dict) else {}
        previous_rows = _normalize_constituent_rows((previous_payload or {}).get("rows", []))
        previous_source = str((previous_payload or {}).get("source") or "").strip()
        previous_sig = _payload_signature(rows=previous_rows, source=previous_source)
        current_sig = _payload_signature(rows=rows, source=source)
        used_trade_date = generated_at.isoformat()
        status = "updated"
        notes = issue
        if error:
            status = "error"
        elif not rows:
            status = "missing"
        elif previous_sig == current_sig:
            status = "unchanged"

        if status == "updated":
            saved_count += int(
                store.save_market_snapshot(
                    dataset_key=TW_ETF_CONSTITUENTS_DATASET_KEY,
                    market="TW",
                    symbol=symbol,
                    interval="constituents",
                    source=source,
                    asof=generated_at,
                    payload={
                        "rows": rows,
                        "source": source,
                        "issue": issue,
                        "row_count": len(rows),
                    },
                )
                or 0
            )
        if status in {"error", "missing"} and issue:
            issues.append(f"{symbol}: {issue}")
        elif status == "error" and error:
            issues.append(f"{symbol}: {error}")
        log_entries.append(
            build_sync_log_entry(
                dataset_name=f"constituents:{symbol}",
                status=status,
                requested_trade_date=generated_at.date().isoformat(),
                used_trade_date=used_trade_date,
                row_count_before=len(previous_rows),
                row_count_after=len(rows),
                updated_rows=len(rows) if status == "updated" else 0,
                source=source,
                notes=notes,
                error=error,
            )
        )

    log_paths = write_sync_log_files(
        log_dir=log_dir or DEFAULT_TW_ETF_CONSTITUENT_LOG_DIR,
        file_stem=f"tw_etf_constituents_sync_{generated_at.strftime('%Y%m%d_%H%M%S')}",
        title="TW ETF Constituents Sync Log",
        entries=log_entries,
        meta={
            "etf_count": len(etf_codes),
            "saved_count": saved_count,
            "force": bool(force),
            "full_refresh": bool(full_refresh),
        },
        generated_at=generated_at,
    )
    return {
        "etf_count": len(etf_codes),
        "saved_count": int(saved_count),
        "updated_count": int(sum(1 for entry in log_entries if entry["status"] == "updated")),
        "unchanged_count": int(sum(1 for entry in log_entries if entry["status"] == "unchanged")),
        "missing_count": int(sum(1 for entry in log_entries if entry["status"] == "missing")),
        "error_count": int(sum(1 for entry in log_entries if entry["status"] == "error")),
        "issues": issues,
        "entries": log_entries,
        "json_log_path": log_paths["json_path"],
        "markdown_log_path": log_paths["markdown_path"],
        "generated_at": generated_at.isoformat(),
        "symbols": list(etf_codes),
    }


def load_latest_tw_etf_constituent_snapshot(
    *,
    store: HistoryStore,
    symbol: str,
) -> dict[str, object] | None:
    snapshot = store.load_latest_market_snapshot(
        dataset_key=TW_ETF_CONSTITUENTS_DATASET_KEY,
        market="TW",
        symbol=str(symbol or "").strip().upper(),
        interval="constituents",
    )
    if not isinstance(snapshot, dict):
        return None
    payload = snapshot.get("payload")
    if not isinstance(payload, dict):
        return None
    return {
        "source": str(payload.get("source") or snapshot.get("source") or "").strip(),
        "rows": _normalize_constituent_rows(payload.get("rows", [])),
        "issue": str(payload.get("issue") or "").strip(),
        "asof": snapshot.get("asof"),
    }


def _resolve_target_etf_codes(*, app_module, symbols: list[str] | None) -> list[str]:
    normalized = [
        str(symbol).strip().upper()
        for symbol in (symbols or [])
        if str(symbol or "").strip()
    ]
    if normalized:
        return sorted(dict.fromkeys(normalized))
    profile_loader = getattr(app_module, "_load_tw_etf_official_profile_map", None)
    if callable(profile_loader):
        trade_day_resolver = getattr(app_module, "_resolve_latest_tw_trade_day_token", None)
        target_trade_day = ""
        if callable(trade_day_resolver):
            try:
                target_trade_day = str(_call_app_quiet(trade_day_resolver, None) or "").strip()
            except Exception:
                target_trade_day = ""
        try:
            profile_map = _call_app_quiet(profile_loader, target_trade_day) if target_trade_day else _call_app_quiet(profile_loader)
        except TypeError:
            profile_map = _call_app_quiet(profile_loader)
        except Exception:
            profile_map = {}
        if isinstance(profile_map, dict) and profile_map:
            return sorted(
                {
                    str(code).strip().upper()
                    for code in profile_map.keys()
                    if str(code).strip() and not str(code).strip().startswith("^")
                }
            )

    build_table = getattr(app_module, "_build_tw_etf_all_types_performance_table", None)
    resolve_trade_day = getattr(app_module, "_resolve_latest_tw_trade_day_token", None)
    if not callable(build_table):
        return []
    if not callable(resolve_trade_day):
        return []
    trade_token = str(_call_app_quiet(resolve_trade_day, None) or "").strip()
    if not trade_token:
        return []
    ytd_start = f"{trade_token[:4]}0101" if len(trade_token) >= 4 else "20260101"
    table_df, _ = _call_app_quiet(
        build_table,
        ytd_start_yyyymmdd=ytd_start,
        ytd_end_yyyymmdd=trade_token,
    )
    if not isinstance(table_df, pd.DataFrame) or table_df.empty or "代碼" not in table_df.columns:
        return []
    return sorted(
        {
            str(code).strip().upper()
            for code in table_df["代碼"].astype(str).tolist()
            if str(code).strip() and not str(code).strip().startswith("^")
        }
    )


def _normalize_constituent_rows(rows: object) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "rank": _normalize_optional_int(row.get("rank")),
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "tw_code": str(row.get("tw_code") or "").strip().upper(),
                "name": str(row.get("name") or "").strip(),
                "market": str(row.get("market") or "").strip().upper(),
                "weight_pct": _normalize_optional_float(row.get("weight_pct")),
                "shares": _normalize_optional_int(row.get("shares")),
            }
        )
    out.sort(key=lambda item: (str(item.get("symbol") or ""), str(item.get("tw_code") or "")))
    return out


def _normalize_optional_float(value: object) -> float | None:
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


def _normalize_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def _payload_signature(*, rows: list[dict[str, object]], source: str) -> str:
    payload = json.dumps(
        {
            "source": str(source or "").strip(),
            "rows": rows,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
