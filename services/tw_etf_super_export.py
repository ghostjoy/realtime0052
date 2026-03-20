from __future__ import annotations

import hashlib
import importlib
import io
import math
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from services.tw_etf_daily_sync import sync_twse_etf_daily_market
from services.tw_etf_margin_sync import sync_twse_etf_margin_daily
from services.tw_etf_mis_sync import TWSE_MIS_ETF_SOURCE, sync_twse_etf_mis_daily
from storage import HistoryStore


@contextmanager
def _suppress_streamlit_noise():
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        yield


def _load_app_module():
    with _suppress_streamlit_noise():
        return importlib.import_module("app")


def _call_app_quiet(func, *args, **kwargs):
    with _suppress_streamlit_noise():
        return func(*args, **kwargs)


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except Exception:
        return None


def _truncate_value(value: object, *, digits: int = 2) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    factor = 10.0 ** max(0, int(digits))
    return math.trunc(float(number) * factor) / factor


def _normalize_yyyymmdd(value: object | None, *, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    if len(text) == 8 and text.isdigit():
        return text
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value}")
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def resolve_tw_etf_super_export_periods(
    *,
    ytd_start: object | None = None,
    ytd_end: object | None = None,
    compare_start: object | None = None,
    compare_end: object | None = None,
    now: datetime | None = None,
) -> dict[str, str]:
    current_dt = now or datetime.now(tz=timezone.utc)
    current_date = current_dt.astimezone().date()
    current_year = current_date.year
    previous_year = current_year - 1
    defaults = {
        "ytd_start": f"{current_year}0101",
        "ytd_end": current_date.strftime("%Y%m%d"),
        "compare_start": f"{previous_year}0101",
        "compare_end": f"{previous_year}1231",
    }
    periods = {
        "ytd_start": _normalize_yyyymmdd(ytd_start, default=defaults["ytd_start"]),
        "ytd_end": _normalize_yyyymmdd(ytd_end, default=defaults["ytd_end"]),
        "compare_start": _normalize_yyyymmdd(compare_start, default=defaults["compare_start"]),
        "compare_end": _normalize_yyyymmdd(compare_end, default=defaults["compare_end"]),
    }
    if periods["ytd_start"] > periods["ytd_end"]:
        raise ValueError("ytd_start cannot be later than ytd_end")
    if periods["compare_start"] > periods["compare_end"]:
        raise ValueError("compare_start cannot be later than compare_end")
    return periods


def resolve_tw_etf_super_export_output_path(
    *,
    trade_token: str,
    out: str | None = None,
) -> Path:
    file_name = f"tw_etf_super_export_{str(trade_token or '').strip() or 'latest'}.csv"
    if not str(out or "").strip():
        return Path.cwd() / file_name
    candidate = Path(str(out).strip()).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate / file_name
    if str(candidate).endswith("/"):
        return candidate / file_name
    return candidate


def _clear_export_related_caches(app_module) -> None:
    clear_fn = getattr(app_module, "_clear_tw_etf_all_types_view_caches", None)
    if callable(clear_fn):
        _call_app_quiet(clear_fn)


def _sync_tw_etf_super_export_sources(
    *,
    store: HistoryStore,
    app_module,
    daily_lookback_days: int,
    force: bool,
    target_trade_date: str | None = None,
) -> dict[str, object]:
    resolve_latest_trade_day_token = app_module._resolve_latest_tw_trade_day_token
    fetch_snapshot_network_single = app_module._fetch_twse_snapshot_network_single
    fetch_snapshot_with_fallback = app_module._fetch_twse_snapshot_with_fallback
    trade_token = str(
        _call_app_quiet(
            resolve_latest_trade_day_token, str(target_trade_date or "").strip() or None
        )
    ).strip()
    summary: dict[str, object] = {
        "main": {"status": "fallback", "used_trade_date": ""},
        "daily_market": {},
        "margin": {},
        "mis": {},
        "three_investors": {},
        "aum_track": {},
        "issues": [],
    }
    issues: list[str] = []
    _clear_export_related_caches(app_module)
    try:
        main_out = _call_app_quiet(fetch_snapshot_network_single, trade_token)
        if main_out is not None:
            used_trade_date, _ = main_out
            summary["main"] = {"status": "synced", "used_trade_date": str(used_trade_date)}
        else:
            used_trade_date, _ = _call_app_quiet(
                fetch_snapshot_with_fallback,
                trade_token,
                lookback_days=14,
            )
            summary["main"] = {"status": "fallback", "used_trade_date": str(used_trade_date)}
    except Exception as exc:
        issues.append(f"main_snapshot: {exc}")
        summary["main"] = {"status": "error", "used_trade_date": ""}

    try:
        summary["daily_market"] = sync_twse_etf_daily_market(
            store=store,
            start=pd.Timestamp(trade_token).date()
            - timedelta(days=max(1, int(daily_lookback_days)))
            if trade_token
            else None,
            end=pd.Timestamp(trade_token).date() if trade_token else None,
            lookback_days=max(1, int(daily_lookback_days)),
            force=bool(force),
        )
    except Exception as exc:
        issues.append(f"daily_market: {exc}")
        summary["daily_market"] = {"status": "error", "issues": [str(exc)]}

    try:
        summary["margin"] = sync_twse_etf_margin_daily(
            store=store,
            start=pd.Timestamp(trade_token).date()
            - timedelta(days=max(1, int(daily_lookback_days)))
            if trade_token
            else None,
            end=pd.Timestamp(trade_token).date() if trade_token else None,
            lookback_days=max(1, int(daily_lookback_days)),
            force=bool(force),
        )
    except Exception as exc:
        issues.append(f"margin: {exc}")
        summary["margin"] = {"status": "error", "issues": [str(exc)]}

    try:
        target_date = pd.Timestamp(trade_token).date() if trade_token else None
        local_coverage = (
            store.load_tw_etf_mis_daily_coverage(start=target_date, end=target_date)
            if target_date is not None
            else {}
        )
        if target_date is not None and int(local_coverage.get("row_count") or 0) > 0 and not force:
            summary["mis"] = {
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "requested_days": 1,
                "synced_days": 0,
                "skipped_days": 1,
                "empty_days": 0,
                "saved_rows": 0,
                "latest_date": target_date.isoformat(),
                "available_date": target_date.isoformat(),
                "issues": [],
                "source": TWSE_MIS_ETF_SOURCE,
            }
        else:
            summary["mis"] = sync_twse_etf_mis_daily(
                store=store,
                start=target_date,
                end=target_date,
                force=bool(force),
            )
    except Exception as exc:
        issues.append(f"mis: {exc}")
        summary["mis"] = {"status": "error", "issues": [str(exc)]}

    fetch_three_investors = getattr(app_module, "_fetch_twse_three_investors_with_fallback", None)
    if callable(fetch_three_investors):
        try:
            used_trade_date, three_frame = _call_app_quiet(
                fetch_three_investors,
                trade_token,
                lookback_days=max(1, int(daily_lookback_days)),
            )
            summary["three_investors"] = {
                "status": "synced" if str(used_trade_date) == str(trade_token) else "fallback",
                "used_trade_date": pd.to_datetime(
                    str(used_trade_date), format="%Y%m%d", errors="coerce"
                ).strftime("%Y-%m-%d")
                if str(used_trade_date).strip()
                else "",
                "row_count": int(len(three_frame)),
            }
        except Exception as exc:
            issues.append(f"three_investors: {exc}")
            summary["three_investors"] = {"status": "error", "issues": [str(exc)]}

    resolve_trade_date_iso = getattr(app_module, "_resolve_latest_tw_trade_date_iso", None)
    load_aum_snapshot_info = getattr(app_module, "_load_tw_etf_aum_snapshot_info", None)
    build_aum_rows_from_snapshot_info = getattr(
        app_module, "_build_tw_etf_aum_rows_from_snapshot_info", None
    )
    if callable(load_aum_snapshot_info) and callable(build_aum_rows_from_snapshot_info):
        try:
            trade_date_iso = (
                str(_call_app_quiet(resolve_trade_date_iso, trade_token)).strip()
                if callable(resolve_trade_date_iso)
                else (
                    pd.Timestamp(trade_token).date().isoformat() if str(trade_token).strip() else ""
                )
            )
            snapshot_info = _call_app_quiet(load_aum_snapshot_info, trade_token)
            aum_rows = _call_app_quiet(build_aum_rows_from_snapshot_info, snapshot_info)
            if aum_rows:
                updated = store.save_tw_etf_aum_snapshot(
                    rows=aum_rows,
                    trade_date=trade_date_iso,
                    keep_days=0,
                )
                summary["aum_track"] = {
                    "status": "synced",
                    "updated": int(updated),
                    "trade_date": trade_date_iso,
                }
            else:
                summary["aum_track"] = {
                    "status": "empty",
                    "updated": 0,
                    "trade_date": trade_date_iso,
                }
        except Exception as exc:
            issues.append(f"aum_track: {exc}")
            summary["aum_track"] = {"status": "error", "issues": [str(exc)]}

    _clear_export_related_caches(app_module)
    summary["issues"] = issues
    return summary


def build_tw_etf_all_types_main_export_frame(
    *,
    table_df: pd.DataFrame,
    meta: dict[str, object],
) -> pd.DataFrame:
    frame = table_df.copy() if isinstance(table_df, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return frame
    frame = frame[[col for col in frame.columns if col not in {"官方主分類", "官方次分類"}]]
    market_2025_return = _safe_float(meta.get("market_2025_return"))
    market_ytd_return = _safe_float(meta.get("market_ytd_return"))
    market_daily_return = _safe_float(meta.get("market_daily_return"))
    market_daily_prev_close = _safe_float(meta.get("market_daily_prev_close"))
    market_daily_open = _safe_float(meta.get("market_daily_open"))
    market_daily_close = _safe_float(meta.get("market_daily_close"))
    benchmark_row = {
        "編號": "—",
        "代碼": "^TWII",
        "ETF": "台股大盤",
        "管理費(%)": None,
        "ETF規模(億)": None,
        "類型": "大盤",
        "2025績效(%)": _truncate_value(market_2025_return, digits=2)
        if market_2025_return is not None
        else None,
        "大盤超額2025(%)": 0.0 if market_2025_return is not None else None,
        "YTD績效(%)": _truncate_value(market_ytd_return, digits=2)
        if market_ytd_return is not None
        else None,
        "大盤超額YTD(%)": 0.0 if market_ytd_return is not None else None,
        "昨收": _truncate_value(market_daily_prev_close, digits=2)
        if market_daily_prev_close is not None
        else None,
        "開盤": _truncate_value(market_daily_open, digits=2)
        if market_daily_open is not None
        else None,
        "收盤": _truncate_value(market_daily_close, digits=2)
        if market_daily_close is not None
        else None,
        "今日漲幅": _truncate_value(market_daily_return, digits=2)
        if market_daily_return is not None
        else None,
        "今日贏大盤%": 0.0 if market_daily_return is not None else None,
    }
    table_view_df = pd.DataFrame.from_records(
        [benchmark_row, *frame.to_dict("records")],
        columns=frame.columns,
    )
    if "編號" in table_view_df.columns:
        serial_numbers = pd.to_numeric(table_view_df["編號"], errors="coerce")
        table_view_df["編號"] = serial_numbers.map(lambda v: "—" if pd.isna(v) else f"{int(v)}")
    return table_view_df


def _is_tw_etf_super_export_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text in {"", "-", "—", "nan", "None", "<NA>"}


def _sanitize_tw_etf_super_export_value(value: object) -> object:
    if _is_tw_etf_super_export_missing(value):
        return "-"
    if not isinstance(value, str):
        return value
    text = str(value).strip()
    if not text.startswith("?"):
        return text
    try:
        from urllib.parse import parse_qs, unquote

        from ui.helpers import strip_symbol_label_token
    except Exception:
        return text
    query = parse_qs(text.lstrip("?"), keep_blank_values=True)
    if not query:
        return text
    bt_label = query.get("bt_label", [])
    if bt_label:
        return strip_symbol_label_token(unquote(str(bt_label[0])))
    hm_label = query.get("hm_label", [])
    if hm_label:
        return unquote(str(hm_label[0]))
    hm_name = query.get("hm_name", [])
    if hm_name:
        return unquote(str(hm_name[0]))
    bt_symbol = query.get("bt_symbol", [])
    if bt_symbol:
        return str(bt_symbol[0]).strip().upper()
    hm_etf = query.get("hm_etf", [])
    if hm_etf:
        return str(hm_etf[0]).strip().upper()
    return text


def _coalesce_tw_etf_super_export_columns(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    primary_missing = primary.map(_is_tw_etf_super_export_missing)
    return primary.where(~primary_missing, fallback)


def build_tw_etf_super_export_table(
    *,
    main_frame: pd.DataFrame,
    daily_market_frame: pd.DataFrame,
    margin_frame: pd.DataFrame | None = None,
    mis_frame: pd.DataFrame,
    three_investors_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sources: list[tuple[str, pd.DataFrame]] = [
        ("main", main_frame),
        ("daily", daily_market_frame),
    ]
    if isinstance(margin_frame, pd.DataFrame):
        sources.append(("margin", margin_frame))
    sources.append(("mis", mis_frame))
    if isinstance(three_investors_frame, pd.DataFrame):
        sources.append(("three_investors", three_investors_frame))
    prepared: list[tuple[str, pd.DataFrame]] = []
    ordered_columns: list[str] = []
    for source_name, frame in sources:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "代碼" not in frame.columns:
            continue
        work = frame.copy()
        work["代碼"] = work["代碼"].astype(str).str.strip().str.upper()
        work = (
            work[work["代碼"] != ""]
            .drop_duplicates(subset=["代碼"], keep="first")
            .reset_index(drop=True)
        )
        if work.empty:
            continue
        prepared.append((source_name, work))
        for column in work.columns:
            if column not in ordered_columns:
                ordered_columns.append(str(column))
    if not prepared:
        return pd.DataFrame()

    result = prepared[0][1].copy()
    for source_name, frame in prepared[1:]:
        merge_suffix = f"__{source_name}"
        result = result.merge(
            frame, on="代碼", how="outer", sort=False, suffixes=("", merge_suffix)
        )
        duplicated_columns = [
            column
            for column in frame.columns
            if column != "代碼" and f"{column}{merge_suffix}" in result.columns
        ]
        for column in duplicated_columns:
            result[column] = _coalesce_tw_etf_super_export_columns(
                result[column],
                result[f"{column}{merge_suffix}"],
            )
            result = result.drop(columns=[f"{column}{merge_suffix}"])

    final_columns = [column for column in ordered_columns if column in result.columns]
    if final_columns:
        result = result[final_columns]
    sanitized = result.copy()
    for column in sanitized.columns:
        sanitized[column] = sanitized[column].map(_sanitize_tw_etf_super_export_value)
    if "代碼" in sanitized.columns:
        code_order = sanitized["代碼"].astype(str).str.strip().str.upper()
        name_order = (
            sanitized["ETF"].astype(str).str.strip()
            if "ETF" in sanitized.columns
            else pd.Series("", index=sanitized.index)
        )
        sanitized = (
            sanitized.assign(
                __sort_bucket=code_order.map(lambda token: 0 if token.startswith("^") else 1),
                __sort_code=code_order,
                __sort_name=name_order,
            )
            .sort_values(
                ["__sort_bucket", "__sort_code", "__sort_name"],
                ascending=[True, True, True],
                na_position="last",
            )
            .drop(columns=["__sort_bucket", "__sort_code", "__sort_name"])
            .reset_index(drop=True)
        )
    serial_width = max(3, len(str(max(1, len(sanitized)))))
    code_series = (
        sanitized["代碼"].astype(str).str.strip().str.upper()
        if "代碼" in sanitized.columns
        else pd.Series("", index=sanitized.index)
    )
    serial_values: list[str] = []
    etf_counter = 0
    for code in code_series.tolist():
        if str(code).startswith("^"):
            serial_values.append("-")
            continue
        etf_counter += 1
        serial_values.append(str(etf_counter).zfill(serial_width))
    if "編號" in sanitized.columns:
        sanitized["編號"] = pd.Series(serial_values, index=sanitized.index, dtype="string")
        ordered = ["編號", *[column for column in sanitized.columns if column != "編號"]]
        sanitized = sanitized[ordered]
    else:
        sanitized.insert(0, "編號", pd.Series(serial_values, index=sanitized.index, dtype="string"))
    return sanitized.reset_index(drop=True)


def build_tw_etf_super_export_csv_bytes(frame: pd.DataFrame) -> bytes:
    export_frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if isinstance(export_frame, pd.DataFrame) and not export_frame.empty:
        export_frame = export_frame.copy()
        for column in ("編號", "代碼"):
            if column not in export_frame.columns:
                continue
            export_frame[column] = export_frame[column].map(
                lambda value: f'="{str(value).replace(chr(34), chr(34) * 2)}"'
            )
    csv_text = export_frame.to_csv(index=False)
    return ("\ufeff" + csv_text).encode("utf-8")


def _frame_payload(frame: pd.DataFrame) -> dict[str, object]:
    safe_frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if safe_frame.empty:
        return {"columns": [], "rows": []}
    safe_frame = safe_frame.astype("object").where(pd.notna(safe_frame), None)
    return {
        "columns": [str(col) for col in safe_frame.columns],
        "rows": safe_frame.to_dict("records"),
    }


def export_tw_etf_super_table_artifact(
    *,
    store: HistoryStore,
    out: str | None = None,
    ytd_start: str | None = None,
    ytd_end: str | None = None,
    compare_start: str | None = None,
    compare_end: str | None = None,
    force: bool = False,
    daily_lookback_days: int = 14,
) -> dict[str, object]:
    app_module = _load_app_module()
    build_performance_table = app_module._build_tw_etf_all_types_performance_table
    build_daily_market_overview = app_module._build_tw_etf_daily_market_overview
    build_margin_overview = app_module._build_tw_etf_margin_overview
    build_mis_overview = app_module._build_tw_etf_mis_overview
    build_three_investors_overview = app_module._build_tw_etf_three_investors_overview
    resolve_latest_trade_day_token = app_module._resolve_latest_tw_trade_day_token
    periods = resolve_tw_etf_super_export_periods(
        ytd_start=ytd_start,
        ytd_end=ytd_end,
        compare_start=compare_start,
        compare_end=compare_end,
    )
    target_trade_token = (
        str(_call_app_quiet(resolve_latest_trade_day_token, periods["ytd_end"])).strip()
        or periods["ytd_end"]
    )
    refresh_summary = _sync_tw_etf_super_export_sources(
        store=store,
        app_module=app_module,
        daily_lookback_days=max(1, int(daily_lookback_days)),
        force=bool(force),
        target_trade_date=target_trade_token,
    )
    table_df, main_meta = _call_app_quiet(
        build_performance_table,
        ytd_start_yyyymmdd=periods["ytd_start"],
        ytd_end_yyyymmdd=periods["ytd_end"],
        compare_start_yyyymmdd=periods["compare_start"],
        compare_end_yyyymmdd=periods["compare_end"],
    )
    daily_market_df, daily_market_meta = _call_app_quiet(
        build_daily_market_overview,
        lookback_days=max(30, int(daily_lookback_days)),
        top_n=0,
        target_trade_date=target_trade_token,
    )
    margin_df, margin_meta = _call_app_quiet(
        build_margin_overview,
        top_n=0,
        target_trade_date=target_trade_token,
    )
    mis_df, mis_meta = _call_app_quiet(
        build_mis_overview,
        top_n=0,
        target_trade_date=target_trade_token,
    )
    etf_codes = tuple(
        sorted(
            {
                str(code).strip().upper()
                for code in table_df.get("代碼", pd.Series(dtype=str)).astype(str).tolist()
                if str(code).strip() and not str(code).strip().startswith("^")
            }
        )
    )
    three_investors_df, three_investors_meta = _call_app_quiet(
        build_three_investors_overview,
        etf_codes=etf_codes,
        lookback_days=max(1, int(daily_lookback_days)),
        top_n=0,
        target_trade_date=target_trade_token,
    )
    main_frame = build_tw_etf_all_types_main_export_frame(table_df=table_df, meta=main_meta)
    super_export_df = build_tw_etf_super_export_table(
        main_frame=main_frame,
        daily_market_frame=daily_market_df,
        margin_frame=margin_df,
        mis_frame=mis_df,
        three_investors_frame=three_investors_df,
    )
    if super_export_df.empty:
        raise RuntimeError("no super export data generated")

    trade_token = target_trade_token
    output_path = resolve_tw_etf_super_export_output_path(trade_token=trade_token, out=out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = build_tw_etf_super_export_csv_bytes(super_export_df)
    output_path.write_bytes(csv_bytes)
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    payload = {
        "frame": _frame_payload(super_export_df),
        "refresh_summary": refresh_summary,
        "main_meta": dict(main_meta or {}),
        "daily_market_meta": dict(daily_market_meta or {}),
        "margin_meta": dict(margin_meta or {}),
        "mis_meta": dict(mis_meta or {}),
        "three_investors_meta": dict(three_investors_meta or {}),
        "csv_sha256": csv_sha256,
    }
    run_id = ""
    save_run = getattr(store, "save_tw_etf_super_export_run", None)
    if callable(save_run):
        run_id = str(
            save_run(
                ytd_start=periods["ytd_start"],
                ytd_end=periods["ytd_end"],
                compare_start=periods["compare_start"],
                compare_end=periods["compare_end"],
                trade_date_anchor=trade_token,
                output_path=str(output_path),
                row_count=int(len(super_export_df)),
                column_count=int(len(super_export_df.columns)),
                payload=payload,
            )
            or ""
        )
    return {
        "run_id": run_id,
        "output_path": str(output_path),
        "trade_date_anchor": trade_token,
        "row_count": int(len(super_export_df)),
        "column_count": int(len(super_export_df.columns)),
        "csv_sha256": csv_sha256,
        "ytd_start": periods["ytd_start"],
        "ytd_end": periods["ytd_end"],
        "compare_start": periods["compare_start"],
        "compare_end": periods["compare_end"],
        "refresh_summary": refresh_summary,
        "issues": list(refresh_summary.get("issues", []))
        if isinstance(refresh_summary, dict)
        else [],
    }
