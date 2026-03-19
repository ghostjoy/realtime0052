from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pandas as pd
import requests

if TYPE_CHECKING:
    from storage import HistoryStore

TWSE_ETF_MARGIN_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
TWSE_ETF_MARGIN_SOURCE = "twse_margin_mi_margn"
TWSE_ETF_MARGIN_START_DATE = date(2001, 1, 1)
TWSE_ETF_MARGIN_COLUMNS = [
    "trade_date",
    "etf_code",
    "etf_name",
    "margin_buy",
    "margin_sell",
    "margin_cash_redemption",
    "margin_prev_balance",
    "margin_balance",
    "margin_next_limit",
    "short_buy",
    "short_sell",
    "short_stock_redemption",
    "short_prev_balance",
    "short_balance",
    "short_next_limit",
    "offset_amount",
    "note",
    "source",
]


def empty_tw_etf_margin_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TWSE_ETF_MARGIN_COLUMNS)


def _coerce_trade_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    ts = pd.Timestamp(value)
    return ts.date()


def _normalize_code(value: object) -> str:
    return str(value or "").strip().upper().replace('="', "").replace('"', "")


def _normalize_name(value: object) -> str:
    return str(value or "").strip()


def _to_int(value: object) -> int | None:
    text = str(value or "").strip().replace(",", "")
    if text in {"", "--", "-"}:
        return None
    if text.startswith("="):
        text = text[1:].strip().strip('"')
    try:
        return int(float(text))
    except Exception:
        return None


def _is_no_data_stat(stat: object) -> bool:
    text = str(stat or "").strip()
    return any(token in text for token in ("沒有符合條件", "查無資料", "無資料"))


def _is_tw_etf_row(code: object, name: object) -> bool:
    code_text = _normalize_code(code)
    name_text = _normalize_name(name)
    return bool(code_text.startswith("00") and name_text)


def fetch_twse_etf_margin_report(
    trade_date: object,
    *,
    timeout_sec: int = 20,
) -> tuple[str, pd.DataFrame, dict[str, object]]:
    trade_day = _coerce_trade_date(trade_date)
    date_token = trade_day.strftime("%Y%m%d")
    resp = requests.get(
        TWSE_ETF_MARGIN_URL,
        params={"response": "json", "date": date_token, "selectType": "ALL"},
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("TWSE ETF margin response is not an object")

    stat = str(payload.get("stat", "") or "").strip()
    tables = payload.get("tables")
    if _is_no_data_stat(stat):
        return (
            trade_day.isoformat(),
            empty_tw_etf_margin_frame(),
            {"stat": stat, "date": date_token},
        )
    if not isinstance(tables, list):
        raise RuntimeError(f"TWSE ETF margin payload missing tables: {stat or 'unknown'}")

    detail_table = next(
        (
            table
            for table in tables
            if isinstance(table, dict)
            and isinstance(table.get("data"), list)
            and len(table.get("fields") or []) >= 16
        ),
        None,
    )
    if not isinstance(detail_table, dict):
        raise RuntimeError(f"TWSE ETF margin payload missing detail table: {stat or 'unknown'}")

    rows = detail_table.get("data")
    if not isinstance(rows, list):
        return (
            trade_day.isoformat(),
            empty_tw_etf_margin_frame(),
            {"stat": stat, "date": date_token},
        )

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        padded = list(row) + [None] * max(0, 16 - len(row))
        code = _normalize_code(padded[0])
        name = _normalize_name(padded[1]) or code
        if not _is_tw_etf_row(code, name):
            continue
        normalized_rows.append(
            {
                "trade_date": trade_day.isoformat(),
                "etf_code": code,
                "etf_name": name,
                "margin_buy": _to_int(padded[2]),
                "margin_sell": _to_int(padded[3]),
                "margin_cash_redemption": _to_int(padded[4]),
                "margin_prev_balance": _to_int(padded[5]),
                "margin_balance": _to_int(padded[6]),
                "margin_next_limit": _to_int(padded[7]),
                "short_buy": _to_int(padded[8]),
                "short_sell": _to_int(padded[9]),
                "short_stock_redemption": _to_int(padded[10]),
                "short_prev_balance": _to_int(padded[11]),
                "short_balance": _to_int(padded[12]),
                "short_next_limit": _to_int(padded[13]),
                "offset_amount": _to_int(padded[14]),
                "note": _normalize_name(padded[15]),
                "source": TWSE_ETF_MARGIN_SOURCE,
            }
        )

    frame = pd.DataFrame(normalized_rows)
    if frame.empty:
        return (
            trade_day.isoformat(),
            empty_tw_etf_margin_frame(),
            {"stat": stat, "date": date_token},
        )

    for column in TWSE_ETF_MARGIN_COLUMNS:
        if column.startswith(("margin_", "short_", "offset_")) and column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["etf_code"]).reset_index(drop=True)
    return (
        trade_day.isoformat(),
        frame[TWSE_ETF_MARGIN_COLUMNS],
        {
            "stat": stat,
            "date": date_token,
            "row_count": int(len(frame)),
            "table_title": str(detail_table.get("title", "") or "").strip(),
        },
    )


def sync_twse_etf_margin_daily(
    *,
    store: HistoryStore,
    start: object | None = None,
    end: object | None = None,
    lookback_days: int = 7,
    force: bool = False,
    timeout_sec: int = 20,
) -> dict[str, object]:
    end_date = _coerce_trade_date(end or datetime.now(tz=timezone.utc))
    if start is not None:
        start_date = _coerce_trade_date(start)
    else:
        coverage = getattr(store, "load_tw_etf_margin_daily_coverage", lambda **kwargs: {})()
        last_date = coverage.get("last_date") if isinstance(coverage, dict) else None
        if isinstance(last_date, datetime):
            start_date = last_date.date() - timedelta(days=max(0, int(lookback_days)))
        elif last_date:
            start_date = _coerce_trade_date(last_date) - timedelta(days=max(0, int(lookback_days)))
        else:
            start_date = end_date - timedelta(days=max(0, int(lookback_days)))

    if start_date < TWSE_ETF_MARGIN_START_DATE:
        start_date = TWSE_ETF_MARGIN_START_DATE
    if start_date > end_date:
        start_date = end_date

    dates = [dt.date() for dt in pd.date_range(start=start_date, end=end_date, freq="D")]
    saved_rows = 0
    synced_dates: list[str] = []
    skipped_dates: list[str] = []
    empty_dates: list[str] = []
    issues: list[str] = []

    for trade_day in dates:
        date_iso = trade_day.isoformat()
        if not force:
            coverage = store.load_tw_etf_margin_daily_coverage(start=trade_day, end=trade_day)
            if int(coverage.get("row_count") or 0) > 0:
                skipped_dates.append(date_iso)
                continue
        try:
            _, frame, _ = fetch_twse_etf_margin_report(trade_day, timeout_sec=timeout_sec)
        except Exception as exc:
            issues.append(f"{date_iso}: {exc}")
            continue
        if frame.empty:
            empty_dates.append(date_iso)
            continue
        saved_rows += int(
            store.save_tw_etf_margin_daily(
                rows=frame.to_dict("records"),
                trade_date=date_iso,
                source=TWSE_ETF_MARGIN_SOURCE,
            )
            or 0
        )
        synced_dates.append(date_iso)

    latest = store.load_tw_etf_margin_daily_coverage()
    latest_date = latest.get("last_date") if isinstance(latest, dict) else None
    latest_token = ""
    if isinstance(latest_date, datetime):
        latest_token = latest_date.date().isoformat()
    elif latest_date:
        latest_token = str(latest_date)

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "requested_days": int(len(dates)),
        "synced_days": int(len(synced_dates)),
        "skipped_days": int(len(skipped_dates)),
        "empty_days": int(len(empty_dates)),
        "saved_rows": int(saved_rows),
        "latest_date": latest_token,
        "issues": issues,
        "synced_dates": synced_dates,
        "skipped_dates": skipped_dates,
        "empty_dates": empty_dates,
        "source": TWSE_ETF_MARGIN_SOURCE,
    }
