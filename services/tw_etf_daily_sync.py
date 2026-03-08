from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pandas as pd
import requests

if TYPE_CHECKING:
    from storage import HistoryStore

TWSE_ETF_DAILY_URL = "https://www.twse.com.tw/ETFReport/ETFDaily"
TWSE_ETF_DAILY_SOURCE = "twse_etf_daily"
TWSE_ETF_DAILY_START_DATE = date(2017, 1, 1)
TWSE_ETF_DAILY_COLUMNS = [
    "trade_date",
    "etf_code",
    "etf_name",
    "trade_value",
    "trade_volume",
    "trade_count",
    "open",
    "high",
    "low",
    "close",
    "change",
    "source",
]


def empty_tw_etf_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TWSE_ETF_DAILY_COLUMNS)


def _coerce_trade_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    ts = pd.Timestamp(value)
    return ts.date()


def _normalize_code(value: object) -> str:
    return str(value or "").strip().upper().replace('="', "").replace('"', "")


def _normalize_name(value: object) -> str:
    return str(value or "").strip()


def _to_float(value: object) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if text in {"", "--", "-", "X0.00"}:
        return None
    if text.startswith("="):
        text = text[1:].strip().strip('"')
    try:
        return float(text)
    except Exception:
        return None


def _to_int(value: object) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except Exception:
        return None


def fetch_twse_etf_daily_report(
    trade_date: object,
    *,
    timeout_sec: int = 20,
) -> tuple[str, pd.DataFrame, dict[str, object]]:
    trade_day = _coerce_trade_date(trade_date)
    date_token = trade_day.strftime("%Y%m%d")
    resp = requests.get(
        TWSE_ETF_DAILY_URL,
        params={"response": "json", "date": date_token},
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("TWSE ETF daily response is not an object")

    stat = str(payload.get("stat", "") or "").strip()
    rows = payload.get("data")
    if not isinstance(rows, list):
        if "沒有符合條件" in stat:
            return trade_day.isoformat(), empty_tw_etf_daily_frame(), {"stat": stat, "date": date_token}
        raise RuntimeError(f"TWSE ETF daily payload missing data rows: {stat or 'unknown'}")

    fields = payload.get("fields")
    field_list = [str(item or "").strip() for item in fields] if isinstance(fields, list) else []
    if not field_list:
        field_list = [
            "證券代號",
            "證券名稱",
            "成交金額",
            "成交股數",
            "成交筆數",
            "開盤價",
            "最高價",
            "最低價",
            "收盤價",
            "漲跌價差",
        ]

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        padded = list(row) + [None] * max(0, len(field_list) - len(row))
        raw = dict(zip(field_list, padded, strict=False))
        code = _normalize_code(raw.get("證券代號"))
        if not code:
            continue
        normalized_rows.append(
            {
                "trade_date": trade_day.isoformat(),
                "etf_code": code,
                "etf_name": _normalize_name(raw.get("證券名稱")) or code,
                "trade_value": _to_float(raw.get("成交金額")),
                "trade_volume": _to_float(raw.get("成交股數")),
                "trade_count": _to_int(raw.get("成交筆數")),
                "open": _to_float(raw.get("開盤價")),
                "high": _to_float(raw.get("最高價")),
                "low": _to_float(raw.get("最低價")),
                "close": _to_float(raw.get("收盤價")),
                "change": _to_float(raw.get("漲跌價差")),
                "source": TWSE_ETF_DAILY_SOURCE,
            }
        )

    frame = pd.DataFrame(normalized_rows)
    if frame.empty:
        return trade_day.isoformat(), empty_tw_etf_daily_frame(), {"stat": stat, "date": date_token}

    for column in ["trade_value", "trade_volume", "trade_count", "open", "high", "low", "close", "change"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"], how="any")
    if frame.empty:
        return trade_day.isoformat(), empty_tw_etf_daily_frame(), {"stat": stat, "date": date_token}
    frame = frame.sort_values(["etf_code"]).reset_index(drop=True)
    return trade_day.isoformat(), frame[TWSE_ETF_DAILY_COLUMNS], {
        "stat": stat,
        "date": date_token,
        "row_count": int(len(frame)),
    }


def sync_twse_etf_daily_market(
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
        coverage = getattr(store, "load_tw_etf_daily_market_coverage", lambda **kwargs: {})()
        last_date = coverage.get("last_date") if isinstance(coverage, dict) else None
        if isinstance(last_date, datetime):
            start_date = (last_date.date() - timedelta(days=max(0, int(lookback_days))))
        elif last_date:
            start_date = _coerce_trade_date(last_date) - timedelta(days=max(0, int(lookback_days)))
        else:
            start_date = TWSE_ETF_DAILY_START_DATE

    if start_date < TWSE_ETF_DAILY_START_DATE:
        start_date = TWSE_ETF_DAILY_START_DATE
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
            coverage = store.load_tw_etf_daily_market_coverage(start=trade_day, end=trade_day)
            if int(coverage.get("row_count") or 0) > 0:
                skipped_dates.append(date_iso)
                continue
        try:
            _, frame, _ = fetch_twse_etf_daily_report(trade_day, timeout_sec=timeout_sec)
        except Exception as exc:
            issues.append(f"{date_iso}: {exc}")
            continue
        if frame.empty:
            empty_dates.append(date_iso)
            continue
        saved_rows += int(
            store.save_tw_etf_daily_market(
                rows=frame.to_dict("records"),
                trade_date=date_iso,
                source=TWSE_ETF_DAILY_SOURCE,
            )
            or 0
        )
        synced_dates.append(date_iso)

    latest = store.load_tw_etf_daily_market_coverage()
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
        "source": TWSE_ETF_DAILY_SOURCE,
    }
