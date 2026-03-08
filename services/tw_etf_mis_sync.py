from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pandas as pd
import requests

if TYPE_CHECKING:
    from storage import HistoryStore

TWSE_MIS_ETF_CATEGORY_URL = "https://mis.twse.com.tw/stock/api/getCategory.jsp"
TWSE_MIS_ETF_DATA_URL = "https://mis.twse.com.tw/stock/data/all_etf.txt"
TWSE_MIS_ETF_SOURCE = "twse_mis_etf_indicator"
TW_TZ = timezone(timedelta(hours=8))
TWSE_MIS_ETF_COLUMNS = [
    "trade_date",
    "etf_code",
    "etf_name",
    "issued_units",
    "creation_redemption_diff",
    "market_price",
    "estimated_nav",
    "premium_discount_pct",
    "previous_nav",
    "reference_url",
    "updated_at",
    "source",
]


def empty_tw_etf_mis_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TWSE_MIS_ETF_COLUMNS)


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
    text = str(value or "").strip().replace(",", "").replace("%", "")
    if text in {"", "--", "-", "X0.00"}:
        return None
    if text.startswith("="):
        text = text[1:].strip().strip('"')
    try:
        return float(text)
    except Exception:
        return None


def _build_updated_at(trade_date_text: object, update_time_text: object) -> str:
    date_token = str(trade_date_text or "").strip()
    time_token = str(update_time_text or "").strip()
    try:
        trade_day = pd.Timestamp(date_token).date()
    except Exception:
        trade_day = datetime.now(tz=TW_TZ).date()
    if not time_token:
        return datetime.combine(trade_day, datetime.min.time(), tzinfo=TW_TZ).isoformat()
    try:
        parsed_time = datetime.strptime(time_token, "%H:%M:%S").time()
    except Exception:
        return datetime.combine(trade_day, datetime.min.time(), tzinfo=TW_TZ).isoformat()
    return datetime.combine(trade_day, parsed_time, tzinfo=TW_TZ).isoformat()


def fetch_twse_etf_mis_report(
    *,
    timeout_sec: int = 20,
) -> tuple[str, pd.DataFrame, dict[str, object]]:
    category_resp = requests.get(
        TWSE_MIS_ETF_CATEGORY_URL,
        params={"ex": "tse", "i": "B0", "lang": "zh_tw"},
        timeout=timeout_sec,
    )
    category_resp.raise_for_status()
    category_payload = category_resp.json()
    if not isinstance(category_payload, dict):
        raise RuntimeError("TWSE MIS ETF category response is not an object")

    category_rows = category_payload.get("msgArray")
    if not isinstance(category_rows, list):
        raise RuntimeError("TWSE MIS ETF category payload missing msgArray")

    category_map: dict[str, dict[str, str]] = {}
    for row in category_rows:
        if not isinstance(row, dict):
            continue
        code = _normalize_code(str(row.get("ch", "")).split(".", 1)[0] or row.get("c") or row.get("a"))
        if not code:
            continue
        category_map[code] = {
            "etf_name": _normalize_name(row.get("n") or row.get("nf") or code) or code,
            "channel": _normalize_name(row.get("ch")),
            "exchange": _normalize_name(row.get("ex")),
        }

    if not category_map:
        raise RuntimeError("TWSE MIS ETF category payload returned no ETF codes")

    data_resp = requests.get(TWSE_MIS_ETF_DATA_URL, timeout=timeout_sec)
    data_resp.raise_for_status()
    data_payload = data_resp.json()
    if not isinstance(data_payload, dict):
        raise RuntimeError("TWSE MIS ETF indicator response is not an object")

    groups = data_payload.get("a1")
    if not isinstance(groups, list):
        raise RuntimeError("TWSE MIS ETF indicator payload missing a1 groups")

    normalized_rows: list[dict[str, object]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        ref_url = _normalize_name(group.get("refURL"))
        msg_array = group.get("msgArray")
        if not isinstance(msg_array, list):
            continue
        for row in msg_array:
            if not isinstance(row, dict):
                continue
            code = _normalize_code(row.get("a"))
            if not code or code not in category_map:
                continue
            trade_date_token = str(row.get("i") or "").strip()
            try:
                trade_day = pd.Timestamp(trade_date_token).date().isoformat()
            except Exception:
                continue
            category_info = category_map.get(code, {})
            normalized_rows.append(
                {
                    "trade_date": trade_day,
                    "etf_code": code,
                    "etf_name": category_info.get("etf_name") or _normalize_name(row.get("b")) or code,
                    "issued_units": _to_float(row.get("c")),
                    "creation_redemption_diff": _to_float(row.get("d")),
                    "market_price": _to_float(row.get("e")),
                    "estimated_nav": _to_float(row.get("f")),
                    "premium_discount_pct": _to_float(row.get("g")),
                    "previous_nav": _to_float(row.get("h")),
                    "reference_url": ref_url,
                    "updated_at": _build_updated_at(trade_date_token, row.get("j")),
                    "source": TWSE_MIS_ETF_SOURCE,
                }
            )

    frame = pd.DataFrame(normalized_rows)
    if frame.empty:
        return "", empty_tw_etf_mis_frame(), {"row_count": 0, "group_count": len(groups)}

    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"])
    if frame.empty:
        return "", empty_tw_etf_mis_frame(), {"row_count": 0, "group_count": len(groups)}

    latest_trade_date = frame["trade_date"].max().date()
    frame = frame.loc[frame["trade_date"].dt.date == latest_trade_date].copy()
    for column in [
        "issued_units",
        "creation_redemption_diff",
        "market_price",
        "estimated_nav",
        "premium_discount_pct",
        "previous_nav",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_date"] = frame["trade_date"].dt.date.astype(str)
    frame = frame.sort_values(["etf_code"]).reset_index(drop=True)
    return latest_trade_date.isoformat(), frame[TWSE_MIS_ETF_COLUMNS], {
        "row_count": int(len(frame)),
        "group_count": len(groups),
        "source": TWSE_MIS_ETF_SOURCE,
    }


def sync_twse_etf_mis_daily(
    *,
    store: HistoryStore,
    start: object | None = None,
    end: object | None = None,
    force: bool = False,
    timeout_sec: int = 20,
) -> dict[str, object]:
    requested_start = _coerce_trade_date(start) if start is not None else None
    requested_end = _coerce_trade_date(end) if end is not None else None
    if requested_start and requested_end and requested_start > requested_end:
        requested_start, requested_end = requested_end, requested_start

    issues: list[str] = []
    try:
        latest_date, frame, fetch_meta = fetch_twse_etf_mis_report(timeout_sec=timeout_sec)
    except Exception as exc:
        return {
            "start_date": requested_start.isoformat() if requested_start else "",
            "end_date": requested_end.isoformat() if requested_end else "",
            "requested_days": 0,
            "synced_days": 0,
            "skipped_days": 0,
            "empty_days": 0,
            "saved_rows": 0,
            "latest_date": "",
            "available_date": "",
            "issues": [str(exc)],
            "source": TWSE_MIS_ETF_SOURCE,
        }

    if not latest_date:
        return {
            "start_date": requested_start.isoformat() if requested_start else "",
            "end_date": requested_end.isoformat() if requested_end else "",
            "requested_days": 1,
            "synced_days": 0,
            "skipped_days": 0,
            "empty_days": 1,
            "saved_rows": 0,
            "latest_date": "",
            "available_date": "",
            "issues": issues,
            "source": TWSE_MIS_ETF_SOURCE,
        }

    available_date = _coerce_trade_date(latest_date)
    if requested_start is None:
        requested_start = available_date
    if requested_end is None:
        requested_end = available_date
    requested_days = 0
    if requested_start <= requested_end:
        requested_days = int(len(pd.date_range(start=requested_start, end=requested_end, freq="D")))

    if available_date < requested_start or available_date > requested_end:
        issues.append(
            f"latest available date {available_date.isoformat()} is outside requested range "
            f"{requested_start.isoformat()} -> {requested_end.isoformat()}"
        )
        return {
            "start_date": requested_start.isoformat(),
            "end_date": requested_end.isoformat(),
            "requested_days": requested_days,
            "synced_days": 0,
            "skipped_days": 0,
            "empty_days": 0,
            "saved_rows": 0,
            "latest_date": available_date.isoformat(),
            "available_date": available_date.isoformat(),
            "issues": issues,
            "source": TWSE_MIS_ETF_SOURCE,
        }

    if not force:
        coverage = store.load_tw_etf_mis_daily_coverage(start=available_date, end=available_date)
        if int(coverage.get("row_count") or 0) > 0:
            return {
                "start_date": requested_start.isoformat(),
                "end_date": requested_end.isoformat(),
                "requested_days": requested_days,
                "synced_days": 0,
                "skipped_days": 1,
                "empty_days": 0,
                "saved_rows": 0,
                "latest_date": available_date.isoformat(),
                "available_date": available_date.isoformat(),
                "issues": issues,
                "source": TWSE_MIS_ETF_SOURCE,
            }

    if frame.empty:
        return {
            "start_date": requested_start.isoformat(),
            "end_date": requested_end.isoformat(),
            "requested_days": requested_days,
            "synced_days": 0,
            "skipped_days": 0,
            "empty_days": 1,
            "saved_rows": 0,
            "latest_date": available_date.isoformat(),
            "available_date": available_date.isoformat(),
            "issues": issues,
            "source": TWSE_MIS_ETF_SOURCE,
        }

    saved_rows = int(
        store.save_tw_etf_mis_daily(
            rows=frame.to_dict("records"),
            trade_date=available_date.isoformat(),
            source=TWSE_MIS_ETF_SOURCE,
        )
        or 0
    )
    latest = store.load_tw_etf_mis_daily_coverage()
    latest_date_raw = latest.get("last_date") if isinstance(latest, dict) else None
    latest_token = ""
    if isinstance(latest_date_raw, datetime):
        latest_token = latest_date_raw.date().isoformat()
    elif latest_date_raw:
        latest_token = str(latest_date_raw)

    return {
        "start_date": requested_start.isoformat(),
        "end_date": requested_end.isoformat(),
        "requested_days": requested_days,
        "synced_days": 1,
        "skipped_days": 0,
        "empty_days": 0,
        "saved_rows": saved_rows,
        "latest_date": latest_token,
        "available_date": available_date.isoformat(),
        "issues": issues,
        "source": TWSE_MIS_ETF_SOURCE,
        "row_count": int(fetch_meta.get("row_count") or 0),
    }
