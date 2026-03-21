from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SYNC_LOG_COLUMNS = [
    "dataset_name",
    "status",
    "requested_trade_date",
    "used_trade_date",
    "row_count_before",
    "row_count_after",
    "updated_rows",
    "source",
    "notes",
    "error",
]


def build_sync_log_entry(
    *,
    dataset_name: str,
    status: str,
    requested_trade_date: object = "",
    used_trade_date: object = "",
    row_count_before: object = None,
    row_count_after: object = None,
    updated_rows: object = None,
    source: object = "",
    notes: object = "",
    error: object = "",
) -> dict[str, object]:
    return {
        "dataset_name": str(dataset_name or "").strip(),
        "status": str(status or "").strip().lower() or "unknown",
        "requested_trade_date": _normalize_scalar(requested_trade_date),
        "used_trade_date": _normalize_scalar(used_trade_date),
        "row_count_before": _normalize_int(row_count_before),
        "row_count_after": _normalize_int(row_count_after),
        "updated_rows": _normalize_int(updated_rows),
        "source": _normalize_scalar(source),
        "notes": _normalize_scalar(notes),
        "error": _normalize_scalar(error),
    }


def build_sync_log_markdown(
    *,
    title: str,
    entries: list[dict[str, object]],
    generated_at: datetime | None = None,
    meta: dict[str, object] | None = None,
) -> str:
    ts = (generated_at or datetime.now(tz=timezone.utc)).astimezone(timezone.utc).isoformat()
    frame = pd.DataFrame(entries or [], columns=SYNC_LOG_COLUMNS)
    if frame.empty:
        table_md = "_No sync log entries._"
    else:
        printable = frame.copy()
        for col in ["row_count_before", "row_count_after", "updated_rows"]:
            printable[col] = printable[col].map(
                lambda value: "" if value is None or pd.isna(value) else str(int(value))
            )
        table_md = _markdown_table(printable)

    lines = [f"# {title}", "", f"- generated_at: `{ts}`"]
    for key, value in (meta or {}).items():
        lines.append(f"- {key}: `{_normalize_scalar(value)}`")
    lines.extend(["", "## Summary", "", table_md])

    detail_entries = [entry for entry in entries or [] if str(entry.get("notes") or "").strip() or str(entry.get("error") or "").strip()]
    if detail_entries:
        lines.extend(["", "## Details", ""])
        for entry in detail_entries:
            dataset_name = str(entry.get("dataset_name") or "").strip() or "unknown"
            lines.append(f"### {dataset_name}")
            lines.append(f"- status: `{str(entry.get('status') or '').strip() or 'unknown'}`")
            notes = str(entry.get("notes") or "").strip()
            error = str(entry.get("error") or "").strip()
            if notes:
                lines.append(f"- notes: {notes}")
            if error:
                lines.append(f"- error: {error}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_sync_log_files(
    *,
    log_dir: str | Path,
    file_stem: str,
    title: str,
    entries: list[dict[str, object]],
    meta: dict[str, object] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, str]:
    target_dir = Path(log_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = (generated_at or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    payload = {
        "title": title,
        "generated_at": ts.isoformat(),
        "meta": _json_safe(meta or {}),
        "entries": [_json_safe(entry) for entry in (entries or [])],
    }
    json_path = target_dir / f"{file_stem}.json"
    md_path = target_dir / f"{file_stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        build_sync_log_markdown(
            title=title,
            entries=entries,
            generated_at=ts,
            meta=meta,
        ),
        encoding="utf-8",
    )
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def _normalize_scalar(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (datetime, pd.Timestamp)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    return str(value).strip()


def _normalize_int(value: object) -> int | None:
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


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return _normalize_scalar(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(col) for col in frame.columns]
    rows = [
        [str(value if value is not None else "") for value in record]
        for record in frame.astype("object").where(pd.notna(frame), "").itertuples(index=False, name=None)
    ]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        escaped = [cell.replace("\n", " ").replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)
