from __future__ import annotations

import re

import pandas as pd

from market_data_types import DataHealth


def _format_as_of_token(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "N/A"
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return text
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d")


def build_data_health(
    *,
    as_of: object,
    data_sources: list[str],
    source_chain: list[str] | None = None,
    degraded: bool = False,
    fallback_depth: int = 0,
    freshness_sec: int | None = None,
    notes: str = "",
) -> DataHealth:
    sources = [str(item).strip() for item in data_sources if str(item).strip()]
    chain_items = source_chain if isinstance(source_chain, list) else []
    chain = [str(item).strip() for item in chain_items if str(item).strip()]
    return DataHealth(
        as_of=_format_as_of_token(as_of),
        data_sources=sources,
        source_chain=chain,
        degraded=bool(degraded),
        fallback_depth=max(0, int(fallback_depth or 0)),
        freshness_sec=freshness_sec,
        notes=str(notes).strip() or None,
    )


def render_data_health_caption(title: str, health: DataHealth) -> str:
    source_text = ",".join(health.data_sources) if health.data_sources else "unknown"
    chain_text = " -> ".join(health.source_chain) if health.source_chain else "—"
    freshness = f"{health.freshness_sec}s" if health.freshness_sec is not None else "—"
    note = f" | note={health.notes}" if health.notes else ""
    return (
        f"{title}：as_of={health.as_of or 'N/A'} | source={source_text} | "
        f"chain={chain_text} | degraded={'yes' if health.degraded else 'no'} | "
        f"fallback_depth={health.fallback_depth} | freshness={freshness}{note}"
    )


def render_sync_issues(prefix: str, issues: list[str], *, preview_limit: int = 3) -> str:
    items = [str(item).strip() for item in (issues or []) if str(item).strip()]
    if not items:
        return ""
    limit = max(1, int(preview_limit))
    preview = [" ".join(str(item).split()) for item in items[:limit]]
    preview_text = " | ".join(
        [item if len(item) <= 120 else f"{item[:117]}..." for item in preview]
    )
    remain = len(items) - len(preview)
    remain_text = f" | 其餘 {remain} 筆請查看終端 log。" if remain > 0 else ""
    return f"{prefix}：{preview_text}{remain_text}"


__all__ = ["build_data_health", "render_data_health_caption", "render_sync_issues"]
