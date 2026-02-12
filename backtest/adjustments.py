from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitEvent:
    date: pd.Timestamp
    ratio: float  # post / pre, e.g. 1/7
    source: str   # known / auto


# Known split events (minimal curated list for symbols we actively use).
# ratio = post-split price / pre-split price.
KNOWN_SPLITS: Dict[Tuple[str, str], Sequence[Tuple[str, float]]] = {
    ("TW", "0052"): (("2025-11-26", 1.0 / 7.0),),
}


def _candidate_ratios() -> List[float]:
    out: List[float] = []
    for n in range(2, 11):
        out.append(1.0 / n)
    for n in range(2, 11):
        out.append(float(n))
    return out


def detect_split_events(
    bars: pd.DataFrame,
    jump_threshold_low: float = 0.55,
    jump_threshold_high: float = 1.8,
    tolerance: float = 0.08,
) -> List[SplitEvent]:
    if bars is None or bars.empty or "close" not in bars.columns:
        return []

    close = pd.to_numeric(bars["close"], errors="coerce")
    ratio = close / close.shift(1)
    ratio = ratio.dropna()
    if ratio.empty:
        return []

    candidates = _candidate_ratios()
    events: List[SplitEvent] = []
    for dt, r in ratio.items():
        if not np.isfinite(r) or r <= 0:
            continue
        if jump_threshold_low < r < jump_threshold_high:
            continue
        nearest = min(candidates, key=lambda c: abs(c - float(r)))
        rel_err = abs(float(r) - nearest) / max(abs(nearest), 1e-9)
        if rel_err <= tolerance:
            events.append(SplitEvent(date=pd.Timestamp(dt), ratio=float(nearest), source="auto"))
    return events


def known_split_events(symbol: str, market: str) -> List[SplitEvent]:
    key = (market.upper(), symbol.upper())
    rows = KNOWN_SPLITS.get(key, ())
    return [SplitEvent(date=pd.Timestamp(d, tz="UTC"), ratio=float(r), source="known") for d, r in rows]


def _merge_events(events: Iterable[SplitEvent]) -> List[SplitEvent]:
    # If same date has multiple sources, keep known over auto.
    by_date: Dict[pd.Timestamp, SplitEvent] = {}
    for e in sorted(events, key=lambda x: (x.date, x.source)):
        old = by_date.get(e.date)
        if old is None:
            by_date[e.date] = e
            continue
        if old.source != "known" and e.source == "known":
            by_date[e.date] = e
    return sorted(by_date.values(), key=lambda x: x.date)


def apply_split_adjustment(
    bars: pd.DataFrame,
    symbol: str,
    market: str,
    use_known: bool = True,
    use_auto_detect: bool = True,
) -> Tuple[pd.DataFrame, List[SplitEvent]]:
    if bars is None or bars.empty:
        return bars.copy(), []

    events: List[SplitEvent] = []
    if use_known:
        events.extend(known_split_events(symbol=symbol, market=market))
    if use_auto_detect:
        events.extend(detect_split_events(bars))
    events = _merge_events(events)
    if not events:
        return bars.copy(), []

    out = bars.copy()
    factor = pd.Series(1.0, index=out.index, dtype=float)
    for ev in events:
        factor.loc[out.index < ev.date] *= ev.ratio

    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * factor

    if "volume" in out.columns:
        safe_factor = factor.replace(0, np.nan)
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce") / safe_factor

    return out, events
