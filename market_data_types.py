from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd

Market = Literal["US", "TW"]


@dataclass(frozen=True)
class QuoteSnapshot:
    symbol: str
    market: Market
    ts: datetime
    price: float | None
    prev_close: float | None
    open: float | None
    high: float | None
    low: float | None
    volume: int | None
    source: str
    is_delayed: bool
    interval: str = "quote"
    currency: str | None = None
    exchange: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def change(self) -> float | None:
        if self.price is None or self.prev_close is None:
            return None
        return self.price - self.prev_close

    @property
    def change_pct(self) -> float | None:
        if self.change is None or not self.prev_close:
            return None
        return self.change / self.prev_close * 100.0


@dataclass(frozen=True)
class OhlcvSnapshot:
    symbol: str
    market: Market
    interval: str
    tz: str
    df: pd.DataFrame
    source: str
    is_delayed: bool
    fetched_at: datetime


@dataclass(frozen=True)
class DataQuality:
    freshness_sec: int | None
    degraded: bool
    fallback_depth: int
    reason: str | None


@dataclass(frozen=True)
class DataHealth:
    as_of: str | None
    data_sources: list[str]
    source_chain: list[str]
    degraded: bool
    fallback_depth: int
    freshness_sec: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class BenchmarkLoadResult:
    bars: pd.DataFrame
    close: pd.Series
    symbol_used: str
    sync_issues: list[str]
    source_chain: list[str]
    candidates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SyncPlanResult:
    synced_symbols: list[str]
    skipped_symbols: list[str]
    issues: list[str]
    source_chain: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LiveContext:
    quote: QuoteSnapshot
    intraday: pd.DataFrame
    daily: pd.DataFrame
    quality: DataQuality
    source_chain: list[str]
    used_fallback: bool
    fundamentals: dict[str, Any] | None = None
    intraday_source: str | None = None
    daily_source: str | None = None
