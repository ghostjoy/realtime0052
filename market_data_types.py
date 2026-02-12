from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

import pandas as pd

Market = Literal["US", "TW"]


@dataclass(frozen=True)
class QuoteSnapshot:
    symbol: str
    market: Market
    ts: datetime
    price: Optional[float]
    prev_close: Optional[float]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    volume: Optional[int]
    source: str
    is_delayed: bool
    interval: str = "quote"
    currency: Optional[str] = None
    exchange: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def change(self) -> Optional[float]:
        if self.price is None or self.prev_close is None:
            return None
        return self.price - self.prev_close

    @property
    def change_pct(self) -> Optional[float]:
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
    freshness_sec: Optional[int]
    degraded: bool
    fallback_depth: int
    reason: Optional[str]


@dataclass(frozen=True)
class LiveContext:
    quote: QuoteSnapshot
    intraday: pd.DataFrame
    daily: pd.DataFrame
    quality: DataQuality
    source_chain: list[str]
    used_fallback: bool
    fundamentals: Optional[Dict[str, Any]] = None
