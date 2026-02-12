from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol

from market_data_types import Market, OhlcvSnapshot, QuoteSnapshot


class ProviderErrorKind(str, Enum):
    AUTH = "AUTH"
    RATE_LIMIT = "RATE_LIMIT"
    NETWORK = "NETWORK"
    EMPTY = "EMPTY"
    PARSE = "PARSE"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ProviderError(RuntimeError):
    source: str
    kind: ProviderErrorKind
    message: str
    cause: Optional[Exception] = None

    def __str__(self) -> str:
        return f"[{self.source}:{self.kind}] {self.message}"


@dataclass(frozen=True)
class ProviderRequest:
    symbol: str
    market: Market
    interval: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    exchange: Optional[str] = None


class MarketDataProvider(Protocol):
    name: str

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        raise NotImplementedError

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        raise NotImplementedError
