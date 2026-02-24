from __future__ import annotations

from datetime import datetime, timezone

from data_sources import fetch_stooq_ohlcv, fetch_stooq_quote
from market_data_types import OhlcvSnapshot, QuoteSnapshot
from providers.base import MarketDataProvider, ProviderError, ProviderErrorKind, ProviderRequest


class UsStooqProvider(MarketDataProvider):
    name = "stooq"

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        try:
            q = fetch_stooq_quote(request.symbol)
        except Exception as exc:
            raise ProviderError(
                self.name, ProviderErrorKind.NETWORK, "Stooq quote request failed", exc
            ) from exc

        return QuoteSnapshot(
            symbol=request.symbol,
            market=request.market,
            ts=q.ts,
            price=q.last,
            prev_close=q.prev_close,
            open=q.open,
            high=q.high,
            low=q.low,
            volume=q.volume,
            source=self.name,
            is_delayed=True,
            interval="1d",
            currency=q.currency,
            exchange=q.exchange,
        )

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        if request.interval != "1d":
            raise ProviderError(
                self.name, ProviderErrorKind.UNSUPPORTED, "Stooq only supports 1d interval"
            )
        try:
            df = fetch_stooq_ohlcv(request.symbol, interval="1d")
        except Exception as exc:
            raise ProviderError(
                self.name, ProviderErrorKind.NETWORK, "Stooq daily request failed", exc
            ) from exc
        if df.empty:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "Stooq returned empty OHLCV")

        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval="1d",
            tz="UTC",
            df=df,
            source=self.name,
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )
