from __future__ import annotations

from datetime import datetime

import pandas as pd

from data_sources import TAIPEI_TZ, fetch_twse_quote
from market_data_types import OhlcvSnapshot, QuoteSnapshot
from providers.base import MarketDataProvider, ProviderError, ProviderErrorKind, ProviderRequest


class TwMisProvider(MarketDataProvider):
    name = "tw_mis"

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        exchange = request.exchange or "tse"
        try:
            quote = fetch_twse_quote(request.symbol, exchange=exchange)
        except Exception as exc:
            raise ProviderError(
                self.name, ProviderErrorKind.NETWORK, "TW MIS quote request failed", exc
            ) from exc

        return QuoteSnapshot(
            symbol=request.symbol,
            market=request.market,
            ts=quote.ts,
            price=quote.last,
            prev_close=quote.prev_close,
            open=quote.open,
            high=quote.high,
            low=quote.low,
            volume=quote.volume,
            source=self.name,
            is_delayed=False,
            interval="tick",
            currency="TWD",
            exchange=exchange.upper(),
            extra={
                "name": quote.name,
                "full_name": quote.full_name,
                "bid_prices": quote.bid_prices,
                "bid_sizes": quote.bid_sizes,
                "ask_prices": quote.ask_prices,
                "ask_sizes": quote.ask_sizes,
            },
        )

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        raise ProviderError(
            self.name,
            ProviderErrorKind.UNSUPPORTED,
            "TW MIS provider does not provide historical OHLCV",
        )

    @staticmethod
    def build_bars_from_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
        if ticks.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ticks = ticks.sort_values("ts").copy()
        ticks["delta_volume"] = ticks["cum_volume"].diff().clip(lower=0).fillna(0)
        grouped = ticks.set_index("ts")
        ohlc = grouped["price"].resample("1min").ohlc()
        vol = grouped["delta_volume"].resample("1min").sum()
        out = ohlc.copy()
        out["volume"] = vol
        out = out.dropna(subset=["open", "high", "low", "close"], how="any")
        if out.index.tz is None:
            out.index = out.index.tz_localize(TAIPEI_TZ)
        return out

    @staticmethod
    def now():
        return datetime.now(tz=TAIPEI_TZ)
