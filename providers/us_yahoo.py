from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from data_sources import TAIPEI_TZ, fetch_yf_fundamentals, fetch_yf_last_close, fetch_yf_ohlcv
from market_data_types import OhlcvSnapshot, QuoteSnapshot
from providers.base import MarketDataProvider, ProviderError, ProviderErrorKind, ProviderRequest


class UsYahooProvider(MarketDataProvider):
    name = "yahoo"

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        try:
            bars = fetch_yf_ohlcv(request.symbol, period="1d", interval="1m")
        except Exception as exc:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, "Yahoo quote request failed", exc) from exc

        if bars.empty:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "Yahoo returned empty quote bars")

        bars = bars.sort_index()
        last_bar = bars.iloc[-1]
        last_ts = pd.Timestamp(bars.index[-1]).to_pydatetime()
        try:
            _, pct = fetch_yf_last_close(request.symbol)
            if pct is None:
                prev_close: Optional[float] = None
            else:
                close_guess = float(last_bar["close"])
                prev_close = close_guess / (1.0 + pct / 100.0)
        except Exception:
            prev_close = None

        return QuoteSnapshot(
            symbol=request.symbol,
            market=request.market,
            ts=last_ts,
            price=float(last_bar["close"]),
            prev_close=prev_close,
            open=float(bars["open"].iloc[0]),
            high=float(bars["high"].max()),
            low=float(bars["low"].min()),
            volume=int(float(bars.get("volume", pd.Series(index=bars.index)).fillna(0).sum())),
            source=self.name,
            is_delayed=True,
            interval="1m",
            currency="USD",
            exchange="YAHOO",
        )

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        period = "10y" if request.interval == "1d" else "1d"
        interval = request.interval
        try:
            df = fetch_yf_ohlcv(request.symbol, period=period, interval=interval)
        except Exception as exc:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, "Yahoo OHLCV request failed", exc) from exc
        if df.empty:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "Yahoo returned empty OHLCV")

        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval=interval,
            tz=str(TAIPEI_TZ),
            df=df,
            source=self.name,
            is_delayed=True,
            fetched_at=datetime.now(tz=TAIPEI_TZ),
        )

    def fundamentals(self, symbol: str):
        try:
            return fetch_yf_fundamentals(symbol)
        except Exception:
            return None
