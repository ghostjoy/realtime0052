from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_sources import TAIPEI_TZ
from market_data_types import DataQuality, LiveContext, OhlcvSnapshot, QuoteSnapshot
from providers import TwMisProvider, TwOpenApiProvider, UsStooqProvider, UsTwelveDataProvider, UsYahooProvider
from providers.base import ProviderError, ProviderRequest


@dataclass(frozen=True)
class LiveOptions:
    use_yahoo: bool
    keep_minutes: int
    exchange: str = "tse"


@dataclass
class _CacheEntry:
    expires_at: datetime
    value: object


class _TtlCache:
    def __init__(self):
        self._store: Dict[str, _CacheEntry] = {}

    def get(self, key: str):
        entry = self._store.get(key)
        if not entry:
            return None
        if entry.expires_at <= datetime.now(tz=timezone.utc):
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: object, ttl_sec: int):
        self._store[key] = _CacheEntry(expires_at=datetime.now(tz=timezone.utc) + timedelta(seconds=ttl_sec), value=value)


class MarketDataService:
    def __init__(self):
        self.yahoo = UsYahooProvider()
        self.us_twelve = UsTwelveDataProvider()
        self.us_stooq = UsStooqProvider()
        self.tw_mis = TwMisProvider()
        self.tw_openapi = TwOpenApiProvider()
        self.cache = _TtlCache()

    @staticmethod
    def _quality(quote: QuoteSnapshot, fallback_depth: int, reason: Optional[str]) -> DataQuality:
        freshness = int((datetime.now(tz=quote.ts.tzinfo or timezone.utc) - quote.ts).total_seconds())
        return DataQuality(
            freshness_sec=max(freshness, 0),
            degraded=fallback_depth > 0 or quote.is_delayed,
            fallback_depth=fallback_depth,
            reason=reason,
        )

    def _try_quote_chain(self, providers, request: ProviderRequest) -> Tuple[QuoteSnapshot, List[str], Optional[str], int]:
        errors: List[str] = []
        for idx, provider in enumerate(providers):
            try:
                quote = provider.quote(request)
                return quote, [p.name for p in providers], errors[-1] if errors else None, idx
            except ProviderError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover
                errors.append(f"[{provider.name}] {exc}")
        raise RuntimeError("; ".join(errors) if errors else "no provider available")

    def _try_ohlcv_chain(self, providers, request: ProviderRequest) -> OhlcvSnapshot:
        errors: List[str] = []
        for provider in providers:
            try:
                return provider.ohlcv(request)
            except ProviderError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover
                errors.append(f"[{provider.name}] {exc}")
        raise RuntimeError("; ".join(errors) if errors else "no provider available")

    def get_us_live_context(
        self,
        symbol: str,
        yahoo_symbol: str,
        options: LiveOptions,
    ) -> LiveContext:
        cache_key = f"us-live:{symbol}:{yahoo_symbol}:{int(options.use_yahoo)}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, LiveContext):
            return cached

        quote_req = ProviderRequest(symbol=symbol, market="US", interval="quote")
        quote, chain, reason, depth = self._try_quote_chain([self.us_twelve, self.yahoo, self.us_stooq], quote_req)

        daily_req = ProviderRequest(symbol=symbol, market="US", interval="1d")
        daily = self._try_ohlcv_chain([self.us_twelve, self.yahoo, self.us_stooq], daily_req).df

        intraday = pd.DataFrame()
        if options.use_yahoo:
            try:
                intraday_req = ProviderRequest(symbol=yahoo_symbol, market="US", interval="1m")
                intraday = self.yahoo.ohlcv(intraday_req).df
            except Exception:
                intraday = pd.DataFrame()
        if intraday.empty:
            intraday = daily.tail(260).copy()

        ctx = LiveContext(
            quote=quote,
            intraday=intraday,
            daily=daily,
            quality=self._quality(quote, depth, reason),
            source_chain=chain,
            used_fallback=depth > 0,
            fundamentals=self.yahoo.fundamentals(yahoo_symbol),
        )
        self.cache.set(cache_key, ctx, ttl_sec=10)
        return ctx

    def get_tw_live_context(
        self,
        symbol: str,
        yahoo_symbol: str,
        ticks: pd.DataFrame,
        options: LiveOptions,
    ) -> Tuple[LiveContext, pd.DataFrame]:
        quote_req = ProviderRequest(symbol=symbol, market="TW", interval="quote", exchange=options.exchange)
        quote, chain, reason, depth = self._try_quote_chain([self.tw_mis, self.tw_openapi], quote_req)

        if quote.price is not None:
            new_tick = pd.DataFrame(
                [{"ts": quote.ts, "price": float(quote.price), "cum_volume": float(quote.volume or 0)}]
            )
            ticks = pd.concat([ticks, new_tick], ignore_index=True)
            ticks = ticks.drop_duplicates(subset=["ts", "price", "cum_volume"], keep="last")
            cutoff = TwMisProvider.now() - pd.Timedelta(minutes=options.keep_minutes)
            ticks = ticks[ticks["ts"] >= cutoff].copy()

        bars_rt = TwMisProvider.build_bars_from_ticks(ticks)
        intraday = bars_rt.copy()
        daily = pd.DataFrame()

        if options.use_yahoo:
            try:
                intraday_req = ProviderRequest(symbol=yahoo_symbol, market="TW", interval="1m")
                intraday = self.yahoo.ohlcv(intraday_req).df
            except Exception:
                pass

        try:
            daily_req = ProviderRequest(symbol=symbol, market="TW", interval="1d")
            daily = self.tw_openapi.ohlcv(daily_req).df
        except Exception:
            if options.use_yahoo:
                try:
                    daily = self.yahoo.ohlcv(ProviderRequest(symbol=yahoo_symbol, market="TW", interval="1d")).df
                except Exception:
                    daily = pd.DataFrame()

        ctx = LiveContext(
            quote=quote,
            intraday=intraday,
            daily=daily,
            quality=self._quality(quote, depth, reason),
            source_chain=chain,
            used_fallback=depth > 0,
            fundamentals=None,
        )
        return ctx, ticks

    def get_reference_context(self, market: str) -> pd.DataFrame:
        import yfinance as yf

        symbols = (
            ["^TWII", "USDTWD=X", "^IXIC", "^VIX"]
            if market == "TW"
            else ["^GSPC", "^IXIC", "^DJI", "^VIX", "DX-Y.NYB", "^TNX"]
        )
        labels = (
            {"^TWII": "加權指數", "USDTWD=X": "美元/台幣", "^IXIC": "NASDAQ", "^VIX": "VIX"}
            if market == "TW"
            else {
                "^GSPC": "S&P 500",
                "^IXIC": "NASDAQ",
                "^DJI": "Dow",
                "^VIX": "VIX",
                "DX-Y.NYB": "美元指數(DXY)",
                "^TNX": "美10年期殖利率(TNX)",
            }
        )
        cache_key = f"ref:{market}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

        rows = []
        try:
            data = yf.download(
                tickers=symbols,
                period="5d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            return pd.DataFrame(columns=["標的", "代碼", "最新收盤", "日變動%"])

        for sym in symbols:
            try:
                frame = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
                closes = frame["Close"].dropna()
                if closes.empty:
                    continue
                last = float(closes.iloc[-1])
                prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
                pct = None if prev in (None, 0) else (last - prev) / prev * 100.0
                rows.append(
                    {
                        "標的": labels.get(sym, sym),
                        "代碼": sym,
                        "最新收盤": round(last, 4),
                        "日變動%": None if pct is None else round(pct, 2),
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(rows)
        self.cache.set(cache_key, df, ttl_sec=90)
        return df

    def get_benchmark_series(
        self,
        market: str,
        start: datetime,
        end: datetime,
        benchmark: str = "auto",
    ) -> pd.DataFrame:
        import yfinance as yf

        def _normalize_close(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame(columns=["close"])
            out = df.copy()
            if isinstance(out.columns, pd.MultiIndex):
                out.columns = [c[0].lower() for c in out.columns]
            else:
                out.columns = [str(c).lower() for c in out.columns]
            if "close" not in out.columns:
                return pd.DataFrame(columns=["close"])
            close_df = out[["close"]].copy()
            close_df.index = pd.to_datetime(close_df.index, utc=True, errors="coerce")
            close_df = close_df.dropna(subset=["close"])
            return close_df

        def _fetch_yf(symbol: str) -> pd.DataFrame:
            try:
                df = yf.download(
                    tickers=symbol,
                    start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            except Exception:
                return pd.DataFrame(columns=["close"])
            out = _normalize_close(df)
            if not out.empty:
                out = out[(out.index >= start) & (out.index <= end)]
                out.attrs["symbol"] = symbol
                out.attrs["source"] = "yfinance"
            return out

        def _fetch_tw_proxy(symbol: str) -> pd.DataFrame:
            try:
                snap = self.tw_openapi.ohlcv(
                    ProviderRequest(symbol=symbol, market="TW", interval="1d", start=start, end=end)
                )
            except Exception:
                return pd.DataFrame(columns=["close"])
            out = _normalize_close(snap.df)
            if not out.empty:
                out = out[(out.index >= start) & (out.index <= end)]
                out.attrs["symbol"] = symbol
                out.attrs["source"] = snap.source
            return out

        def _fetch_us_proxy(symbol: str) -> pd.DataFrame:
            try:
                snap = self.us_stooq.ohlcv(
                    ProviderRequest(symbol=symbol, market="US", interval="1d", start=start, end=end)
                )
            except Exception:
                return pd.DataFrame(columns=["close"])
            out = _normalize_close(snap.df)
            if not out.empty:
                out = out[(out.index >= start) & (out.index <= end)]
                out.attrs["symbol"] = symbol
                out.attrs["source"] = snap.source
            return out

        benchmark = (benchmark or "auto").strip().lower()
        if benchmark == "off":
            return pd.DataFrame(columns=["close"])

        default_symbol = "^TWII" if market == "TW" else "^GSPC"
        cache_key = f"benchmark:{market}:{benchmark}:{start.date().isoformat()}:{end.date().isoformat()}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

        if market == "TW":
            candidates = {
                "auto": [("yf", "^TWII"), ("tw_proxy", "0050"), ("tw_proxy", "006208")],
                "twii": [("yf", "^TWII")],
                "0050": [("tw_proxy", "0050")],
                "006208": [("tw_proxy", "006208")],
            }
        else:
            candidates = {
                "auto": [("yf", "^GSPC"), ("us_proxy", "SPY"), ("us_proxy", "QQQ"), ("us_proxy", "DIA")],
                "gspc": [("yf", "^GSPC")],
                "spy": [("us_proxy", "SPY")],
                "qqq": [("us_proxy", "QQQ")],
                "dia": [("us_proxy", "DIA")],
            }
        chain = candidates.get(benchmark, [("yf", default_symbol)])
        out = pd.DataFrame(columns=["close"])
        for kind, symbol in chain:
            if kind == "yf":
                out = _fetch_yf(symbol)
            elif kind == "tw_proxy":
                out = _fetch_tw_proxy(symbol)
            else:
                out = _fetch_us_proxy(symbol)
            if not out.empty:
                break

        if out.empty:
            return pd.DataFrame(columns=["close"])
        self.cache.set(cache_key, out, ttl_sec=600)
        return out
