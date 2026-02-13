from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
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

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return (symbol or "").strip().upper()

    def _try_quote_chain_with_symbols(
        self,
        market: str,
        candidates: list[tuple[object, str]],
    ) -> Tuple[QuoteSnapshot, List[str], Optional[str], int]:
        errors: List[str] = []
        chain_names = [provider.name for provider, _ in candidates]
        for idx, (provider, symbol) in enumerate(candidates):
            try:
                req = ProviderRequest(symbol=symbol, market=market, interval="quote")
                quote = provider.quote(req)
                return quote, chain_names, errors[-1] if errors else None, idx
            except ProviderError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover
                errors.append(f"[{provider.name}] {exc}")
        raise RuntimeError("; ".join(errors) if errors else "no provider available")

    def _try_ohlcv_chain_with_symbols(
        self,
        market: str,
        interval: str,
        candidates: list[tuple[object, str]],
    ) -> OhlcvSnapshot:
        errors: List[str] = []
        for provider, symbol in candidates:
            try:
                req = ProviderRequest(symbol=symbol, market=market, interval=interval)
                return provider.ohlcv(req)
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
        symbol = self._normalize_symbol(symbol)
        yahoo_symbol = self._normalize_symbol(yahoo_symbol) or symbol
        cache_key = f"us-live:{symbol}:{yahoo_symbol}:{int(options.use_yahoo)}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, LiveContext):
            return cached
        last_good_key = f"us-live:last-good:{symbol}:{yahoo_symbol}"

        quote_candidates: list[tuple[object, str]] = []
        if getattr(self.us_twelve, "api_key", None):
            quote_candidates.append((self.us_twelve, symbol))
        quote_candidates.append((self.yahoo, yahoo_symbol))
        quote_candidates.append((self.us_stooq, symbol))
        try:
            quote, chain, reason, depth = self._try_quote_chain_with_symbols("US", quote_candidates)
        except Exception as exc:
            last_good = self.cache.get(last_good_key)
            if isinstance(last_good, LiveContext):
                return last_good
            raise RuntimeError(
                f"{exc}; 請確認美股代碼是否正確（例如 AAPL/TSLA）或稍後再試"
            ) from exc

        daily_candidates: list[tuple[object, str]] = []
        if getattr(self.us_twelve, "api_key", None):
            daily_candidates.append((self.us_twelve, symbol))
        daily_candidates.append((self.yahoo, yahoo_symbol))
        daily_candidates.append((self.us_stooq, symbol))

        daily = pd.DataFrame()
        try:
            daily = self._try_ohlcv_chain_with_symbols("US", "1d", daily_candidates).df
        except Exception:
            last_good = self.cache.get(last_good_key)
            if isinstance(last_good, LiveContext) and isinstance(last_good.daily, pd.DataFrame):
                daily = last_good.daily.copy()

        intraday = pd.DataFrame()
        if options.use_yahoo:
            try:
                intraday_req = ProviderRequest(symbol=yahoo_symbol, market="US", interval="1m")
                intraday = self.yahoo.ohlcv(intraday_req).df
            except Exception:
                intraday = pd.DataFrame()
        if intraday.empty:
            if not daily.empty:
                intraday = daily.tail(260).copy()
            else:
                last_good = self.cache.get(last_good_key)
                if isinstance(last_good, LiveContext) and isinstance(last_good.intraday, pd.DataFrame):
                    intraday = last_good.intraday.copy()

        if daily.empty and not intraday.empty:
            tmp = intraday.copy()
            tmp = tmp.sort_index()
            daily = (
                tmp.resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna(subset=["open", "high", "low", "close"], how="any")
            )

        if daily.empty and quote.price is not None:
            ts = pd.Timestamp(quote.ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            daily = pd.DataFrame(
                [
                    {
                        "open": float(quote.open if quote.open is not None else quote.price),
                        "high": float(quote.high if quote.high is not None else quote.price),
                        "low": float(quote.low if quote.low is not None else quote.price),
                        "close": float(quote.price),
                        "volume": float(quote.volume or 0),
                    }
                ],
                index=pd.DatetimeIndex([ts]),
            )

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
        self.cache.set(last_good_key, ctx, ttl_sec=1800)
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

    def get_00935_constituents(self, limit: int = 60) -> tuple[list[str], str]:
        import yfinance as yf

        cache_key = f"universe:TW:00935:{int(limit)}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            syms, source = cached
            if isinstance(syms, list) and isinstance(source, str):
                return syms, source

        fallback_symbols = [
            "2330",
            "2454",
            "2317",
            "2308",
            "2382",
            "3711",
            "3034",
            "2379",
            "2301",
            "6669",
            "3231",
            "6415",
            "3443",
            "3017",
            "6531",
            "3661",
            "3037",
            "4919",
            "2327",
            "2408",
            "3008",
            "3533",
            "5269",
            "5274",
            "2376",
            "2439",
            "2377",
            "2357",
            "2360",
            "2345",
        ]

        symbols: list[str] = []
        source = "fallback_manual"
        try:
            ticker = yf.Ticker("00935.TW")
            tables: list[pd.DataFrame] = []
            try:
                top = ticker.funds_data.top_holdings
                if isinstance(top, pd.DataFrame) and not top.empty:
                    tables.append(top)
            except Exception:
                pass
            try:
                eq = ticker.funds_data.equity_holdings
                if isinstance(eq, pd.DataFrame) and not eq.empty:
                    tables.append(eq)
            except Exception:
                pass

            found: list[str] = []
            for table in tables:
                raw_tokens: list[str] = []
                if "Symbol" in table.columns:
                    raw_tokens = [str(x) for x in table["Symbol"].tolist()]
                else:
                    raw_tokens = [str(x) for x in table.index.tolist()]

                for token in raw_tokens:
                    m = re.search(r"(\d{4})", token)
                    if not m:
                        continue
                    code = m.group(1)
                    if code not in found:
                        found.append(code)
            if found:
                symbols = found
                source = "yfinance_funds_data"
        except Exception:
            symbols = []

        if not symbols:
            symbols = fallback_symbols
            source = "fallback_manual"

        symbols = symbols[: max(1, int(limit))]
        self.cache.set(cache_key, (symbols, source), ttl_sec=3600)
        return symbols, source
