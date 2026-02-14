from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import html as html_lib
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_sources import TAIPEI_TZ
from market_data_types import DataQuality, LiveContext, OhlcvSnapshot, QuoteSnapshot
from providers import (
    TwFugleHistoricalProvider,
    TwFugleWebSocketProvider,
    TwMisProvider,
    TwOpenApiProvider,
    TwTpexOpenApiProvider,
    UsStooqProvider,
    UsTwelveDataProvider,
    UsYahooProvider,
)
from providers.base import ProviderError, ProviderRequest


@dataclass(frozen=True)
class LiveOptions:
    use_yahoo: bool
    keep_minutes: int
    exchange: str = "tse"
    use_fugle_ws: bool = True


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
        self.tw_fugle_rest = TwFugleHistoricalProvider()
        self.tw_fugle_ws = TwFugleWebSocketProvider()
        self.tw_mis = TwMisProvider()
        self.tw_openapi = TwOpenApiProvider()
        self.tw_tpex = TwTpexOpenApiProvider()
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
            provider_request = request
            if request.market == "TW" and getattr(provider, "name", "") == "yahoo":
                yahoo_symbol = self._normalize_tw_yahoo_symbol(request.symbol)
                if yahoo_symbol and yahoo_symbol != request.symbol:
                    provider_request = ProviderRequest(
                        symbol=yahoo_symbol,
                        market=request.market,
                        interval=request.interval,
                        start=request.start,
                        end=request.end,
                        exchange=request.exchange,
                    )
            try:
                return provider.ohlcv(provider_request)
            except ProviderError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover
                errors.append(f"[{provider.name}] {exc}")
        raise RuntimeError("; ".join(errors) if errors else "no provider available")

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return (symbol or "").strip().upper()

    @staticmethod
    def _normalize_tw_yahoo_symbol(symbol: str) -> str:
        text = (symbol or "").strip().upper()
        if not text or "." in text:
            return text
        # TW individual stocks/ETFs are typically 4~6 digit codes on Yahoo.
        if re.fullmatch(r"\d{4,6}", text):
            return f"{text}.TW"
        return text

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
        daily_source: Optional[str] = None
        try:
            daily_snap = self._try_ohlcv_chain_with_symbols("US", "1d", daily_candidates)
            daily = daily_snap.df
            daily_source = str(daily_snap.source or "") or None
        except Exception:
            last_good = self.cache.get(last_good_key)
            if isinstance(last_good, LiveContext) and isinstance(last_good.daily, pd.DataFrame):
                daily = last_good.daily.copy()
                daily_source = str(getattr(last_good, "daily_source", "") or "cache:last_good")

        intraday = pd.DataFrame()
        intraday_source: Optional[str] = None
        if options.use_yahoo:
            try:
                intraday_req = ProviderRequest(symbol=yahoo_symbol, market="US", interval="1m")
                intraday_snap = self.yahoo.ohlcv(intraday_req)
                intraday = intraday_snap.df
                if not intraday.empty:
                    intraday_source = str(intraday_snap.source or "") or "yahoo"
            except Exception:
                intraday = pd.DataFrame()
        if intraday.empty:
            if not daily.empty:
                intraday = daily.tail(260).copy()
                intraday_source = f"{daily_source or 'daily'}_tail"
            else:
                last_good = self.cache.get(last_good_key)
                if isinstance(last_good, LiveContext) and isinstance(last_good.intraday, pd.DataFrame):
                    intraday = last_good.intraday.copy()
                    intraday_source = str(getattr(last_good, "intraday_source", "") or "cache:last_good")

        if daily.empty and not intraday.empty:
            tmp = intraday.copy()
            tmp = tmp.sort_index()
            daily = (
                tmp.resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna(subset=["open", "high", "low", "close"], how="any")
            )
            daily_source = f"{intraday_source or 'intraday'}_resampled"

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
            daily_source = f"{quote.source}_quote_derived"

        ctx = LiveContext(
            quote=quote,
            intraday=intraday,
            daily=daily,
            quality=self._quality(quote, depth, reason),
            source_chain=chain,
            used_fallback=depth > 0,
            fundamentals=self.yahoo.fundamentals(yahoo_symbol),
            intraday_source=intraday_source,
            daily_source=daily_source,
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
        intraday_cache_key = f"tw-live:intraday1m:last-good:{symbol}:{yahoo_symbol}"

        quote_providers: list[object] = []
        if options.use_fugle_ws and getattr(self.tw_fugle_ws, "api_key", None):
            quote_providers.append(self.tw_fugle_ws)
        quote_providers.extend([self.tw_mis, self.tw_openapi, self.tw_tpex])

        quote_req = ProviderRequest(symbol=symbol, market="TW", interval="quote", exchange=options.exchange)
        quote, chain, reason, depth = self._try_quote_chain(quote_providers, quote_req)

        if quote.price is not None:
            tick_ts = pd.Timestamp(quote.ts)
            if tick_ts.tzinfo is None:
                tick_ts = tick_ts.tz_localize("UTC")
            else:
                tick_ts = tick_ts.tz_convert("UTC")
            now_ts = pd.Timestamp(datetime.now(tz=timezone.utc))
            quote_source = str(getattr(quote, "source", "") or "").strip().lower()
            # Daily snapshot sources may carry stale trading-date timestamps.
            # Use "now" as tick ts to keep intraday chart visible.
            if quote_source in {"tw_openapi", "tw_tpex"}:
                tick_ts = now_ts
            elif (now_ts - tick_ts) > pd.Timedelta(days=1):
                tick_ts = now_ts

            new_tick = pd.DataFrame(
                [{"ts": tick_ts, "price": float(quote.price), "cum_volume": float(quote.volume or 0)}]
            )
            if ticks.empty:
                ticks = new_tick.copy()
            else:
                ticks = pd.concat([ticks, new_tick], ignore_index=True)
            ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True, errors="coerce")
            ticks = ticks.dropna(subset=["ts"])
            ticks = ticks.drop_duplicates(subset=["ts", "price", "cum_volume"], keep="last")
            cutoff = pd.Timestamp(TwMisProvider.now() - pd.Timedelta(minutes=options.keep_minutes)).tz_convert("UTC")
            ticks = ticks[ticks["ts"] >= cutoff].copy()

        bars_rt = TwMisProvider.build_bars_from_ticks(ticks)
        intraday = bars_rt.copy()
        intraday_source: Optional[str] = f"{quote.source}_ticks" if not bars_rt.empty else None
        daily = pd.DataFrame()
        daily_source: Optional[str] = None

        if options.use_yahoo:
            try:
                intraday_req = ProviderRequest(symbol=yahoo_symbol, market="TW", interval="1m")
                intraday_snap = self.yahoo.ohlcv(intraday_req)
                yahoo_intraday = intraday_snap.df
                if not yahoo_intraday.empty:
                    intraday = yahoo_intraday
                    intraday_source = str(intraday_snap.source or "") or "yahoo"
                    self.cache.set(
                        intraday_cache_key,
                        {"df": intraday.copy(), "source": intraday_source},
                        ttl_sec=1800,
                    )
            except Exception:
                pass

        # If tick aggregation is too sparse, prefer Fugle historical 1m candles for chart readability.
        # This helps cases where only one latest trade tick is available in the current refresh cycle.
        min_intraday_bars = max(6, int(max(1, options.keep_minutes) // 30))
        need_fugle_intraday = (
            getattr(self.tw_fugle_rest, "api_key", None) is not None
            and (intraday.empty or len(intraday) < min_intraday_bars)
        )
        if need_fugle_intraday:
            try:
                now_utc = datetime.now(tz=timezone.utc)
                tw_midnight = datetime.now(tz=TAIPEI_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
                day_start_utc = tw_midnight.astimezone(timezone.utc)
                rolling_start = now_utc - pd.Timedelta(minutes=max(30, int(options.keep_minutes)))
                intraday_start = min(rolling_start, day_start_utc)
                intraday_req = ProviderRequest(
                    symbol=symbol,
                    market="TW",
                    interval="1m",
                    start=intraday_start,
                    end=now_utc,
                )
                fugle_intraday_snap = self.tw_fugle_rest.ohlcv(intraday_req)
                fugle_intraday = fugle_intraday_snap.df
                if not fugle_intraday.empty:
                    intraday = fugle_intraday
                    intraday_source = str(fugle_intraday_snap.source or "") or "tw_fugle_rest"
                    self.cache.set(
                        intraday_cache_key,
                        {"df": intraday.copy(), "source": intraday_source},
                        ttl_sec=1800,
                    )
            except Exception:
                pass

        if intraday.empty or len(intraday) < min_intraday_bars:
            cached_intraday = self.cache.get(intraday_cache_key)
            if isinstance(cached_intraday, dict):
                cached_df = cached_intraday.get("df")
                if isinstance(cached_df, pd.DataFrame) and not cached_df.empty and len(cached_df) > len(intraday):
                    intraday = cached_df.copy()
                    cached_source = str(cached_intraday.get("source") or "1m")
                    intraday_source = f"cache:last_good:{cached_source}"

        if str(options.exchange or "tse").strip().lower() == "otc":
            daily_providers = [self.tw_tpex, self.tw_openapi]
        else:
            daily_providers = [self.tw_openapi, self.tw_tpex]
        if getattr(self.tw_fugle_rest, "api_key", None):
            daily_providers = [self.tw_fugle_rest, *daily_providers]

        try:
            daily_req = ProviderRequest(symbol=symbol, market="TW", interval="1d")
            daily_snap = self._try_ohlcv_chain(daily_providers, daily_req)
            daily = daily_snap.df
            daily_source = str(daily_snap.source or "") or None
        except Exception:
            if options.use_yahoo:
                try:
                    daily_snap = self.yahoo.ohlcv(ProviderRequest(symbol=yahoo_symbol, market="TW", interval="1d"))
                    daily = daily_snap.df
                    daily_source = str(daily_snap.source or "") or "yahoo"
                except Exception:
                    daily = pd.DataFrame()

        # Stable fallback: if real intraday remains sparse, use daily tail bars for chart continuity.
        if (intraday.empty or len(intraday) < min_intraday_bars) and not daily.empty:
            intraday = daily.tail(260).copy()
            intraday_source = f"{daily_source or 'daily'}_tail"

        ctx = LiveContext(
            quote=quote,
            intraday=intraday,
            daily=daily,
            quality=self._quality(quote, depth, reason),
            source_chain=chain,
            used_fallback=depth > 0,
            fundamentals=None,
            intraday_source=intraday_source,
            daily_source=daily_source,
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

    def get_tw_symbol_names(self, symbols: list[str]) -> dict[str, str]:
        import requests

        normalized = []
        for symbol in symbols:
            text = str(symbol or "").strip().upper()
            if not text:
                continue
            if text not in normalized:
                normalized.append(text)
        if not normalized:
            return {}

        cache_key = f"tw_names:{','.join(sorted(normalized))}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            return {str(k): str(v) for k, v in cached.items()}

        out = {symbol: symbol for symbol in normalized}
        wanted = set(normalized)

        twse_ok = False
        twse_url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        try:
            resp = requests.get(twse_url, timeout=12)
            resp.raise_for_status()
            rows = resp.json()
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    code = str(row.get("Code", "")).strip().upper()
                    if code not in wanted:
                        continue
                    name = str(row.get("Name", "")).strip()
                    if name:
                        out[code] = name
                twse_ok = True
        except Exception:
            twse_ok = False

        # TPEx 上櫃名稱補齊（例如 6510 精測）
        tpex_ok = False
        tpex_url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
        try:
            resp = requests.get(tpex_url, timeout=12)
            resp.raise_for_status()
            rows = resp.json()
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    code = str(row.get("SecuritiesCompanyCode", "")).strip().upper()
                    if code not in wanted:
                        continue
                    name = str(row.get("CompanyName", "")).strip()
                    if name:
                        out[code] = name
                tpex_ok = True
        except Exception:
            tpex_ok = False

        ttl = 21600 if (twse_ok or tpex_ok) else 900
        self.cache.set(cache_key, out, ttl_sec=ttl)
        return out

    def get_tw_symbol_industries(self, symbols: list[str]) -> dict[str, str]:
        import requests

        normalized: list[str] = []
        for symbol in symbols:
            text = str(symbol or "").strip().upper()
            if not text:
                continue
            if text not in normalized:
                normalized.append(text)
        if not normalized:
            return {}

        cache_key = f"tw_industries:{','.join(sorted(normalized))}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            return {str(k): str(v) for k, v in cached.items()}

        wanted = set(normalized)
        out = {symbol: "" for symbol in normalized}
        twse_ok = False
        tpex_ok = False

        # TWSE listed company profile (contains industry code)
        twse_url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        try:
            resp = requests.get(twse_url, timeout=12)
            resp.raise_for_status()
            rows = resp.json()
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    code = str(row.get("公司代號", "")).strip().upper()
                    if code not in wanted:
                        continue
                    industry = str(row.get("產業別", "")).strip()
                    if industry:
                        out[code] = industry
                twse_ok = True
        except Exception:
            twse_ok = False

        # TPEx OTC company profile (contains industry code)
        tpex_url = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
        try:
            resp = requests.get(tpex_url, timeout=12)
            resp.raise_for_status()
            rows = resp.json()
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    code = str(row.get("SecuritiesCompanyCode", "")).strip().upper()
                    if code not in wanted:
                        continue
                    industry = str(row.get("SecuritiesIndustryCode", "")).strip()
                    if industry:
                        out[code] = industry
                tpex_ok = True
        except Exception:
            tpex_ok = False

        ttl = 21600 if (twse_ok or tpex_ok) else 900
        self.cache.set(cache_key, out, ttl_sec=ttl)
        return out

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

    @staticmethod
    def _dedupe_4digit_codes(values: list[str]) -> list[str]:
        out: list[str] = []
        for value in values:
            token = str(value or "").strip().upper()
            if not re.fullmatch(r"\d{4}", token):
                continue
            if token not in out:
                out.append(token)
        return out

    def _fetch_00935_nomura_constituents(self) -> list[str]:
        import requests

        url_date = "https://www.nomurafunds.com.tw/API/ETFAPI/api/Fund/GetFundTradeInfoDate"
        url_trade = "https://www.nomurafunds.com.tw/API/ETFAPI/api/Fund/GetFundTradeInfo"
        payload_date = {"Type": 0, "Keyword": None, "FundNo": "00935", "Date": None}
        payload_trade = {"Type": 0, "Keyword": None, "FundNo": "00935", "Date": None}
        headers = {"Content-Type": "application/json"}

        date_resp = requests.post(url_date, json=payload_date, headers=headers, timeout=12)
        date_resp.raise_for_status()
        date_data = date_resp.json() if date_resp.content else {}
        entries = date_data.get("Entries") if isinstance(date_data, dict) else {}
        latest_date = ""
        if isinstance(entries, dict):
            latest_date = str(entries.get("LatestDate") or "").strip()
            if not latest_date:
                all_dates = entries.get("AllDate")
                if isinstance(all_dates, list) and all_dates:
                    latest_date = str(all_dates[0]).strip()
        if not latest_date:
            return []

        payload_trade["Date"] = latest_date
        trade_resp = requests.post(url_trade, json=payload_trade, headers=headers, timeout=16)
        trade_resp.raise_for_status()
        trade_data = trade_resp.json() if trade_resp.content else {}
        trade_entries = trade_data.get("Entries") if isinstance(trade_data, dict) else {}
        stocks = trade_entries.get("Stocks") if isinstance(trade_entries, dict) else None
        if not isinstance(stocks, list):
            return []

        codes: list[str] = []
        for row in stocks:
            if not isinstance(row, dict):
                continue
            code = str(row.get("CStockCode") or "").strip().upper()
            codes.append(code)
        return self._dedupe_4digit_codes(codes)

    def _fetch_0050_yuanta_constituents(self) -> list[str]:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        for _ in range(2):
            html = requests.get("https://www.yuantaetfs.com/tradeInfo/pcf/0050", headers=headers, timeout=20).text
            start_marker = '元大台灣卓越50證券投資信託基金"'
            end_marker = '"en",{},"股票實物申贖"'
            start_idx = html.find(start_marker)
            end_idx = html.find(end_marker, start_idx if start_idx >= 0 else 0)
            if start_idx >= 0 and end_idx > start_idx:
                search_text = html[start_idx:end_idx]
            else:
                search_text = html

            pairs = re.findall(r'"(\d{4})","([^"]*[\u4e00-\u9fff][^"]*)","[^"]+"', search_text)
            codes = [str(code).strip().upper() for code, _ in pairs]
            deduped = self._dedupe_4digit_codes(codes)
            if len(deduped) >= 45:
                return deduped[:50]
        return []

    def _fetch_moneydj_basic0007b_html(self, etf_code: str) -> str:
        import requests

        code = str(etf_code or "").strip().upper()
        if not re.fullmatch(r"\d{4,6}", code):
            return ""

        url = f"https://www.moneydj.com/ETF/X/Basic/Basic0007B.xdjhtm?etfid={code}.TW"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        # MoneyDJ occasionally sends UTF-8 content with legacy headers.
        apparent = (resp.apparent_encoding or "").strip()
        if apparent and (resp.encoding or "").lower() in {"", "iso-8859-1", "big5", "cp950"}:
            resp.encoding = apparent
        return resp.text or ""

    def _fetch_moneydj_tw_constituents(self, etf_code: str) -> list[str]:
        code = str(etf_code or "").strip().upper()
        html = self._fetch_moneydj_basic0007b_html(code)
        if not html:
            return []

        tokens = re.findall(r"etfid=(\d{4})\.TW", html, flags=re.IGNORECASE)
        tokens += re.findall(r"\((\d{4})\.TW\)", html, flags=re.IGNORECASE)
        codes = self._dedupe_4digit_codes(tokens)
        return [c for c in codes if c != code]

    @staticmethod
    def _parse_etf_token_market(symbol: str) -> str:
        token = str(symbol or "").strip().upper()
        if "." in token:
            return token.split(".")[-1]
        return ""

    def _fetch_moneydj_full_constituents(self, etf_code: str) -> list[dict[str, object]]:
        code = str(etf_code or "").strip().upper()
        html = self._fetch_moneydj_basic0007b_html(code)
        if not html:
            return []

        pattern = re.compile(
            rf"<td class=['\"]col05['\"]>\s*<a [^>]*?etfid=([^&\"']+)&back={re.escape(code)}\.TW['\"][^>]*>([^<]+)</a>\s*</td>\s*"
            r"<td class=['\"]col06['\"]>([^<]*)</td>\s*"
            r"<td class=['\"]col07['\"]>([^<]*)</td>",
            flags=re.IGNORECASE,
        )
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for idx, m in enumerate(pattern.finditer(html), start=1):
            raw_symbol = html_lib.unescape((m.group(1) or "").strip())
            if not raw_symbol or raw_symbol in seen:
                continue
            seen.add(raw_symbol)

            label = html_lib.unescape((m.group(2) or "").strip())
            name = label
            if label.endswith(f"({raw_symbol})"):
                name = label[: -(len(raw_symbol) + 2)].strip() or raw_symbol
            weight_raw = html_lib.unescape((m.group(3) or "").strip())
            shares_raw = html_lib.unescape((m.group(4) or "").strip())
            try:
                weight_pct = float(weight_raw.replace(",", ""))
            except Exception:
                weight_pct = None
            try:
                shares = int(float(shares_raw.replace(",", "")))
            except Exception:
                shares = None
            market = self._parse_etf_token_market(raw_symbol)
            tw_code = ""
            tw_match = re.fullmatch(r"(\d{4})\.TW", raw_symbol, flags=re.IGNORECASE)
            if tw_match:
                tw_code = tw_match.group(1)
            rows.append(
                {
                    "rank": idx,
                    "symbol": raw_symbol,
                    "name": name,
                    "market": market,
                    "weight_pct": weight_pct,
                    "shares": shares,
                    "tw_code": tw_code,
                }
            )
        return rows

    def get_etf_constituents_full(
        self, etf_code: str, limit: Optional[int] = None, force_refresh: bool = False
    ) -> tuple[list[dict[str, object]], str]:
        code = str(etf_code or "").strip().upper()
        if not code:
            return [], "invalid_symbol"
        limit_token = "all" if limit is None else str(max(1, int(limit)))
        cache_key = f"universe-full:{code}:{limit_token}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if isinstance(cached, tuple) and len(cached) == 2:
                rows, source = cached
                if isinstance(rows, list) and isinstance(source, str):
                    return rows, source

        rows: list[dict[str, object]] = []
        source = "unavailable"
        try:
            rows = self._fetch_moneydj_full_constituents(code)
            if rows:
                source = "moneydj_basic0007b_full"
        except Exception:
            rows = []

        if limit is not None:
            rows = rows[: max(1, int(limit))]
        self.cache.set(cache_key, (rows, source), ttl_sec=3600)
        return rows, source

    @staticmethod
    def get_tw_etf_expected_count(etf_code: str) -> Optional[int]:
        mapping = {
            "0050": 50,
            "00935": 50,
        }
        return mapping.get(str(etf_code or "").strip().upper())

    def get_tw_etf_constituents(self, etf_code: str, limit: Optional[int] = None) -> tuple[list[str], str]:
        import yfinance as yf

        code = str(etf_code or "").strip().upper()
        if not code:
            return [], "invalid_symbol"
        expected_count = self.get_tw_etf_expected_count(code)
        limit_token = "all" if limit is None else str(max(1, int(limit)))
        cache_key = f"universe:TW:{code}:{limit_token}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            syms, source = cached
            if isinstance(syms, list) and isinstance(source, str):
                return syms, source

        fallback_map: dict[str, list[str]] = {
            "00935": [
                "2330",
                "2454",
                "2308",
                "3711",
                "2345",
                "2303",
                "3017",
                "2327",
                "3037",
                "2360",
                "3231",
                "2449",
                "3653",
                "5274",
                "2368",
                "3665",
                "3661",
                "3034",
                "2379",
                "6223",
                "3443",
                "2313",
                "3293",
                "6239",
                "6515",
                "2356",
                "3533",
                "5347",
                "6442",
                "4958",
                "6488",
                "6510",
                "3324",
                "5483",
                "6285",
                "5269",
                "6531",
                "1560",
                "3023",
                "2492",
                "2352",
                "6147",
                "3131",
                "4966",
                "2458",
                "3374",
                "6789",
                "6526",
                "6548",
                "6412",
            ],
            "0050": [
                "1216",
                "1301",
                "1303",
                "2002",
                "2059",
                "2207",
                "2301",
                "2303",
                "2308",
                "2317",
                "2327",
                "2330",
                "2345",
                "2357",
                "2360",
                "2379",
                "2382",
                "2383",
                "2395",
                "2408",
                "2412",
                "2454",
                "2603",
                "2615",
                "2880",
                "2881",
                "2882",
                "2883",
                "2884",
                "2885",
                "2886",
                "2887",
                "2890",
                "2891",
                "2892",
                "2912",
                "3008",
                "3017",
                "3034",
                "3045",
                "3231",
                "3653",
                "3661",
                "3665",
                "3711",
                "4904",
                "5880",
                "6505",
                "6669",
                "6919",
            ],
            "00910": [
                "2455",
                "6271",
                "6285",
            ],
        }
        fallback_symbols = self._dedupe_4digit_codes(fallback_map.get(code, []))

        symbols: list[str] = []
        source = "fallback_manual"

        try:
            if code == "00935":
                symbols = self._fetch_00935_nomura_constituents()
                if symbols:
                    source = "nomura_etfapi"
            elif code == "0050":
                symbols = self._fetch_0050_yuanta_constituents()
                if symbols:
                    source = "yuanta_pcf_html"
            else:
                symbols = self._fetch_moneydj_tw_constituents(code)
                if symbols:
                    source = "moneydj_basic0007b"
        except Exception:
            symbols = []

        if not symbols:
            try:
                ticker = yf.Ticker(f"{code}.TW")
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
                        if m:
                            found.append(m.group(1))
                symbols = self._dedupe_4digit_codes(found)
                if expected_count is not None and len(symbols) < max(1, expected_count - 5):
                    symbols = []
                if symbols:
                    source = "yfinance_funds_data"
            except Exception:
                symbols = []

        if not symbols and fallback_symbols:
            symbols = fallback_symbols
            source = "fallback_manual"

        if limit is not None:
            symbols = symbols[: max(1, int(limit))]
        self.cache.set(cache_key, (symbols, source), ttl_sec=3600)
        return symbols, source

    def get_00935_constituents(self, limit: Optional[int] = None) -> tuple[list[str], str]:
        return self.get_tw_etf_constituents("00935", limit=limit)

    def get_0050_constituents(self, limit: Optional[int] = None) -> tuple[list[str], str]:
        return self.get_tw_etf_constituents("0050", limit=limit)
