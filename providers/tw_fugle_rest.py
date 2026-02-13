from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from market_data_types import OhlcvSnapshot, QuoteSnapshot
from providers.base import MarketDataProvider, ProviderError, ProviderErrorKind, ProviderRequest


class TwFugleHistoricalProvider(MarketDataProvider):
    name = "tw_fugle_rest"
    default_base_url = "https://api.fugle.tw/marketdata/v1.0/stock"
    default_key_file = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "codexapp" / "fuglekey"
    # Fugle historical candles endpoint is documented for 1-year query range per request (daily).
    max_days_per_request = 360
    # Intraday candles endpoint range is shorter; keep each request within 5 days.
    max_days_per_intraday_request = 5

    @classmethod
    def _resolve_api_key(cls, api_key: Optional[str] = None) -> Optional[str]:
        direct = str(api_key or "").strip()
        if direct:
            return direct

        env_key = str(os.getenv("FUGLE_MARKETDATA_API_KEY") or os.getenv("FUGLE_API_KEY") or "").strip()
        if env_key:
            return env_key

        key_file = str(
            os.getenv("FUGLE_MARKETDATA_API_KEY_FILE") or os.getenv("FUGLE_API_KEY_FILE") or cls.default_key_file
        ).strip()
        if not key_file:
            return None
        try:
            text = Path(key_file).expanduser().read_text(encoding="utf-8")
        except Exception:
            return None
        normalized = text.strip()
        return normalized or None

    def __init__(self, api_key: Optional[str] = None, timeout_sec: int = 15, base_url: Optional[str] = None):
        self.api_key = self._resolve_api_key(api_key)
        self.timeout_sec = timeout_sec
        self.base_url = (base_url or self.default_base_url).rstrip("/")

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        raise ProviderError(self.name, ProviderErrorKind.UNSUPPORTED, "Fugle historical provider does not provide quote")

    def _fetch_chunk(self, symbol: str, start: datetime, end: datetime, *, timeframe: str) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ProviderError(self.name, ProviderErrorKind.AUTH, "FUGLE_MARKETDATA_API_KEY is missing")

        url = f"{self.base_url}/historical/candles/{symbol}"
        params = {
            "from": start.date().isoformat(),
            "to": end.date().isoformat(),
            "timeframe": timeframe,
            "fields": "open,high,low,close,volume",
            "sort": "asc",
        }
        headers = {"X-API-KEY": self.api_key}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout_sec)
        except Exception as exc:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, "Fugle historical request failed", exc) from exc

        if resp.status_code in {401, 403}:
            raise ProviderError(self.name, ProviderErrorKind.AUTH, "Fugle API key unauthorized")
        if resp.status_code == 429:
            raise ProviderError(self.name, ProviderErrorKind.RATE_LIMIT, "Fugle API rate limit")
        if resp.status_code >= 400:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, f"Fugle API http {resp.status_code}")

        try:
            payload = resp.json()
        except Exception as exc:
            raise ProviderError(self.name, ProviderErrorKind.PARSE, "Fugle historical response parse failed", exc) from exc

        if not isinstance(payload, dict):
            raise ProviderError(self.name, ProviderErrorKind.PARSE, "Fugle historical response is not an object")
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        out: list[dict[str, Any]] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            ts = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            open_ = self._to_float(row.get("open"))
            high = self._to_float(row.get("high"))
            low = self._to_float(row.get("low"))
            close = self._to_float(row.get("close"))
            volume = self._to_float(row.get("volume"))
            if None in (open_, high, low, close):
                continue
            out.append(
                {
                    "date": pd.Timestamp(ts).to_pydatetime(),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": 0.0 if volume is None else float(volume),
                }
            )
        return out

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        if request.market != "TW":
            raise ProviderError(self.name, ProviderErrorKind.UNSUPPORTED, "Fugle historical provider only supports TW market")
        interval = str(request.interval or "").strip().lower()
        if interval not in {"1d", "1m"}:
            raise ProviderError(self.name, ProviderErrorKind.UNSUPPORTED, "Fugle historical provider supports 1d/1m interval")

        symbol = str(request.symbol or "").strip().upper()
        if not symbol:
            raise ProviderError(self.name, ProviderErrorKind.PARSE, "missing symbol")
        if not symbol.isdigit():
            raise ProviderError(self.name, ProviderErrorKind.UNSUPPORTED, f"unsupported TW symbol for Fugle: {symbol}")

        end = request.end or datetime.now(tz=timezone.utc)
        if interval == "1d":
            start = request.start or datetime(end.year - 5, 1, 1, tzinfo=timezone.utc)
            timeframe = "D"
            max_span = max(1, int(self.max_days_per_request)) - 1
        else:
            start = request.start or (end - timedelta(days=1))
            timeframe = "1"
            max_span = max(1, int(self.max_days_per_intraday_request)) - 1
        if start > end:
            raise ProviderError(self.name, ProviderErrorKind.PARSE, "start date is after end date")

        rows: list[dict[str, Any]] = []
        cursor = start
        while cursor.date() <= end.date():
            chunk_end = min(end, cursor + timedelta(days=max_span))
            rows.extend(self._fetch_chunk(symbol=symbol, start=cursor, end=chunk_end, timeframe=timeframe))
            cursor = chunk_end + timedelta(days=1)

        if not rows:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "Fugle historical returned empty OHLCV")

        df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
        df = df[(df.index >= start) & (df.index <= end)]
        if df.empty:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "Fugle historical OHLCV filtered empty")

        return OhlcvSnapshot(
            symbol=symbol,
            market="TW",
            interval=interval,
            tz="UTC",
            df=df[["open", "high", "low", "close", "volume"]],
            source=self.name,
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )
