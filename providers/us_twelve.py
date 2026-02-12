from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from market_data_types import OhlcvSnapshot, QuoteSnapshot
from providers.base import MarketDataProvider, ProviderError, ProviderErrorKind, ProviderRequest


class UsTwelveDataProvider(MarketDataProvider):
    name = "twelvedata"
    base_url = "https://api.twelvedata.com"

    def __init__(self, api_key: Optional[str] = None, timeout_sec: int = 12):
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")
        self.timeout_sec = timeout_sec

    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderError(self.name, ProviderErrorKind.AUTH, "TWELVE_DATA_API_KEY is missing")
        params = dict(params)
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = requests.get(url, params=params, timeout=self.timeout_sec)
        except requests.RequestException as exc:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, "Twelve Data network error", exc) from exc

        if resp.status_code == 429:
            raise ProviderError(self.name, ProviderErrorKind.RATE_LIMIT, "Twelve Data rate limited")
        if resp.status_code == 401:
            raise ProviderError(self.name, ProviderErrorKind.AUTH, "Twelve Data auth failed")
        if resp.status_code >= 400:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, f"Twelve Data HTTP {resp.status_code}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise ProviderError(self.name, ProviderErrorKind.PARSE, "Twelve Data invalid JSON", exc) from exc

        if isinstance(payload, dict) and payload.get("status") == "error":
            code = (payload.get("code") or "").lower()
            message = str(payload.get("message") or "Twelve Data returned error")
            if "api" in code or "key" in code:
                kind = ProviderErrorKind.AUTH
            elif "limit" in code:
                kind = ProviderErrorKind.RATE_LIMIT
            else:
                kind = ProviderErrorKind.NETWORK
            raise ProviderError(self.name, kind, message)
        return payload

    @staticmethod
    def _parse_dt(value: str) -> datetime:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).rename(
            columns={"datetime": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        )
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "open", "high", "low", "close"]).set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        payload = self._request("quote", {"symbol": request.symbol})
        if not isinstance(payload, dict) or "close" not in payload:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "Twelve Data quote missing")

        close = payload.get("close")
        prev_close = payload.get("previous_close")
        open_ = payload.get("open")
        high = payload.get("high")
        low = payload.get("low")
        volume = payload.get("volume")
        ts = payload.get("datetime") or payload.get("timestamp")
        if isinstance(ts, str):
            quote_ts = self._parse_dt(ts)
        else:
            quote_ts = datetime.now(tz=timezone.utc)

        return QuoteSnapshot(
            symbol=request.symbol,
            market=request.market,
            ts=quote_ts,
            price=float(close) if close is not None else None,
            prev_close=float(prev_close) if prev_close is not None else None,
            open=float(open_) if open_ is not None else None,
            high=float(high) if high is not None else None,
            low=float(low) if low is not None else None,
            volume=int(float(volume)) if volume is not None else None,
            source=self.name,
            is_delayed=False,
            interval="quote",
            currency=payload.get("currency"),
            exchange=payload.get("exchange"),
        )

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        if request.interval != "1d":
            raise ProviderError(self.name, ProviderErrorKind.UNSUPPORTED, "Twelve Data provider supports 1d in this app")

        payload = self._request(
            "time_series",
            {"symbol": request.symbol, "interval": "1day", "outputsize": 5000, "format": "JSON"},
        )
        rows = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "Twelve Data OHLCV missing values")
        df = self._to_df(rows)
        if df.empty:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "Twelve Data OHLCV empty")

        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval="1d",
            tz="UTC",
            df=df,
            source=self.name,
            is_delayed=False,
            fetched_at=datetime.now(tz=timezone.utc),
        )
