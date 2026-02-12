from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from market_data_types import OhlcvSnapshot, QuoteSnapshot
from providers.base import MarketDataProvider, ProviderError, ProviderErrorKind, ProviderRequest


def _parse_roc_compact(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if len(text) != 7 or not text.isdigit():
        return None
    year = int(text[:3]) + 1911
    month = int(text[3:5])
    day = int(text[5:7])
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_roc_slash(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    parts = text.split("/")
    if len(parts) != 3:
        return None
    try:
        year = int(parts[0]) + 1911
        month = int(parts[1])
        day = int(parts[2])
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "--", "-", "X0.00"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> Optional[int]:
    fv = _to_float(value)
    if fv is None:
        return None
    return int(fv)


class TwOpenApiProvider(MarketDataProvider):
    name = "tw_openapi"

    def __init__(self, timeout_sec: int = 15):
        self.timeout_sec = timeout_sec

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        try:
            resp = requests.get(url, timeout=self.timeout_sec)
            resp.raise_for_status()
            rows = resp.json()
        except Exception as exc:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, "TWSE OpenAPI quote request failed", exc) from exc

        if not isinstance(rows, list):
            raise ProviderError(self.name, ProviderErrorKind.PARSE, "TWSE OpenAPI quote response is not an array")

        row = next((item for item in rows if str(item.get("Code", "")).strip() == request.symbol), None)
        if row is None:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, f"TWSE OpenAPI symbol not found: {request.symbol}")

        ts = _parse_roc_compact(str(row.get("Date") or "")) or datetime.now(tz=timezone.utc)
        price = _to_float(row.get("ClosingPrice"))
        open_ = _to_float(row.get("OpeningPrice"))
        high = _to_float(row.get("HighestPrice"))
        low = _to_float(row.get("LowestPrice"))
        vol = _to_int(row.get("TradeVolume"))
        chg = _to_float(row.get("Change"))
        prev_close = (price - chg) if (price is not None and chg is not None) else None

        return QuoteSnapshot(
            symbol=request.symbol,
            market=request.market,
            ts=ts,
            price=price,
            prev_close=prev_close,
            open=open_,
            high=high,
            low=low,
            volume=vol,
            source=self.name,
            is_delayed=True,
            interval="1d",
            currency="TWD",
            exchange="TWSE",
            extra={"name": row.get("Name")},
        )

    def _fetch_month(self, symbol: str, month_anchor: datetime) -> List[List[Any]]:
        date_str = month_anchor.strftime("%Y%m01")
        url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        params = {"response": "json", "date": date_str, "stockNo": symbol}
        try:
            resp = requests.get(url, params=params, timeout=self.timeout_sec)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, "TWSE monthly history request failed", exc) from exc

        data = payload.get("data")
        if not isinstance(data, list):
            return []
        return data

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        if request.interval != "1d":
            raise ProviderError(self.name, ProviderErrorKind.UNSUPPORTED, "TW OpenAPI provider supports 1d interval")

        end = request.end or datetime.now(tz=timezone.utc)
        start = request.start or datetime(end.year - 5, 1, 1, tzinfo=timezone.utc)
        months = pd.period_range(start=start.date(), end=end.date(), freq="M")

        rows: List[Dict[str, Any]] = []
        for period in months:
            month_anchor = datetime(period.year, period.month, 1, tzinfo=timezone.utc)
            for item in self._fetch_month(request.symbol, month_anchor):
                if not isinstance(item, list) or len(item) < 8:
                    continue
                date = _parse_roc_slash(str(item[0]))
                if date is None:
                    continue
                open_ = _to_float(item[3])
                high = _to_float(item[4])
                low = _to_float(item[5])
                close = _to_float(item[6])
                volume = _to_float(item[1])
                if None in (open_, high, low, close):
                    continue
                rows.append(
                    {
                        "date": date,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume or 0.0,
                    }
                )

        if not rows:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "TW OpenAPI returned empty OHLCV")

        df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
        df = df[(df.index >= start) & (df.index <= end)]
        if df.empty:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "TW OpenAPI OHLCV filtered empty")

        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval="1d",
            tz="UTC",
            df=df[["open", "high", "low", "close", "volume"]],
            source=self.name,
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )
