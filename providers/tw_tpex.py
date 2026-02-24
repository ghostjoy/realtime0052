from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from market_data_types import OhlcvSnapshot, QuoteSnapshot
from providers.base import MarketDataProvider, ProviderError, ProviderErrorKind, ProviderRequest


def _parse_roc_compact(value: str) -> datetime | None:
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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "--", "-", "除權息"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


class TwTpexOpenApiProvider(MarketDataProvider):
    name = "tw_tpex"

    def __init__(self, timeout_sec: int = 15):
        self.timeout_sec = timeout_sec

    def _fetch_rows(self) -> list[dict[str, Any]]:
        url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
        try:
            resp = requests.get(url, timeout=self.timeout_sec)
            resp.raise_for_status()
            rows = resp.json()
        except Exception as exc:
            raise ProviderError(
                self.name, ProviderErrorKind.NETWORK, "TPEx OpenAPI request failed", exc
            ) from exc
        if not isinstance(rows, list):
            raise ProviderError(
                self.name, ProviderErrorKind.PARSE, "TPEx OpenAPI response is not an array"
            )
        return rows

    def _find_row(self, symbol: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        row = next(
            (item for item in rows if str(item.get("SecuritiesCompanyCode", "")).strip() == symbol),
            None,
        )
        if row is None:
            raise ProviderError(
                self.name, ProviderErrorKind.EMPTY, f"TPEx symbol not found: {symbol}"
            )
        return row

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        rows = self._fetch_rows()
        row = self._find_row(request.symbol, rows)

        ts = _parse_roc_compact(str(row.get("Date") or "")) or datetime.now(tz=timezone.utc)
        close = _to_float(row.get("Close"))
        change = _to_float(row.get("Change"))
        prev_close = (close - change) if (close is not None and change is not None) else None

        return QuoteSnapshot(
            symbol=request.symbol,
            market=request.market,
            ts=ts,
            price=close,
            prev_close=prev_close,
            open=_to_float(row.get("Open")),
            high=_to_float(row.get("High")),
            low=_to_float(row.get("Low")),
            volume=int(_to_float(row.get("TradingShares")) or 0),
            source=self.name,
            is_delayed=True,
            interval="1d",
            currency="TWD",
            exchange="TPEx",
            extra={"name": row.get("CompanyName")},
        )

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        if request.interval != "1d":
            raise ProviderError(
                self.name, ProviderErrorKind.UNSUPPORTED, "TPEx provider supports 1d interval"
            )

        end = request.end or datetime.now(tz=timezone.utc)
        start = request.start or (end - pd.Timedelta(days=5))

        # This endpoint only exposes latest daily close rows, not long history.
        if (end.date() - start.date()).days > 7:
            raise ProviderError(
                self.name,
                ProviderErrorKind.UNSUPPORTED,
                "TPEx OpenAPI only provides latest daily snapshot; use another provider for long history",
            )

        rows = self._fetch_rows()
        row = self._find_row(request.symbol, rows)
        date = _parse_roc_compact(str(row.get("Date") or ""))
        if date is None:
            raise ProviderError(
                self.name, ProviderErrorKind.PARSE, "TPEx OpenAPI date parse failed"
            )

        open_ = _to_float(row.get("Open"))
        high = _to_float(row.get("High"))
        low = _to_float(row.get("Low"))
        close = _to_float(row.get("Close"))
        volume = _to_float(row.get("TradingShares"))
        if None in (open_, high, low, close):
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "TPEx OpenAPI OHLCV missing")

        df = pd.DataFrame(
            [
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume or 0.0,
                }
            ],
            index=pd.DatetimeIndex([date]),
        )
        df = df[(df.index >= start) & (df.index <= end)]
        if df.empty:
            raise ProviderError(
                self.name, ProviderErrorKind.EMPTY, "TPEx OpenAPI OHLCV filtered empty"
            )

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
