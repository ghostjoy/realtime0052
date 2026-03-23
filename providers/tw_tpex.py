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


class TwTpexEtfHistoricalProvider(MarketDataProvider):
    name = "tw_tpex_etf"
    endpoint = "https://www.tpex.org.tw/www/zh-tw/ETFReport/historical"

    def __init__(self, timeout_sec: int = 15, stop_missing_days: int = 10):
        self.timeout_sec = timeout_sec
        self.stop_missing_days = max(3, int(stop_missing_days))

    @staticmethod
    def _format_query_date(value: datetime) -> str:
        return pd.Timestamp(value).tz_convert("UTC").strftime("%Y/%m/%d")

    def _fetch_rows_for_date(self, trade_date: datetime) -> list[list[Any]]:
        params = {
            "type": "Daily",
            "cate": "all",
            "date": self._format_query_date(trade_date),
        }
        try:
            resp = requests.get(self.endpoint, params=params, timeout=self.timeout_sec)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise ProviderError(
                self.name, ProviderErrorKind.NETWORK, "TPEx ETF historical request failed", exc
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderError(
                self.name, ProviderErrorKind.PARSE, "TPEx ETF historical response is not an object"
            )
        tables = payload.get("tables")
        if not isinstance(tables, list):
            raise ProviderError(
                self.name, ProviderErrorKind.PARSE, "TPEx ETF historical response missing tables"
            )
        rows: list[list[Any]] = []
        for table in tables:
            if not isinstance(table, dict):
                continue
            data_rows = table.get("data")
            if not isinstance(data_rows, list):
                continue
            for row in data_rows:
                if isinstance(row, list):
                    rows.append(row)
        return rows

    @staticmethod
    def _find_symbol_row(symbol: str, rows: list[list[Any]]) -> list[Any] | None:
        return next(
            (row for row in rows if len(row) >= 11 and str(row[1]).strip().upper() == symbol),
            None,
        )

    def _parse_row(self, row: list[Any]) -> dict[str, Any]:
        date = _parse_roc_compact(str(row[0] or ""))
        if date is None:
            raise ProviderError(self.name, ProviderErrorKind.PARSE, "TPEx ETF trade date parse failed")
        open_ = _to_float(row[5])
        high = _to_float(row[6])
        low = _to_float(row[7])
        close = _to_float(row[8])
        volume_lots = _to_float(row[3])
        if None in (open_, high, low, close):
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "TPEx ETF OHLCV missing")
        return {
            "date": date,
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume_lots or 0.0) * 1000.0,
        }

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        raise ProviderError(
            self.name,
            ProviderErrorKind.UNSUPPORTED,
            "TPEx ETF historical provider does not provide quote",
        )

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        market = str(request.market or "").strip().upper()
        if market != "OTC":
            raise ProviderError(
                self.name,
                ProviderErrorKind.UNSUPPORTED,
                "TPEx ETF historical provider only supports OTC market",
            )
        if request.interval != "1d":
            raise ProviderError(
                self.name,
                ProviderErrorKind.UNSUPPORTED,
                "TPEx ETF historical provider supports 1d interval",
            )
        end = request.end or datetime.now(tz=timezone.utc)
        start = request.start or (end - pd.Timedelta(days=90))
        if start > end:
            raise ProviderError(self.name, ProviderErrorKind.PARSE, "start date is after end date")

        symbol = str(request.symbol or "").strip().upper()
        rows_out: list[dict[str, Any]] = []
        missing_after_hits = 0
        day_index = pd.bdate_range(start=start.date(), end=end.date(), tz="UTC")
        for trade_date in reversed(day_index):
            rows = self._fetch_rows_for_date(pd.Timestamp(trade_date).to_pydatetime())
            row = self._find_symbol_row(symbol, rows)
            if row is None:
                if rows_out:
                    missing_after_hits += 1
                    if missing_after_hits >= self.stop_missing_days:
                        break
                continue
            missing_after_hits = 0
            rows_out.append(self._parse_row(row))

        if not rows_out:
            raise ProviderError(
                self.name, ProviderErrorKind.EMPTY, "TPEx ETF historical returned empty OHLCV"
            )

        df = (
            pd.DataFrame(rows_out)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .set_index("date")
        )
        df = df[(df.index >= start) & (df.index <= end)]
        if df.empty:
            raise ProviderError(
                self.name, ProviderErrorKind.EMPTY, "TPEx ETF historical OHLCV filtered empty"
            )

        return OhlcvSnapshot(
            symbol=symbol,
            market=market,
            interval="1d",
            tz="UTC",
            df=df[["open", "high", "low", "close", "volume"]],
            source=self.name,
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
            asof=pd.Timestamp(df.index.max()).to_pydatetime() if len(df.index) else None,
        )
