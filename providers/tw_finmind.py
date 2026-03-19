from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests

from providers.base import ProviderError, ProviderErrorKind


class TwFinMindClient:
    name = "finmind"
    base_url = "https://api.finmindtrade.com/api/v4/data"
    default_key_file = (
        Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "codexapp"
        / "finmindkey"
    )

    @classmethod
    def _resolve_api_key(cls, api_key: str | None = None) -> str | None:
        direct = str(api_key or "").strip()
        if direct:
            return direct

        env_key = str(os.getenv("FINMIND_API_TOKEN") or os.getenv("FINMIND_TOKEN") or "").strip()
        if env_key:
            return env_key

        key_file = str(
            os.getenv("FINMIND_API_TOKEN_FILE")
            or os.getenv("FINMIND_TOKEN_FILE")
            or cls.default_key_file
        ).strip()
        if not key_file:
            return None
        try:
            text = Path(key_file).expanduser().read_text(encoding="utf-8")
        except Exception:
            return None
        normalized = text.strip()
        return normalized or None

    def __init__(self, api_key: str | None = None, timeout_sec: int = 15):
        self.api_key = self._resolve_api_key(api_key)
        self.timeout_sec = timeout_sec

    @property
    def enabled(self) -> bool:
        return bool(str(self.api_key or "").strip())

    @staticmethod
    def _normalize_date(value: date | datetime | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        text = str(value).strip()
        return text or None

    def _request_dataset(self, dataset: str, **params: Any) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ProviderError(self.name, ProviderErrorKind.AUTH, "FINMIND_API_TOKEN is missing")

        query = {"dataset": dataset}
        for key, value in params.items():
            if value is None:
                continue
            text = self._normalize_date(value)
            query[key] = text if text is not None else value

        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(
                self.base_url, headers=headers, params=query, timeout=self.timeout_sec
            )
        except requests.RequestException as exc:
            raise ProviderError(
                self.name, ProviderErrorKind.NETWORK, "FinMind network error", exc
            ) from exc

        try:
            payload = resp.json()
        except ValueError as exc:
            raise ProviderError(
                self.name, ProviderErrorKind.PARSE, "FinMind invalid JSON", exc
            ) from exc

        status_code = int(getattr(resp, "status_code", 0) or 0)
        status_value = None
        message = ""
        if isinstance(payload, dict):
            status_value = payload.get("status")
            message = str(payload.get("msg") or payload.get("message") or "").strip()

        if status_code in {401, 403} or str(status_value) in {"401", "403"}:
            raise ProviderError(self.name, ProviderErrorKind.AUTH, message or "FinMind auth failed")
        if status_code in {402, 429} or str(status_value) in {"402", "429"}:
            raise ProviderError(
                self.name, ProviderErrorKind.RATE_LIMIT, message or "FinMind rate limited"
            )
        if status_code >= 400:
            raise ProviderError(
                self.name,
                ProviderErrorKind.NETWORK,
                message or f"FinMind HTTP {status_code}",
            )
        if not isinstance(payload, dict):
            raise ProviderError(
                self.name, ProviderErrorKind.PARSE, "FinMind response is not an object"
            )

        data = payload.get("data")
        if data in (None, ""):
            return []
        if not isinstance(data, list):
            raise ProviderError(
                self.name, ProviderErrorKind.PARSE, "FinMind response data is not a list"
            )
        return [row for row in data if isinstance(row, dict)]

    def fetch_stock_info(self) -> list[dict[str, Any]]:
        return self._request_dataset("TaiwanStockInfo")

    def fetch_month_revenue(
        self, stock_id: str, *, start_date: date | datetime | str
    ) -> list[dict[str, Any]]:
        return self._request_dataset(
            "TaiwanStockMonthRevenue",
            data_id=str(stock_id or "").strip().upper(),
            start_date=start_date,
        )

    def fetch_stock_news(
        self, stock_id: str, *, start_date: date | datetime | str
    ) -> list[dict[str, Any]]:
        return self._request_dataset(
            "TaiwanStockNews",
            data_id=str(stock_id or "").strip().upper(),
            start_date=start_date,
        )

    def fetch_institutional_investors(
        self,
        stock_id: str,
        *,
        start_date: date | datetime | str,
        end_date: date | datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        return self._request_dataset(
            "TaiwanStockInstitutionalInvestorsBuySell",
            data_id=str(stock_id or "").strip().upper(),
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_price_adj(
        self,
        stock_id: str,
        *,
        start_date: date | datetime | str,
        end_date: date | datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        return self._request_dataset(
            "TaiwanStockPriceAdj",
            data_id=str(stock_id or "").strip().upper(),
            start_date=start_date,
            end_date=end_date,
        )
