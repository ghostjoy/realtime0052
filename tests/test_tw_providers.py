from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from providers.base import ProviderError, ProviderRequest
from providers.tw_fugle_ws import TwFugleWebSocketProvider
from providers.tw_tpex import TwTpexOpenApiProvider


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeWebSocketConn:
    def __init__(self):
        self.sent: list[str] = []
        self._messages = [
            json.dumps({"event": "authenticated"}),
            json.dumps(
                {
                    "data": {
                        "channel": "books",
                        "symbol": "2330",
                        "bids": [{"price": 190.5, "size": 100}],
                        "asks": [{"price": 191.0, "size": 120}],
                    }
                }
            ),
            json.dumps(
                {
                    "data": {
                        "channel": "trades",
                        "symbol": "2330",
                        "price": 190.8,
                        "previousClose": 189.0,
                        "open": 189.5,
                        "high": 191.2,
                        "low": 188.8,
                        "volume": 900,
                        "timestamp": int(datetime(2026, 2, 13, 6, 0, tzinfo=timezone.utc).timestamp() * 1000),
                    }
                }
            ),
        ]

    def send(self, text: str):
        self.sent.append(text)

    def recv(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        return ""

    def close(self):
        return None


class TwProvidersTests(unittest.TestCase):
    @patch("requests.get")
    def test_tpex_quote_success(self, mock_get):
        mock_get.return_value = _MockResponse(
            [
                {
                    "Date": "1150211",
                    "SecuritiesCompanyCode": "6488",
                    "CompanyName": "環球晶",
                    "Close": "420.5",
                    "Change": "+3.0",
                    "Open": "418",
                    "High": "423",
                    "Low": "417",
                    "TradingShares": "100000",
                }
            ]
        )
        provider = TwTpexOpenApiProvider()
        quote = provider.quote(ProviderRequest(symbol="6488", market="TW", interval="quote"))
        self.assertEqual(quote.source, "tw_tpex")
        self.assertEqual(quote.price, 420.5)
        self.assertEqual(quote.prev_close, 417.5)
        self.assertEqual(quote.extra.get("name"), "環球晶")

    def test_tpex_ohlcv_rejects_long_range(self):
        provider = TwTpexOpenApiProvider()
        with self.assertRaises(ProviderError):
            provider.ohlcv(
                ProviderRequest(
                    symbol="6488",
                    market="TW",
                    interval="1d",
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
            )

    def test_fugle_ws_quote_success(self):
        fake_conn = _FakeWebSocketConn()
        fake_mod = types.SimpleNamespace(create_connection=lambda *_args, **_kwargs: fake_conn)
        provider = TwFugleWebSocketProvider(api_key="fake-key", timeout_sec=2)
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            quote = provider.quote(ProviderRequest(symbol="2330", market="TW", interval="quote", exchange="tse"))
        self.assertEqual(quote.source, "fugle_ws")
        self.assertEqual(quote.price, 190.8)
        self.assertEqual(quote.prev_close, 189.0)
        self.assertEqual(quote.extra.get("bid_prices"), [190.5])
        self.assertEqual(quote.extra.get("ask_prices"), [191.0])
        self.assertGreaterEqual(len(fake_conn.sent), 3)
        auth_payload = json.loads(fake_conn.sent[0])
        self.assertEqual(auth_payload.get("event"), "auth")
        self.assertEqual(auth_payload.get("data", {}).get("apikey"), "fake-key")

    def test_fugle_ws_requires_api_key(self):
        provider = TwFugleWebSocketProvider(api_key=None)
        provider.api_key = None
        with self.assertRaises(ProviderError):
            provider.quote(ProviderRequest(symbol="2330", market="TW", interval="quote"))

    def test_fugle_default_streaming_endpoint(self):
        provider = TwFugleWebSocketProvider(api_key="fake-key")
        self.assertTrue(provider.ws_url.endswith("/streaming"))

    def test_fugle_api_key_from_key_file_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "fuglekey"
            key_file.write_text("  file-key-123  \n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "FUGLE_MARKETDATA_API_KEY_FILE": str(key_file),
                    "FUGLE_MARKETDATA_API_KEY": "",
                    "FUGLE_API_KEY": "",
                },
                clear=True,
            ):
                provider = TwFugleWebSocketProvider(api_key=None)
        self.assertEqual(provider.api_key, "file-key-123")

    def test_fugle_api_key_from_default_key_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "fuglekey"
            key_file.write_text("default-file-key", encoding="utf-8")
            with patch.object(TwFugleWebSocketProvider, "default_key_file", key_file):
                with patch.dict(os.environ, {}, clear=True):
                    provider = TwFugleWebSocketProvider(api_key=None)
        self.assertEqual(provider.api_key, "default-file-key")

    def test_fugle_parse_microsecond_timestamp(self):
        ts_us = 1685338200000000
        parsed = TwFugleWebSocketProvider._parse_ts(ts_us)
        self.assertEqual(parsed.year, 2023)


if __name__ == "__main__":
    unittest.main()
