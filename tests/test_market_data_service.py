from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from market_data_types import OhlcvSnapshot
from market_data_types import QuoteSnapshot
from providers.base import ProviderError, ProviderErrorKind, ProviderRequest
from services.market_data_service import MarketDataService


class _BrokenProvider:
    name = "broken"

    def quote(self, request: ProviderRequest):
        raise ProviderError("broken", ProviderErrorKind.RATE_LIMIT, "rate limited")


class _GoodProvider:
    name = "good"

    def quote(self, request: ProviderRequest):
        return QuoteSnapshot(
            symbol=request.symbol,
            market=request.market,
            ts=datetime.now(tz=timezone.utc),
            price=100.0,
            prev_close=98.0,
            open=99.0,
            high=101.0,
            low=97.0,
            volume=10_000,
            source=self.name,
            is_delayed=False,
        )


class MarketDataServiceTests(unittest.TestCase):
    def test_quote_chain_fallback(self):
        service = MarketDataService()
        request = ProviderRequest(symbol="TSLA", market="US", interval="quote")
        quote, chain, reason, depth = service._try_quote_chain([_BrokenProvider(), _GoodProvider()], request)  # noqa: SLF001
        self.assertEqual(quote.source, "good")
        self.assertEqual(depth, 1)
        self.assertEqual(chain, ["broken", "good"])
        self.assertIn("RATE_LIMIT", reason or "")

    def test_quality_marks_degraded_for_fallback(self):
        quote = QuoteSnapshot(
            symbol="TSLA",
            market="US",
            ts=datetime.now(tz=timezone.utc),
            price=100.0,
            prev_close=99.0,
            open=100.0,
            high=101.0,
            low=98.0,
            volume=1,
            source="good",
            is_delayed=False,
        )
        quality = MarketDataService._quality(quote, fallback_depth=2, reason="fallback")
        self.assertTrue(quality.degraded)
        self.assertEqual(quality.fallback_depth, 2)
        self.assertEqual(quality.reason, "fallback")

    @patch("yfinance.download")
    def test_get_benchmark_series(self, mock_download):
        idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
        mock_download.return_value = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=idx)
        service = MarketDataService()
        out = service.get_benchmark_series(
            market="US",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        )
        self.assertFalse(out.empty)
        self.assertIn("close", out.columns)
        # cache hit path
        out2 = service.get_benchmark_series(
            market="US",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        )
        self.assertEqual(len(out2), len(out))

    @patch("yfinance.download")
    def test_get_benchmark_series_fallback_to_proxy(self, mock_download):
        mock_download.side_effect = RuntimeError("rate limit")
        idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
        snap = OhlcvSnapshot(
            symbol="SPY",
            market="US",
            interval="1d",
            tz="UTC",
            df=pd.DataFrame(
                {"open": [1, 1, 1, 1, 1], "high": [1, 1, 1, 1, 1], "low": [1, 1, 1, 1, 1], "close": [10, 11, 12, 13, 14], "volume": [1, 1, 1, 1, 1]},
                index=idx,
            ),
            source="stooq",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )
        service = MarketDataService()
        with patch.object(service.us_stooq, "ohlcv", return_value=snap):
            out = service.get_benchmark_series(
                market="US",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 31, tzinfo=timezone.utc),
                benchmark="auto",
            )
        self.assertFalse(out.empty)
        self.assertEqual(out.attrs.get("symbol"), "SPY")
        self.assertEqual(out.attrs.get("source"), "stooq")

    @patch("requests.get")
    def test_get_tw_symbol_names(self, mock_get):
        mock_resp = unittest.mock.MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [
            {"Code": "2330", "Name": "台積電"},
            {"Code": "3017", "Name": "奇鋐"},
        ]
        mock_get.return_value = mock_resp

        service = MarketDataService()
        out = service.get_tw_symbol_names(["2330", "3017", "9999"])
        self.assertEqual(out["2330"], "台積電")
        self.assertEqual(out["3017"], "奇鋐")
        self.assertEqual(out["9999"], "9999")
        out2 = service.get_tw_symbol_names(["2330", "3017", "9999"])
        self.assertEqual(out2["2330"], "台積電")
        self.assertTrue(mock_get.called)

    @patch("yfinance.Ticker")
    def test_get_tw_etf_constituents_fallback(self, mock_ticker):
        mock_ticker.side_effect = RuntimeError("upstream down")
        service = MarketDataService()
        with patch.object(service, "_fetch_0050_yuanta_constituents", return_value=[]):
            symbols, source = service.get_tw_etf_constituents("0050", limit=5)
            self.assertEqual(source, "fallback_manual")
            self.assertEqual(len(symbols), 5)
            self.assertTrue(all(len(s) == 4 and s.isdigit() for s in symbols))

    def test_get_tw_etf_constituents_00935_primary_source(self):
        service = MarketDataService()
        with patch.object(service, "_fetch_00935_nomura_constituents", return_value=["2330", "2454", "2317"]):
            symbols, source = service.get_tw_etf_constituents("00935")
        self.assertEqual(source, "nomura_etfapi")
        self.assertEqual(symbols, ["2330", "2454", "2317"])

    def test_get_tw_etf_expected_count(self):
        self.assertEqual(MarketDataService.get_tw_etf_expected_count("0050"), 50)
        self.assertEqual(MarketDataService.get_tw_etf_expected_count("00935"), 50)
        self.assertIsNone(MarketDataService.get_tw_etf_expected_count("9999"))


if __name__ == "__main__":
    unittest.main()
