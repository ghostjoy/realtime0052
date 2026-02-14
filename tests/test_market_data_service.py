from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from market_data_types import OhlcvSnapshot
from market_data_types import QuoteSnapshot
from providers.base import ProviderError, ProviderErrorKind, ProviderRequest
from services.market_data_service import LiveOptions, MarketDataService


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


class _OhlcvProvider:
    def __init__(self, name: str, fail: bool = False):
        self.name = name
        self.fail = fail
        self.last_symbol: str | None = None

    def ohlcv(self, request: ProviderRequest):
        self.last_symbol = request.symbol
        if self.fail:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, "ohlcv failed")
        idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval=request.interval,
            tz="UTC",
            df=pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1.0, 1.0, 1.0],
                },
                index=idx,
            ),
            source=self.name,
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
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

    def test_try_ohlcv_chain_normalizes_tw_symbol_for_yahoo(self):
        service = MarketDataService()
        tw_openapi = _OhlcvProvider("tw_openapi", fail=True)
        yahoo = _OhlcvProvider("yahoo")
        request = ProviderRequest(symbol="0050", market="TW", interval="1d")

        snap = service._try_ohlcv_chain([tw_openapi, yahoo], request)  # noqa: SLF001

        self.assertEqual(tw_openapi.last_symbol, "0050")
        self.assertEqual(yahoo.last_symbol, "0050.TW")
        self.assertEqual(snap.symbol, "0050.TW")

    def test_try_ohlcv_chain_keeps_tw_index_symbol_for_yahoo(self):
        service = MarketDataService()
        tw_openapi = _OhlcvProvider("tw_openapi", fail=True)
        yahoo = _OhlcvProvider("yahoo")
        request = ProviderRequest(symbol="^TWII", market="TW", interval="1d")

        service._try_ohlcv_chain([tw_openapi, yahoo], request)  # noqa: SLF001

        self.assertEqual(yahoo.last_symbol, "^TWII")

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
        twse_resp = unittest.mock.MagicMock()
        twse_resp.raise_for_status.return_value = None
        twse_resp.json.return_value = [
            {"Code": "2330", "Name": "台積電"},
            {"Code": "3017", "Name": "奇鋐"},
        ]
        tpex_resp = unittest.mock.MagicMock()
        tpex_resp.raise_for_status.return_value = None
        tpex_resp.json.return_value = [
            {"SecuritiesCompanyCode": "6510", "CompanyName": "精測"},
        ]

        def _fake_get(url, timeout=12):
            if "openapi.twse.com.tw" in str(url):
                return twse_resp
            if "www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes" in str(url):
                return tpex_resp
            raise RuntimeError(f"unexpected url: {url}")

        mock_get.side_effect = _fake_get

        service = MarketDataService()
        out = service.get_tw_symbol_names(["2330", "3017", "6510", "9999"])
        self.assertEqual(out["2330"], "台積電")
        self.assertEqual(out["3017"], "奇鋐")
        self.assertEqual(out["6510"], "精測")
        self.assertEqual(out["9999"], "9999")
        out2 = service.get_tw_symbol_names(["2330", "3017", "6510", "9999"])
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

    def test_get_tw_live_context_prefers_fugle_when_key_exists(self):
        service = MarketDataService()
        service.tw_fugle_ws.api_key = "fake-key"
        service.tw_mis.quote = unittest.mock.MagicMock(side_effect=RuntimeError("should not be called"))
        service.tw_openapi.quote = unittest.mock.MagicMock(side_effect=RuntimeError("should not be called"))
        service.tw_tpex.quote = unittest.mock.MagicMock(side_effect=RuntimeError("should not be called"))

        quote = QuoteSnapshot(
            symbol="2330",
            market="TW",
            ts=datetime.now(tz=timezone.utc),
            price=100.0,
            prev_close=99.0,
            open=99.0,
            high=101.0,
            low=98.0,
            volume=1000,
            source="fugle_ws",
            is_delayed=False,
            extra={},
        )
        idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
        snap = OhlcvSnapshot(
            symbol="2330",
            market="TW",
            interval="1d",
            tz="UTC",
            df=pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1.0, 1.0, 1.0],
                },
                index=idx,
            ),
            source="tw_openapi",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )
        with patch.object(service.tw_fugle_ws, "quote", return_value=quote):
            with patch.object(service, "_try_ohlcv_chain", return_value=snap):
                ctx, _ = service.get_tw_live_context(
                    symbol="2330",
                    yahoo_symbol="2330.TW",
                    ticks=pd.DataFrame(columns=["ts", "price", "cum_volume"]),
                    options=LiveOptions(use_yahoo=False, keep_minutes=60, exchange="tse", use_fugle_ws=True),
                )
        self.assertEqual(ctx.quote.source, "fugle_ws")
        service.tw_mis.quote.assert_not_called()
        service.tw_openapi.quote.assert_not_called()
        service.tw_tpex.quote.assert_not_called()

    def test_get_tw_live_context_falls_back_to_daily_tail_when_intraday_empty(self):
        service = MarketDataService()
        service.tw_fugle_ws.api_key = "fake-key"
        service.tw_fugle_rest.api_key = None

        quote = QuoteSnapshot(
            symbol="2330",
            market="TW",
            ts=datetime.now(tz=timezone.utc),
            price=612.0,
            prev_close=605.0,
            open=607.0,
            high=613.0,
            low=604.0,
            volume=12345,
            source="fugle_ws",
            is_delayed=False,
            extra={},
        )
        idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
        daily_snap = OhlcvSnapshot(
            symbol="2330",
            market="TW",
            interval="1d",
            tz="UTC",
            df=pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1.0, 1.0, 1.0],
                },
                index=idx,
            ),
            source="tw_openapi",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )
        intraday_empty_snap = OhlcvSnapshot(
            symbol="2330.TW",
            market="TW",
            interval="1m",
            tz="UTC",
            df=pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
            source="yahoo",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )

        with patch.object(service.tw_fugle_ws, "quote", return_value=quote):
            with patch.object(service, "_try_ohlcv_chain", return_value=daily_snap):
                with patch.object(service.yahoo, "ohlcv", return_value=intraday_empty_snap):
                    ctx, _ = service.get_tw_live_context(
                        symbol="2330",
                        yahoo_symbol="2330.TW",
                        ticks=pd.DataFrame(columns=["ts", "price", "cum_volume"]),
                        options=LiveOptions(use_yahoo=True, keep_minutes=60, exchange="tse", use_fugle_ws=True),
                    )

        self.assertFalse(ctx.intraday.empty)
        self.assertEqual(ctx.intraday_source, "tw_openapi_tail")

    def test_get_tw_live_context_prefers_fugle_rest_for_daily_when_key_exists(self):
        service = MarketDataService()
        service.tw_fugle_rest.api_key = "fake-key"
        service.tw_fugle_ws.api_key = None

        quote = QuoteSnapshot(
            symbol="0050",
            market="TW",
            ts=datetime.now(tz=timezone.utc),
            price=100.0,
            prev_close=99.0,
            open=99.0,
            high=101.0,
            low=98.0,
            volume=100,
            source="tw_mis",
            is_delayed=False,
            extra={},
        )
        idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
        daily_snap = OhlcvSnapshot(
            symbol="0050",
            market="TW",
            interval="1d",
            tz="UTC",
            df=pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1.0, 1.0, 1.0],
                },
                index=idx,
            ),
            source="tw_fugle_rest",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )

        with patch.object(service, "_try_quote_chain", return_value=(quote, ["tw_mis"], None, 0)):
            with patch.object(service, "_try_ohlcv_chain", return_value=daily_snap) as mock_daily_chain:
                with patch.object(service.yahoo, "ohlcv", side_effect=RuntimeError("skip intraday yahoo")):
                    service.get_tw_live_context(
                        symbol="0050",
                        yahoo_symbol="0050.TW",
                        ticks=pd.DataFrame(columns=["ts", "price", "cum_volume"]),
                        options=LiveOptions(use_yahoo=True, keep_minutes=60, exchange="tse", use_fugle_ws=False),
                    )
        providers = mock_daily_chain.call_args.args[0]
        provider_names = [str(getattr(p, "name", "")) for p in providers]
        self.assertGreaterEqual(len(provider_names), 1)
        self.assertEqual(provider_names[0], "tw_fugle_rest")

    def test_get_tw_live_context_prefers_fugle_rest_intraday_when_tick_sparse(self):
        service = MarketDataService()
        service.tw_fugle_ws.api_key = "fake-key"
        service.tw_fugle_rest.api_key = "fake-key"

        quote = QuoteSnapshot(
            symbol="2330",
            market="TW",
            ts=datetime.now(tz=timezone.utc),
            price=600.0,
            prev_close=590.0,
            open=595.0,
            high=610.0,
            low=585.0,
            volume=1000,
            source="fugle_ws",
            is_delayed=False,
            extra={},
        )
        daily_idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
        daily_snap = OhlcvSnapshot(
            symbol="2330",
            market="TW",
            interval="1d",
            tz="UTC",
            df=pd.DataFrame(
                {
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [1.0, 1.0],
                    "volume": [1.0, 1.0],
                },
                index=daily_idx,
            ),
            source="tw_openapi",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )
        intraday_idx = pd.date_range(datetime.now(tz=timezone.utc) - pd.Timedelta(minutes=10), periods=20, freq="1min")
        intraday_snap = OhlcvSnapshot(
            symbol="2330",
            market="TW",
            interval="1m",
            tz="UTC",
            df=pd.DataFrame(
                {
                    "open": [600.0] * len(intraday_idx),
                    "high": [601.0] * len(intraday_idx),
                    "low": [599.0] * len(intraday_idx),
                    "close": [600.5] * len(intraday_idx),
                    "volume": [100.0] * len(intraday_idx),
                },
                index=intraday_idx,
            ),
            source="tw_fugle_rest",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )

        with patch.object(service.tw_fugle_ws, "quote", return_value=quote):
            with patch.object(service, "_try_ohlcv_chain", return_value=daily_snap):
                with patch.object(service.yahoo, "ohlcv", side_effect=RuntimeError("empty yahoo")):
                    with patch.object(service.tw_fugle_rest, "ohlcv", return_value=intraday_snap):
                        ctx, _ = service.get_tw_live_context(
                            symbol="2330",
                            yahoo_symbol="2330.TW",
                            ticks=pd.DataFrame(columns=["ts", "price", "cum_volume"]),
                            options=LiveOptions(use_yahoo=True, keep_minutes=180, exchange="tse", use_fugle_ws=True),
                        )

        self.assertEqual(ctx.intraday_source, "tw_fugle_rest")
        self.assertGreaterEqual(len(ctx.intraday), 10)

    def test_get_tw_live_context_uses_now_tick_for_stale_daily_quote(self):
        service = MarketDataService()
        stale_quote = QuoteSnapshot(
            symbol="2330",
            market="TW",
            ts=datetime(2023, 1, 1, tzinfo=timezone.utc),
            price=600.0,
            prev_close=590.0,
            open=595.0,
            high=610.0,
            low=585.0,
            volume=1000,
            source="tw_openapi",
            is_delayed=True,
            extra={},
        )
        idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
        daily_snap = OhlcvSnapshot(
            symbol="2330",
            market="TW",
            interval="1d",
            tz="UTC",
            df=pd.DataFrame(
                {
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [1.0, 1.0],
                    "volume": [1.0, 1.0],
                },
                index=idx,
            ),
            source="tw_openapi",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )
        with patch.object(service, "_try_quote_chain", return_value=(stale_quote, ["tw_openapi"], None, 0)):
            with patch.object(service, "_try_ohlcv_chain", return_value=daily_snap):
                ctx, ticks = service.get_tw_live_context(
                    symbol="2330",
                    yahoo_symbol="2330.TW",
                    ticks=pd.DataFrame(columns=["ts", "price", "cum_volume"]),
                    options=LiveOptions(use_yahoo=False, keep_minutes=60, exchange="tse", use_fugle_ws=False),
                )
        self.assertFalse(ctx.intraday.empty)
        self.assertFalse(ticks.empty)
        latest = pd.to_datetime(ticks["ts"], utc=True, errors="coerce").max()
        self.assertFalse(pd.isna(latest))
        self.assertGreaterEqual(latest, pd.Timestamp(datetime.now(tz=timezone.utc) - pd.Timedelta(minutes=5)))

    def test_get_tw_live_context_uses_cached_last_good_intraday_when_sparse(self):
        service = MarketDataService()
        service.tw_fugle_ws.api_key = "fake-key"
        service.tw_fugle_rest.api_key = None
        cache_key = "tw-live:intraday1m:last-good:2330:2330.TW"
        idx = pd.date_range(datetime.now(tz=timezone.utc) - pd.Timedelta(minutes=30), periods=30, freq="1min")
        cached_df = pd.DataFrame(
            {
                "open": [600.0] * len(idx),
                "high": [601.0] * len(idx),
                "low": [599.0] * len(idx),
                "close": [600.5] * len(idx),
                "volume": [100.0] * len(idx),
            },
            index=idx,
        )
        service.cache.set(cache_key, {"df": cached_df, "source": "yahoo"}, ttl_sec=1800)

        quote = QuoteSnapshot(
            symbol="2330",
            market="TW",
            ts=datetime.now(tz=timezone.utc),
            price=600.0,
            prev_close=590.0,
            open=595.0,
            high=610.0,
            low=585.0,
            volume=1000,
            source="fugle_ws",
            is_delayed=False,
            extra={},
        )
        daily_idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
        daily_snap = OhlcvSnapshot(
            symbol="2330",
            market="TW",
            interval="1d",
            tz="UTC",
            df=pd.DataFrame(
                {
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [1.0, 1.0],
                    "volume": [1.0, 1.0],
                },
                index=daily_idx,
            ),
            source="tw_openapi",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )

        with patch.object(service.tw_fugle_ws, "quote", return_value=quote):
            with patch.object(service, "_try_ohlcv_chain", return_value=daily_snap):
                with patch.object(service.yahoo, "ohlcv", side_effect=RuntimeError("rate limit")):
                    ctx, _ = service.get_tw_live_context(
                        symbol="2330",
                        yahoo_symbol="2330.TW",
                        ticks=pd.DataFrame(columns=["ts", "price", "cum_volume"]),
                        options=LiveOptions(use_yahoo=True, keep_minutes=180, exchange="tse", use_fugle_ws=True),
                    )

        self.assertEqual(ctx.intraday_source, "cache:last_good:yahoo")
        self.assertGreaterEqual(len(ctx.intraday), 20)


if __name__ == "__main__":
    unittest.main()
