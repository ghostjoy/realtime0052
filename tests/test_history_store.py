from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone

import pandas as pd

from market_data_types import OhlcvSnapshot
from providers.base import ProviderRequest
from storage.history_store import HistoryStore


class _Provider:
    def __init__(self, name: str):
        self.name = name


class _FakeService:
    def __init__(self, source: str = "yahoo", fail: bool = False):
        self.us_twelve = _Provider("twelvedata")
        self.yahoo = _Provider("yahoo")
        self.us_stooq = _Provider("stooq")
        self.tw_fugle_rest = _Provider("tw_fugle_rest")
        self.tw_openapi = _Provider("tw_openapi")
        self.tw_tpex = _Provider("tw_tpex")
        self.source = source
        self.fail = fail

    def _try_ohlcv_chain(self, providers, request: ProviderRequest):
        if self.fail:
            raise RuntimeError("upstream unavailable")
        idx = next(i for i, p in enumerate(providers) if p.name == self.source)
        provider = providers[idx]
        dates = pd.date_range("2024-01-01", periods=30, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            },
            index=dates,
        )
        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval="1d",
            tz="UTC",
            df=df,
            source=provider.name,
            is_delayed=provider.name != "twelvedata",
            fetched_at=datetime.now(tz=timezone.utc),
        )


class _CaptureService:
    def __init__(self):
        self.us_twelve = _Provider("twelvedata")
        self.yahoo = _Provider("yahoo")
        self.us_stooq = _Provider("stooq")
        self.tw_fugle_rest = _Provider("tw_fugle_rest")
        self.tw_fugle_rest.api_key = None
        self.tw_openapi = _Provider("tw_openapi")
        self.tw_tpex = _Provider("tw_tpex")
        self.last_request: ProviderRequest | None = None
        self.last_provider_name: str | None = None

    def _try_ohlcv_chain(self, providers, request: ProviderRequest):
        self.last_request = request
        self.last_provider_name = str(getattr(providers[0], "name", "") or "")
        start = request.start or datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = pd.date_range(start=start, periods=5, freq="B", tz="UTC")
        provider = providers[0]
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            },
            index=dates,
        )
        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval="1d",
            tz="UTC",
            df=df,
            source=provider.name,
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )


class _DupColumnService:
    def __init__(self):
        self.us_twelve = _Provider("twelvedata")
        self.yahoo = _Provider("yahoo")
        self.us_stooq = _Provider("stooq")
        self.tw_fugle_rest = _Provider("tw_fugle_rest")
        self.tw_fugle_rest.api_key = None
        self.tw_openapi = _Provider("tw_openapi")
        self.tw_tpex = _Provider("tw_tpex")

    def _try_ohlcv_chain(self, providers, request: ProviderRequest):
        dates = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
        # Simulate malformed upstream shape: duplicated "open" column.
        df = pd.DataFrame(
            data=[
                [100.0, 100.0, 101.0, 99.0, 100.5, 1000.0]
                for _ in range(len(dates))
            ],
            columns=["open", "open", "high", "low", "close", "volume"],
            index=dates,
        )
        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval="1d",
            tz="UTC",
            df=df,
            source="yahoo",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )


class HistoryStoreTests(unittest.TestCase):
    def test_sync_and_load_daily_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService(source="yahoo"))
            report = store.sync_symbol_history(symbol="TSLA", market="US")
            self.assertIsNone(report.error)
            self.assertGreater(report.rows_upserted, 0)
            self.assertEqual(report.fallback_depth, 1)

            bars = store.load_daily_bars("TSLA", "US")
            self.assertFalse(bars.empty)
            self.assertTrue({"open", "high", "low", "close", "volume", "source"}.issubset(bars.columns))

            second = store.sync_symbol_history(symbol="TSLA", market="US")
            self.assertEqual(second.rows_upserted, 0)

    def test_incremental_stale_sync_is_non_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            service = _FakeService(source="yahoo")
            store = HistoryStore(db_path=db_path, service=service)
            first = store.sync_symbol_history(symbol="TSLA", market="US")
            self.assertIsNone(first.error)
            self.assertGreater(first.rows_upserted, 0)

            store.service = _FakeService(source="yahoo", fail=True)
            second = store.sync_symbol_history(
                symbol="TSLA",
                market="US",
                end=datetime(2024, 2, 9, tzinfo=timezone.utc),
            )
            self.assertIsNone(second.error)
            self.assertTrue(second.stale)

    def test_save_and_load_universe_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            store.save_universe_snapshot(
                universe_id="TW:00935",
                symbols=["2330", "2454", " 2317 "],
                source="unit_test",
            )
            snap = store.load_universe_snapshot("TW:00935")
            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(snap.universe_id, "TW:00935")
            self.assertEqual(snap.symbols, ["2330", "2454", "2317"])
            self.assertEqual(snap.source, "unit_test")

    def test_save_and_load_latest_heatmap_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            payload_old = {"rows": [{"symbol": "2330", "excess_pct": 1.2}], "generated_at": "2026-01-01T00:00:00+00:00"}
            payload_new = {"rows": [{"symbol": "2454", "excess_pct": 2.3}], "generated_at": "2026-01-02T00:00:00+00:00"}
            store.save_heatmap_run(universe_id="TW:00935", payload=payload_old)
            store.save_heatmap_run(universe_id="TW:00935", payload=payload_new)
            latest = store.load_latest_heatmap_run("TW:00935")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.universe_id, "TW:00935")
            self.assertEqual(latest.payload.get("generated_at"), "2026-01-02T00:00:00+00:00")

    def test_save_and_load_latest_rotation_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            payload_old = {"generated_at": "2026-01-01T00:00:00+00:00", "top_n": 2}
            payload_new = {"generated_at": "2026-01-02T00:00:00+00:00", "top_n": 3}
            store.save_rotation_run(
                universe_id="TW:ROTATION:CORE6",
                run_key="run:old",
                params={"top_n": 2},
                payload=payload_old,
            )
            store.save_rotation_run(
                universe_id="TW:ROTATION:CORE6",
                run_key="run:new",
                params={"top_n": 3},
                payload=payload_new,
            )
            latest = store.load_latest_rotation_run("TW:ROTATION:CORE6")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.universe_id, "TW:ROTATION:CORE6")
            self.assertEqual(latest.run_key, "run:new")
            self.assertEqual(latest.params.get("top_n"), 3)
            self.assertEqual(latest.payload.get("generated_at"), "2026-01-02T00:00:00+00:00")

    def test_save_and_load_latest_backtest_replay_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            store.save_backtest_replay_run(
                run_key="bt:tw:0050",
                params={"strategy": "buy_hold"},
                payload={"generated_at": "2026-02-17T00:00:00+00:00", "mode": "single"},
            )
            store.save_backtest_replay_run(
                run_key="bt:tw:0050",
                params={"strategy": "sma_cross"},
                payload={"generated_at": "2026-02-18T00:00:00+00:00", "mode": "single"},
            )
            latest = store.load_latest_backtest_replay_run("bt:tw:0050")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.run_key, "bt:tw:0050")
            self.assertEqual(latest.params.get("strategy"), "sma_cross")
            self.assertEqual(latest.payload.get("generated_at"), "2026-02-18T00:00:00+00:00")

    def test_sync_history_backfills_when_start_before_local_first_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            service = _CaptureService()
            store = HistoryStore(db_path=db_path, service=service)
            first_start = datetime(2024, 1, 10, tzinfo=timezone.utc)
            end = datetime(2024, 1, 31, tzinfo=timezone.utc)

            store.sync_symbol_history(symbol="2330", market="TW", start=first_start, end=end)
            assert service.last_request is not None
            self.assertEqual(service.last_request.start.date().isoformat(), "2024-01-10")

            backfill_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            store.sync_symbol_history(symbol="2330", market="TW", start=backfill_start, end=end)
            assert service.last_request is not None
            self.assertEqual(service.last_request.start.date().isoformat(), "2024-01-01")

    def test_sync_history_prefers_fugle_for_tw_when_key_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            service = _CaptureService()
            service.tw_fugle_rest.api_key = "fake-key"
            store = HistoryStore(db_path=db_path, service=service)
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            end = datetime(2024, 1, 31, tzinfo=timezone.utc)

            report = store.sync_symbol_history(symbol="0050", market="TW", start=start, end=end)
            self.assertEqual(report.source, "tw_fugle_rest")
            self.assertEqual(service.last_provider_name, "tw_fugle_rest")

    def test_sync_history_normalizes_duplicate_ohlcv_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_DupColumnService())
            report = store.sync_symbol_history(symbol="^N225", market="US")
            self.assertIsNone(report.error)
            self.assertGreater(report.rows_upserted, 0)
            bars = store.load_daily_bars("^N225", "US")
            self.assertFalse(bars.empty)
            self.assertTrue({"open", "high", "low", "close", "volume"}.issubset(bars.columns))

    def test_symbol_metadata_upsert_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            written = store.upsert_symbol_metadata(
                [
                    {"symbol": "2330", "market": "TW", "name": "台積電", "exchange": "TW", "industry": "24"},
                    {"symbol": "AAPL", "market": "US", "name": "Apple", "exchange": "US", "currency": "USD"},
                ]
            )
            self.assertEqual(written, 2)

            tw_meta = store.load_symbol_metadata(["2330", "2454"], market="TW")
            self.assertEqual(tw_meta["2330"]["name"], "台積電")
            self.assertEqual(tw_meta["2330"]["industry"], "24")
            self.assertNotIn("2454", tw_meta)

            us_symbols = store.list_symbols("US")
            self.assertIn("AAPL", us_symbols)

    def test_bootstrap_run_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            run_id = store.start_bootstrap_run(
                scope="manual:both",
                params={"years": 5, "parallel": True},
            )
            store.finish_bootstrap_run(
                run_id,
                status="completed",
                total_symbols=10,
                synced_symbols=8,
                failed_symbols=2,
                summary={"scope": "both"},
                error=None,
            )
            latest = store.load_latest_bootstrap_run()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.run_id, run_id)
            self.assertEqual(latest.status, "completed")
            self.assertEqual(latest.total_symbols, 10)
            self.assertEqual(latest.synced_symbols, 8)
            self.assertEqual(latest.failed_symbols, 2)
            self.assertEqual(latest.summary.get("scope"), "both")

    def test_save_and_load_intraday_ticks_with_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            now = datetime.now(tz=timezone.utc)
            old_ts = now - pd.Timedelta(days=5)
            rows = [
                {"ts": old_ts.isoformat(), "price": 100.0, "cum_volume": 10.0, "source": "fugle_ws"},
                {"ts": now.isoformat(), "price": 110.0, "cum_volume": 20.0, "source": "fugle_ws"},
            ]
            written = store.save_intraday_ticks("2330", "TW", rows, retain_days=2)
            self.assertEqual(written, 2)
            loaded = store.load_intraday_ticks(
                "2330",
                "TW",
                start=now - pd.Timedelta(days=30),
                end=now + pd.Timedelta(days=1),
            )
            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(float(loaded["price"].iloc[0]), 110.0)
            self.assertEqual(str(loaded["source"].iloc[0]), "fugle_ws")


if __name__ == "__main__":
    unittest.main()
