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
        self.tw_openapi = _Provider("tw_openapi")
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


if __name__ == "__main__":
    unittest.main()
