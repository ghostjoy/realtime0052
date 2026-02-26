from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from storage.duck_store import DuckHistoryStore


class _NoopService:
    def set_metadata_store(self, store):
        self.store = store


class _Provider:
    def __init__(self, name: str, api_key: str | None = None):
        self.name = name
        self.api_key = api_key


class _CaptureSyncService:
    def __init__(self):
        self.us_twelve = _Provider("twelvedata")
        self.yahoo = _Provider("yahoo")
        self.us_stooq = _Provider("stooq")
        self.tw_fugle_rest = _Provider("tw_fugle_rest", api_key="fake-key")
        self.tw_openapi = _Provider("tw_openapi")
        self.tw_tpex = _Provider("tw_tpex")
        self.last_request = None
        self.last_provider_name: str | None = None

    def set_metadata_store(self, store):
        self.store = store

    def _try_ohlcv_chain(self, providers, request):
        self.last_request = request
        self.last_provider_name = str(getattr(providers[0], "name", "") or "")
        idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            },
            index=idx,
        )
        return SimpleNamespace(df=df, source=self.last_provider_name)


def _seed_legacy_sqlite(path: Path):
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS instruments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                name TEXT,
                currency TEXT,
                timezone TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                UNIQUE(symbol, market)
            );
            CREATE TABLE IF NOT EXISTS daily_bars (
                instrument_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                adj_close REAL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY(instrument_id, date)
            );
            """
        )
        conn.execute(
            "INSERT INTO instruments(id, symbol, market, name, currency, timezone, active) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "2330", "TW", "台積電", "TWD", "Asia/Taipei", 1),
        )
        conn.executemany(
            (
                "INSERT INTO daily_bars(instrument_id, date, open, high, low, close, volume, adj_close, source, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            [
                (
                    1,
                    "2024-01-02",
                    100.0,
                    101.0,
                    99.0,
                    100.5,
                    1000.0,
                    100.5,
                    "unit",
                    "2024-01-02T00:00:00+00:00",
                ),
                (
                    1,
                    "2024-01-03",
                    101.0,
                    102.0,
                    100.0,
                    101.5,
                    1100.0,
                    101.5,
                    "unit",
                    "2024-01-03T00:00:00+00:00",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


class DuckStoreTests(unittest.TestCase):
    def test_default_legacy_sqlite_path_prefers_env(self):
        dummy = DuckHistoryStore.__new__(DuckHistoryStore)
        dummy.db_path = Path("market_history.duckdb")
        with patch.dict(
            "os.environ", {"REALTIME0052_DB_PATH": "/tmp/legacy_from_env.sqlite3"}, clear=False
        ):
            out = dummy._default_legacy_sqlite_path()
        self.assertEqual(str(out), "/tmp/legacy_from_env.sqlite3")

    def test_auto_migrates_daily_bars_from_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sqlite_path = tmp_path / "legacy.sqlite3"
            duckdb_path = tmp_path / "history.duckdb"
            parquet_root = tmp_path / "parquet"
            _seed_legacy_sqlite(sqlite_path)

            store = DuckHistoryStore(
                db_path=str(duckdb_path),
                parquet_root=str(parquet_root),
                service=_NoopService(),
                legacy_sqlite_path=str(sqlite_path),
                auto_migrate_legacy_sqlite=True,
            )

            bars = store.load_daily_bars(symbol="2330", market="TW")
            self.assertEqual(len(bars), 2)
            self.assertEqual(bars.index[0].strftime("%Y-%m-%d"), "2024-01-02")
            self.assertEqual(bars.index[-1].strftime("%Y-%m-%d"), "2024-01-03")
            self.assertEqual(store.list_symbols("TW"), ["2330"])

    def test_intraday_ticks_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            written = store.save_intraday_ticks(
                symbol="2330",
                market="TW",
                ticks=[
                    {
                        "ts": "2024-01-03T01:00:00+00:00",
                        "price": 100.0,
                        "cum_volume": 10,
                        "source": "unit",
                    },
                    {
                        "ts": "2024-01-03T01:00:00+00:00",
                        "price": 101.0,
                        "cum_volume": 11,
                        "source": "unit",
                    },
                    {
                        "ts": "2024-01-03T01:01:00+00:00",
                        "price": 102.0,
                        "cum_volume": 15,
                        "source": "unit",
                    },
                ],
            )
            self.assertEqual(written, 2)
            loaded = store.load_intraday_ticks("2330", "TW")
            self.assertEqual(len(loaded), 2)
            self.assertAlmostEqual(float(loaded["price"].iloc[0]), 101.0, places=6)
            self.assertAlmostEqual(float(loaded["price"].iloc[1]), 102.0, places=6)

    def test_queue_daily_bars_writeback_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )

            idx = pd.date_range("2025-01-01", periods=3, freq="B", tz="UTC")
            frame = pd.DataFrame({"close": [100.0, 101.0, 102.5]}, index=idx)
            queued = store.queue_daily_bars_writeback(
                symbol="^GSPC", market="US", bars=frame, source="yfinance"
            )
            self.assertTrue(queued)
            self.assertTrue(store.flush_writeback_queue(timeout_sec=3.0))

            bars = store.load_daily_bars(symbol="^GSPC", market="US")
            self.assertEqual(len(bars), 3)
            self.assertTrue((bars["source"] == "yfinance").all())
            self.assertAlmostEqual(float(bars["close"].iloc[-1]), 102.5, places=6)

    def test_queue_daily_bars_writeback_coalesces_same_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )

            idx_a = pd.date_range("2025-01-01", periods=2, freq="B", tz="UTC")
            idx_b = pd.date_range("2025-01-02", periods=2, freq="B", tz="UTC")
            frame_a = pd.DataFrame({"close": [100.0, 101.0]}, index=idx_a)
            frame_b = pd.DataFrame({"close": [102.0, 103.0]}, index=idx_b)

            queued_a = store.queue_daily_bars_writeback(
                symbol="^GSPC", market="US", bars=frame_a, source="first"
            )
            queued_b = store.queue_daily_bars_writeback(
                symbol="^GSPC", market="US", bars=frame_b, source="second"
            )
            self.assertTrue(queued_a)
            self.assertTrue(queued_b)
            self.assertTrue(store.flush_writeback_queue(timeout_sec=3.0))

            bars = store.load_daily_bars(symbol="^GSPC", market="US")
            self.assertEqual(len(bars), 3)
            self.assertEqual(bars.index[-1].date().isoformat(), "2025-01-03")
            self.assertAlmostEqual(float(bars["close"].iloc[-1]), 103.0, places=6)
            self.assertEqual(str(bars["source"].iloc[0]), "first")
            self.assertEqual(str(bars["source"].iloc[-1]), "second")

    def test_daily_delta_compaction_rolls_up_base_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            store._daily_delta_compact_threshold = 2

            idx_a = pd.date_range("2025-01-01", periods=2, freq="B", tz="UTC")
            idx_b = pd.date_range("2025-01-03", periods=2, freq="B", tz="UTC")
            frame_a = pd.DataFrame({"close": [100.0, 101.0]}, index=idx_a)
            frame_b = pd.DataFrame({"close": [102.0, 103.0]}, index=idx_b)
            self.assertTrue(
                store.queue_daily_bars_writeback(
                    symbol="0050", market="TW", bars=frame_a, source="unit"
                )
            )
            self.assertTrue(store.flush_writeback_queue(timeout_sec=3.0))
            self.assertTrue(
                store.queue_daily_bars_writeback(
                    symbol="0050", market="TW", bars=frame_b, source="unit"
                )
            )
            self.assertTrue(store.flush_writeback_queue(timeout_sec=3.0))

            base_path = store._daily_symbol_path("0050", "TW")
            delta_dir = store._daily_delta_dir("0050", "TW")
            self.assertTrue(base_path.exists())
            self.assertEqual(len(list(delta_dir.glob("*.parquet"))), 0)
            bars = store.load_daily_bars("0050", "TW")
            self.assertEqual(len(bars), 4)

    def test_intraday_delta_compaction_rolls_up_base_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            store._intraday_delta_compact_threshold = 2

            count_a = store.save_intraday_ticks(
                symbol="2330",
                market="TW",
                ticks=[{"ts": "2025-01-02T01:00:00+00:00", "price": 100.0, "cum_volume": 10}],
            )
            count_b = store.save_intraday_ticks(
                symbol="2330",
                market="TW",
                ticks=[{"ts": "2025-01-02T01:01:00+00:00", "price": 101.0, "cum_volume": 15}],
            )
            self.assertEqual(count_a, 1)
            self.assertEqual(count_b, 1)

            base_path = store._intraday_symbol_path("2330", "TW")
            delta_dir = store._intraday_delta_dir("2330", "TW")
            self.assertTrue(base_path.exists())
            self.assertEqual(len(list(delta_dir.glob("*.parquet"))), 0)
            ticks = store.load_intraday_ticks("2330", "TW")
            self.assertEqual(len(ticks), 2)

    def test_load_daily_coverage_and_sync_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )

            idx = pd.date_range("2025-01-01", periods=3, freq="B", tz="UTC")
            frame = pd.DataFrame({"close": [100.0, 101.0, 102.5]}, index=idx)
            queued = store.queue_daily_bars_writeback(
                symbol="^GSPC", market="US", bars=frame, source="yfinance"
            )
            self.assertTrue(queued)
            self.assertTrue(store.flush_writeback_queue(timeout_sec=3.0))

            coverage = store.load_daily_coverage(
                symbol="^GSPC",
                market="US",
                start=idx[1].to_pydatetime(),
                end=idx[2].to_pydatetime(),
            )
            self.assertEqual(int(coverage["row_count"]), 2)
            self.assertEqual(coverage["first_date"].date().isoformat(), idx[1].date().isoformat())
            self.assertEqual(coverage["last_date"].date().isoformat(), idx[2].date().isoformat())

            sync_state = store.load_sync_state(symbol="^GSPC", market="US")
            self.assertIsNotNone(sync_state)
            assert sync_state is not None
            self.assertEqual(sync_state["last_source"], "yfinance")
            self.assertEqual(
                sync_state["last_success_date"].date().isoformat(), idx[2].date().isoformat()
            )

    def test_save_and_load_tw_etf_aum_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            dates = ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
            for idx, trade_date in enumerate(dates):
                saved = store.save_tw_etf_aum_snapshot(
                    rows=[
                        {"etf_code": "0050", "etf_name": "元大台灣50", "aum_billion": 1000 + idx},
                        {"etf_code": "0056", "etf_name": "高股息", "aum_billion": 800 + idx},
                    ],
                    trade_date=trade_date,
                    keep_days=3,
                )
                self.assertEqual(saved, 2)

            hist = store.load_tw_etf_aum_history(etf_codes=["0050", "0056"], keep_days=3)
            self.assertFalse(hist.empty)
            self.assertEqual(set(hist["etf_code"].unique()), {"0050", "0056"})
            self.assertEqual(int((hist["etf_code"] == "0050").sum()), 3)
            self.assertEqual(int((hist["etf_code"] == "0056").sum()), 3)
            self.assertNotIn("2026-01-02", set(hist["trade_date"].astype(str)))

    def test_save_tw_etf_aum_history_keep_days_zero_keeps_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            dates = ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
            for idx, trade_date in enumerate(dates):
                saved = store.save_tw_etf_aum_snapshot(
                    rows=[
                        {"etf_code": "0050", "etf_name": "元大台灣50", "aum_billion": 1000 + idx},
                    ],
                    trade_date=trade_date,
                    keep_days=0,
                )
                self.assertEqual(saved, 1)

            hist_all = store.load_tw_etf_aum_history(etf_codes=["0050"], keep_days=0)
            self.assertEqual(int((hist_all["etf_code"] == "0050").sum()), 4)
            self.assertIn("2026-01-02", set(hist_all["trade_date"].astype(str)))

    def test_clear_tw_etf_aum_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            store.save_tw_etf_aum_snapshot(
                rows=[{"etf_code": "0050", "etf_name": "元大台灣50", "aum_billion": 1000.0}],
                trade_date="2026-01-05",
                keep_days=0,
            )
            removed = store.clear_tw_etf_aum_history()
            self.assertGreaterEqual(int(removed), 1)
            hist = store.load_tw_etf_aum_history(etf_codes=["0050"], keep_days=0)
            self.assertTrue(hist.empty)

    def test_upsert_and_list_heatmap_hub_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            store.upsert_heatmap_hub_entry(
                etf_code="00935", etf_name="野村臺灣新科技50", opened=True
            )
            store.upsert_heatmap_hub_entry(
                etf_code="00935", etf_name="野村臺灣新科技50", opened=True, pin_as_card=True
            )
            store.upsert_heatmap_hub_entry(etf_code="0050", etf_name="元大台灣50", opened=False)

            all_rows = store.list_heatmap_hub_entries()
            self.assertEqual(len(all_rows), 2)
            top = all_rows[0]
            self.assertEqual(top.etf_code, "00935")
            self.assertTrue(top.pin_as_card)
            self.assertEqual(top.open_count, 2)

            pinned_rows = store.list_heatmap_hub_entries(pinned_only=True)
            self.assertEqual(len(pinned_rows), 1)
            self.assertEqual(pinned_rows[0].etf_code, "00935")

            ok = store.set_heatmap_hub_pin(etf_code="0050", pin_as_card=True)
            self.assertTrue(ok)
            pinned_rows = store.list_heatmap_hub_entries(pinned_only=True)
            self.assertEqual({row.etf_code for row in pinned_rows}, {"0050", "00935"})

    def test_sync_tw_index_uses_yahoo_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            service = _CaptureSyncService()
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=service,
                auto_migrate_legacy_sqlite=False,
            )
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            end = datetime(2024, 1, 31, tzinfo=timezone.utc)

            report = store.sync_symbol_history(symbol="^TWII", market="TW", start=start, end=end)
            self.assertEqual(report.source, "yahoo")
            self.assertEqual(service.last_provider_name, "yahoo")
            self.assertEqual(str(getattr(service.last_request, "symbol", "")), "^TWII")


if __name__ == "__main__":
    unittest.main()
