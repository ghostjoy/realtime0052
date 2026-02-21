from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from storage.duck_store import DuckHistoryStore


class _NoopService:
    def set_metadata_store(self, store):
        self.store = store


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
                (1, "2024-01-02", 100.0, 101.0, 99.0, 100.5, 1000.0, 100.5, "unit", "2024-01-02T00:00:00+00:00"),
                (1, "2024-01-03", 101.0, 102.0, 100.0, 101.5, 1100.0, 101.5, "unit", "2024-01-03T00:00:00+00:00"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


class DuckStoreTests(unittest.TestCase):
    def test_default_legacy_sqlite_path_prefers_env(self):
        dummy = DuckHistoryStore.__new__(DuckHistoryStore)
        dummy.db_path = Path("market_history.duckdb")
        with patch.dict("os.environ", {"REALTIME0052_DB_PATH": "/tmp/legacy_from_env.sqlite3"}, clear=False):
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
                    {"ts": "2024-01-03T01:00:00+00:00", "price": 100.0, "cum_volume": 10, "source": "unit"},
                    {"ts": "2024-01-03T01:00:00+00:00", "price": 101.0, "cum_volume": 11, "source": "unit"},
                    {"ts": "2024-01-03T01:01:00+00:00", "price": 102.0, "cum_volume": 15, "source": "unit"},
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
            queued = store.queue_daily_bars_writeback(symbol="^GSPC", market="US", bars=frame, source="yfinance")
            self.assertTrue(queued)
            self.assertTrue(store.flush_writeback_queue(timeout_sec=3.0))

            bars = store.load_daily_bars(symbol="^GSPC", market="US")
            self.assertEqual(len(bars), 3)
            self.assertTrue((bars["source"] == "yfinance").all())
            self.assertAlmostEqual(float(bars["close"].iloc[-1]), 102.5, places=6)


if __name__ == "__main__":
    unittest.main()
