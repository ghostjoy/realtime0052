from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
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
        self.tw_tpex_etf = _Provider("tw_tpex_etf")
        self.tw_tpex = _Provider("tw_tpex")
        self.last_request = None
        self.last_provider_name: str | None = None
        self.last_provider_chain: list[str] = []

    def set_metadata_store(self, store):
        self.store = store

    def _try_ohlcv_chain(self, providers, request):
        self.last_request = request
        self.last_provider_chain = [str(getattr(provider, "name", "") or "") for provider in providers]
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
            self.assertIn("vwap", bars.columns)
            self.assertAlmostEqual(float(bars["vwap"].iloc[0]), 100.1666666667, places=6)
            self.assertEqual(store.list_symbols("TW"), ["2330"])

    def test_save_and_load_latest_tw_etf_super_export_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            first_run_id = store.save_tw_etf_super_export_run(
                ytd_start="20260101",
                ytd_end="20260309",
                compare_start="20250101",
                compare_end="20251231",
                trade_date_anchor="20260309",
                output_path="/tmp/tw_etf_super_export_20260309.csv",
                row_count=100,
                column_count=28,
                payload={"frame": {"rows": [{"代碼": "0050"}]}},
            )
            second_run_id = store.save_tw_etf_super_export_run(
                ytd_start="20260101",
                ytd_end="20260310",
                compare_start="20250101",
                compare_end="20251231",
                trade_date_anchor="20260310",
                output_path="/tmp/tw_etf_super_export_20260310.csv",
                row_count=123,
                column_count=29,
                payload={"csv_sha256": "abc123", "frame": {"rows": [{"代碼": "0052"}]}},
            )

            self.assertNotEqual(first_run_id, second_run_id)
            latest = store.load_latest_tw_etf_super_export_run()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.run_id, second_run_id)
            self.assertEqual(latest.trade_date_anchor, "20260310")
            self.assertEqual(latest.output_path, "/tmp/tw_etf_super_export_20260310.csv")
            self.assertEqual(latest.row_count, 123)
            self.assertEqual(latest.column_count, 29)
            self.assertEqual(latest.payload.get("csv_sha256"), "abc123")

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
            frame = pd.DataFrame(
                {
                    "open": [99.0, 100.0, 101.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [98.0, 99.0, 100.0],
                    "close": [100.0, 101.0, 102.5],
                    "volume": [10.0, 20.0, 30.0],
                },
                index=idx,
            )
            queued = store.queue_daily_bars_writeback(
                symbol="^GSPC", market="US", bars=frame, source="yfinance"
            )
            self.assertTrue(queued)
            self.assertTrue(store.flush_writeback_queue(timeout_sec=3.0))

            bars = store.load_daily_bars(symbol="^GSPC", market="US")
            self.assertEqual(len(bars), 3)
            self.assertTrue((bars["source"] == "yfinance").all())
            self.assertAlmostEqual(float(bars["close"].iloc[-1]), 102.5, places=6)
            self.assertIn("vwap", bars.columns)
            self.assertAlmostEqual(float(bars["vwap"].iloc[0]), 99.6666666667, places=6)

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

    def test_tw_etf_aum_history_defaults_to_keep_all(self):
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
                )
                self.assertEqual(saved, 1)

            hist_all = store.load_tw_etf_aum_history(etf_codes=["0050"])
            self.assertEqual(int((hist_all["etf_code"] == "0050").sum()), 4)
            self.assertIn("2026-01-02", set(hist_all["trade_date"].astype(str)))

    def test_client_visit_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            self.assertTrue(
                store.upsert_client_visit(
                    session_id="sess_1",
                    ip_address="203.0.113.10",
                    forwarded_for="203.0.113.10, 10.0.0.1",
                    user_agent="pytest",
                    last_page="留言板",
                )
            )
            self.assertTrue(
                store.upsert_client_visit(
                    session_id="sess_1",
                    ip_address="203.0.113.10",
                    user_agent="pytest",
                    last_page="台股 ETF 全類型總表",
                )
            )
            visits = store.list_recent_client_visits(limit=10)
            self.assertEqual(len(visits), 1)
            self.assertEqual(visits[0].ip_address, "203.0.113.10")
            self.assertEqual(visits[0].visit_count, 2)
            self.assertEqual(visits[0].last_page, "台股 ETF 全類型總表")

    def test_message_board_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            root_id = store.save_message_board_entry(
                author_name="訪客A",
                body="第一則留言",
                ip_address="203.0.113.10",
            )
            reply_id = store.save_message_board_entry(
                author_name="訪客B",
                body="這是回覆",
                parent_message_id=root_id,
                ip_address="198.51.100.8",
            )
            rows = store.list_message_board_entries(limit=10)
            ids = {row.message_id for row in rows}
            self.assertIn(root_id, ids)
            self.assertIn(reply_id, ids)
            reply_row = next(row for row in rows if row.message_id == reply_id)
            self.assertEqual(reply_row.parent_message_id, root_id)
            self.assertTrue(
                store.update_message_board_entry(message_id=root_id, body="第一則留言（已修改）")
            )
            rows_after_update = store.list_message_board_entries(limit=10)
            root_row = next(row for row in rows_after_update if row.message_id == root_id)
            self.assertEqual(root_row.body, "第一則留言（已修改）")
            deleted = store.delete_message_board_entry(message_id=root_id)
            self.assertEqual(deleted, 2)
            rows_after_delete = store.list_message_board_entries(limit=10)
            self.assertEqual(rows_after_delete, [])

    def test_notebook_entry_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            note_id = store.save_notebook_entry(
                note_id="default",
                title="研究筆記",
                body="# 測試筆記\n\n內容 A",
            )
            first = store.load_notebook_entry(note_id="default")
            self.assertEqual(note_id, "default")
            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first.note_id, "default")
            self.assertEqual(first.title, "研究筆記")
            self.assertEqual(first.body, "# 測試筆記\n\n內容 A")

            store.save_notebook_entry(
                note_id="default",
                title="研究筆記-更新",
                body="# 測試筆記\n\n內容 B",
            )
            second = store.load_notebook_entry(note_id="default")
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(second.title, "研究筆記-更新")
            self.assertEqual(second.body, "# 測試筆記\n\n內容 B")

    def test_list_create_and_delete_notebook_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            first_id = store.create_notebook_entry(title="第一篇", body="內容 A")
            second_id = store.create_notebook_entry(title="第二篇", body="內容 B")
            store.save_notebook_entry(note_id=first_id, title="第一篇-更新", body="內容 A2")

            rows = store.list_notebook_entries(limit=10)
            self.assertEqual([row.note_id for row in rows], [first_id, second_id])
            self.assertEqual(rows[0].title, "第一篇-更新")
            self.assertTrue(store.delete_notebook_entry(note_id=second_id))
            remain = store.list_notebook_entries(limit=10)
            self.assertEqual([row.note_id for row in remain], [first_id])

    def test_notebook_schema_upgrade_adds_title_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "history.duckdb"
            conn = duckdb.connect(str(db_path))
            try:
                conn.execute(
                    """
                    CREATE TABLE notebook_entries (
                        id BIGINT,
                        note_id VARCHAR,
                        body VARCHAR,
                        created_at VARCHAR,
                        updated_at VARCHAR,
                        UNIQUE(note_id)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO notebook_entries(id, note_id, body, created_at, updated_at)
                    VALUES (1, 'legacy', '# 舊筆記', '2026-03-01T00:00:00+00:00', '2026-03-01T00:00:00+00:00')
                    """
                )
            finally:
                conn.close()

            store = DuckHistoryStore(
                db_path=str(db_path),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            legacy = store.load_notebook_entry(note_id="legacy")
            self.assertIsNotNone(legacy)
            assert legacy is not None
            self.assertEqual(legacy.title, "未命名筆記")
            store.save_notebook_entry(note_id="legacy", title="補標題", body="# 新內容")
            upgraded = store.load_notebook_entry(note_id="legacy")
            self.assertIsNotNone(upgraded)
            assert upgraded is not None
            self.assertEqual(upgraded.title, "補標題")

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

    def test_save_and_load_tw_etf_daily_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            saved = store.save_tw_etf_daily_market(
                rows=[
                    {
                        "trade_date": "2026-03-06",
                        "etf_code": "0050",
                        "etf_name": "元大台灣50",
                        "trade_value": 100000000.0,
                        "trade_volume": 123456.0,
                        "trade_count": 789,
                        "open": 75.0,
                        "high": 76.0,
                        "low": 74.5,
                        "close": 75.5,
                        "change": -0.5,
                        "source": "unit_test",
                    },
                    {
                        "trade_date": "2026-03-06",
                        "etf_code": "0056",
                        "etf_name": "元大高股息",
                        "trade_value": 50000000.0,
                        "trade_volume": 654321.0,
                        "trade_count": 321,
                        "open": 38.0,
                        "high": 38.5,
                        "low": 37.8,
                        "close": 38.2,
                        "change": 0.2,
                        "source": "unit_test",
                    },
                ],
                trade_date="2026-03-06",
            )
            self.assertEqual(saved, 2)

            coverage = store.load_tw_etf_daily_market_coverage()
            self.assertEqual(int(coverage["row_count"]), 2)
            self.assertEqual(int(coverage["trade_date_count"]), 1)
            self.assertEqual(int(coverage["symbol_count"]), 2)

            frame = store.load_tw_etf_daily_market(start="2026-03-06", end="2026-03-06")
            self.assertEqual(len(frame), 2)
            self.assertEqual(float(frame.loc[frame["etf_code"] == "0050", "close"].iloc[0]), 75.5)

    def test_save_and_load_tw_etf_mis_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            saved = store.save_tw_etf_mis_daily(
                rows=[
                    {
                        "trade_date": "2026-03-06",
                        "etf_code": "0050",
                        "etf_name": "元大台灣50",
                        "issued_units": 1350000000.0,
                        "creation_redemption_diff": 1000000.0,
                        "market_price": 182.1,
                        "estimated_nav": 181.62,
                        "premium_discount_pct": 0.26,
                        "previous_nav": 180.55,
                        "reference_url": "https://example.com/nav",
                        "updated_at": "2026-03-06T14:30:00+08:00",
                        "source": "unit_test",
                    },
                    {
                        "trade_date": "2026-03-06",
                        "etf_code": "0056",
                        "etf_name": "元大高股息",
                        "issued_units": 2460000000.0,
                        "creation_redemption_diff": -500000.0,
                        "market_price": 40.2,
                        "estimated_nav": 40.05,
                        "premium_discount_pct": 0.37,
                        "previous_nav": 39.88,
                        "reference_url": "https://example.com/nav2",
                        "updated_at": "2026-03-06T14:30:00+08:00",
                        "source": "unit_test",
                    },
                ],
                trade_date="2026-03-06",
            )
            self.assertEqual(saved, 2)

            coverage = store.load_tw_etf_mis_daily_coverage()
            self.assertEqual(int(coverage["row_count"]), 2)
            self.assertEqual(int(coverage["trade_date_count"]), 1)
            self.assertEqual(int(coverage["symbol_count"]), 2)

            frame = store.load_tw_etf_mis_daily(start="2026-03-06", end="2026-03-06")
            self.assertEqual(len(frame), 2)
            self.assertEqual(
                float(frame.loc[frame["etf_code"] == "0050", "estimated_nav"].iloc[0]),
                181.62,
            )

    def test_save_and_load_tw_etf_margin_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            saved = store.save_tw_etf_margin_daily(
                rows=[
                    {
                        "trade_date": "2026-03-18",
                        "etf_code": "0050",
                        "etf_name": "元大台灣50",
                        "margin_buy": 826,
                        "margin_sell": 945,
                        "margin_cash_redemption": 4,
                        "margin_prev_balance": 12753,
                        "margin_balance": 12630,
                        "margin_next_limit": 4480625,
                        "short_buy": 22,
                        "short_sell": 89,
                        "short_stock_redemption": 0,
                        "short_prev_balance": 745,
                        "short_balance": 812,
                        "short_next_limit": 4480625,
                        "offset_amount": 21,
                        "note": "X",
                        "source": "unit_test",
                    },
                    {
                        "trade_date": "2026-03-18",
                        "etf_code": "0056",
                        "etf_name": "元大高股息",
                        "margin_buy": 139,
                        "margin_sell": 46,
                        "margin_cash_redemption": 57,
                        "margin_prev_balance": 3669,
                        "margin_balance": 3705,
                        "margin_next_limit": 3444008,
                        "short_buy": 0,
                        "short_sell": 12,
                        "short_stock_redemption": 0,
                        "short_prev_balance": 184,
                        "short_balance": 196,
                        "short_next_limit": 3444008,
                        "offset_amount": 0,
                        "note": "",
                        "source": "unit_test",
                    },
                ],
                trade_date="2026-03-18",
            )
            self.assertEqual(saved, 2)

            coverage = store.load_tw_etf_margin_daily_coverage()
            self.assertEqual(int(coverage["row_count"]), 2)
            self.assertEqual(int(coverage["trade_date_count"]), 1)
            self.assertEqual(int(coverage["symbol_count"]), 2)

            frame = store.load_tw_etf_margin_daily(start="2026-03-18", end="2026-03-18")
            self.assertEqual(len(frame), 2)
            self.assertEqual(
                int(frame.loc[frame["etf_code"] == "0050", "margin_balance"].iloc[0]),
                12630,
            )
            self.assertEqual(
                int(frame.loc[frame["etf_code"] == "0050", "short_balance"].iloc[0]),
                812,
            )

    def test_save_universe_snapshot_can_overwrite_same_universe_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            store.save_universe_snapshot(
                universe_id="TW:0053",
                symbols=["2330", "2317"],
                source="unit_test_a",
            )
            store.save_universe_snapshot(
                universe_id="TW:0053",
                symbols=["2454"],
                source="unit_test_b",
            )

            snap = store.load_universe_snapshot("TW:0053")
            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(snap.universe_id, "TW:0053")
            self.assertEqual(snap.symbols, ["2454"])
            self.assertEqual(snap.source, "unit_test_b")

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

    def test_connect_ctx_does_not_mask_original_error_on_no_active_transaction_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )

            calls: list[str] = []

            class _FakeConn:
                def commit(self):
                    calls.append("commit")

                def rollback(self):
                    calls.append("rollback")
                    raise RuntimeError(
                        "TransactionContext Error: cannot rollback - no transaction is active"
                    )

                def close(self):
                    calls.append("close")

            store._connect = lambda: _FakeConn()  # type: ignore[method-assign]

            with self.assertRaisesRegex(ValueError, "boom"):
                with store._connect_ctx():
                    raise ValueError("boom")

            self.assertIn("rollback", calls)
            self.assertIn("close", calls)

    def test_connect_retries_retryable_duckdb_lock_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )

            class _FakeConn:
                def execute(self, sql: str):
                    return self

                def close(self):
                    return None

            lock_error = RuntimeError(
                'IO Error: Could not set lock on file "history.duckdb": '
                "Conflicting lock is held in other process"
            )
            fake_conn = _FakeConn()
            sleep_calls: list[float] = []
            with (
                patch(
                    "storage.duck_store.duckdb.connect",
                    side_effect=[lock_error, lock_error, fake_conn],
                ) as connect_mock,
                patch("storage.duck_store.time.sleep", side_effect=lambda value: sleep_calls.append(float(value))),
            ):
                conn = store._connect()
                try:
                    self.assertTrue(hasattr(conn, "execute"))
                finally:
                    conn.close()

            self.assertEqual(int(connect_mock.call_count), 3)
            self.assertEqual(sleep_calls, [1.0, 2.0])

    def test_market_snapshot_roundtrip_and_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )

            store.save_market_snapshot(
                dataset_key="twse_mi_index_allbut0999",
                market="TW",
                symbol="ALL",
                interval="1d",
                source="unit",
                asof="20260301",
                payload={"rows": [{"code": "0050", "name": "元大台灣50", "close": 100.0}]},
                freshness_sec=10,
                quality_score=0.9,
                stale=False,
            )
            store.save_market_snapshot(
                dataset_key="twse_mi_index_allbut0999",
                market="TW",
                symbol="ALL",
                interval="1d",
                source="unit",
                asof="20260302",
                payload={"rows": [{"code": "0050", "name": "元大台灣50", "close": 101.0}]},
                freshness_sec=20,
                quality_score=0.8,
                stale=True,
            )

            latest = store.load_latest_market_snapshot(
                dataset_key="twse_mi_index_allbut0999",
                market="TW",
                symbol="ALL",
                interval="1d",
            )
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(str(latest["source"]), "unit")
            self.assertTrue(bool(latest["stale"]))
            payload = latest.get("payload")
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            rows = payload.get("rows")
            self.assertIsInstance(rows, list)
            assert isinstance(rows, list)
            self.assertEqual(float(rows[0]["close"]), 101.0)

            window = store.load_market_snapshot_window(
                dataset_key="twse_mi_index_allbut0999",
                market="TW",
                symbol="ALL",
                interval="1d",
                asof_start="20260301",
                asof_end="20260302",
                limit=8,
            )
            self.assertEqual(len(window), 2)

    def test_market_snapshot_purge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            store.save_market_snapshot(
                dataset_key="unit_snapshot",
                market="TW",
                symbol="TEST",
                interval="1d",
                source="unit",
                asof=datetime(2000, 1, 1, tzinfo=timezone.utc),
                payload={"ok": True},
            )
            removed = store.purge_market_snapshots(dataset_key="unit_snapshot", keep_days=1)
            self.assertGreaterEqual(int(removed), 1)
            latest = store.load_latest_market_snapshot(
                dataset_key="unit_snapshot",
                market="TW",
                symbol="TEST",
                interval="1d",
            )
            self.assertIsNone(latest)

    def test_market_snapshot_stats_housekeeping_and_maintenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DuckHistoryStore(
                db_path=str(tmp_path / "history.duckdb"),
                parquet_root=str(tmp_path / "parquet"),
                service=_NoopService(),
                auto_migrate_legacy_sqlite=False,
            )
            now_utc = datetime.now(tz=timezone.utc)
            store.save_market_snapshot(
                dataset_key="live_quote",
                market="TW",
                symbol="0050",
                interval="quote",
                source="unit",
                asof=now_utc,
                payload={"price": 100.0},
                freshness_sec=5,
                stale=False,
            )
            store.save_market_snapshot(
                dataset_key="live_quote",
                market="TW",
                symbol="0050",
                interval="quote",
                source="unit",
                asof=datetime(2000, 1, 1, tzinfo=timezone.utc),
                payload={"price": 80.0},
                freshness_sec=5,
                stale=True,
            )
            store.save_market_snapshot(
                dataset_key="twse_mi_index_allbut0999",
                market="TW",
                symbol="ALL",
                interval="1d",
                source="unit",
                asof=datetime(2000, 1, 1, tzinfo=timezone.utc),
                payload={"rows": []},
                freshness_sec=0,
                stale=True,
            )

            stats = store.get_market_snapshot_stats(limit=20)
            self.assertTrue(any(str(item.get("dataset_key")) == "live_quote" for item in stats))
            self.assertTrue(
                any(str(item.get("dataset_key")) == "twse_mi_index_allbut0999" for item in stats)
            )

            report = store.run_market_snapshot_housekeeping(
                policy={"live_quote": 1},
                default_keep_days=50000,
            )
            self.assertGreaterEqual(int(report.get("removed_total", 0) or 0), 1)
            removed_map = report.get("removed_by_dataset", {})
            self.assertIsInstance(removed_map, dict)
            assert isinstance(removed_map, dict)
            self.assertGreaterEqual(int(removed_map.get("live_quote", 0) or 0), 1)
            self.assertEqual(int(removed_map.get("twse_mi_index_allbut0999", 0) or 0), 0)

            latest = store.load_latest_market_snapshot(
                dataset_key="live_quote",
                market="TW",
                symbol="0050",
                interval="quote",
            )
            self.assertIsNotNone(latest)

            maintenance = store.run_duckdb_maintenance(checkpoint=True, vacuum=False)
            actions = maintenance.get("actions")
            self.assertIsInstance(actions, list)
            assert isinstance(actions, list)
            self.assertIn("CHECKPOINT", actions)

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

    def test_sync_otc_prefers_fugle_and_skips_tw_openapi(self):
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

            report = store.sync_symbol_history(symbol="009815", market="OTC", start=start, end=end)
            self.assertEqual(report.source, "tw_fugle_rest")
            self.assertEqual(service.last_provider_name, "tw_fugle_rest")
            self.assertEqual(str(getattr(service.last_request, "market", "")), "OTC")
            self.assertEqual(
                service.last_provider_chain, ["tw_fugle_rest", "tw_tpex_etf", "tw_tpex", "yahoo"]
            )


if __name__ == "__main__":
    unittest.main()
