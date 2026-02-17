from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from market_data_types import SyncPlanResult
from services.bootstrap_loader import run_incremental_refresh, run_market_data_bootstrap
from storage.history_store import HistoryStore


class _NoopService:
    def set_metadata_store(self, store):
        self.store = store


class _Report:
    def __init__(self, error: str | None = None):
        self.error = error


class BootstrapLoaderTests(unittest.TestCase):
    @patch("services.bootstrap_loader.sync_symbols_if_needed")
    @patch("services.bootstrap_loader.fetch_tw_symbol_metadata")
    def test_run_market_data_bootstrap_records_run(self, mock_fetch_tw, mock_sync):
        mock_fetch_tw.return_value = (
            [
                {
                    "symbol": "2330",
                    "market": "TW",
                    "name": "台積電",
                    "exchange": "TW",
                    "industry": "24",
                    "currency": "TWD",
                    "source": "unit_test",
                }
            ],
            [],
        )

        def _fake_sync(*, symbols, **kwargs):
            reports = {symbol: _Report() for symbol in symbols}
            plan = SyncPlanResult(
                synced_symbols=list(symbols),
                skipped_symbols=[],
                issues=[],
                source_chain=["unit"],
            )
            return reports, plan

        mock_sync.side_effect = _fake_sync

        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_NoopService())
            summary = run_market_data_bootstrap(
                store=store,
                scope="both",
                years=5,
                parallel=False,
                max_workers=1,
                us_symbols=["AAPL"],
            )

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["tw_symbols"], 1)
            self.assertEqual(summary["us_symbols"], 1)
            self.assertEqual(summary["synced_success"], 2)

            latest = store.load_latest_bootstrap_run()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.status, "completed")
            self.assertEqual(latest.total_symbols, 2)

            tw_meta = store.load_symbol_metadata(["2330"], market="TW")
            self.assertEqual(tw_meta["2330"]["name"], "台積電")

    def test_run_incremental_refresh_without_seed_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_NoopService())
            summary = run_incremental_refresh(store=store, years=5, parallel=False, max_workers=1)
            self.assertEqual(summary["status"], "skipped_no_symbols")
            self.assertEqual(summary["total_symbols"], 0)


if __name__ == "__main__":
    unittest.main()
