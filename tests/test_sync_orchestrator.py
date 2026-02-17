from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from typing import Optional

import pandas as pd

from services.sync_orchestrator import bars_need_backfill, sync_symbols_if_needed


def _bars(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0] * periods,
            "high": [101.0] * periods,
            "low": [99.0] * periods,
            "close": [100.5] * periods,
            "volume": [1000.0] * periods,
        },
        index=idx,
    )


class _FakeStore:
    def __init__(self, bars_map: dict[str, pd.DataFrame], errors: Optional[dict[str, str]] = None):
        self._bars_map = bars_map
        self._errors = errors or {}
        self.sync_calls: list[str] = []

    def load_daily_bars(self, symbol, market, start=None, end=None):
        return self._bars_map.get(str(symbol).upper(), pd.DataFrame())

    def sync_symbol_history(self, symbol, market, start=None, end=None):
        code = str(symbol).upper()
        self.sync_calls.append(code)
        err = self._errors.get(code, "")
        return SimpleNamespace(error=err, rows_upserted=5, source="unit", fallback_depth=0)


class SyncOrchestratorTests(unittest.TestCase):
    def test_bars_need_backfill(self):
        start = datetime(2026, 1, 2, tzinfo=timezone.utc)
        end = datetime(2026, 1, 8, tzinfo=timezone.utc)
        covered = _bars("2026-01-01", 10)
        self.assertFalse(bars_need_backfill(covered, start=start, end=end))
        self.assertTrue(bars_need_backfill(pd.DataFrame(), start=start, end=end))

    def test_sync_symbols_if_needed_backfill(self):
        start = datetime(2026, 1, 2, tzinfo=timezone.utc)
        end = datetime(2026, 1, 8, tzinfo=timezone.utc)
        store = _FakeStore(
            bars_map={
                "AAA": _bars("2026-01-01", 10),
                "BBB": pd.DataFrame(),
            },
            errors={"BBB": "timeout"},
        )
        reports, plan = sync_symbols_if_needed(
            store=store,
            market="TW",
            symbols=["aaa", "bbb"],
            start=start,
            end=end,
            parallel=False,
            mode="backfill",
        )
        self.assertEqual(plan.synced_symbols, ["BBB"])
        self.assertEqual(plan.skipped_symbols, ["AAA"])
        self.assertIn("BBB: timeout", plan.issues)
        self.assertIn("BBB", reports)
        self.assertEqual(store.sync_calls, ["BBB"])

    def test_sync_symbols_if_needed_min_rows(self):
        start = datetime(2026, 1, 2, tzinfo=timezone.utc)
        end = datetime(2026, 1, 8, tzinfo=timezone.utc)
        store = _FakeStore(
            bars_map={
                "AAA": _bars("2026-01-01", 1),
                "BBB": _bars("2026-01-01", 5),
            }
        )
        reports, plan = sync_symbols_if_needed(
            store=store,
            market="TW",
            symbols=["AAA", "BBB"],
            start=start,
            end=end,
            parallel=False,
            mode="min_rows",
            min_rows=2,
        )
        self.assertEqual(plan.synced_symbols, ["AAA"])
        self.assertEqual(plan.skipped_symbols, ["BBB"])
        self.assertIn("AAA", reports)


if __name__ == "__main__":
    unittest.main()
