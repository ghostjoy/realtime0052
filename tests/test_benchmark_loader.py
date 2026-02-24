from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from services.benchmark_loader import (
    benchmark_candidates_tw,
    load_tw_benchmark_bars,
    load_tw_benchmark_close,
)


def _sample_bars(rows: int = 2) -> pd.DataFrame:
    idx = pd.date_range("2026-01-02", periods=rows, freq="D", tz="UTC")
    close = [100.0 + i for i in range(rows)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [x + 1.0 for x in close],
            "low": [x - 1.0 for x in close],
            "close": close,
            "volume": [1000.0] * rows,
        },
        index=idx,
    )


class BenchmarkLoaderTests(unittest.TestCase):
    def test_benchmark_candidates_tw_modes(self):
        self.assertEqual(benchmark_candidates_tw("twii"), ["^TWII"])
        self.assertEqual(
            benchmark_candidates_tw("twii", allow_twii_fallback=True), ["^TWII", "0050", "006208"]
        )
        self.assertEqual(benchmark_candidates_tw("0050"), ["0050"])
        self.assertEqual(benchmark_candidates_tw("unknown"), ["^TWII"])

    def test_load_tw_benchmark_bars_fallback_and_sync_issues(self):
        class _FakeStore:
            def __init__(self):
                self.sync_calls: list[str] = []

            def sync_symbol_history(self, symbol, market, start=None, end=None):
                self.sync_calls.append(str(symbol))
                if str(symbol) == "^TWII":
                    return SimpleNamespace(error="timeout")
                return SimpleNamespace(error=None)

            def load_daily_bars(self, symbol, market, start=None, end=None):
                if str(symbol) == "0050":
                    return _sample_bars(rows=3)
                return pd.DataFrame()

        store = _FakeStore()
        with patch(
            "services.benchmark_loader.apply_split_adjustment",
            side_effect=lambda bars, **kwargs: (bars, []),
        ):
            result = load_tw_benchmark_bars(
                store=store,
                choice="twii",
                start_dt=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_dt=datetime(2026, 2, 1, tzinfo=timezone.utc),
                sync_first=False,
                allow_twii_fallback=True,
                min_rows=2,
            )

        self.assertEqual(result.symbol_used, "0050")
        self.assertEqual(len(result.bars), 3)
        self.assertEqual(len(result.close), 3)
        self.assertIn("^TWII: timeout", result.sync_issues)
        self.assertEqual(store.sync_calls, ["^TWII"])
        self.assertTrue(result.source_chain and result.source_chain[0].startswith("^TWII:"))
        self.assertEqual(result.candidates, ["^TWII", "0050", "006208"])

    def test_load_tw_benchmark_close_requires_two_rows(self):
        class _FakeStore:
            def sync_symbol_history(self, symbol, market, start=None, end=None):
                return SimpleNamespace(error=None)

            def load_daily_bars(self, symbol, market, start=None, end=None):
                return _sample_bars(rows=1)

        with patch(
            "services.benchmark_loader.apply_split_adjustment",
            side_effect=lambda bars, **kwargs: (bars, []),
        ):
            result = load_tw_benchmark_close(
                store=_FakeStore(),
                choice="0050",
                start_dt=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_dt=datetime(2026, 2, 1, tzinfo=timezone.utc),
                sync_first=False,
                allow_twii_fallback=False,
            )

        self.assertTrue(result.close.empty)
        self.assertEqual(result.symbol_used, "")
        self.assertEqual(result.candidates, ["0050"])


if __name__ == "__main__":
    unittest.main()
