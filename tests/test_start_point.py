from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backtest.start_point import apply_start_to_bars_map, resolve_start_index


def _bars(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(100, 100 + n),
            "high": range(101, 101 + n),
            "low": range(99, 99 + n),
            "close": range(100, 100 + n),
            "volume": [1000] * n,
        },
        index=idx,
    )


class StartPointTests(unittest.TestCase):
    def test_resolve_by_k(self):
        bars = _bars()
        idx, dt = resolve_start_index(bars, mode="指定第幾根K", start_k=10)
        self.assertEqual(idx, 10)
        self.assertEqual(dt, bars.index[10])

    def test_resolve_by_date(self):
        bars = _bars()
        idx, dt = resolve_start_index(bars, mode="指定日期", start_date=date(2024, 2, 1))
        self.assertGreaterEqual(dt, pd.Timestamp("2024-02-01", tz="UTC"))
        self.assertEqual(dt, bars.index[idx])

    def test_apply_to_map(self):
        bars_map = {"A": _bars(120), "B": _bars(90)}
        out, info = apply_start_to_bars_map(bars_map, mode="指定第幾根K", start_k=20)
        self.assertEqual(set(out.keys()), {"A", "B"})
        self.assertTrue((info["status"] == "OK").all())
        self.assertEqual(len(out["A"]), 100)
        self.assertEqual(len(out["B"]), 70)


if __name__ == "__main__":
    unittest.main()
