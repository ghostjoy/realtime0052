from __future__ import annotations

import unittest

import pandas as pd

import app


class TotalReturnAdjustmentTests(unittest.TestCase):
    def test_apply_total_return_adjustment_uses_adj_close(self):
        idx = pd.date_range("2026-01-01", periods=4, freq="B", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": [100.0, 102.0, 104.0, 106.0],
                "high": [101.0, 103.0, 105.0, 107.0],
                "low": [99.0, 101.0, 103.0, 105.0],
                "close": [100.0, 102.0, 104.0, 106.0],
                "adj_close": [95.0, 98.0, 101.0, 104.0],
                "volume": [1000.0, 1000.0, 1000.0, 1000.0],
            },
            index=idx,
        )
        adjusted, info = app._apply_total_return_adjustment(bars, min_coverage_ratio=0.5)
        self.assertTrue(bool(info.get("applied")))
        self.assertAlmostEqual(float(adjusted["close"].iloc[0]), 95.0, places=6)
        self.assertAlmostEqual(float(adjusted["close"].iloc[-1]), 104.0, places=6)

    def test_apply_total_return_adjustment_skips_without_adj_close(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="B", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1000.0, 1000.0],
            },
            index=idx,
        )
        adjusted, info = app._apply_total_return_adjustment(bars)
        self.assertFalse(bool(info.get("applied")))
        pd.testing.assert_frame_equal(adjusted, bars)


if __name__ == "__main__":
    unittest.main()
