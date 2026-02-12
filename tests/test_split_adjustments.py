from __future__ import annotations

import unittest

import pandas as pd

from backtest.adjustments import apply_split_adjustment, detect_split_events


class SplitAdjustmentTests(unittest.TestCase):
    def test_known_split_adjusts_0052_history(self):
        idx = pd.to_datetime(
            [
                "2025-11-18",
                "2025-11-19",
                "2025-11-20",
                "2025-11-26",
                "2025-11-27",
            ],
            utc=True,
        )
        bars = pd.DataFrame(
            {
                "open": [246.0, 247.0, 245.0, 35.1, 35.5],
                "high": [248.0, 248.5, 246.0, 35.8, 36.0],
                "low": [244.0, 246.0, 244.0, 34.8, 35.0],
                "close": [245.3, 246.5, 245.1, 35.04, 35.40],
                "volume": [1000.0, 1100.0, 1200.0, 7000.0, 7200.0],
            },
            index=idx,
        )
        out, events = apply_split_adjustment(bars, symbol="0052", market="TW", use_known=True, use_auto_detect=False)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].ratio, 1.0 / 7.0, places=8)
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2025-11-18", tz="UTC"), "close"]), 245.3 / 7.0, places=4)
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2025-11-26", tz="UTC"), "close"]), 35.04, places=4)

    def test_auto_detect_split_event(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="B", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 25.4, 25.8],
                "high": [101.0, 102.0, 103.0, 26.0, 26.2],
                "low": [99.0, 100.0, 101.0, 25.0, 25.5],
                "close": [100.0, 101.0, 102.0, 25.5, 26.0],
                "volume": [1000, 1100, 1200, 5000, 5100],
            },
            index=idx,
        )
        events = detect_split_events(bars)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].ratio, 0.25, places=8)

    def test_no_adjustment_when_disabled(self):
        idx = pd.date_range("2025-01-01", periods=3, freq="B", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [1000, 1100, 1200],
            },
            index=idx,
        )
        out, events = apply_split_adjustment(bars, symbol="0052", market="TW", use_known=False, use_auto_detect=False)
        self.assertTrue(out.equals(bars))
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
