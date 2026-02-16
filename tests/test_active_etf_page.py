from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from app import (
    ACTIVE_ETF_LINE_COLORS,
    _build_symbol_line_styles,
    _build_tw_active_etf_ytd_between,
    _is_tw_active_etf,
)


class ActiveEtfPageTests(unittest.TestCase):
    def test_is_tw_active_etf_rule(self):
        self.assertTrue(_is_tw_active_etf("00993A", "安聯台灣高息成長主動式ETF"))
        self.assertFalse(_is_tw_active_etf("00993", "安聯台灣高息成長主動式ETF"))
        self.assertFalse(_is_tw_active_etf("00993A", "安聯台灣高息成長ETF"))

    def test_build_symbol_line_styles_assigns_distinct_colors(self):
        symbols = [f"00{i:03d}A" for i in range(1, 11)]
        style_map = _build_symbol_line_styles(symbols)
        self.assertEqual(len(style_map), len(symbols))

        used_colors = [style_map[s]["color"] for s in sorted(style_map.keys())]
        self.assertEqual(len(set(used_colors)), len(used_colors))
        for color in used_colors:
            self.assertIn(color, ACTIVE_ETF_LINE_COLORS)

    def test_build_tw_active_etf_ytd_between_filters_and_sorts(self):
        _build_tw_active_etf_ytd_between.clear()

        start_df = pd.DataFrame(
            [
                {"code": "00993A", "name": "安聯台灣高息成長主動式ETF", "close": 100.0},
                {"code": "00980A", "name": "野村臺灣主動式ETF", "close": 50.0},
                {"code": "00935", "name": "科技ETF", "close": 80.0},
                {"code": "00981A", "name": "統一台股主動式ETF", "close": 80.0},
            ]
        )
        end_df = pd.DataFrame(
            [
                {"code": "00993A", "name": "安聯台灣高息成長主動式ETF", "close": 120.0},
                {"code": "00980A", "name": "野村臺灣主動式ETF", "close": 45.0},
                {"code": "00935", "name": "科技ETF", "close": 150.0},
                {"code": "00981A", "name": "統一台股主動式ETF", "close": 100.0},
            ]
        )

        with patch(
            "app._fetch_twse_snapshot_with_fallback",
            side_effect=[("20260102", start_df), ("20260214", end_df)],
        ), patch("app.known_split_events", return_value=[]):
            out, start_used, end_used = _build_tw_active_etf_ytd_between("20260101", "20260216")

        self.assertEqual(start_used, "20260102")
        self.assertEqual(end_used, "20260214")
        self.assertFalse(out.empty)
        self.assertEqual(list(out["代碼"]), ["00981A", "00993A", "00980A"])
        self.assertEqual(float(out.iloc[0]["YTD報酬(%)"]), 25.0)
        self.assertEqual(float(out.iloc[1]["YTD報酬(%)"]), 20.0)
        self.assertEqual(float(out.iloc[2]["YTD報酬(%)"]), -10.0)
        self.assertNotIn("00935", list(out["代碼"]))


if __name__ == "__main__":
    unittest.main()
