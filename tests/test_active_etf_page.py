from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from app import (
    ACTIVE_ETF_LINE_COLORS,
    _decorate_tw_etf_top10_ytd_table,
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
                {"code": "00980A", "name": "主動野村臺灣優選", "close": 50.0},
                {"code": "00981A", "name": "主動統一台股增長", "close": 80.0},
                {"code": "00935", "name": "科技ETF", "close": 80.0},
            ]
        )
        end_df = pd.DataFrame(
            [
                {"code": "00993A", "name": "主動安聯台灣", "close": 120.0},
                {"code": "00980A", "name": "主動野村臺灣優選", "close": 45.0},
                {"code": "00935", "name": "科技ETF", "close": 150.0},
                {"code": "00981A", "name": "主動統一台股增長", "close": 100.0},
            ]
        )

        def _bars(rows: list[tuple[str, float]]) -> pd.DataFrame:
            idx = pd.to_datetime([d for d, _ in rows], utc=True)
            closes = [float(v) for _, v in rows]
            return pd.DataFrame(
                {
                    "open": closes,
                    "high": closes,
                    "low": closes,
                    "close": closes,
                    "volume": [0.0] * len(closes),
                },
                index=idx,
            )

        class _FakeStore:
            def __init__(self):
                self._bars_map = {
                    "00981A": _bars([("2026-01-02", 80.0), ("2026-02-14", 100.0)]),
                    "00993A": _bars([("2026-02-03", 100.0), ("2026-02-14", 120.0)]),
                    "00980A": _bars([("2026-01-02", 50.0), ("2026-02-14", 45.0)]),
                }

            def load_daily_bars(self, symbol, market, start=None, end=None):
                return self._bars_map.get(str(symbol), pd.DataFrame())

            def sync_symbol_history(self, symbol, market, start=None, end=None):
                return SimpleNamespace(error=None)

        with patch(
            "app._fetch_twse_snapshot_with_fallback",
            side_effect=[("20251231", start_df), ("20260214", end_df)],
        ), patch("app._history_store", return_value=_FakeStore()), patch("app.known_split_events", return_value=[]):
            out, start_used, end_used = _build_tw_active_etf_ytd_between("20260101", "20260216")

        self.assertEqual(start_used, "20251231")
        self.assertEqual(end_used, "20260214")
        self.assertFalse(out.empty)
        self.assertEqual(list(out["代碼"]), ["00981A", "00993A", "00980A"])
        self.assertEqual(list(out["績效起算日"]), ["20260102", "20260203", "20260102"])
        self.assertEqual(float(out.iloc[0]["YTD報酬(%)"]), 25.0)
        self.assertEqual(float(out.iloc[1]["YTD報酬(%)"]), 20.0)
        self.assertEqual(float(out.iloc[2]["YTD報酬(%)"]), -10.0)
        self.assertNotIn("00935", list(out["代碼"]))

    def test_decorate_top10_ytd_table_with_benchmark_row(self):
        top10 = pd.DataFrame(
            [
                {
                    "排名": 1,
                    "代碼": "0050",
                    "ETF": "元大台灣50",
                    "類型": "市值型",
                    "期初收盤": 100.0,
                    "復權期初": 100.0,
                    "期末收盤": 112.0,
                    "復權事件": "—",
                    "區間報酬(%)": 12.0,
                },
                {
                    "排名": 2,
                    "代碼": "0056",
                    "ETF": "元大高股息",
                    "類型": "股利型",
                    "期初收盤": 30.0,
                    "復權期初": 30.0,
                    "期末收盤": 32.4,
                    "復權事件": "—",
                    "區間報酬(%)": 8.0,
                },
            ]
        )
        out = _decorate_tw_etf_top10_ytd_table(
            top10,
            y2025_map={"0050": 18.5, "0056": 10.25},
            market_return_pct=5.0,
            market_2025_return_pct=9.0,
            benchmark_code="^TWII",
            end_used="20260214",
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(str(out.iloc[0]["ETF"]), "台股大盤")
        self.assertEqual(str(out.iloc[0]["排名"]), "—")
        self.assertEqual(float(out.iloc[0]["YTD報酬(%)"]), 5.0)
        self.assertEqual(float(out.iloc[1]["贏輸台股大盤(%)"]), 7.0)
        self.assertIn("2025績效(%)", out.columns)
        self.assertIn("YTD報酬(%)", out.columns)


if __name__ == "__main__":
    unittest.main()
