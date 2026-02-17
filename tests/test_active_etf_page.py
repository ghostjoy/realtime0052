from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from app import (
    ACTIVE_ETF_LINE_COLORS,
    BACKTEST_REPLAY_SCHEMA_VERSION,
    _benchmark_candidates_tw,
    _decorate_tw_etf_top10_ytd_table,
    _build_data_health,
    _build_snapshot_health,
    _build_replay_source_hash,
    _build_symbol_line_styles,
    _build_tw_active_etf_ytd_between,
    _classify_issue_level,
    _is_tw_active_etf,
    _load_cached_backtest_payload,
    _load_tw_benchmark_bars,
    _snapshot_fallback_depth,
)


class ActiveEtfPageTests(unittest.TestCase):
    def test_benchmark_candidates_tw_modes(self):
        self.assertEqual(_benchmark_candidates_tw("twii"), ["^TWII"])
        self.assertEqual(_benchmark_candidates_tw("twii", allow_twii_fallback=True), ["^TWII", "0050", "006208"])
        self.assertEqual(_benchmark_candidates_tw("0050"), ["0050"])
        self.assertEqual(_benchmark_candidates_tw("unknown"), ["^TWII"])
        self.assertEqual(_benchmark_candidates_tw("unknown", allow_twii_fallback=True), ["^TWII", "0050", "006208"])

    def test_load_tw_benchmark_bars_fallback_chain(self):
        idx = pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
        bars_0050 = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1200.0],
            },
            index=idx,
        )

        class _FakeStore:
            def __init__(self):
                self.sync_calls: list[str] = []

            def sync_symbol_history(self, symbol, market, start=None, end=None):
                self.sync_calls.append(str(symbol))
                if str(symbol) == "^TWII":
                    return SimpleNamespace(error="network down")
                return SimpleNamespace(error=None)

            def load_daily_bars(self, symbol, market, start=None, end=None):
                if str(symbol) == "0050":
                    return bars_0050
                return pd.DataFrame()

        store = _FakeStore()
        with patch("app.apply_split_adjustment", side_effect=lambda bars, **kwargs: (bars, [])):
            out, symbol, issues = _load_tw_benchmark_bars(
                store=store,
                choice="twii",
                start_dt=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_dt=datetime(2026, 2, 1, tzinfo=timezone.utc),
                sync_first=False,
                allow_twii_fallback=True,
                min_rows=2,
            )

        self.assertEqual(symbol, "0050")
        self.assertEqual(len(out), 2)
        self.assertIn("^TWII: network down", issues)
        self.assertEqual(store.sync_calls, ["^TWII"])

    def test_classify_issue_level(self):
        self.assertEqual(_classify_issue_level("無法建立 ETF 排行：資料缺失"), "error")
        self.assertEqual(_classify_issue_level("部分 ETF 同步失敗，已盡量使用本地可用資料"), "warning")
        self.assertEqual(_classify_issue_level("快取已更新"), "info")

    def test_build_data_health_and_replay_hash(self):
        health = _build_data_health(
            as_of="20260217",
            data_sources=["twse_mi_index"],
            source_chain=["start:20260102", "end:20260217"],
            degraded=True,
            fallback_depth=1,
            notes="target=2026-02-18",
        )
        self.assertEqual(health.as_of, "2026-02-17")
        self.assertEqual(health.fallback_depth, 1)
        self.assertTrue(bool(health.degraded))
        self.assertEqual(list(health.data_sources), ["twse_mi_index"])

        base = {
            "market": "TW",
            "symbols": ["0050"],
            "strategy": "buy_hold",
            "schema_version": BACKTEST_REPLAY_SCHEMA_VERSION,
        }
        h1 = _build_replay_source_hash(base)
        h2 = _build_replay_source_hash(dict(base))
        self.assertEqual(h1, h2)
        mutated = dict(base)
        mutated["strategy"] = "sma_cross"
        self.assertNotEqual(h1, _build_replay_source_hash(mutated))

    def test_snapshot_health_helpers(self):
        self.assertEqual(_snapshot_fallback_depth("20260217", "20260217"), 0)
        self.assertEqual(_snapshot_fallback_depth("20260217", "20260215"), 2)
        health = _build_snapshot_health(
            start_used="20260102",
            end_used="20260215",
            target_yyyymmdd="20260217",
        )
        self.assertEqual(health.as_of, "2026-02-15")
        self.assertEqual(health.fallback_depth, 2)
        self.assertTrue(bool(health.degraded))

    def test_load_cached_backtest_payload_rejects_incompatible_signature(self):
        cached = SimpleNamespace(
            params={"schema_version": BACKTEST_REPLAY_SCHEMA_VERSION - 1, "source_hash": "abc"},
            payload={"mode": "single"},
        )

        class _FakeStore:
            def load_latest_backtest_replay_run(self, run_key):
                self._run_key = run_key
                return cached

        payload, message = _load_cached_backtest_payload(
            store=_FakeStore(),
            run_key="bt:tw:0050",
            expected_schema=BACKTEST_REPLAY_SCHEMA_VERSION,
            expected_hash="def",
        )
        self.assertIsNone(payload)
        self.assertIn("重新計算", message)

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
