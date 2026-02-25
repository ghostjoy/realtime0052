from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs

import pandas as pd
import plotly.graph_objects as go

from app import (
    ACTIVE_ETF_LINE_COLORS,
    BACKTEST_REPLAY_SCHEMA_VERSION,
    _apply_unified_benchmark_hover,
    _attach_tw_etf_aum_column,
    _attach_tw_etf_management_fee_column,
    _benchmark_candidates_tw,
    _build_consensus_representative_between,
    _build_data_health,
    _build_replay_source_hash,
    _build_snapshot_health,
    _build_symbol_line_styles,
    _build_tw_active_etf_ytd_between,
    _build_tw_etf_all_types_performance_table,
    _build_tw_etf_aum_history_wide,
    _build_tw_etf_top10_between,
    _build_two_etf_aggressive_picks,
    _classify_issue_level,
    _classify_tw_etf,
    _compute_jaccard_pct,
    _compute_tw_equal_weight_compare_payload,
    _compute_tw_etf_aum_alert_mask,
    _consensus_threshold_candidates,
    _consume_heatmap_drilldown_query,
    _decorate_dataframe_backtest_links,
    _decorate_tw_etf_aum_history_links,
    _decorate_tw_etf_name_heatmap_links,
    _decorate_tw_etf_top10_ytd_table,
    _dynamic_heatmap_page_renderers,
    _fill_unresolved_tw_names,
    _format_weight_pct_label,
    _infer_market_target_from_symbols,
    _is_tw_active_etf,
    _load_cached_backtest_payload,
    _load_tw_benchmark_bars,
    _recent_twse_trading_days,
    _render_heatmap_constituent_intro_sections,
    _resolve_tw_symbol_names,
    _snapshot_fallback_depth,
)
from ui.helpers import (
    build_heatmap_drill_url,
    normalize_heatmap_etf_code,
    parse_drill_symbol,
    strip_symbol_label_token,
)


class ActiveEtfPageTests(unittest.TestCase):
    def test_resolve_tw_symbol_names_uses_full_rows_fallback(self):
        class _FakeService:
            def get_tw_symbol_names(self, symbols):
                return {str(s): str(s) for s in symbols}

        out = _resolve_tw_symbol_names(
            service=_FakeService(),
            symbols=["2330", "2454"],
            full_rows=[
                {"symbol": "2330.TW", "tw_code": "2330", "name": "台積電"},
                {"symbol": "2454.TW", "tw_code": "2454", "name": "聯發科"},
            ],
        )
        self.assertEqual(out.get("2330"), "台積電")
        self.assertEqual(out.get("2454"), "聯發科")

    def test_fill_unresolved_tw_names_rehydrates_cached_code_names(self):
        class _FakeService:
            def get_tw_symbol_names(self, symbols):
                # Simulate unresolved upstream response.
                return {str(s): str(s) for s in symbols}

        frame = pd.DataFrame(
            [
                {"symbol": "2330", "name": "2330", "excess_pct": 1.2},
                {"symbol": "2454", "name": "", "excess_pct": 0.8},
            ]
        )
        out = _fill_unresolved_tw_names(
            frame=frame,
            service=_FakeService(),
            full_rows=[
                {"symbol": "2330.TW", "tw_code": "2330", "name": "台積電"},
                {"symbol": "2454.TW", "tw_code": "2454", "name": "聯發科"},
            ],
        )
        self.assertEqual(str(out.loc[0, "name"]), "台積電")
        self.assertEqual(str(out.loc[1, "name"]), "聯發科")

    def test_benchmark_candidates_tw_modes(self):
        self.assertEqual(_benchmark_candidates_tw("twii"), ["^TWII"])
        self.assertEqual(
            _benchmark_candidates_tw("twii", allow_twii_fallback=True), ["^TWII", "0050", "006208"]
        )
        self.assertEqual(_benchmark_candidates_tw("0050"), ["0050"])
        self.assertEqual(_benchmark_candidates_tw("unknown"), ["^TWII"])
        self.assertEqual(
            _benchmark_candidates_tw("unknown", allow_twii_fallback=True),
            ["^TWII", "0050", "006208"],
        )

    def test_infer_market_target_from_symbols(self):
        with patch("app._infer_tw_symbol_exchanges", return_value={"8069": "OTC", "4123": "OTC"}):
            self.assertEqual(_infer_market_target_from_symbols(["8069", "4123"]), "OTC")

        with patch("app._infer_tw_symbol_exchanges", return_value={"2330": "TW", "2454": "TW"}):
            self.assertEqual(_infer_market_target_from_symbols(["2330", "2454"]), "TW")

        with patch("app._infer_tw_symbol_exchanges", return_value={"8069": "OTC", "2330": "TW"}):
            self.assertEqual(_infer_market_target_from_symbols(["8069", "2330"]), "TW")

        self.assertEqual(_infer_market_target_from_symbols(["AAPL", "MSFT"]), "US")
        self.assertEqual(_infer_market_target_from_symbols(["SPY", "^GSPC"]), "US")
        self.assertEqual(_infer_market_target_from_symbols(["2330.TW"]), "TW")
        self.assertEqual(_infer_market_target_from_symbols(["8069.TWO"]), "OTC")
        self.assertIsNone(_infer_market_target_from_symbols(["8069", "AAPL"]))
        self.assertIsNone(_infer_market_target_from_symbols([]))

    def testparse_drill_symbol(self):
        self.assertEqual(parse_drill_symbol("0050 元大台灣50"), ("0050", "TW"))
        self.assertEqual(parse_drill_symbol("8069.TWO"), ("8069", "OTC"))
        self.assertEqual(parse_drill_symbol("AAPL Apple"), ("AAPL", "US"))
        self.assertEqual(parse_drill_symbol("—"), ("", ""))

    def test_decorate_dataframe_backtest_links(self):
        source = pd.DataFrame(
            [
                {"代碼": "0050 元大台灣50", "市場": "台股上市(TWSE)"},
                {"代碼": "8069", "市場": "台股上櫃(OTC)"},
                {"代碼": "AAPL", "市場": "US"},
                {"代碼": "—", "市場": "TW"},
            ]
        )
        out, cfg = _decorate_dataframe_backtest_links(source)
        self.assertIn("代碼", cfg)
        self.assertTrue(str(out.iloc[0]["代碼"]).startswith("?bt_symbol=0050&bt_market=TW"))
        self.assertTrue(str(out.iloc[1]["代碼"]).startswith("?bt_symbol=8069&bt_market=OTC"))
        self.assertTrue(str(out.iloc[2]["代碼"]).startswith("?bt_symbol=AAPL&bt_market=US"))
        self.assertTrue(pd.isna(out.iloc[3]["代碼"]))

    def testnormalize_heatmap_etf_code(self):
        self.assertEqual(normalize_heatmap_etf_code("00935"), "00935")
        self.assertEqual(normalize_heatmap_etf_code("00993a"), "00993A")
        self.assertEqual(normalize_heatmap_etf_code("AAPL"), "")
        self.assertEqual(normalize_heatmap_etf_code(""), "")

    def testbuild_heatmap_drill_url(self):
        url = build_heatmap_drill_url("00935", "野村臺灣新科技50")
        self.assertIn("hm_etf=00935", url)
        self.assertIn("hm_open=1", url)
        self.assertIn("hm_src=all_types_table", url)

    def test_decorate_tw_etf_name_heatmap_links(self):
        source = pd.DataFrame(
            [
                {"代碼": "00935", "ETF": "野村臺灣新科技50", "類型": "科技型"},
                {"代碼": "0050", "ETF": "元大台灣50", "類型": "市值型"},
            ]
        )
        out, cfg = _decorate_tw_etf_name_heatmap_links(source)
        self.assertIn("ETF", cfg)
        self.assertTrue(str(out.iloc[0]["ETF"]).startswith("?hm_etf=00935&hm_name="))
        self.assertIn("hm_open=1", str(out.iloc[0]["ETF"]))

    def test_render_heatmap_constituent_intro_sections_generic_etf(self):
        fake_service = SimpleNamespace()
        with (
            patch("app.st.markdown") as markdown_mock,
            patch("app._render_tw_constituent_intro_table") as tw_intro_mock,
            patch("app._render_00910_constituent_intro_table") as intro_00910_mock,
        ):
            _render_heatmap_constituent_intro_sections(
                etf_code="00735",
                snapshot_symbols=["00735", "2330"],
                service=fake_service,  # type: ignore[arg-type]
                full_rows_00910=[],
            )

        intro_00910_mock.assert_not_called()
        tw_intro_mock.assert_called_once_with(
            etf_code="00735",
            symbols=["00735", "2330"],
            service=fake_service,
        )
        markdown_texts = [str(call.args[0]) for call in markdown_mock.call_args_list]
        self.assertEqual(markdown_texts, ["---", "#### 成分股公司簡介"])

    def test_render_heatmap_constituent_intro_sections_for_00910(self):
        fake_service = SimpleNamespace()
        full_rows = [{"symbol": "AAPL.US", "market": "US"}]
        with (
            patch("app.st.markdown") as markdown_mock,
            patch("app._render_tw_constituent_intro_table") as tw_intro_mock,
            patch("app._render_00910_constituent_intro_table") as intro_00910_mock,
        ):
            _render_heatmap_constituent_intro_sections(
                etf_code="00910",
                snapshot_symbols=["2330", "2454"],
                service=fake_service,  # type: ignore[arg-type]
                full_rows_00910=full_rows,
            )

        intro_00910_mock.assert_called_once_with(service=fake_service, full_rows=full_rows)
        tw_intro_mock.assert_called_once_with(
            etf_code="00910",
            symbols=["2330", "2454"],
            service=fake_service,
        )
        markdown_texts = [str(call.args[0]) for call in markdown_mock.call_args_list]
        self.assertEqual(
            markdown_texts,
            [
                "---",
                "#### 00910 全球成分股公司簡介",
                "#### 成分股公司簡介",
            ],
        )

    def test_render_heatmap_constituent_intro_sections_skip_when_no_symbols(self):
        fake_service = SimpleNamespace()
        with (
            patch("app.st.markdown") as markdown_mock,
            patch("app._render_tw_constituent_intro_table") as tw_intro_mock,
            patch("app._render_00910_constituent_intro_table") as intro_00910_mock,
        ):
            _render_heatmap_constituent_intro_sections(
                etf_code="00735",
                snapshot_symbols=[],
                service=fake_service,  # type: ignore[arg-type]
                full_rows_00910=[],
            )

        markdown_mock.assert_not_called()
        tw_intro_mock.assert_not_called()
        intro_00910_mock.assert_not_called()

    def test_heatmap_drilldown_query_routes_to_dynamic_heatmap_renderer(self):
        url = build_heatmap_drill_url("00735", "國泰臺韓科技", src="all_types_table")
        self.assertTrue(url.startswith("?"))
        query = parse_qs(url.lstrip("?"))

        def _query_value(name: str) -> str:
            values = query.get(name, [])
            return str(values[0]) if values else ""

        old_active_page = None
        old_active_payload = None
        has_old_active_page = False
        has_old_active_payload = False

        import app

        if "active_page" in app.st.session_state:
            has_old_active_page = True
            old_active_page = app.st.session_state.get("active_page")
        if "heatmap_hub_active_etf" in app.st.session_state:
            has_old_active_payload = True
            old_active_payload = app.st.session_state.get("heatmap_hub_active_etf")

        try:
            with (
                patch("app._query_param_first", side_effect=_query_value),
                patch("app._upsert_heatmap_hub_entry") as upsert_mock,
                patch("app._clear_query_params") as clear_mock,
            ):
                _consume_heatmap_drilldown_query()

            self.assertEqual(str(app.st.session_state.get("active_page", "")), "00735 熱力圖")
            active_payload = app.st.session_state.get("heatmap_hub_active_etf")
            self.assertIsInstance(active_payload, dict)
            assert isinstance(active_payload, dict)
            self.assertEqual(str(active_payload.get("code", "")), "00735")
            self.assertEqual(str(active_payload.get("name", "")), "國泰臺韓科技")

            upsert_mock.assert_called_once_with(
                etf_code="00735",
                etf_name="國泰臺韓科技",
                opened=True,
            )
            clear_mock.assert_called_once()

            with patch("app._load_heatmap_hub_entries", return_value=[]):
                renderers = _dynamic_heatmap_page_renderers()
                self.assertNotIn("ETF熱力圖:00735", renderers)
        finally:
            if has_old_active_page:
                app.st.session_state["active_page"] = old_active_page
            else:
                app.st.session_state.pop("active_page", None)
            if has_old_active_payload:
                app.st.session_state["heatmap_hub_active_etf"] = old_active_payload
            else:
                app.st.session_state.pop("heatmap_hub_active_etf", None)

    def test_heatmap_drilldown_query_blocks_00993a(self):
        url = build_heatmap_drill_url("00993A", "安聯台灣主動", src="all_types_table")
        query = parse_qs(url.lstrip("?"))

        def _query_value(name: str) -> str:
            values = query.get(name, [])
            return str(values[0]) if values else ""

        with (
            patch("app._query_param_first", side_effect=_query_value),
            patch("app._upsert_heatmap_hub_entry") as upsert_mock,
            patch("app._clear_query_params") as clear_mock,
            patch("app.st.warning") as warning_mock,
        ):
            _consume_heatmap_drilldown_query()

        upsert_mock.assert_not_called()
        clear_mock.assert_called_once()
        warning_mock.assert_called_once()

    def test_attach_management_fee_column(self):
        source = pd.DataFrame(
            [
                {"代碼": "0050", "ETF": "元大台灣50"},
                {"代碼": "^TWII", "ETF": "台股大盤"},
            ]
        )
        with patch(
            "app._get_tw_etf_management_fee_whitelist",
            return_value={"0050": "0.4567%"},
        ):
            out = _attach_tw_etf_management_fee_column(source)
        self.assertIn("管理費(%)", out.columns)
        self.assertIn("ETF規模(億)", out.columns)
        self.assertEqual(float(out.loc[out["代碼"] == "0050", "管理費(%)"].iloc[0]), 0.45)
        self.assertTrue(pd.isna(out.loc[out["代碼"] == "^TWII", "管理費(%)"].iloc[0]))

    def test_attach_aum_column(self):
        source = pd.DataFrame(
            [
                {"代碼": "0050", "ETF": "元大台灣50"},
                {"代碼": "^TWII", "ETF": "台股大盤"},
            ]
        )
        with patch("app._load_tw_etf_aum_billion_map", return_value={"0050": 12491.64}):
            out = _attach_tw_etf_aum_column(source)
        self.assertIn("ETF規模(億)", out.columns)
        self.assertEqual(int(out.loc[out["代碼"] == "0050", "ETF規模(億)"].iloc[0]), 12491)
        self.assertTrue(pd.isna(out.loc[out["代碼"] == "^TWII", "ETF規模(億)"].iloc[0]))

    def test_build_tw_etf_aum_history_wide(self):
        history = pd.DataFrame(
            [
                {
                    "etf_code": "0050",
                    "etf_name": "元大台灣50",
                    "trade_date": "2026-01-03",
                    "aum_billion": 1200.0,
                },
                {
                    "etf_code": "0050",
                    "etf_name": "元大台灣50",
                    "trade_date": "2026-01-02",
                    "aum_billion": 1100.0,
                },
                {
                    "etf_code": "0056",
                    "etf_name": "高股息",
                    "trade_date": "2026-01-03",
                    "aum_billion": 900.0,
                },
            ]
        )
        out = _build_tw_etf_aum_history_wide(history)
        self.assertEqual(
            list(out.columns), ["編號", "台股代號", "ETF名稱", "2026-01-02(億)", "2026-01-03(億)"]
        )
        self.assertEqual(int(out.iloc[0]["編號"]), 1)
        self.assertEqual(str(out.loc[out["台股代號"] == "0050", "ETF名稱"].iloc[0]), "元大台灣50")

    def test_compute_tw_etf_aum_alert_mask(self):
        frame = pd.DataFrame(
            [
                {
                    "編號": 1,
                    "台股代號": "0050",
                    "ETF名稱": "元大台灣50",
                    "2026-01-02(億)": 100.0,
                    "2026-01-03(億)": 111.0,
                    "2026-01-04(億)": 95.0,
                }
            ]
        )
        out = _compute_tw_etf_aum_alert_mask(frame, up_threshold=0.10)
        self.assertEqual(out.get((0, "2026-01-03(億)")), "#ffd1dc")
        self.assertNotIn((0, "2026-01-04(億)"), out)

    def test_decorate_tw_etf_aum_history_links(self):
        source = pd.DataFrame(
            [
                {"編號": 1, "台股代號": "0050", "ETF名稱": "元大台灣50", "2026-01-03(億)": 1200},
                {"編號": 2, "台股代號": "00935", "ETF名稱": "野村臺灣新科技50", "2026-01-03(億)": 800},
            ]
        )
        out, cfg = _decorate_tw_etf_aum_history_links(source)
        self.assertIn("台股代號", cfg)
        self.assertIn("ETF名稱", cfg)
        self.assertTrue(str(out.iloc[0]["台股代號"]).startswith("?bt_symbol=0050"))
        self.assertIn("hm_etf=00935", str(out.iloc[1]["ETF名稱"]))

    def test_recent_twse_trading_days(self):
        is_trading = {
            "20260112": False,
            "20260111": False,
            "20260110": False,
            "20260109": True,
            "20260108": True,
            "20260107": True,
            "20260106": True,
        }
        with patch("app._is_twse_trading_day", side_effect=lambda token: bool(is_trading.get(token))):
            out = _recent_twse_trading_days(anchor_yyyymmdd="20260112", count=3, max_scan_days=7)
        self.assertEqual(out, ["20260107", "20260108", "20260109"])

    def test_format_weight_pct_label(self):
        self.assertEqual(_format_weight_pct_label(3.456), "3.46%")
        self.assertEqual(_format_weight_pct_label(""), "—")
        self.assertEqual(_format_weight_pct_label(None), "—")

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
        self.assertEqual(
            _classify_issue_level("部分 ETF 同步失敗，已盡量使用本地可用資料"), "warning"
        )
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
        self.assertEqual(_classify_tw_etf("某某高股息月月配息ETF"), "股利型")
        self.assertEqual(_classify_tw_etf("某某AI科技ETF"), "科技型")
        self.assertEqual(_classify_tw_etf("某某永續ESG ETF"), "永續ESG型")
        self.assertEqual(_classify_tw_etf("某某金融ETF"), "金融型")
        self.assertEqual(_classify_tw_etf("某某主動ETF"), "主動式")
        self.assertEqual(_classify_tw_etf("完全不含關鍵字", code="0050"), "市值型")
        self.assertEqual(_classify_tw_etf("完全不含關鍵字", code="0056"), "股利型")
        self.assertEqual(_classify_tw_etf("完全不含關鍵字", code="00935"), "科技型")
        self.assertEqual(_classify_tw_etf("完全不含關鍵字", code="00920"), "永續ESG型")
        self.assertEqual(_classify_tw_etf("完全不含關鍵字", code="00917"), "金融型")
        self.assertEqual(_classify_tw_etf("完全不含關鍵字", code="0061"), "海外市場型")
        self.assertEqual(_classify_tw_etf("完全不含關鍵字", code="00981T"), "平衡收益型")
        self.assertEqual(_classify_tw_etf("完全不含關鍵字", code="00995A"), "主動式")
        self.assertEqual(_classify_tw_etf("完全不含關鍵字", code="00999A"), "主動式")

    def test_build_symbol_line_styles_assigns_distinct_colors(self):
        symbols = [f"00{i:03d}A" for i in range(1, 11)]
        style_map = _build_symbol_line_styles(symbols)
        self.assertEqual(len(style_map), len(symbols))

        used_colors = [style_map[s]["color"] for s in sorted(style_map.keys())]
        self.assertEqual(len(set(used_colors)), len(used_colors))
        for color in used_colors:
            self.assertIn(color, ACTIVE_ETF_LINE_COLORS)

    def test_compute_tw_equal_weight_compare_payload_enables_twii_fallback(self):
        idx = pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
        bars = pd.DataFrame(
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
            def load_daily_bars(self, symbol, market, start=None, end=None):
                return bars if str(symbol) == "00980A" else pd.DataFrame()

            def sync_symbol_history(self, symbol, market, start=None, end=None):
                return SimpleNamespace(error=None)

        def _fake_load_tw_benchmark_bars(**kwargs):
            self.assertTrue(bool(kwargs.get("allow_twii_fallback")))
            return bars, "0050", ["^TWII: unsupported"]

        with (
            patch("app._history_store", return_value=_FakeStore()),
            patch(
                "app._load_tw_benchmark_bars",
                side_effect=_fake_load_tw_benchmark_bars,
            ),
            patch("app.apply_split_adjustment", side_effect=lambda bars, **kwargs: (bars, [])),
        ):
            payload = _compute_tw_equal_weight_compare_payload(
                symbols=["00980A"],
                start_dt=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_dt=datetime(2026, 2, 1, tzinfo=timezone.utc),
                benchmark_choice="twii",
                sync_before_run=False,
                insufficient_msg="insufficient",
                initial_capital=1_000_000.0,
            )

        self.assertEqual(str(payload.get("error", "")), "")
        self.assertEqual(str(payload.get("benchmark_symbol", "")), "0050")
        benchmark_series = payload.get("benchmark_equity")
        self.assertIsInstance(benchmark_series, pd.Series)
        assert isinstance(benchmark_series, pd.Series)
        self.assertGreaterEqual(len(benchmark_series), 2)

    def test_apply_unified_benchmark_hover_sets_layout(self):
        fig = go.Figure()
        palette = {
            "benchmark": "#64748b",
            "text_muted": "#4b5563",
            "is_dark": False,
        }
        _apply_unified_benchmark_hover(fig, palette)
        self.assertEqual(str(fig.layout.hovermode), "x unified")
        self.assertEqual(int(fig.layout.spikedistance), -1)
        self.assertTrue(bool(fig.layout.xaxis.showspikes))

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

        with (
            patch(
                "app._fetch_twse_snapshot_with_fallback",
                side_effect=[("20251231", start_df), ("20260214", end_df)],
            ),
            patch("app._history_store", return_value=_FakeStore()),
            patch("app.known_split_events", return_value=[]),
        ):
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

    def test_build_tw_etf_top10_between_returns_universe_count(self):
        _build_tw_etf_top10_between.clear()

        start_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 100.0},
                {"code": "0056", "name": "元大高股息", "close": 30.0},
                {"code": "00935", "name": "野村台灣創新科技50", "close": 50.0},
                {"code": "00632R", "name": "元大台灣50反1", "close": 5.0},
                {"code": "00858", "name": "永豐美國500大", "close": 20.0},
            ]
        )
        end_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 110.0},
                {"code": "0056", "name": "元大高股息", "close": 33.0},
                {"code": "00935", "name": "野村台灣創新科技50", "close": 55.0},
                {"code": "0052", "name": "富邦科技", "close": 120.0},
                {"code": "00632R", "name": "元大台灣50反1", "close": 4.0},
            ]
        )
        class _EmptyStore:
            def load_daily_bars(self, symbol, market, start=None, end=None):
                return pd.DataFrame()

            def sync_symbol_history(self, symbol, market, start=None, end=None):
                return SimpleNamespace(error=None)

        with (
            patch(
                "app._fetch_twse_snapshot_with_fallback",
                side_effect=[("20251231", start_df), ("20260214", end_df)],
            ),
            patch("app._history_store", return_value=_EmptyStore()),
            patch("app.known_split_events", return_value=[]),
        ):
            out, start_used, end_used, universe_count = _build_tw_etf_top10_between(
                "20260101", "20260216"
            )

        self.assertEqual(start_used, "20251231")
        self.assertEqual(end_used, "20260214")
        self.assertEqual(universe_count, 3)
        self.assertEqual(len(out), 3)
        self.assertEqual(list(out["代碼"]), ["0050", "0056", "00935"])

    def test_build_tw_etf_top10_between_includes_newly_listed_etf(self):
        _build_tw_etf_top10_between.clear()

        start_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 100.0},
                {"code": "0056", "name": "元大高股息", "close": 30.0},
            ]
        )
        end_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 110.0},
                {"code": "0056", "name": "元大高股息", "close": 33.0},
                {"code": "00993A", "name": "主動安聯台灣", "close": 120.0},
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
                    "00993A": _bars([("2026-02-03", 100.0), ("2026-02-14", 120.0)]),
                }

            def load_daily_bars(self, symbol, market, start=None, end=None):
                return self._bars_map.get(str(symbol), pd.DataFrame())

            def sync_symbol_history(self, symbol, market, start=None, end=None):
                return SimpleNamespace(error=None)

        with (
            patch(
                "app._fetch_twse_snapshot_with_fallback",
                side_effect=[("20251231", start_df), ("20260214", end_df)],
            ),
            patch("app._history_store", return_value=_FakeStore()),
            patch("app.known_split_events", return_value=[]),
        ):
            out, start_used, end_used, universe_count = _build_tw_etf_top10_between(
                "20260101", "20260216", include_all_etf=True
            )

        self.assertEqual(start_used, "20251231")
        self.assertEqual(end_used, "20260214")
        self.assertEqual(universe_count, 3)
        self.assertEqual(list(out["代碼"]), ["00993A", "0050", "0056"])
        self.assertEqual(float(out.loc[out["代碼"] == "00993A", "區間報酬(%)"].iloc[0]), 20.0)

    def test_build_tw_etf_top10_between_empty_intersection(self):
        _build_tw_etf_top10_between.clear()

        start_df = pd.DataFrame([{"code": "0050", "name": "元大台灣50", "close": 100.0}])
        end_df = pd.DataFrame([{"code": "0056", "name": "元大高股息", "close": 30.0}])
        class _EmptyStore:
            def load_daily_bars(self, symbol, market, start=None, end=None):
                return pd.DataFrame()

            def sync_symbol_history(self, symbol, market, start=None, end=None):
                return SimpleNamespace(error=None)

        with (
            patch(
                "app._fetch_twse_snapshot_with_fallback",
                side_effect=[("20251231", start_df), ("20260214", end_df)],
            ),
            patch("app._history_store", return_value=_EmptyStore()),
            patch("app.known_split_events", return_value=[]),
        ):
            out, start_used, end_used, universe_count = _build_tw_etf_top10_between(
                "20260101", "20260216"
            )

        self.assertEqual(start_used, "20251231")
        self.assertEqual(end_used, "20260214")
        self.assertTrue(out.empty)
        self.assertEqual(universe_count, 0)

    def test_build_tw_etf_top10_between_type_filter(self):
        _build_tw_etf_top10_between.clear()

        start_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 100.0},
                {"code": "0056", "name": "元大高股息", "close": 30.0},
                {"code": "00929", "name": "復華台灣科技季配息", "close": 20.0},
            ]
        )
        end_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 120.0},
                {"code": "0056", "name": "元大高股息", "close": 33.0},
                {"code": "00929", "name": "復華台灣科技季配息", "close": 25.0},
            ]
        )

        with (
            patch(
                "app._fetch_twse_snapshot_with_fallback",
                side_effect=[("20251231", start_df), ("20260214", end_df)],
            ),
            patch("app.known_split_events", return_value=[]),
        ):
            out, start_used, end_used, universe_count = _build_tw_etf_top10_between(
                "20260101",
                "20260216",
                type_filter="股利型",
            )

        self.assertEqual(start_used, "20251231")
        self.assertEqual(end_used, "20260214")
        self.assertEqual(universe_count, 2)
        self.assertEqual(list(out["代碼"]), ["00929", "0056"])
        self.assertTrue((out["類型"] == "股利型").all())

    def test_build_tw_etf_top10_between_bottom_n(self):
        _build_tw_etf_top10_between.clear()

        start_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 100.0},
                {"code": "0056", "name": "元大高股息", "close": 30.0},
                {"code": "00929", "name": "復華台灣科技季配息", "close": 20.0},
                {"code": "00935", "name": "野村台灣創新科技50", "close": 50.0},
            ]
        )
        end_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 120.0},  # +20%
                {"code": "0056", "name": "元大高股息", "close": 27.0},  # -10%
                {"code": "00929", "name": "復華台灣科技季配息", "close": 16.0},  # -20%
                {"code": "00935", "name": "野村台灣創新科技50", "close": 55.0},  # +10%
            ]
        )

        with (
            patch(
                "app._fetch_twse_snapshot_with_fallback",
                side_effect=[("20241231", start_df), ("20251231", end_df)],
            ),
            patch("app.known_split_events", return_value=[]),
        ):
            out, start_used, end_used, universe_count = _build_tw_etf_top10_between(
                "20250101",
                "20251231",
                top_n=2,
                sort_ascending=True,
            )

        self.assertEqual(start_used, "20241231")
        self.assertEqual(end_used, "20251231")
        self.assertEqual(universe_count, 4)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out["代碼"]), ["00929", "0056"])

    def test_build_tw_etf_top10_between_exclude_split_event(self):
        _build_tw_etf_top10_between.clear()

        start_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 100.0},
                {"code": "0056", "name": "元大高股息", "close": 30.0},
            ]
        )
        end_df = pd.DataFrame(
            [
                {"code": "0050", "name": "元大台灣50", "close": 60.0},
                {"code": "0056", "name": "元大高股息", "close": 27.0},
            ]
        )

        split_event = SimpleNamespace(date="2025-06-18", ratio=0.2)
        with (
            patch(
                "app._fetch_twse_snapshot_with_fallback",
                side_effect=[("20241231", start_df), ("20251231", end_df)],
            ),
            patch(
                "app.known_split_events",
                side_effect=lambda symbol, market: [split_event] if str(symbol) == "0050" else [],
            ),
        ):
            out, _, _, universe_count = _build_tw_etf_top10_between(
                "20250101",
                "20251231",
                top_n=10,
                sort_ascending=True,
                exclude_split_event=True,
            )

        self.assertEqual(universe_count, 1)
        self.assertEqual(list(out["代碼"]), ["0056"])

    def test_consensus_threshold_candidates(self):
        self.assertEqual(_consensus_threshold_candidates(10), [10, 8, 7])
        self.assertEqual(_consensus_threshold_candidates(3), [3, 2])
        self.assertEqual(_consensus_threshold_candidates(2), [2])
        self.assertEqual(_consensus_threshold_candidates(1), [1])
        self.assertEqual(_consensus_threshold_candidates(0), [])

    def test_build_consensus_representative_between_strict_intersection(self):
        _build_consensus_representative_between.clear()

        top10_df = pd.DataFrame(
            [
                {"排名": 1, "代碼": "0050", "ETF": "元大台灣50"},
                {"排名": 2, "代碼": "0056", "ETF": "元大高股息"},
                {"排名": 3, "代碼": "00935", "ETF": "野村台灣創新科技50"},
            ]
        )

        class _FakeService:
            def get_etf_constituents_full(self, etf_code, limit=None, force_refresh=False):
                data = {
                    "0050": [
                        {"symbol": "A", "name": "A", "weight_pct": 40.0},
                        {"symbol": "B", "name": "B", "weight_pct": 30.0},
                        {"symbol": "C", "name": "C", "weight_pct": 30.0},
                    ],
                    "0056": [
                        {"symbol": "A", "name": "A", "weight_pct": 20.0},
                        {"symbol": "B", "name": "B", "weight_pct": 20.0},
                        {"symbol": "D", "name": "D", "weight_pct": 60.0},
                    ],
                    "00935": [
                        {"symbol": "A", "name": "A", "weight_pct": 10.0},
                        {"symbol": "B", "name": "B", "weight_pct": 10.0},
                        {"symbol": "E", "name": "E", "weight_pct": 80.0},
                    ],
                }
                return data.get(str(etf_code), []), "mock_full"

            def get_tw_etf_constituents(self, etf_code, limit=None):
                return [], "mock_fallback"

        with (
            patch(
                "app._build_tw_etf_top10_between",
                return_value=(top10_df, "20251231", "20260214", 120),
            ),
            patch("app._market_service", return_value=_FakeService()),
        ):
            payload = _build_consensus_representative_between(
                start_yyyymmdd="20251231",
                end_yyyymmdd="20260214",
                force_refresh_constituents=False,
            )

        self.assertEqual(str(payload.get("error", "")), "")
        self.assertEqual(int(payload.get("threshold_used", 0)), 3)
        self.assertEqual(int(payload.get("consensus_count", 0)), 2)
        self.assertFalse(bool(payload.get("fallback_applied", True)))
        top_pick = payload.get("top_pick", {})
        self.assertEqual(str(top_pick.get("ETF代碼", "")), "0050")

    def test_build_consensus_representative_between_fallback_threshold(self):
        _build_consensus_representative_between.clear()

        top10_df = pd.DataFrame(
            [
                {"排名": 1, "代碼": "0050", "ETF": "元大台灣50"},
                {"排名": 2, "代碼": "0056", "ETF": "元大高股息"},
                {"排名": 3, "代碼": "00935", "ETF": "野村台灣創新科技50"},
            ]
        )

        class _FakeService:
            def get_etf_constituents_full(self, etf_code, limit=None, force_refresh=False):
                data = {
                    "0050": [
                        {"symbol": "A", "name": "A", "weight_pct": 30.0},
                        {"symbol": "B", "name": "B", "weight_pct": 30.0},
                        {"symbol": "C", "name": "C", "weight_pct": 40.0},
                    ],
                    "0056": [
                        {"symbol": "A", "name": "A", "weight_pct": 50.0},
                        {"symbol": "D", "name": "D", "weight_pct": 30.0},
                        {"symbol": "E", "name": "E", "weight_pct": 20.0},
                    ],
                    "00935": [
                        {"symbol": "B", "name": "B", "weight_pct": 20.0},
                        {"symbol": "D", "name": "D", "weight_pct": 20.0},
                        {"symbol": "F", "name": "F", "weight_pct": 60.0},
                    ],
                }
                return data.get(str(etf_code), []), "mock_full"

            def get_tw_etf_constituents(self, etf_code, limit=None):
                return [], "mock_fallback"

        with (
            patch(
                "app._build_tw_etf_top10_between",
                return_value=(top10_df, "20251231", "20260214", 120),
            ),
            patch("app._market_service", return_value=_FakeService()),
        ):
            payload = _build_consensus_representative_between(
                start_yyyymmdd="20251231",
                end_yyyymmdd="20260214",
                force_refresh_constituents=False,
            )

        self.assertEqual(str(payload.get("error", "")), "")
        self.assertEqual(int(payload.get("threshold_used", 0)), 2)
        self.assertEqual(int(payload.get("consensus_count", 0)), 3)
        self.assertTrue(bool(payload.get("fallback_applied", False)))
        top_pick = payload.get("top_pick", {})
        self.assertEqual(str(top_pick.get("ETF代碼", "")), "0056")

    def test_compute_jaccard_pct(self):
        self.assertAlmostEqual(
            _compute_jaccard_pct({"A", "B"}, {"B", "C"}), 33.3333333333, places=3
        )
        self.assertEqual(_compute_jaccard_pct(set(), set()), 0.0)

    def test_build_two_etf_aggressive_picks_normal(self):
        _build_two_etf_aggressive_picks.clear()

        top10_df = pd.DataFrame(
            [
                {"排名": 1, "代碼": "0050", "ETF": "元大台灣50", "區間報酬(%)": 25.0},
                {"排名": 2, "代碼": "0056", "ETF": "元大高股息", "區間報酬(%)": 22.0},
                {"排名": 3, "代碼": "00935", "ETF": "野村臺灣新科技50", "區間報酬(%)": 20.0},
                {"排名": 4, "代碼": "00878", "ETF": "國泰永續高股息", "區間報酬(%)": 19.0},
            ]
        )

        consensus_payload = {
            "error": "",
            "top_pick": {"ETF代碼": "0050", "代表性分數(覆蓋率%)": 22.8},
        }

        class _FakeService:
            def get_etf_constituents_full(self, etf_code, limit=None, force_refresh=False):
                data = {
                    "0050": [
                        {"symbol": "A", "market": "TW"},
                        {"symbol": "B", "market": "TW"},
                        {"symbol": "C", "market": "TW"},
                    ],
                    "0056": [
                        {"symbol": "A", "market": "TW"},
                        {"symbol": "B", "market": "TW"},
                        {"symbol": "D", "market": "TW"},
                    ],
                    "00935": [
                        {"symbol": "X", "market": "TW"},
                        {"symbol": "Y", "market": "TW"},
                        {"symbol": "Z", "market": "TW"},
                    ],
                    "00878": [
                        {"symbol": "A", "market": "TW"},
                        {"symbol": "X", "market": "TW"},
                        {"symbol": "Z", "market": "TW"},
                    ],
                }
                return data.get(str(etf_code), []), "mock_full"

            def get_tw_etf_constituents(self, etf_code, limit=None):
                return [], "mock_fallback"

        with (
            patch(
                "app._build_tw_etf_top10_between",
                return_value=(top10_df, "20251231", "20260214", 120),
            ),
            patch("app._build_consensus_representative_between", return_value=consensus_payload),
            patch("app._market_service", return_value=_FakeService()),
        ):
            payload = _build_two_etf_aggressive_picks(
                start_yyyymmdd="20251231",
                end_yyyymmdd="20260214",
                allow_overseas=False,
                overlap_cap_pct=10.0,
                force_refresh_constituents=False,
            )

        self.assertEqual(str(payload.get("error", "")), "")
        self.assertEqual(str(payload.get("fallback_mode", "")), "strict_overlap")
        pick_1 = payload.get("pick_1", {})
        pick_2 = payload.get("pick_2", {})
        self.assertEqual(str(pick_1.get("ETF代碼", "")), "0050")
        self.assertEqual(str(pick_2.get("ETF代碼", "")), "00935")
        self.assertAlmostEqual(float(pick_2.get("與核心重疊度(%)")), 0.0, places=2)

    def test_build_two_etf_aggressive_picks_fallback_overlap_20(self):
        _build_two_etf_aggressive_picks.clear()

        top10_df = pd.DataFrame(
            [
                {"排名": 1, "代碼": "0050", "ETF": "元大台灣50", "區間報酬(%)": 24.0},
                {"排名": 2, "代碼": "0056", "ETF": "元大高股息", "區間報酬(%)": 22.0},
                {"排名": 3, "代碼": "00935", "ETF": "野村臺灣新科技50", "區間報酬(%)": 21.0},
                {"排名": 4, "代碼": "00878", "ETF": "國泰永續高股息", "區間報酬(%)": 20.5},
            ]
        )

        consensus_payload = {
            "error": "",
            "top_pick": {"ETF代碼": "0050", "代表性分數(覆蓋率%)": 20.1},
        }

        class _FakeService:
            def get_etf_constituents_full(self, etf_code, limit=None, force_refresh=False):
                data = {
                    "0050": [{"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}, {"symbol": "D"}],
                    "0056": [{"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}, {"symbol": "E"}],
                    "00935": [{"symbol": "A"}, {"symbol": "X"}, {"symbol": "Y"}, {"symbol": "Z"}],
                    "00878": [{"symbol": "B"}, {"symbol": "C"}, {"symbol": "F"}, {"symbol": "G"}],
                }
                return data.get(str(etf_code), []), "mock_full"

            def get_tw_etf_constituents(self, etf_code, limit=None):
                return [], "mock_fallback"

        with (
            patch(
                "app._build_tw_etf_top10_between",
                return_value=(top10_df, "20251231", "20260214", 120),
            ),
            patch("app._build_consensus_representative_between", return_value=consensus_payload),
            patch("app._market_service", return_value=_FakeService()),
        ):
            payload = _build_two_etf_aggressive_picks(
                start_yyyymmdd="20251231",
                end_yyyymmdd="20260214",
                allow_overseas=False,
                overlap_cap_pct=10.0,
                force_refresh_constituents=False,
            )

        self.assertEqual(str(payload.get("error", "")), "")
        self.assertEqual(str(payload.get("fallback_mode", "")), "relaxed_overlap_20")
        pick_2 = payload.get("pick_2", {})
        self.assertEqual(str(pick_2.get("ETF代碼", "")), "00935")
        self.assertLessEqual(float(payload.get("overlap_cap_used", 0.0)), 20.0)

    def test_build_two_etf_aggressive_picks_fallback_top_return(self):
        _build_two_etf_aggressive_picks.clear()

        top10_df = pd.DataFrame(
            [
                {"排名": 1, "代碼": "0050", "ETF": "元大台灣50", "區間報酬(%)": 23.0},
                {"排名": 2, "代碼": "0056", "ETF": "元大高股息", "區間報酬(%)": 21.0},
                {"排名": 3, "代碼": "00935", "ETF": "野村臺灣新科技50", "區間報酬(%)": 19.0},
            ]
        )

        consensus_payload = {
            "error": "",
            "top_pick": {"ETF代碼": "0050", "代表性分數(覆蓋率%)": 19.8},
        }

        class _FakeService:
            def get_etf_constituents_full(self, etf_code, limit=None, force_refresh=False):
                data = {
                    "0050": [{"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}],
                    "0056": [{"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}, {"symbol": "D"}],
                    "00935": [{"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}, {"symbol": "E"}],
                }
                return data.get(str(etf_code), []), "mock_full"

            def get_tw_etf_constituents(self, etf_code, limit=None):
                return [], "mock_fallback"

        with (
            patch(
                "app._build_tw_etf_top10_between",
                return_value=(top10_df, "20251231", "20260214", 120),
            ),
            patch("app._build_consensus_representative_between", return_value=consensus_payload),
            patch("app._market_service", return_value=_FakeService()),
        ):
            payload = _build_two_etf_aggressive_picks(
                start_yyyymmdd="20251231",
                end_yyyymmdd="20260214",
                allow_overseas=False,
                overlap_cap_pct=10.0,
                force_refresh_constituents=False,
            )

        self.assertEqual(str(payload.get("error", "")), "")
        self.assertEqual(str(payload.get("fallback_mode", "")), "top_return_fallback")
        pick_2 = payload.get("pick_2", {})
        self.assertEqual(str(pick_2.get("ETF代碼", "")), "0056")

    def test_build_two_etf_aggressive_picks_exclude_overseas(self):
        _build_two_etf_aggressive_picks.clear()

        top10_df = pd.DataFrame(
            [
                {"排名": 1, "代碼": "0050", "ETF": "元大台灣50", "區間報酬(%)": 25.0},
                {"排名": 2, "代碼": "00910", "ETF": "第一金太空衛星", "區間報酬(%)": 24.5},
                {"排名": 3, "代碼": "00941", "ETF": "中信上游半導體", "區間報酬(%)": 24.0},
            ]
        )

        consensus_payload = {
            "error": "",
            "top_pick": {"ETF代碼": "0050", "代表性分數(覆蓋率%)": 21.2},
        }

        class _FakeService:
            def get_etf_constituents_full(self, etf_code, limit=None, force_refresh=False):
                data = {
                    "0050": [
                        {"symbol": "2330", "market": "TW"},
                        {"symbol": "2454", "market": "TW"},
                    ],
                    "00910": [
                        {"symbol": "AAPL.US", "market": "US"},
                        {"symbol": "GOOGL.US", "market": "US"},
                    ],
                    "00941": [
                        {"symbol": "3017", "market": "TW"},
                        {"symbol": "3661", "market": "TW"},
                    ],
                }
                return data.get(str(etf_code), []), "mock_full"

            def get_tw_etf_constituents(self, etf_code, limit=None):
                return [], "mock_fallback"

        with (
            patch(
                "app._build_tw_etf_top10_between",
                return_value=(top10_df, "20251231", "20260214", 120),
            ),
            patch("app._build_consensus_representative_between", return_value=consensus_payload),
            patch("app._market_service", return_value=_FakeService()),
        ):
            payload = _build_two_etf_aggressive_picks(
                start_yyyymmdd="20251231",
                end_yyyymmdd="20260214",
                allow_overseas=False,
                overlap_cap_pct=10.0,
                force_refresh_constituents=False,
            )

        self.assertEqual(str(payload.get("error", "")), "")
        pick_2 = payload.get("pick_2", {})
        self.assertEqual(str(pick_2.get("ETF代碼", "")), "00941")
        excluded = payload.get("excluded_overseas_codes", [])
        self.assertIn("00910", list(excluded))

    def test_build_two_etf_aggressive_picks_consensus_fail_fallback_top1(self):
        _build_two_etf_aggressive_picks.clear()

        top10_df = pd.DataFrame(
            [
                {"排名": 1, "代碼": "0050", "ETF": "元大台灣50", "區間報酬(%)": 25.0},
                {"排名": 2, "代碼": "0056", "ETF": "元大高股息", "區間報酬(%)": 22.0},
                {"排名": 3, "代碼": "00935", "ETF": "野村臺灣新科技50", "區間報酬(%)": 20.0},
            ]
        )

        consensus_payload = {"error": "consensus down", "top_pick": {}}

        class _FakeService:
            def get_etf_constituents_full(self, etf_code, limit=None, force_refresh=False):
                data = {
                    "0050": [{"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}],
                    "0056": [{"symbol": "A"}, {"symbol": "B"}, {"symbol": "D"}],
                    "00935": [{"symbol": "X"}, {"symbol": "Y"}, {"symbol": "Z"}],
                }
                return data.get(str(etf_code), []), "mock_full"

            def get_tw_etf_constituents(self, etf_code, limit=None):
                return [], "mock_fallback"

        with (
            patch(
                "app._build_tw_etf_top10_between",
                return_value=(top10_df, "20251231", "20260214", 120),
            ),
            patch("app._build_consensus_representative_between", return_value=consensus_payload),
            patch("app._market_service", return_value=_FakeService()),
        ):
            payload = _build_two_etf_aggressive_picks(
                start_yyyymmdd="20251231",
                end_yyyymmdd="20260214",
                allow_overseas=False,
                overlap_cap_pct=10.0,
                force_refresh_constituents=False,
            )

        self.assertEqual(str(payload.get("error", "")), "")
        pick_1 = payload.get("pick_1", {})
        self.assertEqual(str(pick_1.get("ETF代碼", "")), "0050")
        self.assertIn("回退", str(pick_1.get("說明", "")))
        issues = payload.get("issues", [])
        self.assertTrue(any("consensus:" in str(item) for item in issues))

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
            compare_return_map={"0050": 18.5, "0056": 10.25},
            market_return_pct=5.0,
            market_compare_return_pct=9.0,
            benchmark_code="^TWII",
            end_used="20260214",
            underperform_col_label="輸給台股大盤(%)",
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(str(out.iloc[0]["ETF"]), "台股大盤")
        self.assertEqual(str(out.iloc[0]["排名"]), "—")
        self.assertEqual(float(out.iloc[0]["YTD報酬(%)"]), 5.0)
        self.assertEqual(float(out.iloc[1]["贏輸台股大盤(%)"]), 7.0)
        self.assertEqual(float(out.iloc[1]["輸給台股大盤(%)"]), 0.0)
        self.assertEqual(float(out.iloc[2]["輸給台股大盤(%)"]), 0.0)
        self.assertIn("2025績效(%)", out.columns)
        self.assertIn("YTD報酬(%)", out.columns)

    def test_build_tw_etf_all_types_performance_table(self):
        _build_tw_etf_all_types_performance_table.clear()
        ytd_df = pd.DataFrame(
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
                    "區間報酬(%)": 12.349,
                },
                {
                    "排名": 2,
                    "代碼": "00935",
                    "ETF": "野村臺灣新科技50",
                    "類型": "科技型",
                    "期初收盤": 50.0,
                    "復權期初": 50.0,
                    "期末收盤": 57.5,
                    "復權事件": "—",
                    "區間報酬(%)": 15.678,
                },
            ]
        )
        y2025_df = pd.DataFrame(
            [
                {"代碼": "0050", "區間報酬(%)": 30.456},
                {"代碼": "00935", "區間報酬(%)": 20.987},
            ]
        )
        with (
            patch(
                "app._build_tw_etf_top10_between",
                side_effect=[
                    (ytd_df, "20251231", "20260214", 2),
                    (y2025_df, "20241231", "20251231", 2),
                ],
            ) as top10_build_mock,
            patch(
                "app._load_tw_market_return_between",
                side_effect=[
                    (10.555, "0050", []),
                    (18.111, "0050", []),
                ],
            ),
            patch(
                "app._load_tw_etf_daily_change_map",
                return_value=({"0050": 1.5, "00935": 4.2}, "20260214", "20260213"),
            ),
            patch(
                "app._load_tw_snapshot_open_map",
                return_value=("20260214", {"0050": 113.0, "00935": 58.0}),
            ),
            patch(
                "app._load_tw_market_daily_return",
                return_value=(0.8, "0050", "20260213", "20260214", []),
            ),
        ):
            out, meta = _build_tw_etf_all_types_performance_table(
                ytd_start_yyyymmdd="20251231",
                ytd_end_yyyymmdd="20260214",
                compare_start_yyyymmdd="20241231",
                compare_end_yyyymmdd="20251231",
            )

        self.assertEqual(len(out), 2)
        self.assertIn("編號", out.columns)
        self.assertNotIn("排名", out.columns)
        self.assertIn("2025績效(%)", out.columns)
        self.assertIn("2026YTD績效(%)", out.columns)
        self.assertIn("輸贏大盤2025(%)", out.columns)
        self.assertIn("輸贏大盤2026YTD(%)", out.columns)
        self.assertIn("管理費(%)", out.columns)
        self.assertIn("開盤", out.columns)
        self.assertIn("收盤", out.columns)
        self.assertIn("今日漲幅", out.columns)
        self.assertIn("今日贏大盤%", out.columns)
        self.assertEqual(float(out.loc[out["代碼"] == "0050", "2025績效(%)"].iloc[0]), 30.45)
        self.assertEqual(float(out.loc[out["代碼"] == "00935", "2026YTD績效(%)"].iloc[0]), 15.67)
        self.assertEqual(float(out.loc[out["代碼"] == "0050", "開盤"].iloc[0]), 113.0)
        self.assertEqual(float(out.loc[out["代碼"] == "00935", "開盤"].iloc[0]), 58.0)
        self.assertEqual(float(out.loc[out["代碼"] == "0050", "輸贏大盤2025(%)"].iloc[0]), 12.33)
        self.assertEqual(
            float(out.loc[out["代碼"] == "00935", "輸贏大盤2026YTD(%)"].iloc[0]), 5.11
        )
        self.assertEqual(float(out.loc[out["代碼"] == "00935", "今日漲幅"].iloc[0]), 4.2)
        self.assertEqual(float(out.loc[out["代碼"] == "0050", "今日贏大盤%"].iloc[0]), 0.7)
        self.assertEqual(str(meta.get("market_2025_symbol", "")), "0050")
        self.assertEqual(str(meta.get("market_ytd_symbol", "")), "0050")
        self.assertEqual(str(meta.get("market_daily_symbol", "")), "0050")
        self.assertEqual(int(top10_build_mock.call_count), 2)
        for call in top10_build_mock.call_args_list:
            self.assertTrue(bool(call.kwargs.get("include_all_etf", False)))


if __name__ == "__main__":
    unittest.main()
