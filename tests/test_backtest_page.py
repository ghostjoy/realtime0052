from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go

from ui.pages.backtest import (
    _add_equity_summary_annotation,
    _build_finmind_backtest_insight,
    _build_tw_chip_filter_series,
    _build_tw_chip_history_frame,
    _build_tw_chip_replay_insight,
)


class FinMindBacktestInsightTests(unittest.TestCase):
    def test_add_equity_summary_annotation_stays_inside_plot_frame(self):
        fig = go.Figure()
        equity = pd.Series([100_000.0, 120_000.0], index=pd.date_range("2026-01-01", periods=2))
        benchmark = pd.Series([100_000.0, 110_000.0], index=pd.date_range("2026-01-01", periods=2))

        with patch("ui.pages.backtest._to_rgba", side_effect=lambda color, alpha: color):
            _add_equity_summary_annotation(
                fig,
                equity_series=equity,
                benchmark_series=benchmark,
                palette={
                    "equity": "#22C55E",
                    "benchmark": "#0EA5E9",
                    "text_color": "#0F172A",
                    "paper_bg": "#FFFFFF",
                },
            )

        self.assertEqual(len(fig.layout.annotations), 1)
        annotation = fig.layout.annotations[0]
        self.assertGreater(float(annotation.x), 1.0)
        self.assertEqual(str(annotation.xanchor), "left")

    def test_build_finmind_backtest_insight_with_positive_research_data(self):
        insight = _build_finmind_backtest_insight(
            {
                "enabled": True,
                "month_revenue": [
                    {"date": "2025-02-01", "revenue": 100_000_000},
                    {"date": "2026-01-01", "revenue": 110_000_000},
                    {"date": "2026-02-01", "revenue": 130_000_000},
                ],
                "institutional_investors": [
                    {
                        "date": "2026-03-14",
                        "name": "Foreign_Investor",
                        "buy": 2000,
                        "sell": 500,
                    },
                    {
                        "date": "2026-03-14",
                        "name": "Investment_Trust",
                        "buy": 500,
                        "sell": 300,
                    },
                    {
                        "date": "2026-03-14",
                        "name": "Dealer_self",
                        "buy": 100,
                        "sell": 80,
                    },
                ],
                "news": [
                    {"date": "2026-03-14", "title": "台積電法說聚焦 AI 需求"},
                    {"date": "2026-03-13", "title": "晶圓代工報價展望穩定"},
                ],
            }
        )

        self.assertTrue(insight["enabled"])
        self.assertIn("營收動能延續", "\n".join(insight["indicator_lines"]))
        self.assertIn("最新營收與法人同步偏正向", "\n".join(insight["conclusion_lines"]))
        self.assertEqual(len(insight["decision_rows"]), 3)
        self.assertEqual(insight["decision_rows"][0][0], "最新月營收 YoY > 0 ?")
        self.assertEqual(insight["decision_rows"][0][1], "True")
        self.assertEqual(insight["decision_rows"][2][0], "最新法人籌碼偏多 ?")
        self.assertEqual(insight["decision_rows"][2][1], "True")

    def test_build_finmind_backtest_insight_handles_missing_data(self):
        insight = _build_finmind_backtest_insight(
            {
                "enabled": True,
                "month_revenue": [],
                "institutional_investors": [],
                "news": [],
            }
        )

        self.assertTrue(insight["enabled"])
        self.assertIn("最新月營收資料不足", "\n".join(insight["indicator_lines"]))
        self.assertIn("最新法人籌碼資料不足", "\n".join(insight["indicator_lines"]))
        self.assertIn("研究資料不足", "\n".join(insight["conclusion_lines"]))
        self.assertEqual(len(insight["decision_rows"]), 3)
        self.assertEqual(insight["decision_rows"][0][1], "N/A")
        self.assertEqual(insight["decision_rows"][2][1], "N/A")

    def test_build_finmind_backtest_insight_when_disabled(self):
        insight = _build_finmind_backtest_insight({"enabled": False})

        self.assertFalse(insight["enabled"])
        self.assertIn("未啟用", insight["caption"])
        self.assertEqual(insight["indicator_lines"], [])
        self.assertEqual(insight["conclusion_lines"], [])
        self.assertEqual(insight["decision_rows"], [])

    def test_build_finmind_backtest_insight_prefers_official_three_investors(self):
        insight = _build_finmind_backtest_insight(
            {"enabled": False},
            official_institutional={
                "data_date": "2026-03-14",
                "foreign_net_lots": -101500.1,
                "investment_trust_net_lots": 3214.0,
                "dealer_net_lots": -22166.17,
                "total_net_lots": -120452.27,
            },
        )

        self.assertTrue(insight["enabled"])
        self.assertIn("TWSE：最新三大法人 2026-03-14", "\n".join(insight["indicator_lines"]))
        self.assertIn("官方三大法人顯示 籌碼偏空", "\n".join(insight["conclusion_lines"]))
        self.assertEqual(len(insight["decision_rows"]), 4)
        self.assertEqual(insight["decision_rows"][0][0], "外資淨買賣超 > 0 ?")
        self.assertEqual(insight["decision_rows"][3][0], "三大法人合計 > 0 ?")

    def test_build_tw_chip_filter_series_standard_mode(self):
        bars = pd.DataFrame(
            {"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1]},
            index=pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"], utc=True),
        )
        rows = [
            {"date": "2026-03-10", "name": "Foreign_Investor", "buy": 2000, "sell": 1000},
            {"date": "2026-03-10", "name": "Investment_Trust", "buy": 500, "sell": 100},
            {"date": "2026-03-11", "name": "Foreign_Investor", "buy": 100, "sell": 500},
            {"date": "2026-03-11", "name": "Investment_Trust", "buy": 100, "sell": 50},
            {"date": "2026-03-12", "name": "Foreign_Investor", "buy": 500, "sell": 100},
            {"date": "2026-03-12", "name": "Investment_Trust", "buy": 0, "sell": 200},
        ]
        with patch("ui.pages.backtest._load_finmind_institutional_history", return_value=rows):
            gate, meta = _build_tw_chip_filter_series("0050", bars, filter_mode="標準")

        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual(gate.tolist(), [1, 0, 1])
        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["mode"], "標準")

    def test_build_finmind_backtest_insight_can_skip_institutional_lines(self):
        insight = _build_finmind_backtest_insight(
            {
                "enabled": True,
                "month_revenue": [
                    {"date": "2025-02-01", "revenue": 100_000_000},
                    {"date": "2026-02-01", "revenue": 130_000_000},
                ],
                "institutional_investors": [
                    {"date": "2026-03-14", "name": "Foreign_Investor", "buy": 2000, "sell": 500},
                ],
                "news": [{"date": "2026-03-14", "title": "AI 需求延續"}],
            },
            include_institutional=False,
        )

        indicator_text = "\n".join(insight["indicator_lines"])
        self.assertIn("最新月營收", indicator_text)
        self.assertNotIn("最新法人籌碼", indicator_text)
        self.assertEqual(len(insight["decision_rows"]), 2)
        self.assertEqual(insight["decision_rows"][0][0], "最新月營收 YoY > 0 ?")

    def test_build_tw_chip_history_frame_uses_latest_available_past_day(self):
        bars = pd.DataFrame(
            {"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1]},
            index=pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"], utc=True),
        )
        rows = [
            {"date": "2026-03-10", "name": "Foreign_Investor", "buy": 3000, "sell": 1000},
            {"date": "2026-03-10", "name": "Investment_Trust", "buy": 2000, "sell": 500},
            {"date": "2026-03-12", "name": "Foreign_Investor", "buy": 500, "sell": 2500},
            {"date": "2026-03-12", "name": "Investment_Trust", "buy": 200, "sell": 1200},
        ]
        with patch("ui.pages.backtest._load_finmind_institutional_history", return_value=rows):
            history, meta = _build_tw_chip_history_frame("0050", bars)

        self.assertIsNotNone(history)
        assert history is not None
        self.assertTrue(meta["enabled"])
        self.assertEqual(
            pd.Timestamp(history.iloc[1]["資料日期"]).strftime("%Y-%m-%d"), "2026-03-10"
        )
        self.assertEqual(
            pd.Timestamp(history.iloc[2]["資料日期"]).strftime("%Y-%m-%d"), "2026-03-12"
        )

    def test_build_tw_chip_replay_insight_scores_and_labels(self):
        index = pd.to_datetime(["2026-03-10", "2026-03-11"], utc=True)
        chip_history = pd.DataFrame(
            {
                "外資": [2.0, -4.0],
                "投信": [1.0, -1.0],
                "自營商": [0.5, -0.5],
                "三大法人合計": [3.5, -5.5],
                "近3日法人合計": [3.5, -2.0],
                "近5日法人合計": [3.5, -2.0],
                "資料日期": pd.to_datetime(["2026-03-10", "2026-03-11"]),
            },
            index=index,
        )

        positive = _build_tw_chip_replay_insight(chip_history, index[0])
        negative = _build_tw_chip_replay_insight(chip_history, index[1])

        self.assertTrue(positive["enabled"])
        self.assertEqual(positive["chip_state"], "籌碼順風")
        self.assertEqual(positive["chip_score"], 100)
        self.assertEqual(positive["decision_rows"][5][0], "Chip 分數 >= 65 ?")
        self.assertEqual(positive["decision_rows"][5][1], "True")
        self.assertEqual(negative["chip_state"], "籌碼逆風")
        self.assertLess(int(negative["chip_score"]), 40)
