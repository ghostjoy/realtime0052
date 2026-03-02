from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from ui.charts.lightweight_adapter import (
    _series_type,
    render_lightweight_kline_equity_chart,
    render_lightweight_live_chart,
    render_lightweight_multi_line_chart,
)


class LightweightAdapterTests(unittest.TestCase):
    def test_series_type_uses_chart_name_tokens(self):
        self.assertEqual(_series_type("Line"), "Line")
        self.assertEqual(_series_type("Candlestick"), "Candlestick")
        self.assertEqual(_series_type("Histogram"), "Histogram")

    def test_render_multi_line_emits_line_type_token(self):
        idx = pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
        series = pd.Series([100.0, 101.0], index=idx)
        captured = {}

        def _fake_render(charts, key=None):
            captured["charts"] = charts
            captured["key"] = key
            return None

        with patch(
            "ui.charts.lightweight_adapter.renderLightweightCharts", side_effect=_fake_render
        ):
            ok = render_lightweight_multi_line_chart(
                lines=[{"name": "Strategy", "series": series, "color": "#16a34a"}],
                palette={
                    "grid": "rgba(120,120,120,0.2)",
                    "plot_bg": "#ffffff",
                    "paper_bg": "#ffffff",
                    "text_color": "#111827",
                },
                key="ut:lw:line",
                height=320,
            )

        self.assertTrue(ok)
        charts = captured.get("charts")
        self.assertIsInstance(charts, list)
        assert isinstance(charts, list)
        self.assertEqual(charts[0]["series"][0]["type"], "Line")

    def test_render_live_chart_renders_price_and_volume_in_one_component(self):
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
        calls = []

        def _fake_render(charts, key=None):
            calls.append((charts, key))
            return None

        with patch(
            "ui.charts.lightweight_adapter.renderLightweightCharts", side_effect=_fake_render
        ):
            ok = render_lightweight_live_chart(
                ind=bars,
                palette={
                    "grid": "rgba(120,120,120,0.2)",
                    "plot_bg": "#ffffff",
                    "paper_bg": "#ffffff",
                    "text_color": "#111827",
                    "price_up_line": "#5FA783",
                    "price_down_line": "#D78C95",
                },
                key="ut:lw:live",
                overlays=None,
            )

        self.assertTrue(ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "ut:lw:live")
        charts = calls[0][0]
        self.assertIsInstance(charts, list)
        assert isinstance(charts, list)
        self.assertEqual(len(charts), 2)

    def test_render_kline_equity_renders_price_and_equity_in_one_component(self):
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
        strategy = pd.Series([1_000_000.0, 1_010_000.0], index=idx)
        benchmark = pd.Series([1_000_000.0, 1_005_000.0], index=idx)
        twii_overlay = pd.Series([100.0, 101.5], index=idx)
        calls = []

        def _fake_render(charts, key=None):
            calls.append((charts, key))
            return None

        with patch(
            "ui.charts.lightweight_adapter.renderLightweightCharts", side_effect=_fake_render
        ):
            ok = render_lightweight_kline_equity_chart(
                bars=bars,
                strategy=strategy,
                benchmark=benchmark,
                price_overlays=[
                    {
                        "name": "TWII（同基準價）",
                        "series": twii_overlay,
                        "color": "rgba(100,100,100,0.4)",
                        "width": 1,
                        "dash": "dash",
                    }
                ],
                palette={
                    "grid": "rgba(120,120,120,0.2)",
                    "plot_bg": "#ffffff",
                    "paper_bg": "#ffffff",
                    "text_color": "#111827",
                    "price_up_line": "#5FA783",
                    "price_down_line": "#D78C95",
                    "equity": "#16a34a",
                    "benchmark": "#64748b",
                    "benchmark_dash": "dash",
                },
                key="ut:lw:replay",
            )

        self.assertTrue(ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "ut:lw:replay")
        charts = calls[0][0]
        self.assertIsInstance(charts, list)
        assert isinstance(charts, list)
        self.assertEqual(len(charts), 2)
        price_series = charts[0]["series"]
        self.assertEqual(len(price_series), 2)
        self.assertEqual(price_series[0]["type"], "Candlestick")
        self.assertEqual(price_series[1]["type"], "Line")


if __name__ == "__main__":
    unittest.main()
