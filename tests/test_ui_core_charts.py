from __future__ import annotations

from typing import Any

import pandas as pd

from ui.core import charts


def _configure_test_runtime(captured: dict[str, Any]) -> None:
    charts.configure_runtime(
        {
            "_apply_plotly_watermark": lambda fig, text, palette: None,
            "_apply_unified_benchmark_hover": lambda fig, palette: None,
            "_benchmark_renderer": lambda: "plotly",
            "_enable_plotly_draw_tools": lambda fig: None,
            "_hovertemplate_with_code": lambda code, value_label, y_format: (
                f"{code}:{value_label}:{y_format}"
            ),
            "_live_kline_renderer": lambda: "plotly",
            "_render_plotly_chart": lambda fig, **kwargs: captured.setdefault("fig", fig),
            "_to_rgba": lambda color, alpha: color,
            "_ui_palette": lambda: {
                "bb_lower": "#60A5FA",
                "bb_upper": "#2563EB",
                "buy_hold": "#F59E0B",
                "equity": "#2563EB",
                "kd_d": "#EF4444",
                "kd_k": "#22C55E",
                "macd_line": "#2563EB",
                "macd_signal": "#F59E0B",
                "plot_template": "plotly_white",
                "paper_bg": "#FFFFFF",
                "plot_bg": "#FFFFFF",
                "signal_buy": "#22C55E",
                "signal_sell": "#EF4444",
                "sma20": "#0EA5E9",
                "text_color": "#111827",
                "text_muted": "#64748B",
                "grid": "#CBD5E1",
                "volume_down": "#DC2626",
                "volume_up": "#16A34A",
            },
        }
    )


def test_render_benchmark_lines_chart_uses_right_axis_for_second_series():
    captured: dict[str, Any] = {}
    _configure_test_runtime(captured)
    idx = pd.date_range("2026-03-01", periods=3, freq="D", tz="UTC")
    charts._render_benchmark_lines_chart(
        lines=[
            {
                "name": "Strategy Equity",
                "series": pd.Series([100, 110, 120], index=idx),
                "color": "#2563EB",
            },
            {
                "name": "Benchmark Equity",
                "series": pd.Series([100, 105, 108], index=idx),
                "color": "#475569",
            },
        ],
        height=420,
        chart_key="unit_test_chart",
        enable_lightweight=False,
    )

    fig = captured["fig"]
    assert fig.layout.yaxis.side == "left"
    assert fig.layout.yaxis2.side == "right"
    assert fig.layout.yaxis2.matches == "y"
    assert fig.data[1].yaxis == "y2"
    assert fig.layout.xaxis.range is not None
    assert pd.Timestamp(fig.layout.xaxis.range[1]) > idx[-1]
    assert fig.layout.legend.x > 1.09
    assert len(fig.layout.shapes) == 1
    assert pd.Timestamp(fig.layout.shapes[0].x0) == idx[-1]


def test_render_benchmark_lines_chart_adds_hidden_right_axis_mirror_for_single_series():
    captured: dict[str, Any] = {}
    _configure_test_runtime(captured)
    idx = pd.date_range("2026-03-01", periods=3, freq="D", tz="UTC")
    charts._render_benchmark_lines_chart(
        lines=[
            {
                "name": "Strategy Equity",
                "series": pd.Series([100, 110, 120], index=idx),
                "color": "#2563EB",
            },
        ],
        height=420,
        chart_key="unit_test_single_series_chart",
        enable_lightweight=False,
    )

    fig = captured["fig"]
    assert fig.layout.yaxis2.side == "right"
    assert len(fig.data) == 2
    assert fig.data[1].yaxis == "y2"
    assert fig.data[1].showlegend is False
    assert fig.data[1].hoverinfo == "skip"
    assert fig.layout.xaxis.range is not None
    assert pd.Timestamp(fig.layout.xaxis.range[1]) > idx[-1]
    assert fig.layout.legend.x > 1.09
    assert len(fig.layout.shapes) == 1
    assert pd.Timestamp(fig.layout.shapes[0].x0) == idx[-1]


def test_render_benchmark_lines_chart_marks_boxes_above_benchmark_with_star():
    captured: dict[str, Any] = {}
    _configure_test_runtime(captured)
    idx = pd.date_range("2026-03-01", periods=3, freq="D", tz="UTC")
    charts._render_benchmark_lines_chart(
        lines=[
            {
                "name": "Strategy Equity",
                "series": pd.Series([100, 112, 128], index=idx),
                "color": "#2563EB",
            },
            {
                "name": "Benchmark Equity",
                "series": pd.Series([100, 106, 109], index=idx),
                "color": "#475569",
                "is_benchmark": True,
            },
            {
                "name": "Buy-and-Hold（0052）",
                "series": pd.Series([100, 104, 105], index=idx),
                "color": "#F59E0B",
            },
        ],
        height=420,
        chart_key="unit_test_highlight_boxes_chart",
        enable_lightweight=False,
        highlight_above_benchmark_boxes=True,
    )

    fig = captured["fig"]
    trace_names = [str(trace.name) for trace in fig.data]
    assert trace_names[0] == "Benchmark Equity"
    assert trace_names[1] == "Strategy Equity *"
    assert any(name.startswith("Buy-and-Hold（0052）") for name in trace_names)
    assert "* = final value above benchmark" in trace_names


def test_build_multi_line_styles_keeps_all_asset_lines_solid():
    styles = charts.build_multi_line_styles(
        ["00988A", "00992A", "00994A", "00991A", "00987A", "00981A", "00985A", "00990A", "00995A"]
    )
    assert styles["00981A"]["dash"] == "solid"
    assert styles["00995A"]["dash"] == "solid"


def test_render_indicator_panels_aligns_left_and_right_margins_with_main_chart():
    captured: dict[str, Any] = {}
    _configure_test_runtime(captured)
    idx = pd.date_range("2026-03-01", periods=3, freq="D", tz="UTC")
    charts._render_indicator_panels(
        pd.DataFrame(
            {
                "close": [100.0, 102.0, 101.0],
                "bb_upper": [104.0, 105.0, 104.5],
                "bb_mid": [100.0, 101.0, 101.0],
                "bb_lower": [96.0, 97.0, 97.5],
                "rsi_14": [45.0, 55.0, 50.0],
                "stoch_k": [40.0, 60.0, 55.0],
                "stoch_d": [38.0, 58.0, 52.0],
                "macd": [0.1, 0.3, 0.2],
                "macd_signal": [0.0, 0.2, 0.15],
                "macd_hist": [0.1, -0.1, 0.05],
            },
            index=idx,
        ),
        chart_key="unit_test_indicator_chart",
    )

    fig = captured["fig"]
    trace_by_name = {str(trace.name): trace for trace in fig.data}
    assert len(fig.layout.annotations) == 1
    annotation = fig.layout.annotations[0]
    assert fig.layout.height == 680
    assert fig.layout.margin.l == 60
    assert fig.layout.margin.r == 220
    assert float(annotation.x) > 1.0
    assert str(annotation.xanchor) == "left"
    assert "Close" in str(annotation.text)
    assert "101.00" in str(annotation.text)
    assert "RSI14" in str(annotation.text)
    assert "50.00" in str(annotation.text)
    assert "MACD Signal" in str(annotation.text)
    assert "0.1500" in str(annotation.text)
    assert "MACD Hist" in str(annotation.text)
    assert "0.0500" in str(annotation.text)
    assert "#334155" in str(annotation.text)
    assert "#1D4ED8" in str(annotation.text)
    assert "#059669" in str(annotation.text)
    assert trace_by_name["Close"].line.color == "#334155"
    assert trace_by_name["BB中軌"].line.color == "#2563EB"
    assert trace_by_name["BB上軌"].line.color == "#B91C1C"
    assert trace_by_name["BB下軌"].line.color == "#0F766E"
    assert trace_by_name["RSI14"].line.color == "#1D4ED8"
    assert trace_by_name["KD-K"].line.color == "#0F766E"
    assert trace_by_name["KD-D"].line.color == "#C2410C"
    assert trace_by_name["MACD"].line.color == "#059669"
    assert trace_by_name["MACD Signal"].line.color == "#B45309"
    assert tuple(trace_by_name["MACD Hist"].marker.color) == (
        "rgba(5,150,105,0.45)",
        "rgba(185,28,28,0.38)",
        "rgba(5,150,105,0.45)",
    )
