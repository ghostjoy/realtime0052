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
                "equity": "#2563EB",
                "plot_template": "plotly_white",
                "paper_bg": "#FFFFFF",
                "plot_bg": "#FFFFFF",
                "text_color": "#111827",
                "grid": "#CBD5E1",
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
