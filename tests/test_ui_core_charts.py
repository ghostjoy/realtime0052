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
