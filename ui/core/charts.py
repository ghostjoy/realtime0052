from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.charts import render_lightweight_live_chart, render_lightweight_multi_line_chart
from ui.shared.runtime import configure_module_runtime

MULTI_LINE_COLORS: tuple[str, ...] = (
    "#1D4ED8",
    "#DC2626",
    "#059669",
    "#D97706",
    "#7C3AED",
    "#0891B2",
    "#BE123C",
    "#65A30D",
)

INDICATOR_PANEL_COLORS: dict[str, str] = {
    "close": "#334155",
    "bb_mid": "#2563EB",
    "bb_upper": "#B91C1C",
    "bb_lower": "#0F766E",
    "rsi": "#1D4ED8",
    "kd_k": "#0F766E",
    "kd_d": "#C2410C",
    "macd": "#059669",
    "macd_signal": "#B45309",
    "macd_hist_up": "rgba(5,150,105,0.45)",
    "macd_hist_down": "rgba(185,28,28,0.38)",
}

REQUIRED_RUNTIME_NAMES = (
    "_apply_plotly_watermark",
    "_apply_unified_benchmark_hover",
    "_benchmark_renderer",
    "_enable_plotly_draw_tools",
    "_hovertemplate_with_code",
    "_live_kline_renderer",
    "_render_plotly_chart",
    "_to_rgba",
    "_ui_palette",
)

_apply_plotly_watermark: Any = None
_apply_unified_benchmark_hover: Any = None
_benchmark_renderer: Any = None
_enable_plotly_draw_tools: Any = None
_hovertemplate_with_code: Any = None
_live_kline_renderer: Any = None
_render_plotly_chart: Any = None
_to_rgba: Any = None
_ui_palette: Any = None


def configure_runtime(values: Mapping[str, Any]) -> None:
    configure_module_runtime(globals(), REQUIRED_RUNTIME_NAMES, values, module_name=__name__)


def _to_utc_timestamp(value: object) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _plotly_datetime_axis_range_with_right_padding(
    index_values: object,
    *,
    x_start: object | None = None,
    x_end: object | None = None,
    pad_ratio: float = 0.05,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    idx = pd.to_datetime(index_values, utc=True, errors="coerce")
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    idx = idx.dropna().sort_values().unique()
    if len(idx) == 0:
        return None

    range_start = _to_utc_timestamp(x_start) or pd.Timestamp(idx[0])
    range_end = _to_utc_timestamp(x_end) or pd.Timestamp(idx[-1])
    if range_end < range_start:
        range_start, range_end = range_end, range_start

    visible = idx[(idx >= range_start) & (idx <= range_end)]
    if len(visible) == 0:
        visible = idx
    visible = pd.DatetimeIndex(visible).sort_values()
    min_pad = pd.Timedelta(days=1)
    if len(visible) >= 2:
        deltas = visible.to_series().diff().dropna()
        median_delta = deltas.median() if not deltas.empty else pd.NaT
        if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
            median_delta = min_pad
        span = pd.Timestamp(visible[-1]) - pd.Timestamp(visible[0])
        ratio_pad = span * float(max(pad_ratio, 0.0)) if span > pd.Timedelta(0) else median_delta
        if ratio_pad <= pd.Timedelta(0):
            ratio_pad = median_delta
        right_pad = median_delta if median_delta >= ratio_pad else ratio_pad
    else:
        right_pad = min_pad
    return (range_start, range_end + right_pad)


def _plotly_right_edge_marker_x(index_values: object) -> pd.Timestamp | None:
    idx = pd.to_datetime(index_values, utc=True, errors="coerce")
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    idx = idx.dropna().sort_values()
    if len(idx) == 0:
        return None
    return pd.Timestamp(idx[-1])


def build_multi_line_styles(
    symbols: list[str],
    *,
    colors: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, str]]:
    ordered = sorted({str(sym or "").strip().upper() for sym in symbols if str(sym or "").strip()})
    if not ordered:
        return {}
    color_cycle = [str(color) for color in (colors or MULTI_LINE_COLORS) if str(color).strip()]
    if not color_cycle:
        color_cycle = list(MULTI_LINE_COLORS)
    styles: dict[str, dict[str, str]] = {}
    for idx, sym in enumerate(ordered):
        styles[sym] = {
            "color": color_cycle[idx % len(color_cycle)],
            "dash": "solid",
        }
    return styles


def _last_valid_numeric(series: object) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if isinstance(numeric, pd.Series):
        numeric = numeric.dropna()
        if numeric.empty:
            return None
        return float(numeric.iloc[-1])
    if pd.isna(numeric):
        return None
    return float(numeric)


def _format_indicator_value(value: float | None, *, digits: int) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{int(digits)}f}"


def _add_indicator_summary_annotation(
    fig: go.Figure,
    *,
    frame: pd.DataFrame,
    palette: Mapping[str, Any],
    x_paper: float = 1.01,
    y_paper: float = 0.985,
) -> None:
    bb_mid_col = "bb_mid" if "bb_mid" in frame.columns else ("sma_20" if "sma_20" in frame.columns else "")
    items: list[tuple[str, str, float | None, int]] = [
        ("Close", INDICATOR_PANEL_COLORS["close"], _last_valid_numeric(frame.get("close")), 2),
        (
            "BB中軌",
            INDICATOR_PANEL_COLORS["bb_mid"],
            _last_valid_numeric(frame.get(bb_mid_col)) if bb_mid_col else None,
            2,
        ),
        ("BB上軌", INDICATOR_PANEL_COLORS["bb_upper"], _last_valid_numeric(frame.get("bb_upper")), 2),
        ("BB下軌", INDICATOR_PANEL_COLORS["bb_lower"], _last_valid_numeric(frame.get("bb_lower")), 2),
        ("RSI14", INDICATOR_PANEL_COLORS["rsi"], _last_valid_numeric(frame.get("rsi_14")), 2),
        ("KD-K", INDICATOR_PANEL_COLORS["kd_k"], _last_valid_numeric(frame.get("stoch_k")), 2),
        ("KD-D", INDICATOR_PANEL_COLORS["kd_d"], _last_valid_numeric(frame.get("stoch_d")), 2),
        ("MACD", INDICATOR_PANEL_COLORS["macd"], _last_valid_numeric(frame.get("macd")), 4),
        (
            "MACD Signal",
            INDICATOR_PANEL_COLORS["macd_signal"],
            _last_valid_numeric(frame.get("macd_signal")),
            4,
        ),
        (
            "MACD Hist",
            INDICATOR_PANEL_COLORS["macd_hist_up"],
            _last_valid_numeric(frame.get("macd_hist")),
            4,
        ),
    ]
    lines = [
        "<span style='color:"
        f"{color}"
        ";'><b>"
        f"{label}"
        "</b></span>: "
        f"{_format_indicator_value(value, digits=digits)}"
        for label, color, value, digits in items
    ]
    fig.add_annotation(
        x=x_paper,
        y=y_paper,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text="<br>".join(lines),
        font=dict(color=str(palette["text_color"]), size=15),
        bgcolor=_to_rgba(str(palette["paper_bg"]), 0.94),
        bordercolor=_to_rgba(str(palette["text_color"]), 0.25),
        borderwidth=1,
        borderpad=8,
    )


def _render_benchmark_lines_chart(
    *,
    lines: list[dict[str, Any]],
    height: int,
    chart_key: str,
    enable_lightweight: bool = True,
    annotate_extrema: bool = False,
    extrema_series_name: str | None = None,
    watermark_text: str = "",
    highlight_above_benchmark_boxes: bool = False,
):
    palette = _ui_palette()
    renderer = _benchmark_renderer()
    if enable_lightweight and renderer == "lightweight":
        ok = render_lightweight_multi_line_chart(
            lines=lines,
            palette=palette,
            key=f"lw:{chart_key}",
            height=height,
        )
        if ok:
            return
        st.caption("lightweight-charts 渲染失敗，已自動回退 Plotly。")

    benchmark_latest: float | None = None
    if highlight_above_benchmark_boxes:
        for line in lines:
            series = line.get("series")
            if not isinstance(series, pd.Series) or series.empty or not bool(line.get("is_benchmark")):
                continue
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                continue
            benchmark_latest = float(numeric.iloc[-1])
            break

    prepared_lines: list[dict[str, Any]] = []
    for line in lines:
        series = line.get("series")
        if not isinstance(series, pd.Series) or series.empty:
            continue
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            continue
        prepared_lines.append(
            {
                "line": line,
                "series": series,
                "numeric": numeric,
                "latest_value": float(numeric.iloc[-1]),
                "is_benchmark": bool(line.get("is_benchmark")),
            }
        )

    if highlight_above_benchmark_boxes and len(prepared_lines) > 1:
        benchmark_lines = [item for item in prepared_lines if item["is_benchmark"]]
        other_lines = [item for item in prepared_lines if not item["is_benchmark"]]
        other_lines.sort(key=lambda item: item["latest_value"], reverse=True)
        prepared_lines = benchmark_lines + other_lines

    fig = go.Figure()
    plotted_series: list[tuple[str, pd.Series]] = []
    secondary_axis_trace_idx: int | None = None
    for item in prepared_lines:
        line = item["line"]
        series = item["series"]
        numeric = item["numeric"]
        color = str(line.get("color", palette["equity"]))
        width = float(line.get("width", 2.0) or 2.0)
        dash = str(line.get("dash", "solid"))
        name = str(line.get("name", "Series"))
        if (
            highlight_above_benchmark_boxes
            and benchmark_latest is not None
            and not item["is_benchmark"]
            and item["latest_value"] > benchmark_latest
        ):
            name = f"{name} *"
        line_color = color
        if (
            highlight_above_benchmark_boxes
            and benchmark_latest is not None
            and not item["is_benchmark"]
            and item["latest_value"] <= benchmark_latest
        ):
            line_color = _to_rgba(color, 0.58)
        hover_code = str(line.get("hover_code", name))
        value_label = str(line.get("value_label", "Value"))
        y_format = str(line.get("y_format", ",.0f"))
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
                line=dict(color=line_color, width=width, dash=dash),
                hovertemplate=_hovertemplate_with_code(
                    hover_code, value_label=value_label, y_format=y_format
                ),
            )
        )
        if secondary_axis_trace_idx is None and len(plotted_series) >= 1:
            secondary_axis_trace_idx = len(fig.data) - 1
        plotted_series.append((name, series))

    if secondary_axis_trace_idx is not None:
        fig.data[secondary_axis_trace_idx].update(yaxis="y2")
    elif plotted_series:
        mirror_name, mirror_series = plotted_series[0]
        fig.add_trace(
            go.Scatter(
                x=mirror_series.index,
                y=mirror_series.values,
                mode="lines",
                name=f"{mirror_name} (axis mirror)",
                line=dict(color="rgba(0,0,0,0)", width=0.1),
                hoverinfo="skip",
                showlegend=False,
                yaxis="y2",
            )
        )
    if highlight_above_benchmark_boxes:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="* = final value above benchmark",
                line=dict(color="rgba(0,0,0,0)", width=0.1),
                hoverinfo="skip",
            )
        )

    if annotate_extrema and plotted_series:
        target_name = str(extrema_series_name or "").strip()
        target_pair = next((item for item in plotted_series if item[0] == target_name), None)
        if target_pair is None:
            target_pair = plotted_series[0]
        extrema_name, extrema_series = target_pair
        extrema_vals = pd.to_numeric(extrema_series, errors="coerce").dropna()
        if not extrema_vals.empty:

            def _fmt_extrema_date(ts_val: object) -> str:
                try:
                    return pd.Timestamp(ts_val).strftime("%Y-%m-%d")
                except Exception:
                    return str(ts_val)

            high_idx = extrema_vals.idxmax()
            low_idx = extrema_vals.idxmin()
            high_val = float(extrema_vals.loc[high_idx])
            low_val = float(extrema_vals.loc[low_idx])
            high_date = _fmt_extrema_date(high_idx)
            low_date = _fmt_extrema_date(low_idx)
            high_color = str(palette.get("signal_sell", "#D6465A"))
            low_color = str(palette.get("signal_buy", "#2F9E6B"))
            val_fmt = ",.4f" if max(abs(high_val), abs(low_val)) < 10 else ",.2f"
            low_ax = 0
            low_ay = 48
            fig.add_annotation(
                x=high_idx,
                y=high_val,
                text=f"{extrema_name} 最高 {high_val:{val_fmt}}<br>日期 {high_date}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowwidth=1.8,
                arrowcolor=high_color,
                ax=0,
                ay=-48,
                font=dict(color=high_color, size=11),
                bgcolor=_to_rgba(str(palette["paper_bg"]), 0.78),
                bordercolor=high_color,
                borderwidth=1,
            )
            if low_idx != high_idx:
                fig.add_annotation(
                    x=low_idx,
                    y=low_val,
                    text=f"{extrema_name} 最低 {low_val:{val_fmt}}<br>日期 {low_date}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.0,
                    arrowwidth=1.8,
                    arrowcolor=low_color,
                    ax=low_ax,
                    ay=low_ay,
                    font=dict(color=low_color, size=11),
                    bgcolor=_to_rgba(str(palette["paper_bg"]), 0.78),
                    bordercolor=low_color,
                    borderwidth=1,
                )

    legend_name_len = max((len(str(trace.name or "")) for trace in fig.data), default=12)
    axis_tick_gutter = 108
    legend_right_margin = int(min(620, max(300, axis_tick_gutter + legend_name_len * 7 + 56)))
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=legend_right_margin, t=36, b=12),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"])),
        legend=dict(
            orientation="v",
            y=1.0,
            yanchor="top",
            x=1.10,
            xanchor="left",
            bgcolor=_to_rgba(str(palette["paper_bg"]), 0.65),
            bordercolor=str(palette["grid"]),
            borderwidth=1,
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            matches="y",
            showgrid=False,
            showticklabels=True,
            ticks="outside",
            tickfont=dict(color=str(palette["text_color"])),
        ),
    )
    fig.update_xaxes(gridcolor=str(palette["grid"]))
    fig.update_yaxes(gridcolor=str(palette["grid"]))
    fig.update_layout(
        yaxis=dict(gridcolor=str(palette["grid"]), side="left"),
        yaxis2=dict(
            overlaying="y",
            side="right",
            matches="y",
            showgrid=False,
            showticklabels=True,
            ticks="outside",
            tickfont=dict(color=str(palette["text_color"])),
        ),
    )
    x_range = _plotly_datetime_axis_range_with_right_padding(
        [ts for _, series in plotted_series for ts in series.index]
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    edge_marker_x = _plotly_right_edge_marker_x(
        [ts for _, series in plotted_series for ts in series.index]
    )
    if edge_marker_x is not None:
        fig.add_vline(
            x=edge_marker_x,
            line_width=1,
            line_dash="dot",
            line_color=_to_rgba(
                str(palette["text_muted"]) if "text_muted" in palette else str(palette["grid"]), 0.9
            ),
        )
    _apply_unified_benchmark_hover(fig, palette)
    _enable_plotly_draw_tools(fig)
    if not str(watermark_text).strip():
        fallback_code = str(lines[0].get("hover_code", "")) if lines else ""
        watermark_text = fallback_code or str(lines[0].get("name", "")) if lines else ""
    _apply_plotly_watermark(fig, text=str(watermark_text), palette=palette)
    safe_name = str(chart_key).replace(":", "_").replace("/", "_")
    _render_plotly_chart(
        fig,
        chart_key=f"plotly:{chart_key}",
        filename=safe_name,
        scale=2,
        width="stretch",
        watermark_text=str(watermark_text),
        palette=palette,
    )


def _render_live_chart(ind: pd.DataFrame, *, watermark_text: str = ""):
    palette = _ui_palette()
    if _live_kline_renderer() == "lightweight":
        overlays: list[dict[str, object]] = []
        for col, name, color in [
            ("sma_20", "SMA20", str(palette["sma20"])),
            ("sma_60", "SMA60", str(palette["sma60"])),
            ("vwap", "VWAP", str(palette["vwap"])),
            ("bb_upper", "BB上軌", str(palette["bb_upper"])),
            ("bb_lower", "BB下軌", str(palette["bb_lower"])),
        ]:
            if col in ind.columns:
                overlays.append(
                    {
                        "name": name,
                        "series": pd.to_numeric(ind[col], errors="coerce"),
                        "color": color,
                        "width": 2,
                    }
                )
        ok = render_lightweight_live_chart(
            ind=ind,
            palette=palette,
            key="live_chart",
            overlays=overlays,
        )
        if ok:
            return
        st.caption("lightweight-charts 渲染失敗，已自動回退 Plotly。")

    price_up_line = str(palette.get("price_up_line", palette["price_up"]))
    price_down_line = str(palette.get("price_down_line", palette["price_down"]))
    price_up_fill = str(palette.get("price_up_fill", _to_rgba(price_up_line, 0.42)))
    price_down_fill = str(palette.get("price_down_fill", _to_rgba(price_down_line, 0.42)))
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.72, 0.28]
    )
    fig.add_trace(
        go.Candlestick(
            x=ind.index,
            open=ind["open"],
            high=ind["high"],
            low=ind["low"],
            close=ind["close"],
            name="K線",
            increasing_line_color=price_up_line,
            increasing_fillcolor=price_up_fill,
            decreasing_line_color=price_down_line,
            decreasing_fillcolor=price_down_fill,
        ),
        row=1,
        col=1,
    )

    for col, name, color in [
        ("sma_20", "SMA20", str(palette["sma20"])),
        ("sma_60", "SMA60", str(palette["sma60"])),
        ("vwap", "VWAP", str(palette["vwap"])),
        ("bb_upper", "BB上軌", str(palette["bb_upper"])),
        ("bb_lower", "BB下軌", str(palette["bb_lower"])),
    ]:
        if col in ind.columns:
            fig.add_trace(
                go.Scatter(
                    x=ind.index,
                    y=ind[col],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1.3),
                ),
                row=1,
                col=1,
            )

    volume = ind.get("volume", pd.Series(index=ind.index)).fillna(0)
    close_diff = (
        pd.to_numeric(ind.get("close", pd.Series(index=ind.index)), errors="coerce")
        .diff()
        .fillna(0.0)
    )
    volume_colors = np.where(
        close_diff >= 0, str(palette["volume_up"]), str(palette["volume_down"])
    )
    fig.add_trace(
        go.Bar(x=ind.index, y=volume, name="Volume", marker_color=volume_colors),
        row=2,
        col=1,
    )
    fig.update_xaxes(gridcolor=str(palette["grid"]))
    fig.update_yaxes(gridcolor=str(palette["grid"]))
    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_y=1.02,
        margin=dict(l=10, r=10, t=30, b=10),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"])),
    )
    _enable_plotly_draw_tools(fig)
    _apply_plotly_watermark(fig, text=str(watermark_text), palette=palette)
    _render_plotly_chart(
        fig,
        chart_key="plotly:live_chart_main",
        filename="live_chart",
        scale=2,
        width="stretch",
        watermark_text=str(watermark_text),
        palette=palette,
    )


def _render_indicator_panels(
    ind: pd.DataFrame,
    *,
    chart_key: str,
    height: int = 600,
    x_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    watermark_text: str = "",
):
    if not isinstance(ind, pd.DataFrame) or ind.empty or "close" not in ind.columns:
        st.caption("指標副圖：資料不足。")
        return

    frame = ind.copy().sort_index()
    for col in [
        "close",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "sma_20",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "macd",
        "macd_signal",
        "macd_hist",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    palette = _ui_palette()
    indicator_colors = INDICATOR_PANEL_COLORS
    panel_height = max(420, min(int(height), 700))
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.32, 0.22, 0.22, 0.24],
    )

    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            mode="lines",
            name="Close",
            line=dict(color=indicator_colors["close"], width=1.35),
        ),
        row=1,
        col=1,
    )
    bb_mid_col = (
        "bb_mid" if "bb_mid" in frame.columns else ("sma_20" if "sma_20" in frame.columns else "")
    )
    if bb_mid_col:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[bb_mid_col],
                mode="lines",
                name="BB中軌",
                line=dict(color=indicator_colors["bb_mid"], width=1.05),
            ),
            row=1,
            col=1,
        )
    if "bb_upper" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["bb_upper"],
                mode="lines",
                name="BB上軌",
                line=dict(color=indicator_colors["bb_upper"], width=1.05),
            ),
            row=1,
            col=1,
        )
    if "bb_lower" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["bb_lower"],
                mode="lines",
                name="BB下軌",
                line=dict(color=indicator_colors["bb_lower"], width=1.05),
            ),
            row=1,
            col=1,
        )
    panel1_cols = [c for c in ["close", "bb_upper", "bb_mid", "bb_lower"] if c in frame.columns]
    if panel1_cols:
        panel1_vals = (
            pd.concat([pd.to_numeric(frame[c], errors="coerce") for c in panel1_cols], axis=1)
            .stack(future_stack=True)
            .dropna()
        )
        if not panel1_vals.empty:
            y_min = float(panel1_vals.min())
            y_max = float(panel1_vals.max())
            span = y_max - y_min
            pad = max(span * 0.08, max(abs(y_max), 1.0) * 0.01)
            fig.update_yaxes(range=[y_min - pad, y_max + pad], row=1, col=1)

    if "rsi_14" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["rsi_14"],
                mode="lines",
                name="RSI14",
                line=dict(color=indicator_colors["rsi"], width=1.25),
            ),
            row=2,
            col=1,
        )
    fig.add_hline(
        y=70, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=2, col=1
    )
    fig.add_hline(
        y=30, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=2, col=1
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    if "stoch_k" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["stoch_k"],
                mode="lines",
                name="KD-K",
                line=dict(color=indicator_colors["kd_k"], width=1.25),
            ),
            row=3,
            col=1,
        )
    if "stoch_d" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["stoch_d"],
                mode="lines",
                name="KD-D",
                line=dict(color=indicator_colors["kd_d"], width=1.25),
            ),
            row=3,
            col=1,
        )
    fig.add_hline(
        y=80, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=3, col=1
    )
    fig.add_hline(
        y=20, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=3, col=1
    )
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    if "macd_hist" in frame.columns:
        hist_vals = pd.to_numeric(frame["macd_hist"], errors="coerce").fillna(0.0)
        hist_colors = np.where(
            hist_vals >= 0,
            indicator_colors["macd_hist_up"],
            indicator_colors["macd_hist_down"],
        )
        fig.add_trace(
            go.Bar(x=frame.index, y=hist_vals.values, name="MACD Hist", marker_color=hist_colors),
            row=4,
            col=1,
        )
    if "macd" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["macd"],
                mode="lines",
                name="MACD",
                line=dict(color=indicator_colors["macd"], width=1.25),
            ),
            row=4,
            col=1,
        )
    if "macd_signal" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["macd_signal"],
                mode="lines",
                name="MACD Signal",
                line=dict(color=indicator_colors["macd_signal"], width=1.2),
            ),
            row=4,
            col=1,
        )
    fig.add_hline(
        y=0, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=4, col=1
    )

    if x_range is not None:
        x0, x1 = x_range
        for row in (1, 2, 3, 4):
            fig.update_xaxes(range=[x0, x1], row=row, col=1)

    fig.update_xaxes(gridcolor=str(palette["grid"]))
    fig.update_yaxes(gridcolor=str(palette["grid"]))
    fig.update_layout(
        height=panel_height,
        margin=dict(l=60, r=220, t=64, b=8),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"])),
        legend_orientation="h",
        legend_y=1.13,
        legend_yanchor="bottom",
        legend_x=0.0,
        legend_xanchor="left",
        uirevision=f"indicators:{chart_key}",
    )
    _add_indicator_summary_annotation(fig, frame=frame, palette=palette)
    _enable_plotly_draw_tools(fig)
    _apply_plotly_watermark(fig, text=str(watermark_text), palette=palette)
    _render_plotly_chart(
        fig,
        chart_key=str(chart_key),
        filename=str(chart_key),
        scale=2,
        width="stretch",
        watermark_text=str(watermark_text),
        palette=palette,
    )


__all__ = [
    "build_multi_line_styles",
    "_plotly_datetime_axis_range_with_right_padding",
    "_plotly_right_edge_marker_x",
    "_render_benchmark_lines_chart",
    "_render_live_chart",
    "_render_indicator_panels",
]
