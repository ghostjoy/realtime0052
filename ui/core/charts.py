from __future__ import annotations

import warnings

_CTX_BOUND = False


def _bind_ctx(ctx: object):
    """Deprecated: Use explicit imports instead of ctx injection."""
    warnings.warn(
        "_bind_ctx is deprecated. Pass dependencies explicitly instead of using ctx=globals()",
        DeprecationWarning,
        stacklevel=2,
    )
    global _CTX_BOUND
    if _CTX_BOUND:
        return
    items = []
    if isinstance(ctx, dict):
        items = list(ctx.items())
    else:
        attrs = getattr(ctx, "__dict__", None)
        if isinstance(attrs, dict):
            items = list(attrs.items())
    module_globals = globals()
    for key, value in items:
        name = str(key or "")
        if not name or name.startswith("__") or (name in module_globals):
            continue
        module_globals[name] = value
    _CTX_BOUND = True

def _render_benchmark_lines_chart(
    *,
    ctx: object,
    lines: list[dict[str, Any]],
    height: int,
    chart_key: str,
    enable_lightweight: bool = True,
    annotate_extrema: bool = False,
    extrema_series_name: Optional[str] = None,
    watermark_text: str = "",
):
    _bind_ctx(ctx)
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

    fig = go.Figure()
    plotted_series: list[tuple[str, pd.Series]] = []
    for line in lines:
        series = line.get("series")
        if not isinstance(series, pd.Series) or series.empty:
            continue
        color = str(line.get("color", palette["equity"]))
        width = float(line.get("width", 2.0) or 2.0)
        dash = str(line.get("dash", "solid"))
        name = str(line.get("name", "Series"))
        hover_code = str(line.get("hover_code", name))
        value_label = str(line.get("value_label", "Value"))
        y_format = str(line.get("y_format", ",.0f"))
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=_hovertemplate_with_code(hover_code, value_label=value_label, y_format=y_format),
            )
        )
        plotted_series.append((name, series))

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

    legend_name_len = max((len(str(line.get("name", ""))) for line in lines), default=12)
    legend_right_margin = int(min(420, max(190, legend_name_len * 7 + 40)))
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
            x=1.01,
            xanchor="left",
            bgcolor=_to_rgba(str(palette["paper_bg"]), 0.65),
            bordercolor=str(palette["grid"]),
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor=str(palette["grid"]))
    fig.update_yaxes(gridcolor=str(palette["grid"]))
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

def _render_live_chart(ind: pd.DataFrame, *, ctx: object, watermark_text: str = ""):
    _bind_ctx(ctx)
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
                overlays.append({"name": name, "series": pd.to_numeric(ind[col], errors="coerce"), "color": color, "width": 2})
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
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.72, 0.28])
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
                go.Scatter(x=ind.index, y=ind[col], mode="lines", name=name, line=dict(color=color, width=1.3)),
                row=1,
                col=1,
            )

    volume = ind.get("volume", pd.Series(index=ind.index)).fillna(0)
    close_diff = pd.to_numeric(ind.get("close", pd.Series(index=ind.index)), errors="coerce").diff().fillna(0.0)
    volume_colors = np.where(close_diff >= 0, str(palette["volume_up"]), str(palette["volume_down"]))
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
    ctx: object,
    chart_key: str,
    height: int = 460,
    x_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    watermark_text: str = "",
):
    _bind_ctx(ctx)
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
    rsi_color = str(palette.get("rsi_line", palette["sma20"]))
    kd_k_color = str(palette.get("kd_k", palette["signal_buy"]))
    kd_d_color = str(palette.get("kd_d", palette["signal_sell"]))
    macd_line_color = str(palette.get("macd_line", palette["equity"]))
    macd_signal_color = str(palette.get("macd_signal", palette["buy_hold"]))
    panel_height = max(320, min(int(height), 580))
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.34, 0.2, 0.2, 0.26],
    )

    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            mode="lines",
            name="Close",
            line=dict(color=str(palette["text_muted"]), width=1.35),
        ),
        row=1,
        col=1,
    )
    bb_mid_col = "bb_mid" if "bb_mid" in frame.columns else ("sma_20" if "sma_20" in frame.columns else "")
    if bb_mid_col:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[bb_mid_col],
                mode="lines",
                name="BB中軌",
                line=dict(color=str(palette["sma20"]), width=1.0),
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
                line=dict(color=str(palette["bb_upper"]), width=1.0),
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
                line=dict(color=str(palette["bb_lower"]), width=1.0),
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
                line=dict(color=rsi_color, width=1.2),
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=70, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    if "stoch_k" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["stoch_k"],
                mode="lines",
                name="KD-K",
                line=dict(color=kd_k_color, width=1.2),
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
                line=dict(color=kd_d_color, width=1.2),
            ),
            row=3,
            col=1,
        )
    fig.add_hline(y=80, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=20, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    if "macd_hist" in frame.columns:
        hist_vals = pd.to_numeric(frame["macd_hist"], errors="coerce").fillna(0.0)
        hist_colors = np.where(hist_vals >= 0, str(palette["volume_up"]), str(palette["volume_down"]))
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
                line=dict(color=macd_line_color, width=1.2),
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
                line=dict(color=macd_signal_color, width=1.1),
            ),
            row=4,
            col=1,
        )
    fig.add_hline(y=0, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=4, col=1)

    if x_range is not None:
        x0, x1 = x_range
        for row in (1, 2, 3, 4):
            fig.update_xaxes(range=[x0, x1], row=row, col=1)

    fig.update_xaxes(gridcolor=str(palette["grid"]))
    fig.update_yaxes(gridcolor=str(palette["grid"]))
    fig.update_layout(
        height=panel_height,
        margin=dict(l=10, r=10, t=64, b=8),
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

__all__ = ["_render_benchmark_lines_chart", "_render_live_chart", "_render_indicator_panels"]
