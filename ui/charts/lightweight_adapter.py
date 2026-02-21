from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    from streamlit_lightweight_charts import Chart, renderLightweightCharts
except Exception:  # pragma: no cover
    Chart = None  # type: ignore[assignment]
    renderLightweightCharts = None  # type: ignore[assignment]


def _series_type(name: str) -> str:
    # The component frontend switch-cases "Line"/"Candlestick"/... by name.
    # Use enum.name when available, otherwise keep a plain string fallback.
    if Chart is None:
        return name
    item = getattr(Chart, name, None)
    return str(getattr(item, "name", name))

def _to_unix_seconds(ts: pd.Timestamp) -> int:
    return int(ts.tz_convert("UTC").timestamp())


def _line_style_token(dash: str) -> int:
    token = str(dash or "").strip().lower()
    if token in {"dash", "dashed"}:
        return 2
    if token in {"dot", "dotted"}:
        return 1
    if token in {"dashdot", "dash-dot"}:
        return 4
    return 0


def _chart_layout_options(palette: dict[str, object], *, height: int) -> dict[str, object]:
    grid = str(palette.get("grid", "rgba(120,120,120,0.2)"))
    return {
        "height": int(height),
        "layout": {
            "background": {"type": "solid", "color": str(palette.get("plot_bg", palette.get("paper_bg", "#ffffff")))},
            "textColor": str(palette.get("text_color", "#111827")),
        },
        "grid": {
            "vertLines": {"color": grid},
            "horzLines": {"color": grid},
        },
        "rightPriceScale": {
            "borderColor": grid,
            "scaleMargins": {"top": 0.08, "bottom": 0.12},
        },
        "timeScale": {"borderColor": grid},
        "crosshair": {
            "vertLine": {"color": grid, "labelBackgroundColor": str(palette.get("paper_bg", "#ffffff"))},
            "horzLine": {"color": grid, "labelBackgroundColor": str(palette.get("paper_bg", "#ffffff"))},
        },
    }


def _build_candlestick_points(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    bars = frame.copy()
    bars["__time"] = pd.to_datetime(bars.index, utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    bars = bars.dropna(subset=["__time", "open", "high", "low", "close"])
    bars = bars.sort_values("__time")
    points: list[dict[str, object]] = []
    for _, row in bars.iterrows():
        ts = pd.Timestamp(row["__time"])
        points.append(
            {
                "time": _to_unix_seconds(ts),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
    return points


def _build_volume_points(frame: pd.DataFrame, *, up_color: str, down_color: str) -> list[dict[str, object]]:
    if frame.empty:
        return []
    bars = frame.copy()
    bars["__time"] = pd.to_datetime(bars.index, utc=True, errors="coerce")
    bars["volume"] = pd.to_numeric(bars.get("volume"), errors="coerce").fillna(0.0)
    bars["close"] = pd.to_numeric(bars.get("close"), errors="coerce")
    bars["open"] = pd.to_numeric(bars.get("open"), errors="coerce")
    bars = bars.dropna(subset=["__time", "close"]).sort_values("__time")
    points: list[dict[str, object]] = []
    for _, row in bars.iterrows():
        ts = pd.Timestamp(row["__time"])
        is_up = float(row["close"]) >= float(row["open"]) if pd.notna(row["open"]) else True
        points.append(
            {
                "time": _to_unix_seconds(ts),
                "value": float(row["volume"]),
                "color": up_color if is_up else down_color,
            }
        )
    return points


def _build_line_points(series: pd.Series) -> list[dict[str, object]]:
    if series is None or series.empty:
        return []
    frame = pd.DataFrame({"value": pd.to_numeric(series, errors="coerce")})
    frame["time"] = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.dropna(subset=["time", "value"]).sort_values("time")
    if frame.empty:
        return []
    points: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        points.append({"time": _to_unix_seconds(pd.Timestamp(row["time"])), "value": float(row["value"])})
    return points


def render_lightweight_live_chart(
    *,
    ind: pd.DataFrame,
    palette: dict[str, object],
    key: str,
    overlays: Optional[list[dict[str, object]]] = None,
) -> bool:
    if Chart is None or renderLightweightCharts is None:
        return False
    if ind is None or ind.empty or not {"open", "high", "low", "close"}.issubset(ind.columns):
        return False
    candle_points = _build_candlestick_points(ind)
    if not candle_points:
        return False

    price_up = str(palette.get("price_up_line", palette.get("price_up", "#5FA783")))
    price_down = str(palette.get("price_down_line", palette.get("price_down", "#D78C95")))
    volume_up = str(palette.get("volume_up", "rgba(22,163,74,0.35)"))
    volume_down = str(palette.get("volume_down", "rgba(220,38,38,0.28)"))

    price_series: list[dict[str, object]] = [
        {
            "type": _series_type("Candlestick"),
            "data": candle_points,
            "options": {
                "upColor": price_up,
                "downColor": price_down,
                "borderUpColor": price_up,
                "borderDownColor": price_down,
                "wickUpColor": price_up,
                "wickDownColor": price_down,
            },
        }
    ]
    for overlay in overlays or []:
        series = overlay.get("series")
        if not isinstance(series, pd.Series):
            continue
        line_points = _build_line_points(series)
        if not line_points:
            continue
        price_series.append(
            {
                "type": _series_type("Line"),
                "data": line_points,
                "options": {
                    "title": str(overlay.get("name", "")),
                    "color": str(overlay.get("color", "#3b82f6")),
                    "lineWidth": int(overlay.get("width", 2) or 2),
                    "lineStyle": _line_style_token(str(overlay.get("dash", "solid"))),
                    "priceLineVisible": False,
                    "lastValueVisible": True,
                },
            }
        )

    volume_points = _build_volume_points(ind, up_color=volume_up, down_color=volume_down)
    price_chart = {
        "chart": _chart_layout_options(palette, height=540),
        "series": price_series,
    }
    volume_chart = {
        "chart": _chart_layout_options(palette, height=220),
        "series": [
            {
                "type": _series_type("Histogram"),
                "data": volume_points,
                "options": {
                    "title": "Volume",
                    "priceFormat": {"type": "volume"},
                    "priceLineVisible": False,
                    "lastValueVisible": True,
                },
            }
        ],
    }
    try:
        renderLightweightCharts([price_chart, volume_chart], key=key)
        return True
    except Exception:
        return False


def render_lightweight_kline_equity_chart(
    *,
    bars: pd.DataFrame,
    strategy: pd.Series,
    benchmark: Optional[pd.Series],
    palette: dict[str, object],
    key: str,
) -> bool:
    if Chart is None or renderLightweightCharts is None:
        return False
    if bars is None or bars.empty or strategy is None or strategy.empty:
        return False

    candle_points = _build_candlestick_points(bars)
    strategy_points = _build_line_points(strategy)
    benchmark_points = _build_line_points(benchmark if isinstance(benchmark, pd.Series) else pd.Series(dtype=float))
    if not candle_points or not strategy_points:
        return False

    price_up = str(palette.get("price_up_line", palette.get("price_up", "#5FA783")))
    price_down = str(palette.get("price_down_line", palette.get("price_down", "#D78C95")))
    price_chart = {
        "chart": _chart_layout_options(palette, height=520),
        "series": [
            {
                "type": _series_type("Candlestick"),
                "data": candle_points,
                "options": {
                    "upColor": price_up,
                    "downColor": price_down,
                    "borderUpColor": price_up,
                    "borderDownColor": price_down,
                    "wickUpColor": price_up,
                    "wickDownColor": price_down,
                },
            }
        ],
    }
    equity_chart = {
        "chart": _chart_layout_options(palette, height=280),
        "series": [
            {
                "type": _series_type("Line"),
                "data": strategy_points,
                "options": {
                    "title": "Strategy Equity",
                    "color": str(palette.get("equity", "#16a34a")),
                    "lineWidth": 2,
                    "priceLineVisible": False,
                    "lastValueVisible": True,
                },
            }
        ],
    }
    if benchmark_points:
        equity_chart["series"].append(
            {
                "type": _series_type("Line"),
                "data": benchmark_points,
                "options": {
                    "title": "Benchmark Equity",
                    "color": str(palette.get("benchmark", "#64748b")),
                    "lineWidth": 2,
                    "lineStyle": _line_style_token(str(palette.get("benchmark_dash", "dash"))),
                    "priceLineVisible": False,
                    "lastValueVisible": True,
                },
            }
        )
    try:
        renderLightweightCharts([price_chart, equity_chart], key=key)
        return True
    except Exception:
        return False


def render_lightweight_multi_line_chart(
    *,
    lines: list[dict[str, object]],
    palette: dict[str, object],
    key: str,
    height: int = 420,
) -> bool:
    if Chart is None or renderLightweightCharts is None:
        return False
    series_cfg: list[dict[str, object]] = []
    for line in lines:
        series = line.get("series")
        if not isinstance(series, pd.Series):
            continue
        points = _build_line_points(series)
        if not points:
            continue
        series_cfg.append(
            {
                "type": _series_type("Line"),
                "data": points,
                "options": {
                    "title": str(line.get("name", "")),
                    "color": str(line.get("color", "#3b82f6")),
                    "lineWidth": int(line.get("width", 2) or 2),
                    "lineStyle": _line_style_token(str(line.get("dash", "solid"))),
                    "priceLineVisible": False,
                    "lastValueVisible": True,
                },
            }
        )
    if not series_cfg:
        return False
    charts = [
        {
            "chart": _chart_layout_options(palette, height=height),
            "series": series_cfg,
        }
    ]
    try:
        renderLightweightCharts(charts, key=key)
        return True
    except Exception:
        return False
