from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest import CostModel, build_buy_hold_equity, run_backtest, run_portfolio_backtest
from services.backtest_runner import (
    benchmark_candidates,
    default_cost_params,
    load_and_prepare_symbol_bars,
)
from services.sync_orchestrator import normalize_symbols, sync_symbols_if_needed
from utils import normalize_ohlcv_frame


class HistoryStoreLike(Protocol):
    def load_daily_bars(
        self, symbol: str, market: str, start: datetime, end: datetime
    ) -> pd.DataFrame: ...

    def sync_symbol_history(self, symbol: str, market: str, start: datetime, end: datetime): ...

    def load_symbol_metadata(self, symbols: list[str], market: str) -> dict[str, dict[str, object]]: ...


DEFAULT_CHART_THEME = "soft-gray"
SUPPORTED_CHART_LAYOUTS = ("single", "combined", "split")
SUPPORTED_CHART_THEMES = ("paper-light", "soft-gray", "data-dark")
SUPPORTED_CHART_STRATEGIES = (
    "buy_hold",
    "sma_trend_filter",
    "donchian_breakout",
    "sma_cross",
)
_SUPPORTED_BENCHMARK_CHOICES = {
    "TW": {"auto", "twii", "0050", "006208"},
    "US": {"auto", "gspc", "spy", "qqq", "dia"},
}
_DEFAULT_PALETTE = {
    "paper_bg": "#F7F8FA",
    "plot_bg": "#F7F8FA",
    "text_color": "#1F2937",
    "grid": "rgba(107,114,128,0.16)",
    "price_up": "#5F9F84",
    "price_down": "#CF8F98",
    "equity": "#16A34A",
    "buy_hold": "#0F766E",
    "benchmark": "#6B7280",
    "benchmark_dash": "dash",
    "signal_buy": "#0F9D58",
    "signal_sell": "#DC2626",
    "fill_buy": "#16A34A",
    "fill_sell": "#DC2626",
    "trade_path": "#64748B",
    "asset_palette": ["#4B5563", "#0EA5E9", "#10B981", "#F59E0B", "#6366F1", "#E11D48"],
    "plot_template": "plotly_white",
}
_THEME_PALETTES = {
    "paper-light": {
        **_DEFAULT_PALETTE,
        "paper_bg": "#FFFFFF",
        "plot_bg": "#FFFFFF",
        "text_color": "#0F172A",
        "buy_hold": "#0F766E",
        "benchmark": "#64748B",
    },
    "soft-gray": {
        **dict(_DEFAULT_PALETTE),
        "asset_palette": [
            "#1D4ED8",
            "#DC2626",
            "#059669",
            "#D97706",
            "#7C3AED",
            "#0891B2",
            "#BE123C",
            "#65A30D",
        ],
    },
    "data-dark": {
        **_DEFAULT_PALETTE,
        "paper_bg": "#0F1419",
        "plot_bg": "#111821",
        "text_color": "#E7E9EA",
        "grid": "rgba(139,152,165,0.22)",
        "price_up": "#5FA783",
        "price_down": "#D78C95",
        "equity": "#22C55E",
        "buy_hold": "#2DD4BF",
        "benchmark": "#8B98A5",
        "trade_path": "#94A3B8",
        "asset_palette": [
            "#38BDF8",
            "#F97316",
            "#34D399",
            "#F43F5E",
            "#A78BFA",
            "#FACC15",
            "#22C55E",
            "#60A5FA",
        ],
        "plot_template": "plotly_dark",
    },
}


@dataclass(frozen=True)
class SymbolChartPayload:
    symbol: str
    market_code: str
    bars: pd.DataFrame
    strategy_equity: pd.Series
    benchmark_overlay: pd.Series
    benchmark_equity: pd.Series
    benchmark_symbol: str
    signals: pd.Series
    trades: list[object]
    symbol_name: str


@dataclass(frozen=True)
class CombinedChartPayload:
    market_code: str
    strategy_equity: pd.Series
    benchmark_equity: pd.Series
    per_symbol_buy_hold: dict[str, pd.Series]
    benchmark_symbol: str
    symbol_names: dict[str, str]


def _apply_total_return_adjustment_chart(
    bars: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return bars, {"applied": False, "reason": "empty"}
    if "adj_close" not in bars.columns or "close" not in bars.columns:
        return bars, {"applied": False, "reason": "no_adj_close"}

    close = pd.to_numeric(bars["close"], errors="coerce")
    adj_close = pd.to_numeric(bars["adj_close"], errors="coerce")
    valid = close.gt(0) & adj_close.gt(0)
    coverage_pct = float(valid.mean() * 100.0) if len(valid) else 0.0
    if coverage_pct < 60.0:
        return bars, {"applied": False, "reason": "coverage_low", "coverage_pct": coverage_pct}

    out = bars.copy()
    factor = (adj_close / close).where(valid)
    for col in ["open", "high", "low", "close"]:
        if col in out.columns:
            base = pd.to_numeric(out[col], errors="coerce")
            adjusted = (base * factor).where(valid)
            out[col] = adjusted.where(valid, base)
    out["close"] = adj_close.where(valid, out["close"])
    return out, {"applied": True, "coverage_pct": coverage_pct}


def _normalize_market(market: str) -> str:
    token = str(market or "TW").strip().upper()
    if token in {"TW", "OTC"}:
        return "TW"
    if token in {"US", "USA"}:
        return "US"
    raise ValueError(f"unsupported market: {market}")


def _infer_market_from_symbol(symbol: str) -> str:
    token = str(symbol or "").strip().upper()
    if token.isdigit() and len(token) in {4, 5, 6}:
        return "TW"
    return "US"


def _safe_token(value: str, *, fallback: str) -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    token = token.strip("._-")
    return token or fallback


def _to_rgba(color: str, alpha: float) -> str:
    text = str(color).strip()
    a = min(max(float(alpha), 0.0), 1.0)
    if text.startswith("#") and len(text) == 7:
        r = int(text[1:3], 16)
        g = int(text[3:5], 16)
        b = int(text[5:7], 16)
        return f"rgba({r},{g},{b},{a:.3f})"
    if text.startswith("#") and len(text) == 4:
        r = int(text[1] * 2, 16)
        g = int(text[2] * 2, 16)
        b = int(text[3] * 2, 16)
        return f"rgba({r},{g},{b},{a:.3f})"
    if text.startswith("rgb(") and text.endswith(")"):
        inner = text[4:-1]
        return f"rgba({inner},{a:.3f})"
    if text.startswith("rgba("):
        return text
    return text


def _benchmark_line_style(palette: dict[str, object], *, width: float = 2.0) -> dict[str, object]:
    return {
        "color": str(palette["benchmark"]),
        "width": float(width),
        "dash": str(palette.get("benchmark_dash", "dash")),
    }


def _plotly_datetime_axis_range_with_right_padding(
    index_values: Iterable[object],
    *,
    pad_ratio: float = 0.05,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    idx = pd.to_datetime(index_values, utc=True, errors="coerce")
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    idx = idx.dropna().sort_values().unique()
    if len(idx) == 0:
        return None

    visible = pd.DatetimeIndex(idx).sort_values()
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
    return (pd.Timestamp(visible[0]), pd.Timestamp(visible[-1]) + right_pad)


def _build_benchmark_overlay(price_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    common_index = price_close.index.intersection(benchmark_close.index)
    if len(common_index) < 2:
        return pd.Series(dtype=float)
    focus_close = pd.to_numeric(price_close.reindex(common_index).ffill(), errors="coerce").dropna()
    bench_close = (
        pd.to_numeric(benchmark_close.reindex(common_index).ffill(), errors="coerce").dropna()
    )
    common_index = focus_close.index.intersection(bench_close.index)
    if len(common_index) < 2:
        return pd.Series(dtype=float)
    focus_base = float(focus_close.iloc[0])
    bench_base = float(bench_close.iloc[0])
    if not math.isfinite(focus_base) or focus_base <= 0 or not math.isfinite(bench_base) or bench_base <= 0:
        return pd.Series(dtype=float)
    return ((bench_close / bench_base) * focus_base).dropna()


def _build_benchmark_equity(
    benchmark_close: pd.Series, strategy_index: pd.Index, initial_capital: float
) -> pd.Series:
    aligned = pd.to_numeric(benchmark_close.reindex(strategy_index).ffill(), errors="coerce").dropna()
    if len(aligned) < 2:
        return pd.Series(dtype=float)
    base_val = float(aligned.iloc[0])
    if not math.isfinite(base_val) or base_val <= 0:
        return pd.Series(dtype=float)
    return (aligned / base_val) * float(initial_capital)


def _normalize_to_base_100(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    base = float(clean.iloc[0])
    if not math.isfinite(base) or base <= 0:
        return pd.Series(dtype=float)
    return (clean / base) * 100.0


def _normalize_to_base_one(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    base = float(clean.iloc[0])
    if not math.isfinite(base) or base <= 0:
        return pd.Series(dtype=float)
    return clean / base


def _resolve_symbol_names(
    store: HistoryStoreLike, symbols: list[str], market_code: str
) -> dict[str, str]:
    loader = getattr(store, "load_symbol_metadata", None)
    if not callable(loader) or not symbols:
        return {}
    try:
        metadata = loader(symbols, market_code)
    except Exception:
        return {}
    if not isinstance(metadata, dict):
        return {}
    out: dict[str, str] = {}
    for symbol in symbols:
        row = metadata.get(symbol)
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if name:
            out[symbol] = name
    return out


def _format_symbol_label(symbol: str, symbol_names: dict[str, str]) -> str:
    name = str(symbol_names.get(symbol, "")).strip()
    if name and name != symbol:
        return f"{symbol} {name}"
    return symbol


def _display_benchmark_symbol(symbol: str) -> str:
    return str(symbol or "Benchmark").strip().lstrip("^") or "Benchmark"


def _format_compact_value(value: float) -> str:
    amount = float(value)
    abs_amount = abs(amount)
    if abs_amount >= 1_000_000_000:
        text = f"{amount / 1_000_000_000:.6f}".rstrip("0").rstrip(".")
        return f"{text}B"
    if abs_amount >= 1_000_000:
        text = f"{amount / 1_000_000:.6f}".rstrip("0").rstrip(".")
        return f"{text}M"
    if abs_amount >= 1_000:
        text = f"{amount / 1_000:.6f}".rstrip("0").rstrip(".")
        return f"{text}K"
    return f"{amount:,.0f}"


def _add_equity_summary_box(
    fig: go.Figure,
    *,
    equity_series: pd.Series,
    benchmark_series: pd.Series,
    palette: dict[str, object],
    x_paper: float = 1.01,
    y_paper: float = 0.315,
) -> None:
    equity_clean = pd.to_numeric(equity_series, errors="coerce").dropna()
    benchmark_clean = pd.to_numeric(benchmark_series, errors="coerce").dropna()
    lines: list[str] = []
    if not equity_clean.empty:
        lines.append(
            "<span style='color:"
            f"{str(palette['equity'])}"
            ";'><b>Equity</b></span>: "
            f"{_format_compact_value(float(equity_clean.iloc[-1]))}"
        )
    if not benchmark_clean.empty:
        lines.append(
            "<span style='color:"
            f"{str(palette['benchmark'])}"
            ";'><b>Benchmark Eq.</b></span>: "
            f"{_format_compact_value(float(benchmark_clean.iloc[-1]))}"
        )
    if not lines:
        return
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
        font=dict(color=str(palette.get("text_color", "#0F172A")), size=15),
        bgcolor=_to_rgba(str(palette["paper_bg"]), 0.94),
        bordercolor=_to_rgba(str(palette.get("text_color", "#0F172A")), 0.25),
        borderwidth=1,
        borderpad=8,
    )


def _plotly_right_edge_marker_x(index_values: Iterable[object]) -> pd.Timestamp | None:
    idx = pd.to_datetime(index_values, utc=True, errors="coerce")
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    idx = idx.dropna().sort_values()
    if len(idx) == 0:
        return None
    return pd.Timestamp(idx[-1])


def _resolve_palette(theme: str) -> dict[str, object]:
    token = str(theme or DEFAULT_CHART_THEME).strip().lower()
    if token not in _THEME_PALETTES:
        raise ValueError(f"unsupported theme: {theme}")
    return dict(_THEME_PALETTES[token])


def _resolve_symbol_markets(symbols: list[str], market: str) -> dict[str, str]:
    token = str(market or "auto").strip().upper()
    if token == "AUTO":
        return {symbol: _infer_market_from_symbol(symbol) for symbol in symbols}
    market_code = _normalize_market(token)
    return {symbol: market_code for symbol in symbols}


def _validate_benchmark_choice(choice: str, market_code: str) -> str:
    token = str(choice or "auto").strip().lower()
    supported = _SUPPORTED_BENCHMARK_CHOICES[str(market_code).strip().upper()]
    if token not in supported:
        raise ValueError(
            f"unsupported benchmark choice for {market_code}: {choice}. "
            f"supported={','.join(sorted(supported))}"
        )
    return token


def _build_strategy_params(
    *,
    strategy: str,
    fast: int | None,
    slow: int | None,
    trend: int | None,
    entry_n: int | None,
    exit_n: int | None,
) -> dict[str, float]:
    token = str(strategy or "").strip()
    if token == "buy_hold":
        return {}
    if token == "sma_cross":
        return {"fast": float(fast if fast is not None else 10), "slow": float(slow if slow is not None else 30)}
    if token == "sma_trend_filter":
        return {
            "fast": float(fast if fast is not None else 20),
            "slow": float(slow if slow is not None else 60),
            "trend": float(trend if trend is not None else 120),
        }
    if token == "donchian_breakout":
        entry_val = int(entry_n if entry_n is not None else 55)
        exit_default = min(20, max(5, entry_val - 1))
        exit_val = int(exit_n if exit_n is not None else exit_default)
        if exit_val >= entry_val:
            raise ValueError("exit_n must be smaller than entry_n for donchian_breakout")
        return {
            "entry_n": float(entry_val),
            "exit_n": float(exit_val),
            "trend": float(trend if trend is not None else 120),
        }
    raise ValueError(f"unsupported strategy: {strategy}")


def _load_benchmark_bars(
    *,
    store: HistoryStoreLike,
    market_code: str,
    start: datetime,
    end: datetime,
    choice: str,
    sync_before_run: bool,
) -> tuple[pd.DataFrame, str]:
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    for benchmark_symbol in benchmark_candidates(market_code, choice):
        bars = normalize_ohlcv_frame(
            store.load_daily_bars(symbol=benchmark_symbol, market=market_code, start=start, end=end)
        )
        needs_sync = bool(sync_before_run) and (
            bars.empty or pd.Timestamp(bars.index.max()).tz_convert("UTC") < end_ts
        )
        if needs_sync:
            store.sync_symbol_history(symbol=benchmark_symbol, market=market_code, start=start, end=end)
            bars = normalize_ohlcv_frame(
                store.load_daily_bars(
                    symbol=benchmark_symbol, market=market_code, start=start, end=end
                )
            )
        if bars.empty or "close" not in bars.columns:
            continue
        return bars.sort_index(), benchmark_symbol
    return pd.DataFrame(), ""


def _build_symbol_payloads(
    *,
    store: HistoryStoreLike,
    symbols: list[str],
    market_code: str,
    start: datetime,
    end: datetime,
    strategy: str,
    strategy_params: dict[str, float],
    initial_capital: float,
    benchmark_choice: str,
    fee_rate: float | None,
    sell_tax: float | None,
    slippage: float | None,
    sync_before_run: bool,
    use_split_adjustment: bool,
    use_total_return_adjustment: bool,
) -> dict[str, SymbolChartPayload]:
    clean_symbols = normalize_symbols(symbols)
    if not clean_symbols:
        return {}
    symbol_names = _resolve_symbol_names(store, clean_symbols, market_code)

    if sync_before_run:
        sync_symbols_if_needed(
            store=store,
            market=market_code,
            symbols=clean_symbols,
            start=start,
            end=end,
            parallel=True,
            max_workers=min(max(len(clean_symbols), 1), 6),
            mode="backfill",
            min_rows=120,
        )

    prepared = load_and_prepare_symbol_bars(
        store=store,
        market_code=market_code,
        symbols=clean_symbols,
        start=start,
        end=end,
        use_total_return_adjustment=use_total_return_adjustment,
        use_split_adjustment=use_split_adjustment,
        auto_detect_split=True,
        apply_total_return_adjustment=_apply_total_return_adjustment_chart,
    )
    bars_by_symbol = prepared.bars_by_symbol
    if not bars_by_symbol:
        return {}

    benchmark_bars, benchmark_symbol = _load_benchmark_bars(
        store=store,
        market_code=market_code,
        start=start,
        end=end,
        choice=benchmark_choice,
        sync_before_run=sync_before_run,
    )
    benchmark_close = pd.to_numeric(
        benchmark_bars.get("close", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if benchmark_close.empty:
        raise ValueError(f"benchmark data unavailable for market={market_code} choice={benchmark_choice}")

    fee_default, tax_default, slip_default = default_cost_params(market_code, clean_symbols)
    cost_model = CostModel(
        fee_rate=float(fee_default if fee_rate is None else fee_rate),
        sell_tax_rate=float(tax_default if sell_tax is None else sell_tax),
        slippage_rate=float(slip_default if slippage is None else slippage),
    )

    payloads: dict[str, SymbolChartPayload] = {}
    for symbol, bars in bars_by_symbol.items():
        result = run_backtest(
            bars=bars,
            strategy_name=strategy,
            strategy_params=dict(strategy_params),
            cost_model=cost_model,
            initial_capital=float(initial_capital),
        )
        strategy_equity = result.equity_curve.get("equity", pd.Series(dtype=float))
        if not isinstance(strategy_equity, pd.Series) or strategy_equity.empty:
            continue
        price_close = pd.to_numeric(bars.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
        benchmark_overlay = _build_benchmark_overlay(price_close, benchmark_close)
        benchmark_equity = _build_benchmark_equity(
            benchmark_close, strategy_equity.index, float(initial_capital)
        )
        payloads[symbol] = SymbolChartPayload(
            symbol=symbol,
            market_code=market_code,
            bars=bars,
            strategy_equity=strategy_equity,
            benchmark_overlay=benchmark_overlay,
            benchmark_equity=benchmark_equity,
            benchmark_symbol=benchmark_symbol,
            signals=result.signals if isinstance(result.signals, pd.Series) else pd.Series(dtype=int),
            trades=list(result.trades or []),
            symbol_name=str(symbol_names.get(symbol, "")).strip(),
        )
    return payloads


def _build_combined_chart_payload(
    *,
    store: HistoryStoreLike,
    symbols: list[str],
    market_code: str,
    start: datetime,
    end: datetime,
    strategy: str,
    strategy_params: dict[str, float],
    initial_capital: float,
    benchmark_choice: str,
    fee_rate: float | None,
    sell_tax: float | None,
    slippage: float | None,
    sync_before_run: bool,
    use_split_adjustment: bool,
    use_total_return_adjustment: bool,
) -> CombinedChartPayload:
    clean_symbols = normalize_symbols(symbols)
    if not clean_symbols:
        raise ValueError("no valid symbols provided")
    symbol_names = _resolve_symbol_names(store, clean_symbols, market_code)

    if sync_before_run:
        sync_symbols_if_needed(
            store=store,
            market=market_code,
            symbols=clean_symbols,
            start=start,
            end=end,
            parallel=True,
            max_workers=min(max(len(clean_symbols), 1), 6),
            mode="backfill",
            min_rows=120,
        )

    prepared = load_and_prepare_symbol_bars(
        store=store,
        market_code=market_code,
        symbols=clean_symbols,
        start=start,
        end=end,
        use_total_return_adjustment=use_total_return_adjustment,
        use_split_adjustment=use_split_adjustment,
        auto_detect_split=True,
        apply_total_return_adjustment=_apply_total_return_adjustment_chart,
    )
    bars_by_symbol = prepared.bars_by_symbol
    if not bars_by_symbol:
        raise ValueError("no symbols produced chart data")

    target_index = pd.DatetimeIndex([])
    for bars in bars_by_symbol.values():
        target_index = target_index.union(pd.DatetimeIndex(bars.index))
    target_index = target_index.sort_values()
    if len(target_index) < 2:
        raise ValueError("not enough overlapping bars for combined chart")

    fee_default, tax_default, slip_default = default_cost_params(market_code, clean_symbols)
    cost_model = CostModel(
        fee_rate=float(fee_default if fee_rate is None else fee_rate),
        sell_tax_rate=float(tax_default if sell_tax is None else sell_tax),
        slippage_rate=float(slip_default if slippage is None else slippage),
    )
    if len(bars_by_symbol) == 1:
        only_symbol = next(iter(bars_by_symbol.keys()))
        strategy_result = run_backtest(
            bars=bars_by_symbol[only_symbol],
            strategy_name=strategy,
            strategy_params=dict(strategy_params),
            cost_model=cost_model,
            initial_capital=float(initial_capital),
        )
        strategy_equity = strategy_result.equity_curve.get("equity", pd.Series(dtype=float))
    else:
        portfolio_result = run_portfolio_backtest(
            bars_by_symbol=bars_by_symbol,
            strategy_name=strategy,
            strategy_params=dict(strategy_params),
            cost_model=cost_model,
            initial_capital=float(initial_capital),
        )
        strategy_equity = portfolio_result.equity_curve.get("equity", pd.Series(dtype=float))
    strategy_equity = pd.to_numeric(strategy_equity, errors="coerce").dropna().sort_index()
    if len(strategy_equity) < 2:
        raise ValueError("strategy equity unavailable for combined chart")

    per_symbol_buy_hold: dict[str, pd.Series] = {}
    for symbol, bars in bars_by_symbol.items():
        eq_sym = build_buy_hold_equity(
            bars_by_symbol={symbol: bars},
            target_index=target_index,
            initial_capital=float(initial_capital),
        ).dropna()
        if len(eq_sym) >= 2:
            per_symbol_buy_hold[symbol] = eq_sym

    benchmark_bars, benchmark_symbol = _load_benchmark_bars(
        store=store,
        market_code=market_code,
        start=start,
        end=end,
        choice=benchmark_choice,
        sync_before_run=sync_before_run,
    )
    benchmark_close = pd.to_numeric(
        benchmark_bars.get("close", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    benchmark_equity = _build_benchmark_equity(benchmark_close, target_index, float(initial_capital))
    if len(benchmark_equity) < 2:
        raise ValueError("benchmark equity unavailable for combined chart")

    common_index = pd.DatetimeIndex(strategy_equity.index).intersection(pd.DatetimeIndex(benchmark_equity.index))
    for series in per_symbol_buy_hold.values():
        common_index = common_index.intersection(pd.DatetimeIndex(series.index))
    common_index = common_index.sort_values()
    if len(common_index) < 2:
        raise ValueError("strategy/buy-hold/benchmark missing enough overlap for combined chart")

    aligned_strategy = strategy_equity.reindex(common_index).ffill().dropna()
    aligned_benchmark = benchmark_equity.reindex(common_index).ffill().dropna()
    aligned_per_symbol: dict[str, pd.Series] = {}
    for symbol, series in per_symbol_buy_hold.items():
        aligned = series.reindex(common_index).ffill().dropna()
        if len(aligned) >= 2:
            aligned_per_symbol[symbol] = aligned

    return CombinedChartPayload(
        market_code=market_code,
        strategy_equity=aligned_strategy,
        benchmark_equity=aligned_benchmark,
        per_symbol_buy_hold=aligned_per_symbol,
        benchmark_symbol=benchmark_symbol,
        symbol_names=symbol_names,
    )


def _style_figure(fig: go.Figure, *, palette: dict[str, object], height: int, title: str) -> None:
    fig.update_xaxes(
        gridcolor=str(palette["grid"]),
        tickfont=dict(size=14),
        automargin=True,
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=str(palette["grid"]),
        tickfont=dict(size=14),
        automargin=True,
        zeroline=False,
    )
    fig.update_layout(
        height=int(height),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=15),
            bordercolor=_to_rgba(str(palette["grid"]), 0.85),
            borderwidth=1,
            bgcolor=_to_rgba(str(palette["paper_bg"]), 0.78),
        ),
        margin=dict(l=16, r=16, t=64, b=20),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"]), size=15),
        title=dict(text=title, x=0.01, font=dict(size=18)),
    )


def _build_single_symbol_backtest_figure(
    payload: SymbolChartPayload,
    *,
    strategy: str,
    start: datetime,
    end: datetime,
    palette: dict[str, object],
    height: int,
    annotate_extrema: bool = False,
    show_signals: bool = False,
    show_fills: bool = False,
    show_trade_path: bool = False,
    show_end_marker: bool = False,
) -> go.Figure:
    bars = payload.bars.sort_index()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.68, 0.32],
    )
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="Price",
            increasing_line_color=str(palette["price_up"]),
            increasing_fillcolor=_to_rgba(str(palette["price_up"]), 0.35),
            decreasing_line_color=str(palette["price_down"]),
            decreasing_fillcolor=_to_rgba(str(palette["price_down"]), 0.35),
        ),
        row=1,
        col=1,
    )
    if not payload.benchmark_overlay.empty:
        fig.add_trace(
            go.Scatter(
                x=payload.benchmark_overlay.index,
                y=payload.benchmark_overlay.values,
                mode="lines",
                name=f"{_display_benchmark_symbol(payload.benchmark_symbol)}（同基準價）",
                line=_benchmark_line_style(palette, width=1.6),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=payload.strategy_equity.index,
            y=payload.strategy_equity.values,
            mode="lines",
            name="Equity",
            line={"color": str(palette["equity"]), "width": 2.1},
        ),
        row=2,
        col=1,
    )
    if not payload.benchmark_equity.empty:
        fig.add_trace(
            go.Scatter(
                x=payload.benchmark_equity.index,
                y=payload.benchmark_equity.values,
                mode="lines",
                name="Benchmark Equity",
                line=_benchmark_line_style(palette, width=2.0),
            ),
            row=2,
            col=1,
        )
    high_series = pd.to_numeric(bars.get("high", pd.Series(dtype=float)), errors="coerce")
    low_series = pd.to_numeric(bars.get("low", pd.Series(dtype=float)), errors="coerce")
    if annotate_extrema and high_series.notna().any():
        highest_idx = high_series.idxmax()
        highest_val = float(high_series.loc[highest_idx])
        highest_date = pd.Timestamp(highest_idx).strftime("%Y-%m-%d")
        fig.add_annotation(
            x=highest_idx,
            y=highest_val,
            text=f"最高價 {highest_val:.2f}<br>日期 {highest_date}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=1.8,
            arrowcolor=str(palette.get("signal_sell", palette["price_down"])),
            ax=0,
            ay=-42,
            font=dict(color=str(palette.get("signal_sell", palette["price_down"])), size=11),
            bgcolor=_to_rgba(str(palette["paper_bg"]), 0.78),
            bordercolor=str(palette.get("signal_sell", palette["price_down"])),
            borderwidth=1,
            row=1,
            col=1,
        )
    if annotate_extrema and low_series.notna().any():
        lowest_idx = low_series.idxmin()
        lowest_val = float(low_series.loc[lowest_idx])
        lowest_date = pd.Timestamp(lowest_idx).strftime("%Y-%m-%d")
        fig.add_annotation(
            x=lowest_idx,
            y=lowest_val,
            text=f"最低價 {lowest_val:.2f}<br>日期 {lowest_date}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=1.8,
            arrowcolor=str(palette.get("signal_buy", palette["price_up"])),
            ax=0,
            ay=42,
            font=dict(color=str(palette.get("signal_buy", palette["price_up"])), size=11),
            bgcolor=_to_rgba(str(palette["paper_bg"]), 0.78),
            bordercolor=str(palette.get("signal_buy", palette["price_up"])),
            borderwidth=1,
            row=1,
            col=1,
        )
    if show_signals and isinstance(payload.signals, pd.Series) and not payload.signals.empty:
        sig_now = payload.signals.reindex(bars.index).ffill().fillna(0).astype(int)
        buy_idx = sig_now[(sig_now == 1) & (sig_now.shift(1).fillna(0) == 0)].index
        sell_idx = sig_now[(sig_now == 0) & (sig_now.shift(1).fillna(0) == 1)].index
        if len(buy_idx) > 0:
            buy_px = bars.loc[buy_idx.intersection(bars.index), "close"]
            fig.add_trace(
                go.Scatter(
                    x=buy_px.index,
                    y=buy_px.values,
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(
                        color=str(palette.get("signal_buy", palette["price_up"])),
                        size=11,
                        symbol="triangle-up",
                        line=dict(color=str(palette.get("text_color", "#0F172A")), width=1),
                    ),
                ),
                row=1,
                col=1,
            )
        if len(sell_idx) > 0:
            sell_px = bars.loc[sell_idx.intersection(bars.index), "close"]
            fig.add_trace(
                go.Scatter(
                    x=sell_px.index,
                    y=sell_px.values,
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(
                        color=str(palette.get("signal_sell", palette["price_down"])),
                        size=11,
                        symbol="triangle-down",
                        line=dict(color=str(palette.get("text_color", "#0F172A")), width=1),
                    ),
                ),
                row=1,
                col=1,
            )
    if show_fills or show_trade_path:
        eq_series = pd.to_numeric(payload.strategy_equity, errors="coerce").dropna()
        buy_x: list[pd.Timestamp] = []
        buy_y: list[float] = []
        sell_x: list[pd.Timestamp] = []
        sell_y: list[float] = []
        first_line = True
        for trade in payload.trades:
            entry_dt = pd.Timestamp(getattr(trade, "entry_date", pd.NaT))
            exit_dt = pd.Timestamp(getattr(trade, "exit_date", pd.NaT))
            if entry_dt in eq_series.index:
                buy_x.append(entry_dt)
                buy_y.append(float(eq_series.loc[entry_dt]))
            else:
                entry_val = eq_series.reindex([entry_dt], method="ffill")
                if not entry_val.empty and pd.notna(entry_val.iloc[0]):
                    buy_x.append(entry_dt)
                    buy_y.append(float(entry_val.iloc[0]))
            if exit_dt in eq_series.index:
                sell_x.append(exit_dt)
                sell_y.append(float(eq_series.loc[exit_dt]))
            else:
                exit_val = eq_series.reindex([exit_dt], method="ffill")
                if not exit_val.empty and pd.notna(exit_val.iloc[0]):
                    sell_x.append(exit_dt)
                    sell_y.append(float(exit_val.iloc[0]))
            if show_trade_path:
                entry_val = eq_series.reindex([entry_dt], method="ffill")
                exit_val = eq_series.reindex([exit_dt], method="ffill")
                if (
                    not entry_val.empty
                    and not exit_val.empty
                    and pd.notna(entry_val.iloc[0])
                    and pd.notna(exit_val.iloc[0])
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=[entry_dt, exit_dt],
                            y=[float(entry_val.iloc[0]), float(exit_val.iloc[0])],
                            mode="lines",
                            name="Trade Path",
                            line=dict(color=str(palette.get("trade_path", palette["benchmark"])), width=1),
                            showlegend=first_line,
                        ),
                        row=2,
                        col=1,
                    )
                    first_line = False
        if show_fills and buy_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode="markers",
                    name="Buy Fill",
                    marker=dict(
                        color=str(palette.get("fill_buy", palette["price_up"])),
                        size=13,
                        symbol="triangle-up",
                        line=dict(color=str(palette.get("text_color", "#0F172A")), width=1),
                    ),
                ),
                row=2,
                col=1,
            )
        if show_fills and sell_x:
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode="markers",
                    name="Sell Fill",
                    marker=dict(
                        color=str(palette.get("fill_sell", palette["price_down"])),
                        size=13,
                        symbol="triangle-down",
                        line=dict(color=str(palette.get("text_color", "#0F172A")), width=1),
                    ),
                ),
                row=2,
                col=1,
            )
    padded_x_range = _plotly_datetime_axis_range_with_right_padding(bars.index)
    if padded_x_range is not None:
        fig.update_xaxes(range=list(padded_x_range), row=1, col=1)
        fig.update_xaxes(range=list(padded_x_range), row=2, col=1)
    _style_figure(
        fig,
        palette=palette,
        height=height,
        title=(
            f"{payload.symbol} 回測圖 | {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')} "
            f"| strategy={strategy} | benchmark={payload.benchmark_symbol}"
        ),
    )
    fig.update_layout(
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01),
        margin=dict(l=16, r=260, t=64, b=20),
    )
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(tickformat="~s", row=2, col=1)
    _add_equity_summary_box(
        fig,
        equity_series=payload.strategy_equity,
        benchmark_series=payload.benchmark_equity,
        palette=palette,
        x_paper=1.01,
        y_paper=0.315,
    )
    if show_end_marker:
        edge_marker_x = _plotly_right_edge_marker_x(bars.index)
        if edge_marker_x is not None:
            fig.add_shape(
                type="line",
                x0=edge_marker_x,
                x1=edge_marker_x,
                y0=0.0,
                y1=1.0,
                xref="x",
                yref="paper",
                line=dict(
                    color=_to_rgba(str(palette.get("text_color", palette["benchmark"])), 0.65),
                    width=1,
                    dash="dot",
                ),
            )
    return fig


def _build_combined_backtest_figure(
    payload: CombinedChartPayload,
    *,
    strategy: str,
    start: datetime,
    end: datetime,
    palette: dict[str, object],
    height: int,
    include_ew_portfolio: bool = False,
) -> go.Figure:
    fig = go.Figure()
    palette_cycle = list(palette["asset_palette"])
    benchmark_label = payload.benchmark_symbol or "Benchmark"
    benchmark_series = _normalize_to_base_one(payload.benchmark_equity)
    strategy_series = _normalize_to_base_one(payload.strategy_equity)
    strategy_line_name = "Buy and Hold (EW Portfolio)" if strategy == "buy_hold" else "Strategy Equity"
    if not benchmark_series.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_series.index,
                y=benchmark_series.values,
                mode="lines",
                name=f"Benchmark ({benchmark_label})",
                line=_benchmark_line_style(palette, width=2.0),
            )
        )
    if include_ew_portfolio and not strategy_series.empty:
        fig.add_trace(
            go.Scatter(
                x=strategy_series.index,
                y=strategy_series.values,
                mode="lines",
                name=strategy_line_name,
                line={
                    "color": str(palette["buy_hold"] if strategy == "buy_hold" else palette["equity"]),
                    "width": 2.2,
                },
            )
        )
    all_index: list[pd.Index] = []
    if not benchmark_series.empty:
        all_index.append(benchmark_series.index)
    if include_ew_portfolio and not strategy_series.empty:
        all_index.append(strategy_series.index)
    for idx, symbol in enumerate(sorted(payload.per_symbol_buy_hold.keys())):
        series = _normalize_to_base_one(payload.per_symbol_buy_hold[symbol])
        if series.empty:
            continue
        color = palette_cycle[idx % len(palette_cycle)]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=f"Buy and Hold ({_format_symbol_label(symbol, payload.symbol_names)})",
                line={"color": color, "width": 1.9},
            )
        )
        all_index.append(series.index)
    if all_index:
        merged_index = pd.DatetimeIndex(sorted(set().union(*[list(idx) for idx in all_index])))
        padded_x_range = _plotly_datetime_axis_range_with_right_padding(merged_index)
        if padded_x_range is not None:
            fig.update_xaxes(range=list(padded_x_range))
    _style_figure(
        fig,
        palette=palette,
        height=height,
        title=(
            f"多標的回測圖 | {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')} "
            f"| strategy={strategy} | benchmark={benchmark_label}"
        ),
    )
    fig.update_layout(
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01),
        margin=dict(l=16, r=300, t=64, b=20),
    )
    fig.update_yaxes(tickformat=".3~g")
    edge_marker_x = _plotly_right_edge_marker_x(strategy_series.index if not strategy_series.empty else [])
    if edge_marker_x is not None:
        fig.add_shape(
            type="line",
            x0=edge_marker_x,
            x1=edge_marker_x,
            y0=0.0,
            y1=1.0,
            xref="x",
            yref="paper",
            line=dict(
                color=_to_rgba(str(palette.get("text_color", palette["benchmark"])), 0.65),
                width=1,
                dash="dot",
            ),
        )
    return fig


def _default_output_root() -> Path:
    return Path("artifacts") / "charts"


def _resolve_output_path(
    *,
    layout: str,
    symbol: str,
    start: datetime,
    end: datetime,
    out: str | None,
    out_dir: str | None,
    combined_count: int | None = None,
) -> Path:
    if out:
        return Path(out).expanduser()
    target_dir = Path(out_dir).expanduser() if out_dir else _default_output_root()
    target_dir.mkdir(parents=True, exist_ok=True)
    if layout == "combined":
        filename = (
            f"chart_backtest_combined_{int(combined_count or 1)}symbols_"
            f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.png"
        )
    else:
        filename = f"chart_backtest_{_safe_token(symbol, fallback='symbol')}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.png"
    return target_dir / filename


def _write_figure_image(
    fig: go.Figure,
    *,
    output_path: Path,
    width: int,
    height: int,
    scale: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", width=int(width), height=int(height), scale=int(scale))


def export_backtest_chart_artifact(
    *,
    store: HistoryStoreLike,
    symbols: list[str],
    layout: str,
    market: str,
    start: datetime,
    end: datetime,
    strategy: str,
    benchmark_choice: str,
    initial_capital: float,
    fee_rate: float | None,
    sell_tax: float | None,
    slippage: float | None,
    sync_before_run: bool,
    use_split_adjustment: bool,
    use_total_return_adjustment: bool,
    theme: str,
    width: int,
    height: int,
    scale: int,
    out: str | None = None,
    out_dir: str | None = None,
    fast: int | None = None,
    slow: int | None = None,
    trend: int | None = None,
    entry_n: int | None = None,
    exit_n: int | None = None,
    annotate_extrema: bool = False,
    show_signals: bool = False,
    show_fills: bool = False,
    show_trade_path: bool = False,
    show_end_marker: bool = False,
    reference_annotations: bool = False,
    include_ew_portfolio: bool = False,
) -> dict[str, object]:
    clean_symbols = normalize_symbols(symbols)
    if not clean_symbols:
        raise ValueError("no valid symbols provided")

    layout_token = str(layout or "single").strip().lower()
    if layout_token not in SUPPORTED_CHART_LAYOUTS:
        raise ValueError(f"unsupported layout: {layout}")
    if layout_token == "single" and len(clean_symbols) != 1:
        raise ValueError("single layout requires exactly one symbol")
    if layout_token == "split" and out:
        raise ValueError("split layout does not support --out; use --out-dir instead")

    strategy_token = str(strategy or "").strip().lower()
    if strategy_token not in SUPPORTED_CHART_STRATEGIES:
        raise ValueError(f"unsupported strategy: {strategy}")
    strategy_params = _build_strategy_params(
        strategy=strategy_token,
        fast=fast,
        slow=slow,
        trend=trend,
        entry_n=entry_n,
        exit_n=exit_n,
    )
    palette = _resolve_palette(theme)
    if reference_annotations:
        annotate_extrema = True
        show_signals = True
        show_fills = True
        show_trade_path = True
        show_end_marker = True
    symbol_markets = _resolve_symbol_markets(clean_symbols, market)
    market_groups: dict[str, list[str]] = defaultdict(list)
    for symbol, market_code in symbol_markets.items():
        market_groups[market_code].append(symbol)

    if layout_token == "combined" and len(market_groups) != 1:
        raise ValueError("combined layout requires symbols from the same market")

    for market_code in market_groups:
        _validate_benchmark_choice(benchmark_choice, market_code)

    output_paths: list[str] = []
    items: list[dict[str, str]] = []
    issues: list[str] = []

    if layout_token in {"single", "combined"}:
        market_code = next(iter(market_groups.keys()))
        validated_benchmark = _validate_benchmark_choice(benchmark_choice, market_code)
        if layout_token == "single":
            payload_map = _build_symbol_payloads(
                store=store,
                symbols=market_groups[market_code],
                market_code=market_code,
                start=start,
                end=end,
                strategy=strategy_token,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
                benchmark_choice=validated_benchmark,
                fee_rate=fee_rate,
                sell_tax=sell_tax,
                slippage=slippage,
                sync_before_run=sync_before_run,
                use_split_adjustment=use_split_adjustment,
                use_total_return_adjustment=use_total_return_adjustment,
            )
            if not payload_map:
                raise ValueError("no symbols produced chart data")
            payloads = [payload_map[symbol] for symbol in market_groups[market_code] if symbol in payload_map]
            payload = payloads[0]
            fig = _build_single_symbol_backtest_figure(
                payload,
                strategy=strategy_token,
                start=start,
                end=end,
                palette=palette,
                height=height,
                annotate_extrema=annotate_extrema,
                show_signals=show_signals,
                show_fills=show_fills,
                show_trade_path=show_trade_path,
                show_end_marker=show_end_marker,
            )
            output_path = _resolve_output_path(
                layout=layout_token,
                symbol=payload.symbol,
                start=start,
                end=end,
                out=out,
                out_dir=out_dir,
            )
            _write_figure_image(fig, output_path=output_path, width=width, height=height, scale=scale)
            output_paths.append(str(output_path))
            items.append({"symbol": payload.symbol, "path": str(output_path)})
        else:
            payload = _build_combined_chart_payload(
                store=store,
                symbols=market_groups[market_code],
                market_code=market_code,
                start=start,
                end=end,
                strategy=strategy_token,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
                benchmark_choice=validated_benchmark,
                fee_rate=fee_rate,
                sell_tax=sell_tax,
                slippage=slippage,
                sync_before_run=sync_before_run,
                use_split_adjustment=use_split_adjustment,
                use_total_return_adjustment=use_total_return_adjustment,
            )
            fig = _build_combined_backtest_figure(
                payload,
                strategy=strategy_token,
                start=start,
                end=end,
                palette=palette,
                height=height,
                include_ew_portfolio=include_ew_portfolio,
            )
            output_path = _resolve_output_path(
                layout=layout_token,
                symbol="combined",
                start=start,
                end=end,
                out=out,
                out_dir=out_dir,
                combined_count=len(payload.per_symbol_buy_hold),
            )
            _write_figure_image(fig, output_path=output_path, width=width, height=height, scale=scale)
            output_paths.append(str(output_path))
            items.append(
                {
                    "symbol": ",".join(sorted(payload.per_symbol_buy_hold.keys())),
                    "path": str(output_path),
                }
            )
    else:
        base_dir = Path(out_dir).expanduser() if out_dir else _default_output_root() / (
            f"chart_backtest_split_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        for market_code, market_symbols in market_groups.items():
            payload_map = _build_symbol_payloads(
                store=store,
                symbols=market_symbols,
                market_code=market_code,
                start=start,
                end=end,
                strategy=strategy_token,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
                benchmark_choice=_validate_benchmark_choice(benchmark_choice, market_code),
                fee_rate=fee_rate,
                sell_tax=sell_tax,
                slippage=slippage,
                sync_before_run=sync_before_run,
                use_split_adjustment=use_split_adjustment,
                use_total_return_adjustment=use_total_return_adjustment,
            )
            for symbol in market_symbols:
                payload = payload_map.get(symbol)
                if payload is None:
                    issues.append(f"{symbol}: no chart payload")
                    continue
                fig = _build_single_symbol_backtest_figure(
                    payload,
                    strategy=strategy_token,
                    start=start,
                    end=end,
                    palette=palette,
                    height=height,
                    annotate_extrema=annotate_extrema,
                    show_signals=show_signals,
                    show_fills=show_fills,
                    show_trade_path=show_trade_path,
                    show_end_marker=show_end_marker,
                )
                output_path = _resolve_output_path(
                    layout="split",
                    symbol=symbol,
                    start=start,
                    end=end,
                    out=None,
                    out_dir=str(base_dir),
                )
                _write_figure_image(fig, output_path=output_path, width=width, height=height, scale=scale)
                output_paths.append(str(output_path))
                items.append({"symbol": symbol, "path": str(output_path)})

    return {
        "layout": layout_token,
        "format": "png",
        "paths": output_paths,
        "items": items,
        "issues": issues,
        "exported_count": len(output_paths),
        "requested_count": len(clean_symbols),
    }


__all__ = [
    "DEFAULT_CHART_THEME",
    "SUPPORTED_CHART_LAYOUTS",
    "SUPPORTED_CHART_STRATEGIES",
    "SUPPORTED_CHART_THEMES",
    "export_backtest_chart_artifact",
]
