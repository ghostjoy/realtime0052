from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from backtest import (
    CostModel,
    apply_split_adjustment,
    run_backtest,
    run_portfolio_backtest,
    walk_forward_portfolio,
    walk_forward_single,
)
from services.benchmark_loader import benchmark_candidates_tw
from utils import normalize_ohlcv_frame


class HistoryStoreLike(Protocol):
    def load_daily_bars(
        self, symbol: str, market: str, start: datetime, end: datetime
    ) -> pd.DataFrame: ...

    def sync_symbol_history(self, symbol: str, market: str, start: datetime, end: datetime): ...


@dataclass(frozen=True)
class BacktestExecutionInput:
    mode: str
    strategy: str
    strategy_params: dict[str, float]
    enable_walk_forward: bool
    train_ratio: float
    objective: str
    initial_capital: float
    cost_model: CostModel


@dataclass(frozen=True)
class BacktestPreparedBars:
    bars_by_symbol: dict[str, pd.DataFrame]
    availability_rows: list[dict[str, object]]


def parse_symbols(text: str) -> list[str]:
    symbols = [s.strip().upper() for s in str(text or "").replace("，", ",").split(",")]
    out: list[str] = []
    for sym in symbols:
        if sym and sym not in out:
            out.append(sym)
    return out


def _is_tw_etf(symbol: str) -> bool:
    token = str(symbol or "").strip().upper()
    return len(token) == 4 and token.isdigit() and token.startswith("00")


def default_cost_params(market_code: str, symbol_list: list[str]) -> tuple[float, float, float]:
    market_token = str(market_code or "").strip().upper()
    if market_token == "US":
        return 0.0005, 0.0, 0.0010
    tw_tax = 0.001 if symbol_list and all(_is_tw_etf(s) for s in symbol_list) else 0.003
    return 0.001425, tw_tax, 0.0005


def series_metrics(series: pd.Series) -> dict[str, float]:
    if series is None or series.empty:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    returns = series.pct_change().fillna(0.0)
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 365.25)
    total_return = series.iloc[-1] / series.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    sharpe = (returns.mean() / returns.std() * np.sqrt(252.0)) if returns.std() > 0 else 0.0
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(sharpe),
    }


def benchmark_candidates(market_code: str, choice: str) -> list[str]:
    selected = str(choice or "auto").strip().lower()
    if selected == "off":
        return []
    if str(market_code or "").strip().upper() == "TW":
        if selected == "auto":
            return benchmark_candidates_tw("twii", allow_twii_fallback=True)
        return benchmark_candidates_tw(selected, allow_twii_fallback=False)
    mapping = {
        "auto": ["^GSPC", "SPY", "QQQ", "DIA"],
        "gspc": ["^GSPC"],
        "spy": ["SPY"],
        "qqq": ["QQQ"],
        "dia": ["DIA"],
    }
    return mapping.get(selected, ["^GSPC"])


def load_benchmark_from_store(
    *,
    store: HistoryStoreLike,
    market_code: str,
    start: datetime,
    end: datetime,
    choice: str,
) -> pd.DataFrame:
    candidates = benchmark_candidates(market_code, choice)
    if not candidates:
        return pd.DataFrame(columns=["close"])

    for bench_symbol in candidates:
        bars = normalize_ohlcv_frame(
            store.load_daily_bars(symbol=bench_symbol, market=market_code, start=start, end=end)
        )
        end_ts = pd.Timestamp(end).tz_convert("UTC")
        needs_sync = bars.empty or pd.Timestamp(bars.index.max()).tz_convert("UTC") < end_ts
        if needs_sync:
            store.sync_symbol_history(symbol=bench_symbol, market=market_code, start=start, end=end)
            bars = normalize_ohlcv_frame(
                store.load_daily_bars(symbol=bench_symbol, market=market_code, start=start, end=end)
            )
        if bars.empty or "close" not in bars.columns:
            continue

        keep_cols = ["close"]
        if "adj_close" in bars.columns:
            keep_cols.append("adj_close")
        out = bars[keep_cols].copy()
        source_text = ""
        if "source" in bars.columns:
            source_vals = sorted(set(bars["source"].dropna().astype(str)))
            if source_vals:
                source_text = ",".join(source_vals)
        out.attrs["symbol"] = bench_symbol
        store_backend = str(getattr(store, "backend_name", "store") or "store").strip().lower()
        out.attrs["source"] = f"{store_backend}:{source_text}" if source_text else store_backend
        return out

    return pd.DataFrame(columns=["close"])


def queue_benchmark_writeback(
    *, store: HistoryStoreLike, market_code: str, benchmark: pd.DataFrame
) -> bool:
    if not isinstance(benchmark, pd.DataFrame) or benchmark.empty:
        return False
    queue_fn = getattr(store, "queue_daily_bars_writeback", None)
    if not callable(queue_fn):
        return False
    symbol = str(getattr(benchmark, "attrs", {}).get("symbol", "")).strip().upper()
    if not symbol:
        return False
    source = str(getattr(benchmark, "attrs", {}).get("source", "")).strip()
    try:
        return bool(
            queue_fn(
                symbol=symbol,
                market=market_code,
                bars=benchmark,
                source=source or "benchmark_fallback",
            )
        )
    except Exception:
        return False


def load_and_prepare_symbol_bars(
    *,
    store: HistoryStoreLike,
    market_code: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
    use_total_return_adjustment: bool,
    use_split_adjustment: bool,
    auto_detect_split: bool,
    apply_total_return_adjustment: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict[str, object]]],
) -> BacktestPreparedBars:
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    availability_rows: list[dict[str, object]] = []
    for symbol in symbols:
        bars = store.load_daily_bars(symbol=symbol, market=market_code, start=start, end=end)
        if bars.empty:
            availability_rows.append(
                {"symbol": symbol, "rows": 0, "sources": "", "status": "EMPTY", "adj_mode": ""}
            )
            continue
        bars = bars.sort_index()
        adj_info: dict[str, object] = {"applied": False}
        if use_total_return_adjustment:
            bars, adj_info = apply_total_return_adjustment(bars)
        if use_split_adjustment and not bool(adj_info.get("applied")):
            bars, split_events = apply_split_adjustment(
                bars=bars,
                symbol=symbol,
                market=market_code,
                use_known=True,
                use_auto_detect=auto_detect_split,
            )
        else:
            split_events = []
        if bool(adj_info.get("applied")):
            adj_mode = f"ON ({adj_info.get('coverage_pct', 0)}%)"
        elif use_total_return_adjustment:
            reason = str(adj_info.get("reason", "") or "")
            adj_mode = "OFF(no adj_close)" if reason == "no_adj_close" else "OFF(coverage low)"
        else:
            adj_mode = "OFF"
        bars_by_symbol[symbol] = bars
        availability_rows.append(
            {
                "symbol": symbol,
                "rows": int(len(bars)),
                "sources": ",".join(sorted(set(bars["source"].dropna().astype(str))))
                if "source" in bars.columns
                else "",
                "status": "OK",
                "adj_mode": adj_mode,
                "splits": ", ".join(
                    [
                        f"{pd.Timestamp(ev.date).date()} x{ev.ratio:.6f}({ev.source})"
                        for ev in split_events
                    ]
                )
                if split_events
                else "",
            }
        )
    return BacktestPreparedBars(bars_by_symbol=bars_by_symbol, availability_rows=availability_rows)


def execute_backtest_run(
    *, bars_by_symbol: dict[str, pd.DataFrame], config: BacktestExecutionInput
) -> dict[str, object]:
    mode = str(config.mode or "單一標的")
    strategy = str(config.strategy or "")
    initial_capital = float(config.initial_capital)
    strategy_params = config.strategy_params if isinstance(config.strategy_params, dict) else {}
    if bool(config.enable_walk_forward):
        if mode == "單一標的":
            symbol = list(bars_by_symbol.keys())[0]
            wf_result = walk_forward_single(
                bars=bars_by_symbol[symbol],
                strategy_name=strategy,
                cost_model=config.cost_model,
                train_ratio=float(config.train_ratio),
                objective=str(config.objective),
                initial_capital=initial_capital,
            )
            return {
                "mode": "single",
                "walk_forward": True,
                "initial_capital": initial_capital,
                "symbol": symbol,
                "bars_by_symbol": bars_by_symbol,
                "result": wf_result.test_result,
                "train_result": wf_result.train_result,
                "split_date": wf_result.split_date,
                "best_params": wf_result.best_params,
                "candidates": wf_result.candidates,
            }
        wf_portfolio = walk_forward_portfolio(
            bars_by_symbol=bars_by_symbol,
            strategy_name=strategy,
            cost_model=config.cost_model,
            train_ratio=float(config.train_ratio),
            objective=str(config.objective),
            initial_capital=initial_capital,
        )
        return {
            "mode": "portfolio",
            "walk_forward": True,
            "initial_capital": initial_capital,
            "symbols": list(bars_by_symbol.keys()),
            "bars_by_symbol": bars_by_symbol,
            "result": wf_portfolio.test_portfolio,
            "train_result": wf_portfolio.train_portfolio,
            "split_date": wf_portfolio.split_date,
            "best_params": {s: wf.best_params for s, wf in wf_portfolio.symbol_results.items()},
            "candidates": {s: wf.candidates for s, wf in wf_portfolio.symbol_results.items()},
        }

    if mode == "單一標的":
        symbol = list(bars_by_symbol.keys())[0]
        result = run_backtest(
            bars=bars_by_symbol[symbol],
            strategy_name=strategy,
            strategy_params=strategy_params,
            cost_model=config.cost_model,
            initial_capital=initial_capital,
        )
        return {
            "mode": "single",
            "walk_forward": False,
            "initial_capital": initial_capital,
            "symbol": symbol,
            "bars_by_symbol": bars_by_symbol,
            "result": result,
        }
    result = run_portfolio_backtest(
        bars_by_symbol=bars_by_symbol,
        strategy_name=strategy,
        strategy_params=strategy_params,
        cost_model=config.cost_model,
        initial_capital=initial_capital,
    )
    return {
        "mode": "portfolio",
        "walk_forward": False,
        "initial_capital": initial_capital,
        "symbols": list(bars_by_symbol.keys()),
        "bars_by_symbol": bars_by_symbol,
        "result": result,
    }


__all__ = [
    "BacktestExecutionInput",
    "BacktestPreparedBars",
    "benchmark_candidates",
    "default_cost_params",
    "execute_backtest_run",
    "load_and_prepare_symbol_bars",
    "load_benchmark_from_store",
    "parse_symbols",
    "queue_benchmark_writeback",
    "series_metrics",
]
