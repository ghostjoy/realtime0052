from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from indicators import ema, macd, rsi, sma


def buy_hold(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=bars.index, dtype=int)


def sma_cross(bars: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    fast = int(fast)
    slow = int(slow)
    close = bars["close"]
    fast_sma = sma(close, fast)
    slow_sma = sma(close, slow)
    return (fast_sma > slow_sma).astype(int).fillna(0)


def ema_cross(bars: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    fast = int(fast)
    slow = int(slow)
    close = bars["close"]
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    return (fast_ema > slow_ema).astype(int).fillna(0)


def rsi_reversion(bars: pd.DataFrame, buy_below: float = 30.0, sell_above: float = 55.0) -> pd.Series:
    close = bars["close"]
    r = rsi(close, window=14)
    signal = pd.Series(0, index=bars.index, dtype=int)
    in_position = False
    for idx in bars.index:
        val = r.loc[idx]
        if pd.isna(val):
            signal.loc[idx] = 1 if in_position else 0
            continue
        if not in_position and val <= buy_below:
            in_position = True
        elif in_position and val >= sell_above:
            in_position = False
        signal.loc[idx] = 1 if in_position else 0
    return signal


def macd_trend(bars: pd.DataFrame) -> pd.Series:
    close = bars["close"]
    line, sig, _ = macd(close)
    return (line > sig).astype(int).fillna(0)


def sma_trend_filter(bars: pd.DataFrame, fast: int = 20, slow: int = 60, trend: int = 120) -> pd.Series:
    fast = int(fast)
    slow = int(slow)
    trend = int(trend)
    close = bars["close"]
    fast_sma = sma(close, fast)
    slow_sma = sma(close, slow)
    trend_sma = sma(close, trend)
    signal = pd.Series(0, index=bars.index, dtype=int)
    in_position = False
    for idx in bars.index:
        f = fast_sma.loc[idx]
        s = slow_sma.loc[idx]
        t = trend_sma.loc[idx]
        c = close.loc[idx]
        if pd.isna(f) or pd.isna(s) or pd.isna(t) or pd.isna(c):
            signal.loc[idx] = 1 if in_position else 0
            continue
        if not in_position and f > s and c > t:
            in_position = True
        elif in_position and (f < s or c < t):
            in_position = False
        signal.loc[idx] = 1 if in_position else 0
    return signal


def donchian_breakout(
    bars: pd.DataFrame,
    entry_n: int = 55,
    exit_n: int = 20,
    trend: int = 120,
) -> pd.Series:
    entry_n = int(entry_n)
    exit_n = int(exit_n)
    trend = int(trend)
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    trend_sma = sma(close, trend)
    entry_high = high.rolling(window=entry_n, min_periods=entry_n).max().shift(1)
    exit_low = low.rolling(window=exit_n, min_periods=exit_n).min().shift(1)
    signal = pd.Series(0, index=bars.index, dtype=int)
    in_position = False
    for idx in bars.index:
        c = close.loc[idx]
        th = trend_sma.loc[idx]
        eh = entry_high.loc[idx]
        xl = exit_low.loc[idx]
        if pd.isna(c) or pd.isna(th) or pd.isna(eh) or pd.isna(xl):
            signal.loc[idx] = 1 if in_position else 0
            continue
        if not in_position and c >= eh and c > th:
            in_position = True
        elif in_position and c <= xl:
            in_position = False
        signal.loc[idx] = 1 if in_position else 0
    return signal


STRATEGIES: Dict[str, Callable[..., pd.Series]] = {
    "buy_hold": buy_hold,
    "sma_cross": sma_cross,
    "ema_cross": ema_cross,
    "rsi_reversion": rsi_reversion,
    "macd_trend": macd_trend,
    "sma_trend_filter": sma_trend_filter,
    "donchian_breakout": donchian_breakout,
}

STRATEGY_MIN_BARS: Dict[str, int] = {
    "buy_hold": 2,
    "sma_cross": 40,
    "ema_cross": 40,
    "rsi_reversion": 40,
    "macd_trend": 40,
    "sma_trend_filter": 120,
    "donchian_breakout": 120,
}


def get_strategy_min_bars(strategy_name: str) -> int:
    return int(STRATEGY_MIN_BARS.get(str(strategy_name or "").strip(), 40))
