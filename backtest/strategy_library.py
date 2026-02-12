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


STRATEGIES: Dict[str, Callable[..., pd.Series]] = {
    "buy_hold": buy_hold,
    "sma_cross": sma_cross,
    "ema_cross": ema_cross,
    "rsi_reversion": rsi_reversion,
    "macd_trend": macd_trend,
}
