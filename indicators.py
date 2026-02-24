from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def bollinger(close: pd.Series, window: int = 20, stdev: float = 2.0):
    mid = sma(close, window)
    sd = close.rolling(window=window, min_periods=window).std()
    upper = mid + stdev * sd
    lower = mid - stdev * sd
    return mid, upper, lower


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3.0
    pv = (typical * volume).cumsum()
    vv = volume.cumsum().replace(0, np.nan)
    return pv / vv


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.rolling(window=window, min_periods=window).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    signed_vol = volume * direction
    return signed_vol.cumsum()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
):
    ll = low.rolling(window=window, min_periods=window).min()
    hh = high.rolling(window=window, min_periods=window).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k_s = k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k_s.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return k_s, d


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14
) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    direction = np.sign(tp.diff()).fillna(0)
    pos = mf.where(direction > 0, 0.0)
    neg = mf.where(direction < 0, 0.0).abs()
    pos_sum = pos.rolling(window=window, min_periods=window).sum()
    neg_sum = neg.rolling(window=window, min_periods=window).sum().replace(0, np.nan)
    mfr = pos_sum / neg_sum
    out = 100 - (100 / (1 + mfr))
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df.get("volume", pd.Series(index=df.index, dtype=float)).fillna(0.0)

    df["sma_5"] = sma(close, 5)
    df["sma_20"] = sma(close, 20)
    df["sma_60"] = sma(close, 60)
    df["ema_12"] = ema(close, 12)
    df["ema_26"] = ema(close, 26)
    macd_line, macd_sig, macd_hist = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist
    df["rsi_14"] = rsi(close, 14)
    bb_mid, bb_upper, bb_lower = bollinger(close, 20, 2.0)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["vwap"] = vwap(high, low, close, vol)
    df["atr_14"] = atr(high, low, close, 14)
    df["obv"] = obv(close, vol)
    k, d = stochastic(high, low, close, 14, 3, 3)
    df["stoch_k"] = k
    df["stoch_d"] = d
    df["mfi_14"] = mfi(high, low, close, vol, 14)

    return df
