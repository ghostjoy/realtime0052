from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


TAIPEI_TZ = ZoneInfo("Asia/Taipei") if ZoneInfo else timezone.utc


class DataSourceError(RuntimeError):
    pass


class _YfQuiet:
    def __init__(self):
        self._logger = logging.getLogger("yfinance")
        self._prev_level = self._logger.level

    def __enter__(self):
        self._prev_level = self._logger.level
        self._logger.setLevel(logging.CRITICAL + 1)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._logger.setLevel(self._prev_level)
        return False


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "-":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _split_levels(levels: str | None) -> list[float]:
    if not levels:
        return []
    parts = [p for p in levels.split("_") if p]
    out: list[float] = []
    for p in parts:
        v = _to_float(p)
        if v is not None:
            out.append(v)
    return out


def _split_sizes(levels: str | None) -> list[int]:
    if not levels:
        return []
    parts = [p for p in levels.split("_") if p]
    out: list[int] = []
    for p in parts:
        v = _to_int(p)
        if v is not None:
            out.append(v)
    return out


@dataclass(frozen=True)
class TwseQuote:
    stock_id: str
    name: str
    full_name: str
    ts: datetime
    last: float | None
    prev_close: float | None
    open: float | None
    high: float | None
    low: float | None
    volume: int | None  # cumulative volume
    upper_limit: float | None
    lower_limit: float | None
    bid_prices: list[float]
    bid_sizes: list[int]
    ask_prices: list[float]
    ask_sizes: list[int]

    @property
    def change(self) -> float | None:
        if self.last is None or self.prev_close is None:
            return None
        return self.last - self.prev_close

    @property
    def change_pct(self) -> float | None:
        if self.change is None or not self.prev_close:
            return None
        return self.change / self.prev_close * 100.0


@dataclass(frozen=True)
class YfQuote:
    symbol: str
    name: str
    ts: datetime
    last: float | None
    prev_close: float | None
    open: float | None
    high: float | None
    low: float | None
    volume: int | None
    currency: str | None
    exchange: str | None

    @property
    def change(self) -> float | None:
        if self.last is None or self.prev_close is None:
            return None
        return self.last - self.prev_close

    @property
    def change_pct(self) -> float | None:
        if self.change is None or not self.prev_close:
            return None
        return self.change / self.prev_close * 100.0


@lru_cache(maxsize=1)
def _twse_session():
    import requests

    s = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://mis.twse.com.tw/stock/fibest.jsp",
        "Accept": "application/json,text/plain,*/*",
    }
    try:
        s.get("https://mis.twse.com.tw/stock/fibest.jsp", headers=headers, timeout=10)
    except Exception:
        # Cookie pre-flight失敗時，後續仍可能成功；不在此直接拋錯
        pass
    return s


def fetch_twse_quote(stock_id: str, exchange: str = "tse") -> TwseQuote:
    import time

    s = _twse_session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://mis.twse.com.tw/stock/fibest.jsp",
        "Accept": "application/json,text/plain,*/*",
    }

    ex_ch = f"{exchange}_{stock_id}.tw"
    url = (
        "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
        f"?ex_ch={ex_ch}&json=1&delay=0&_={int(time.time() * 1000)}"
    )

    try:
        r = s.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        raise DataSourceError(f"TWSE即時報價取得失敗：{e}") from e

    if payload.get("rtcode") != "0000" or not payload.get("msgArray"):
        raise DataSourceError(
            f"TWSE回應異常：rtcode={payload.get('rtcode')} message={payload.get('rtmessage')}"
        )

    msg: dict[str, Any] = payload["msgArray"][0]
    ts_ms = _to_int(msg.get("tlong"))
    if ts_ms is None:
        ts = datetime.now(tz=TAIPEI_TZ)
    else:
        ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(TAIPEI_TZ)

    bid_prices = _split_levels(msg.get("b"))
    ask_prices = _split_levels(msg.get("a"))
    bid_sizes = _split_sizes(msg.get("g"))
    ask_sizes = _split_sizes(msg.get("f"))

    return TwseQuote(
        stock_id=stock_id,
        name=str(msg.get("n") or stock_id),
        full_name=str(msg.get("nf") or ""),
        ts=ts,
        last=_to_float(msg.get("z")),
        prev_close=_to_float(msg.get("y")),
        open=_to_float(msg.get("o")),
        high=_to_float(msg.get("h")),
        low=_to_float(msg.get("l")),
        volume=_to_int(msg.get("v")),
        upper_limit=_to_float(msg.get("u")),
        lower_limit=_to_float(msg.get("w")),
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
    )


def fetch_yf_ohlcv(symbol: str, period: str, interval: str):
    import pandas as pd
    import yfinance as yf

    # Avoid noisy yfinance "Failed downloads" stderr output by preferring
    # per-symbol history API instead of batch download API.
    try:
        with _YfQuiet():
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # type: ignore[assignment]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    keep = [c for c in ["open", "high", "low", "close", "volume", "adj_close"] if c in df.columns]
    df = df[keep].copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(TAIPEI_TZ)

    return df


def fetch_yf_quote(symbol: str) -> YfQuote:
    import pandas as pd
    import yfinance as yf

    t = yf.Ticker(symbol)

    try:
        info = getattr(t, "fast_info", None)
        currency = getattr(info, "currency", None) if info is not None else None
        exchange = getattr(info, "exchange", None) if info is not None else None
    except Exception:
        currency = None
        exchange = None

    name = symbol
    try:
        # shortName常見但不保證存在
        meta = getattr(t, "info", None) or {}
        name = meta.get("shortName") or meta.get("longName") or symbol
        currency = currency or meta.get("currency")
        exchange = exchange or meta.get("exchange")
    except Exception:
        pass

    try:
        intra = t.history(period="1d", interval="1m")
    except Exception as e:
        raise DataSourceError(f"Yahoo 走勢取得失敗：{e}") from e

    if intra is None or intra.empty:
        raise DataSourceError("Yahoo 回傳空資料（可能為非交易時段、代碼錯誤或資料源暫時不可用）")

    if intra.index.tz is None:
        intra.index = intra.index.tz_localize("UTC")

    last_ts = intra.index[-1]
    last = float(intra["Close"].iloc[-1])
    open_ = float(intra["Open"].iloc[0])
    high = float(intra["High"].max())
    low = float(intra["Low"].min())
    vol = float(intra["Volume"].sum())

    prev_close = None
    try:
        daily = t.history(period="5d", interval="1d")
        if daily is not None and not daily.empty and "Close" in daily.columns:
            closes = daily["Close"].dropna()
            if len(closes) >= 2:
                prev_close = float(closes.iloc[-2])
            elif len(closes) == 1:
                prev_close = float(closes.iloc[-1])
    except Exception:
        prev_close = None

    return YfQuote(
        symbol=symbol,
        name=str(name),
        ts=pd.Timestamp(last_ts).to_pydatetime(),
        last=last,
        prev_close=prev_close,
        open=open_,
        high=high,
        low=low,
        volume=int(vol),
        currency=currency,
        exchange=exchange,
    )


def fetch_yf_last_close(symbol: str) -> tuple[float | None, float | None]:
    import yfinance as yf

    t = yf.Ticker(symbol)
    try:
        hist = t.history(period="5d", interval="1d")
    except Exception:
        return None, None
    if hist is None or hist.empty:
        return None, None

    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return float(closes.iloc[-1]), None

    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2])
    return last, (last - prev) / prev * 100.0


def fetch_yf_fundamentals(symbol: str) -> dict[str, Any]:
    import yfinance as yf

    t = yf.Ticker(symbol)

    info: dict[str, Any] = {}
    try:
        raw = getattr(t, "info", None) or {}
        if isinstance(raw, dict):
            info = raw
    except Exception as e:
        raise DataSourceError(f"Yahoo 基本面取得失敗：{e}") from e

    def pick(key: str):
        v = info.get(key)
        if v is None:
            return None
        if isinstance(v, (int, float, str)):
            return v
        return str(v)

    out: dict[str, Any] = {
        "symbol": symbol,
        "name": pick("shortName") or pick("longName"),
        "sector": pick("sector"),
        "industry": pick("industry"),
        "marketCap": pick("marketCap"),
        "trailingPE": pick("trailingPE"),
        "forwardPE": pick("forwardPE"),
        "priceToBook": pick("priceToBook"),
        "beta": pick("beta"),
        "dividendYield": pick("dividendYield"),
        "profitMargins": pick("profitMargins"),
        "operatingMargins": pick("operatingMargins"),
        "grossMargins": pick("grossMargins"),
        "revenueGrowth": pick("revenueGrowth"),
        "earningsGrowth": pick("earningsGrowth"),
        "freeCashflow": pick("freeCashflow"),
        "totalCash": pick("totalCash"),
        "totalDebt": pick("totalDebt"),
    }
    return out


def _stooq_symbol(symbol: str) -> str:
    s = symbol.strip()
    if not s:
        return s
    if "." not in s:
        return f"{s.lower()}.us"
    return s.lower()


def fetch_stooq_quote(symbol: str) -> YfQuote:
    import csv
    import io

    import requests

    stooq = _stooq_symbol(symbol)
    url = f"https://stooq.com/q/l/?s={stooq}&f=sd2t2ohlcv&h&e=csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise DataSourceError(f"Stooq 報價取得失敗：{e}") from e

    text = r.text.strip()
    if not text or "No data" in text:
        raise DataSourceError("Stooq 回傳空資料（代碼可能不支援）")

    reader = csv.DictReader(io.StringIO(text))
    row = next(reader, None)
    if not row:
        raise DataSourceError("Stooq 回傳資料解析失敗")

    # Date=YYYY-MM-DD, Time=HH:MM:SS（來源時區未明，這裡以 UTC 表示以免誤導）
    date_s = (row.get("Date") or "").strip()
    time_s = (row.get("Time") or "00:00:00").strip()
    try:
        ts = datetime.fromisoformat(f"{date_s}T{time_s}").replace(tzinfo=timezone.utc)
    except Exception:
        ts = datetime.now(tz=timezone.utc)

    open_ = _to_float(row.get("Open"))
    high = _to_float(row.get("High"))
    low = _to_float(row.get("Low"))
    last = _to_float(row.get("Close"))
    vol = _to_int(row.get("Volume"))

    return YfQuote(
        symbol=symbol,
        name=symbol,
        ts=ts,
        last=last,
        prev_close=None,
        open=open_,
        high=high,
        low=low,
        volume=vol,
        currency="USD",
        exchange="STOOQ",
    )


def fetch_stooq_ohlcv(symbol: str, interval: str = "1d"):
    import io

    import pandas as pd
    import requests

    if interval != "1d":
        raise DataSourceError("Stooq 目前僅提供日K（interval=1d）")

    stooq = _stooq_symbol(symbol)
    url = f"https://stooq.com/q/d/l/?s={stooq}&i=d"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise DataSourceError(f"Stooq 走勢取得失敗：{e}") from e

    df = pd.read_csv(io.StringIO(r.text))
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df = df.sort_index()
    df.index = df.index.tz_localize(timezone.utc)
    return df[["open", "high", "low", "close", "volume"]]
