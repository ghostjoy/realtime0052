from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from advice import Profile, render_advice, render_advice_scai_style
from backtest import apply_split_adjustment
from indicators import add_indicators
from market_data_types import DataQuality, LiveContext, QuoteSnapshot
from services import LiveOptions
from ui.constants import COMPACT_CHART_HEIGHT, INDICATOR_CHART_HEIGHT
from ui.shared import get_or_set
from ui.shared.runtime import configure_module_runtime

REQUIRED_RUNTIME_NAMES = (
    "_current_theme_name",
    "_format_int",
    "_format_price",
    "_history_store",
    "_load_intraday_bars_from_sqlite",
    "_market_service",
    "_normalize_ohlcv_frame",
    "_persist_live_tick_buffer",
    "_render_card_section_header",
    "_render_indicator_panels",
    "_render_live_chart",
    "_render_quality_bar",
    "_resolve_live_change_metrics",
    "_symbol_watermark_text",
)

_current_theme_name: Any = None
_format_int: Any = None
_format_price: Any = None
_history_store: Any = None
_load_intraday_bars_from_sqlite: Any = None
_market_service: Any = None
_normalize_ohlcv_frame: Any = None
_persist_live_tick_buffer: Any = None
_render_card_section_header: Any = None
_render_indicator_panels: Any = None
_render_live_chart: Any = None
_render_quality_bar: Any = None
_resolve_live_change_metrics: Any = None
_symbol_watermark_text: Any = None

TW_LIVE_SNAPSHOT_DATASET_KEY = "tw_live_context_v1"
_LIVE_BG_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="live-bg")


def configure_runtime(values: Mapping[str, Any]) -> None:
    configure_module_runtime(globals(), REQUIRED_RUNTIME_NAMES, values, module_name=__name__)


def _serialize_live_frame(frame: pd.DataFrame, *, limit: int) -> list[dict[str, object]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    safe = frame.tail(max(1, int(limit))).copy()
    safe = safe.reset_index()
    ts_col = str(safe.columns[0])
    safe = safe.rename(columns={ts_col: "ts"})

    def _to_float(value: object) -> float:
        num = pd.to_numeric(value, errors="coerce")
        if pd.isna(num):
            return 0.0
        try:
            return float(num)
        except Exception:
            return 0.0

    out: list[dict[str, object]] = []
    for _, row in safe.iterrows():
        ts = pd.Timestamp(row.get("ts"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        out.append(
            {
                "ts": ts.isoformat(),
                "open": _to_float(row.get("open")),
                "high": _to_float(row.get("high")),
                "low": _to_float(row.get("low")),
                "close": _to_float(row.get("close")),
                "volume": _to_float(row.get("volume")),
            }
        )
    return out


def _deserialize_live_frame(rows: object) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows)
    if frame.empty or "ts" not in frame.columns:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame["ts"] = pd.to_datetime(frame.get("ts"), utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts"])
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = frame.set_index("ts").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"], how="any")
    return frame[["open", "high", "low", "close", "volume"]]


def _persist_tw_live_context_snapshot(*, store, symbol: str, ctx: LiveContext) -> None:
    writer = getattr(store, "save_market_snapshot", None)
    if not callable(writer):
        return
    quote = ctx.quote
    now_utc = datetime.now(tz=timezone.utc)
    quote_ts = pd.Timestamp(quote.ts)
    if quote_ts.tzinfo is None:
        quote_ts = quote_ts.tz_localize("UTC")
    else:
        quote_ts = quote_ts.tz_convert("UTC")
    freshness = max(0, int((now_utc - quote_ts.to_pydatetime()).total_seconds()))
    payload = {
        "quote": {
            "symbol": str(quote.symbol or symbol),
            "market": str(quote.market or "TW"),
            "ts": quote_ts.isoformat(),
            "price": quote.price,
            "prev_close": quote.prev_close,
            "open": quote.open,
            "high": quote.high,
            "low": quote.low,
            "volume": quote.volume,
            "source": str(quote.source or "unknown"),
            "is_delayed": bool(quote.is_delayed),
            "interval": str(quote.interval or "quote"),
            "currency": quote.currency,
            "exchange": quote.exchange,
            "extra": dict(getattr(quote, "extra", {}) or {}),
        },
        "intraday": _serialize_live_frame(ctx.intraday, limit=720),
        "daily": _serialize_live_frame(ctx.daily, limit=520),
        "source_chain": list(getattr(ctx, "source_chain", []) or []),
        "intraday_source": str(getattr(ctx, "intraday_source", "") or ""),
        "daily_source": str(getattr(ctx, "daily_source", "") or ""),
        "fallback_depth": int(getattr(ctx.quality, "fallback_depth", 0) or 0),
        "reason": str(getattr(ctx.quality, "reason", "") or ""),
    }
    try:
        writer(
            dataset_key=TW_LIVE_SNAPSHOT_DATASET_KEY,
            market="TW",
            symbol=symbol,
            interval="live",
            source=str(quote.source or "unknown"),
            asof=quote_ts.to_pydatetime(),
            payload=payload,
            freshness_sec=freshness,
            quality_score=quote.quality_score,
            stale=freshness > 60,
            raw_json=getattr(quote, "raw_json", {}),
        )
    except Exception:
        return


def _load_tw_live_context_snapshot(*, store, symbol: str) -> LiveContext | None:
    loader = getattr(store, "load_latest_market_snapshot", None)
    if not callable(loader):
        return None
    try:
        snap = loader(
            dataset_key=TW_LIVE_SNAPSHOT_DATASET_KEY,
            market="TW",
            symbol=symbol,
            interval="live",
        )
    except Exception:
        return None
    if not isinstance(snap, dict):
        return None
    payload = snap.get("payload")
    if not isinstance(payload, dict):
        return None
    quote_payload = payload.get("quote")
    if not isinstance(quote_payload, dict):
        return None

    def _to_optional_float(value: object) -> float | None:
        num = pd.to_numeric(value, errors="coerce")
        if pd.isna(num):
            return None
        try:
            return float(num)
        except Exception:
            return None

    quote_ts = pd.Timestamp(quote_payload.get("ts"))
    if quote_ts.tzinfo is None:
        quote_ts = quote_ts.tz_localize("UTC")
    else:
        quote_ts = quote_ts.tz_convert("UTC")
    quote = QuoteSnapshot(
        symbol=str(quote_payload.get("symbol") or symbol),
        market="TW",
        ts=quote_ts.to_pydatetime(),
        price=_to_optional_float(quote_payload.get("price")),
        prev_close=_to_optional_float(quote_payload.get("prev_close")),
        open=_to_optional_float(quote_payload.get("open")),
        high=_to_optional_float(quote_payload.get("high")),
        low=_to_optional_float(quote_payload.get("low")),
        volume=(
            int(float(quote_payload.get("volume")))
            if pd.notna(pd.to_numeric(quote_payload.get("volume"), errors="coerce"))
            else None
        ),
        source=str(quote_payload.get("source") or "duckdb_local"),
        is_delayed=bool(quote_payload.get("is_delayed", True)),
        interval=str(quote_payload.get("interval") or "quote"),
        currency=str(quote_payload.get("currency") or "TWD"),
        exchange=str(quote_payload.get("exchange") or "TWSE"),
        extra=(
            dict(quote_payload.get("extra")) if isinstance(quote_payload.get("extra"), dict) else {}
        ),
    )
    intraday = _deserialize_live_frame(payload.get("intraday"))
    daily = _deserialize_live_frame(payload.get("daily"))
    freshness = snap.get("freshness_sec")
    if not isinstance(freshness, int):
        freshness = max(
            0, int((datetime.now(tz=timezone.utc) - quote_ts.to_pydatetime()).total_seconds())
        )
    quality = DataQuality(
        freshness_sec=freshness,
        degraded=bool(snap.get("stale", False)),
        fallback_depth=int(payload.get("fallback_depth") or 0),
        reason=str(payload.get("reason") or "") or None,
    )
    return LiveContext(
        quote=quote,
        intraday=intraday,
        daily=daily,
        quality=quality,
        source_chain=[str(x) for x in list(payload.get("source_chain") or []) if str(x).strip()],
        used_fallback=True,
        fundamentals=None,
        intraday_source=str(payload.get("intraday_source") or "cache:duckdb"),
        daily_source=str(payload.get("daily_source") or "cache:duckdb"),
    )


def _run_tw_live_refresh(
    *,
    service,
    symbol: str,
    yahoo_symbol: str,
    ticks: pd.DataFrame,
    options: LiveOptions,
) -> tuple[LiveContext, pd.DataFrame]:
    return service.get_tw_live_context(symbol, yahoo_symbol, ticks=ticks, options=options)


def _render_live_view():
    service = _market_service()
    store = _history_store()

    with st.sidebar:
        st.subheader("介面")
        current_theme = _current_theme_name()
        st.caption(f"配色主題：{current_theme}（固定）")
        st.subheader("即時模式")
        market = st.selectbox("市場", options=["美股(US)", "台股(TWSE)"], index=0)
        if market == "台股(TWSE)":
            stock_id = st.text_input("股票代號", value="0052")
            exchange = st.selectbox("交易所", options=["tse", "otc"], index=0)
            yahoo_symbol = st.text_input("Yahoo 代碼（歷史/補K）", value=f"{stock_id}.TW")
            us_symbol = None
        else:
            us_symbol = st.text_input("美股代碼", value="TSLA").strip().upper()
            stock_id = None
            exchange = "tse"
            yahoo_symbol = st.text_input("Yahoo 代碼（歷史/補K）", value=us_symbol).strip().upper()
            if not yahoo_symbol:
                yahoo_symbol = us_symbol

        refresh_sec = st.slider(
            "自動更新（秒）",
            min_value=10,
            max_value=300,
            value=60,
            step=5,
            help="預設 60 秒更新一次，可降低免費資料源被限流的機率。",
        )
        keep_minutes = st.slider(
            "保留即時資料（分鐘）", min_value=30, max_value=360, value=180, step=30
        )
        use_yahoo = st.checkbox("允許 Yahoo 補K", value=True)
        use_fugle_ws = False
        if market == "台股(TWSE)":
            use_fugle_ws = st.checkbox(
                "啟用 Fugle WebSocket（需 API Key）",
                value=True,
                help="有設定 FUGLE_MARKETDATA_API_KEY，或已放置 key 檔（iCloud 預設路徑）時，台股即時會優先走 Fugle；失敗會自動回退。",
            )

        st.subheader("建議偏好")
        advice_mode = st.selectbox(
            "建議模式", options=["一般(技術面)", "股癌風格(心法/檢核)"], index=1
        )
        horizon = st.selectbox("時間尺度", options=["短線", "中期", "長期"], index=1)
        risk = st.selectbox("風險偏好", options=["保守", "一般", "積極"], index=1)
        style = st.selectbox("操作風格", options=["定期定額", "波段", "趨勢"], index=2)

    profile = Profile(horizon=horizon, risk=risk, style=style)
    run_every = f"{refresh_sec}s"

    @st.fragment(run_every=run_every)
    def live_fragment():
        options = LiveOptions(
            use_yahoo=use_yahoo,
            keep_minutes=keep_minutes,
            exchange=exchange,
            use_fugle_ws=use_fugle_ws,
        )
        if market == "台股(TWSE)":
            assert stock_id is not None
            tick_key = f"ticks:TW:{stock_id}:{exchange}"
            refresh_future_key = f"tw_live_future:{stock_id}:{exchange}:{yahoo_symbol}"
            refresh_next_key = f"tw_live_future_next:{stock_id}:{exchange}:{yahoo_symbol}"

            ticks_raw = st.session_state.get(
                tick_key, pd.DataFrame(columns=["ts", "price", "cum_volume"])
            )
            ticks = (
                ticks_raw.copy()
                if isinstance(ticks_raw, pd.DataFrame)
                else pd.DataFrame(columns=["ts", "price", "cum_volume"])
            )

            ctx: LiveContext | None = _load_tw_live_context_snapshot(store=store, symbol=stock_id)

            inflight = st.session_state.get(refresh_future_key)
            if isinstance(inflight, Future) and inflight.done():
                try:
                    ctx_rt, ticks_rt = inflight.result()
                    if isinstance(ticks_rt, pd.DataFrame):
                        ticks = ticks_rt
                        st.session_state[tick_key] = ticks
                    _persist_tw_live_context_snapshot(store=store, symbol=stock_id, ctx=ctx_rt)
                    ctx = ctx_rt
                except Exception:
                    pass
                finally:
                    st.session_state[refresh_future_key] = None

            now_ts = datetime.now(tz=timezone.utc).timestamp()
            next_allowed = float(st.session_state.get(refresh_next_key, 0.0) or 0.0)
            inflight = st.session_state.get(refresh_future_key)
            if (not isinstance(inflight, Future) or inflight.done()) and now_ts >= next_allowed:
                ticks_refresh = (
                    ticks.copy()
                    if isinstance(ticks, pd.DataFrame)
                    else pd.DataFrame(columns=["ts", "price", "cum_volume"])
                )
                try:
                    future = _LIVE_BG_EXECUTOR.submit(
                        _run_tw_live_refresh,
                        service=service,
                        symbol=stock_id,
                        yahoo_symbol=yahoo_symbol,
                        ticks=ticks_refresh,
                        options=options,
                    )
                    st.session_state[refresh_future_key] = future
                    st.session_state[refresh_next_key] = now_ts + max(5.0, float(refresh_sec) * 0.5)
                except Exception:
                    st.session_state[refresh_next_key] = now_ts + 10.0

            if ctx is None:
                try:
                    ctx, ticks = service.get_tw_live_context(
                        stock_id, yahoo_symbol, ticks=ticks, options=options
                    )
                    st.session_state[tick_key] = ticks
                    _persist_tw_live_context_snapshot(store=store, symbol=stock_id, ctx=ctx)
                except Exception as exc:
                    st.error(f"台股資料取得失敗：{exc}")
                    return
            else:
                st.caption("本輪先顯示 DuckDB 本地快照，背景持續更新中。")

            quote = ctx.quote
            buffer_key = f"intraday_buffer:TW:{stock_id}:{exchange}"
            flush_key = f"intraday_buffer_flush:TW:{stock_id}:{exchange}"
            _persist_live_tick_buffer(
                store=store,
                symbol=stock_id,
                market="TW",
                quote=quote,
                buffer_key=buffer_key,
                flush_key=flush_key,
            )
        else:
            assert us_symbol is not None
            if us_symbol.isdigit():
                st.warning(
                    "目前為美股模式，請輸入美股代碼（例如 AAPL、TSLA）；若要查台股請切換到台股模式。"
                )
                return
            try:
                ctx = service.get_us_live_context(us_symbol, yahoo_symbol, options)
            except Exception as exc:
                st.error(f"美股資料取得失敗：{exc}")
                return
            quote = ctx.quote

        with st.container(border=True):
            _render_card_section_header("即時行情卡", "來源鏈路、最新報價與漲跌資訊。")
            _render_quality_bar(ctx, refresh_sec=refresh_sec)
            display_name = str(quote.extra.get("name") or "").strip()
            if not display_name:
                display_name = str(quote.extra.get("full_name") or "").strip()
            if not display_name and market == "台股(TWSE)" and stock_id:
                try:
                    display_name = str(
                        service.get_tw_symbol_names([stock_id]).get(stock_id, "") or ""
                    ).strip()
                except Exception:
                    display_name = ""
            if not display_name:
                display_name = str(quote.symbol or "—")
            col1, col2, col3, col4 = st.columns(4)
            live_change, live_change_pct, live_change_basis = _resolve_live_change_metrics(
                quote,
                intraday=getattr(ctx, "intraday", pd.DataFrame()),
                daily=getattr(ctx, "daily", pd.DataFrame()),
            )
            col1.metric("名稱", display_name)
            col2.metric("最新價", _format_price(quote.price))
            col3.metric("漲跌", "—" if live_change is None else f"{live_change:+.2f}")
            col4.metric("漲跌幅", "—" if live_change_pct is None else f"{live_change_pct:+.2f}%")
            st.caption(f"漲跌計算依據：{live_change_basis}")
            col5, col6 = st.columns(2)
            col5.metric("成交量", _format_int(quote.volume))
            col6.metric("時間", quote.ts.strftime("%Y-%m-%d %H:%M:%S"))
            st.caption("非投資建議；僅供教育/研究。")

        bars_intraday = _normalize_ohlcv_frame(ctx.intraday)
        if market == "台股(TWSE)":
            assert stock_id is not None
            min_k_for_chart = max(8, int(max(30, keep_minutes) // 15))
            need_sqlite_fill = bars_intraday.empty or len(bars_intraday) < min_k_for_chart
            if need_sqlite_fill:
                sqlite_bars = _normalize_ohlcv_frame(
                    _load_intraday_bars_from_sqlite(
                        store=store,
                        symbol=stock_id,
                        market="TW",
                        keep_minutes=keep_minutes,
                    )
                )
                if not sqlite_bars.empty and len(sqlite_bars) > len(bars_intraday):
                    before = int(len(bars_intraday))
                    bars_intraday = sqlite_bars
                    if before <= 0:
                        st.caption("即時走勢：本輪改用本地即時快取。")
                    else:
                        st.caption(
                            f"即時走勢：K數偏少，已用本地快取補齊（{before} -> {len(bars_intraday)}）。"
                        )
        if bars_intraday.empty:
            st.warning("目前無法取得走勢資料。")
            return

        intraday_split_events = []
        if market == "台股(TWSE)":
            assert stock_id is not None
            adjust_symbol = stock_id
            adjust_market = "TW"
        else:
            adjust_symbol = (us_symbol or quote.symbol).strip().upper()
            adjust_market = "US"
        # Live chart also applies the same split-adjustment policy as backtest:
        # known events + auto-detection.
        bars_intraday, intraday_split_events = apply_split_adjustment(
            bars=bars_intraday,
            symbol=adjust_symbol,
            market=adjust_market,
            use_known=True,
            use_auto_detect=True,
        )
        if intraday_split_events:
            ev_txt = ", ".join(
                [f"{pd.Timestamp(ev.date).date()} x{ev.ratio:.6f}" for ev in intraday_split_events]
            )
            st.caption(f"即時走勢已套用分割調整：{ev_txt}")

        ind = add_indicators(bars_intraday)

        daily_long = _normalize_ohlcv_frame(ctx.daily)
        daily_split_events = []
        if not daily_long.empty:
            if market == "台股(TWSE)":
                assert stock_id is not None
                adjust_symbol = stock_id
                adjust_market = "TW"
            else:
                adjust_symbol = (us_symbol or quote.symbol).strip().upper()
                adjust_market = "US"
            daily_long, daily_split_events = apply_split_adjustment(
                bars=daily_long,
                symbol=adjust_symbol,
                market=adjust_market,
                use_known=True,
                use_auto_detect=True,
            )

        main_col, side_col = st.columns([0.66, 0.34], gap="large")
        with main_col:
            with st.container(border=True):
                _render_card_section_header("即時趨勢卡", "主圖使用分K；下方補上日K長期視角。")
                live_watermark_symbol = (
                    stock_id if market == "台股(TWSE)" else (us_symbol or quote.symbol or "")
                )
                live_watermark_market = "TW" if market == "台股(TWSE)" else "US"
                live_watermark_text = _symbol_watermark_text(
                    symbol=live_watermark_symbol,
                    market=live_watermark_market,
                    service=service,
                )
                _render_live_chart(ind, watermark_text=live_watermark_text)
                live_symbol_key = (
                    (stock_id if market == "台股(TWSE)" else (us_symbol or quote.symbol or "US"))
                    .strip()
                    .upper()
                )
                live_indicator_toggle_key = f"live_indicator_panel:{market}:{live_symbol_key}"
                get_or_set(live_indicator_toggle_key, lambda: True)
                st.checkbox(
                    "顯示技術指標副圖（RSI / MACD / 布林 / KD）",
                    key=live_indicator_toggle_key,
                )
                live_indicator_compact_key = f"live_indicator_compact:{market}:{live_symbol_key}"
                get_or_set(live_indicator_compact_key, lambda: True)
                st.checkbox(
                    "緊湊指標副圖（筆電建議）",
                    key=live_indicator_compact_key,
                )
                if st.session_state.get(live_indicator_toggle_key):
                    _render_indicator_panels(
                        ind,
                        chart_key=f"live_indicator_chart:{market}:{live_symbol_key}",
                        height=(
                            COMPACT_CHART_HEIGHT
                            if st.session_state.get(live_indicator_compact_key)
                            else INDICATOR_CHART_HEIGHT
                        ),
                        watermark_text=live_watermark_text,
                    )
                if not daily_long.empty:
                    st.markdown("#### 長期視角（Daily）")
                    if daily_split_events:
                        ev_txt = ", ".join(
                            [
                                f"{pd.Timestamp(ev.date).date()} x{ev.ratio:.6f}"
                                for ev in daily_split_events
                            ]
                        )
                        st.caption(f"已套用分割調整（復權）：{ev_txt}")
                    daily = add_indicators(daily_long)
                    st.line_chart(daily[["close", "sma_20", "sma_60"]].dropna(how="all"))

        with side_col:
            with st.container(border=True):
                st.markdown("#### 建議")
                if advice_mode.startswith("股癌"):
                    if render_advice_scai_style is None:
                        st.warning("股癌風格載入失敗，已改用一般模式。")
                        st.text(render_advice(ind, profile))
                    else:
                        st.text(
                            render_advice_scai_style(
                                ind, profile, symbol=yahoo_symbol, fundamentals=ctx.fundamentals
                            )
                        )
                else:
                    st.text(render_advice(ind, profile))

            with st.container(border=True):
                st.markdown("#### 技術快照")

                def _val(col: str, ndigits: int = 2) -> str:
                    if col not in ind.columns:
                        return "—"
                    value = pd.to_numeric(ind.iloc[-1][col], errors="coerce")
                    if pd.isna(value):
                        return "—"
                    return f"{float(value):.{ndigits}f}"

                m1, m2 = st.columns(2)
                m1.metric("Close", _val("close"))
                m2.metric("RSI14", _val("rsi_14"))
                m3, m4 = st.columns(2)
                m3.metric("SMA20", _val("sma_20"))
                m4.metric("SMA60", _val("sma_60"))
                m5, m6 = st.columns(2)
                m5.metric("MACD", _val("macd"))
                m6.metric("ATR14", _val("atr_14"))

                show_cols = [
                    "close",
                    "sma_5",
                    "sma_20",
                    "sma_60",
                    "rsi_14",
                    "stoch_k",
                    "stoch_d",
                    "mfi_14",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                    "bb_mid",
                    "bb_upper",
                    "bb_lower",
                    "vwap",
                    "atr_14",
                ]
                with st.expander("查看完整指標表", expanded=False):
                    st.dataframe(ind.iloc[-1][show_cols].to_frame("value").T, width="stretch")

            if market == "台股(TWSE)":
                with st.container(border=True):
                    st.markdown("#### 買賣五檔（即時來源）")
                    bid_prices = quote.extra.get("bid_prices", [])
                    bid_sizes = quote.extra.get("bid_sizes", [])
                    ask_prices = quote.extra.get("ask_prices", [])
                    ask_sizes = quote.extra.get("ask_sizes", [])
                    ob = pd.DataFrame(
                        {
                            "bid_price": bid_prices[:5] + [np.nan] * max(0, 5 - len(bid_prices)),
                            "bid_size": bid_sizes[:5] + [np.nan] * max(0, 5 - len(bid_sizes)),
                            "ask_price": ask_prices[:5] + [np.nan] * max(0, 5 - len(ask_prices)),
                            "ask_size": ask_sizes[:5] + [np.nan] * max(0, 5 - len(ask_sizes)),
                        }
                    )
                    st.dataframe(ob, width="stretch", hide_index=True)
            elif ctx.fundamentals:
                with st.container(border=True):
                    st.markdown("#### 基本面快照（Yahoo）")
                    st.dataframe(pd.DataFrame([ctx.fundamentals]), width="stretch", hide_index=True)

            with st.container(border=True):
                st.markdown("#### 外部參考")
                refs = service.get_reference_context("TW" if market == "台股(TWSE)" else "US")
                if refs.empty:
                    st.caption("目前無法取得外部參考資料。")
                else:
                    st.dataframe(refs, width="stretch", hide_index=True)

    live_fragment()


__all__ = ["_render_live_view"]
