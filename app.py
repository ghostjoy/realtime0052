from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest import (
    CostModel,
    apply_start_to_bars_map,
    apply_split_adjustment,
    build_buy_hold_equity,
    interval_return,
    run_backtest,
    run_portfolio_backtest,
    walk_forward_portfolio,
    walk_forward_single,
)
from indicators import add_indicators
from services import LiveOptions, MarketDataService
from storage import HistoryStore

try:
    from advice import Profile, render_advice, render_advice_scai_style
except ImportError:
    from advice import Profile, render_advice

    render_advice_scai_style = None  # type: ignore[assignment]


@st.cache_resource
def _market_service() -> MarketDataService:
    return MarketDataService()


@st.cache_resource
def _history_store() -> HistoryStore:
    return HistoryStore(service=_market_service())


def _format_price(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.2f}"


def _format_int(v: Optional[int]) -> str:
    if v is None:
        return "—"
    return f"{v:,}"


def _palette_with(base: dict[str, object], **overrides: object) -> dict[str, object]:
    out = dict(base)
    out.update(overrides)
    return out


_BASE_DARK_PALETTE: dict[str, object] = {
    "is_dark": True,
    "background": "#232B3A",
    "sidebar_bg": "#273245",
    "text_color": "#F1F5FF",
    "text_muted": "#CCD6EA",
    "card_bg": "rgba(49, 62, 86, 0.74)",
    "card_border": "rgba(162, 178, 204, 0.44)",
    "control_bg": "rgba(58, 74, 102, 0.94)",
    "control_border": "rgba(170, 185, 209, 0.45)",
    "tab_bg": "rgba(63, 80, 109, 0.82)",
    "tab_text": "#EFF4FF",
    "accent": "#9BB9FF",
    "plot_template": "plotly_dark",
    "paper_bg": "#2A354A",
    "plot_bg": "#2A354A",
    "grid": "rgba(199,212,235,0.20)",
    "price_up": "#87D2B2",
    "price_down": "#F0A2A8",
    "sma20": "#A7D4FF",
    "sma60": "#F6CE91",
    "vwap": "#A8DFC0",
    "bb_upper": "rgba(240,162,168,0.36)",
    "bb_lower": "rgba(167,212,255,0.34)",
    "volume_up": "rgba(135,210,178,0.48)",
    "volume_down": "rgba(240,162,168,0.46)",
    "equity": "#7ccaa8",
    "benchmark": "#b3a2ef",
    "buy_hold": "#f1c27d",
    "asset_palette": ["#98c1d9", "#b8c0ff", "#9bc53d", "#f4a261", "#80ed99", "#f6bd60"],
    "signal_buy": "#90c4f4",
    "signal_sell": "#f4b184",
    "fill_buy": "#7ccaa8",
    "fill_sell": "#e5989b",
    "marker_edge": "#e5ecf6",
    "trade_path": "rgba(203,213,225,0.50)",
    "fill_link": "rgba(176,190,210,0.34)",
}

_BASE_LIGHT_PALETTE: dict[str, object] = {
    "is_dark": False,
    "background": "#ffffff",
    "sidebar_bg": "#f8fafc",
    "text_color": "#0f172a",
    "text_muted": "#475569",
    "card_bg": "rgba(12, 18, 28, 0.035)",
    "card_border": "rgba(49, 61, 82, 0.16)",
    "control_bg": "#ffffff",
    "control_border": "rgba(49, 61, 82, 0.22)",
    "tab_bg": "rgba(148, 163, 184, 0.12)",
    "tab_text": "#0f172a",
    "accent": "#2563eb",
    "plot_template": "plotly_white",
    "paper_bg": "#ffffff",
    "plot_bg": "#ffffff",
    "grid": "rgba(15,23,42,0.10)",
    "price_up": "#16a34a",
    "price_down": "#dc2626",
    "sma20": "#0284c7",
    "sma60": "#d97706",
    "vwap": "#059669",
    "bb_upper": "rgba(220,38,38,0.28)",
    "bb_lower": "rgba(37,99,235,0.24)",
    "volume_up": "rgba(22,163,74,0.35)",
    "volume_down": "rgba(220,38,38,0.28)",
    "equity": "#16a34a",
    "benchmark": "#7c3aed",
    "buy_hold": "#d97706",
    "asset_palette": ["#0284c7", "#be123c", "#0f766e", "#7c2d12", "#4f46e5", "#0369a1"],
    "signal_buy": "#0284c7",
    "signal_sell": "#ea580c",
    "fill_buy": "#16a34a",
    "fill_sell": "#dc2626",
    "marker_edge": "#0f172a",
    "trade_path": "rgba(71,85,105,0.45)",
    "fill_link": "rgba(100,116,139,0.30)",
}

_THEME_PALETTES: dict[str, dict[str, object]] = {
    "夜間專業（Slate Pro）": _palette_with(_BASE_DARK_PALETTE),
    "北歐夜色（Nord Calm）": _palette_with(
        _BASE_DARK_PALETTE,
        background="#2E3440",
        sidebar_bg="#323B4B",
        paper_bg="#3B4252",
        plot_bg="#3B4252",
        accent="#81A1C1",
        text_color="#ECEFF4",
        text_muted="#D8DEE9",
        card_bg="rgba(67, 76, 94, 0.78)",
        control_bg="rgba(74, 84, 105, 0.95)",
        sma20="#88C0D0",
        sma60="#EBCB8B",
        benchmark="#B48EAD",
        asset_palette=["#8FBCBB", "#D08770", "#A3BE8C", "#5E81AC", "#EBCB8B", "#BF616A"],
    ),
    "深海藍（Ocean Night）": _palette_with(
        _BASE_DARK_PALETTE,
        background="#1E293B",
        sidebar_bg="#243349",
        paper_bg="#223047",
        plot_bg="#223047",
        accent="#7DD3FC",
        text_color="#F8FBFF",
        text_muted="#D6E3F2",
        price_up="#86EFAC",
        price_down="#FDA4AF",
        sma20="#7DD3FC",
        sma60="#FDE68A",
        buy_hold="#FDE68A",
        asset_palette=["#67E8F9", "#A7F3D0", "#C4B5FD", "#FDBA74", "#93C5FD", "#F9A8D4"],
    ),
    "日光白（Paper Light）": _palette_with(_BASE_LIGHT_PALETTE),
    "暖陽米白（Warm Paper）": _palette_with(
        _BASE_LIGHT_PALETTE,
        background="#F8F4EA",
        sidebar_bg="#F3ECDD",
        paper_bg="#FFF9EE",
        plot_bg="#FFF9EE",
        card_bg="rgba(120, 90, 35, 0.08)",
        card_border="rgba(132, 105, 60, 0.24)",
        control_bg="#FFF6E5",
        control_border="rgba(120, 90, 35, 0.28)",
        tab_bg="rgba(193, 154, 90, 0.20)",
        tab_text="#3F2F1D",
        text_color="#2E2518",
        text_muted="#685742",
        accent="#AF7E39",
        grid="rgba(120,90,35,0.16)",
        signal_buy="#1D8F5A",
        signal_sell="#C66A2B",
        fill_buy="#2F9D6B",
        fill_sell="#C44D4D",
    ),
    "薄荷清晨（Mint Day）": _palette_with(
        _BASE_LIGHT_PALETTE,
        background="#F2F9F6",
        sidebar_bg="#E7F5EF",
        paper_bg="#F8FCFA",
        plot_bg="#F8FCFA",
        card_bg="rgba(16, 94, 79, 0.07)",
        card_border="rgba(16, 94, 79, 0.20)",
        control_bg="#F6FCF9",
        control_border="rgba(16, 94, 79, 0.25)",
        tab_bg="rgba(16, 94, 79, 0.16)",
        tab_text="#0F4A3E",
        text_color="#0F2F2A",
        text_muted="#3E6960",
        accent="#0E8B74",
        price_up="#129A64",
        price_down="#D1495B",
        sma20="#1976D2",
        sma60="#C27803",
        vwap="#0E8B74",
        benchmark="#5E35B1",
        asset_palette=["#028090", "#00A896", "#5C7AEA", "#D1495B", "#2E7D32", "#9C6644"],
    ),
}


def _theme_options() -> list[str]:
    return list(_THEME_PALETTES.keys())


def _current_theme_name() -> str:
    legacy_dark = bool(st.session_state.get("ui_dark_mode", False))
    default_theme = "夜間專業（Slate Pro）" if legacy_dark else "日光白（Paper Light）"
    theme = str(st.session_state.get("ui_theme", default_theme))
    if theme not in _THEME_PALETTES:
        theme = default_theme
    st.session_state["ui_theme"] = theme
    return theme


def _is_dark_mode() -> bool:
    theme = _current_theme_name()
    return bool(_THEME_PALETTES[theme].get("is_dark", False))


def _ui_palette() -> dict[str, object]:
    theme = _current_theme_name()
    return _THEME_PALETTES[theme]


def _inject_ui_styles():
    palette = _ui_palette()
    background = str(palette["background"])
    sidebar_bg = str(palette["sidebar_bg"])
    text_color = str(palette["text_color"])
    text_muted = str(palette["text_muted"])
    card_bg = str(palette["card_bg"])
    card_border = str(palette["card_border"])
    control_bg = str(palette["control_bg"])
    control_border = str(palette["control_border"])
    tab_bg = str(palette["tab_bg"])
    tab_text = str(palette["tab_text"])
    accent = str(palette["accent"])

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: {background};
            color: {text_color};
        }}
        [data-testid="stSidebar"] {{
            background: {sidebar_bg};
        }}
        .main .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
            max-width: 1360px;
        }}
        .stMarkdown, .stMetricLabel, .stMetricValue {{
            color: {text_color} !important;
        }}
        .stCaption {{
            color: {text_muted} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}
        a {{
            color: {accent} !important;
        }}
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stDateInput"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stMultiSelect"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label,
        [data-testid="stSlider"] label,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {{
            color: {text_color} !important;
        }}
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-testid="stDateInput"] div {{
            background: {control_bg} !important;
            border-color: {control_border} !important;
        }}
        [data-baseweb="input"] input,
        [data-baseweb="select"] span,
        [data-baseweb="select"] div,
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea {{
            color: {text_color} !important;
            caret-color: {text_color} !important;
        }}
        [data-baseweb="popover"] *,
        [role="listbox"] *,
        [data-baseweb="menu"] * {{
            color: {text_color} !important;
        }}
        [data-baseweb="popover"] > div,
        [role="listbox"],
        [data-baseweb="menu"] {{
            background: {control_bg} !important;
            border: 1px solid {control_border} !important;
        }}
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {{
            background: {control_bg} !important;
            color: {text_color} !important;
            border: 1px solid {control_border} !important;
        }}
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {{
            border-color: {accent} !important;
        }}
        [data-testid="stAlert"] {{
            color: {text_color} !important;
        }}
        [data-testid="stCodeBlock"] pre,
        code {{
            color: {text_color} !important;
        }}
        [data-testid="stDataFrame"] * {{
            color: {text_color} !important;
        }}
        div[data-testid="stMetric"] {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 10px;
            padding: 8px 12px;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        div[data-baseweb="tab"] {{
            background: {tab_bg};
            color: {tab_text};
            border-radius: 10px;
            padding-left: 14px;
            padding-right: 14px;
        }}
        .stTabs [aria-selected="true"] {{
            box-shadow: inset 0 -2px 0 {accent};
        }}
        [data-testid="stExpander"] details {{
            border-radius: 10px;
            border: 1px solid {card_border};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_quality_bar(ctx, refresh_sec: int):
    quote = ctx.quote
    quality = ctx.quality
    freshness = "—" if quality.freshness_sec is None else f"{quality.freshness_sec}s"
    st.caption(
        f"資料品質：source={quote.source} | delayed={'yes' if quote.is_delayed else 'no'} | "
        f"fallback_depth={quality.fallback_depth} | freshness={freshness} | refresh={refresh_sec}s"
    )
    st.caption(f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（fragment 局部刷新）")


def _render_live_chart(ind: pd.DataFrame):
    palette = _ui_palette()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.72, 0.28])
    fig.add_trace(
        go.Candlestick(
            x=ind.index,
            open=ind["open"],
            high=ind["high"],
            low=ind["low"],
            close=ind["close"],
            name="K線",
            increasing_line_color=str(palette["price_up"]),
            increasing_fillcolor=str(palette["price_up"]),
            decreasing_line_color=str(palette["price_down"]),
            decreasing_fillcolor=str(palette["price_down"]),
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
    st.plotly_chart(fig, use_container_width=True)


def _render_live_view():
    service = _market_service()

    with st.sidebar:
        st.subheader("介面")
        theme_options = _theme_options()
        current_theme = _current_theme_name()
        default_idx = theme_options.index(current_theme) if current_theme in theme_options else 0
        st.selectbox(
            "配色主題",
            options=theme_options,
            index=default_idx,
            key="ui_theme",
            help="提供多種專業配色（含夜間與日間），可依閱讀情境快速切換。",
        )
        st.caption(f"目前主題：{st.session_state.get('ui_theme', current_theme)}")
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
        keep_minutes = st.slider("保留即時資料（分鐘）", min_value=30, max_value=360, value=180, step=30)
        use_yahoo = st.checkbox("允許 Yahoo 補K", value=True)

        st.subheader("建議偏好")
        advice_mode = st.selectbox("建議模式", options=["一般(技術面)", "股癌風格(心法/檢核)"], index=1)
        horizon = st.selectbox("時間尺度", options=["短線", "中期", "長期"], index=1)
        risk = st.selectbox("風險偏好", options=["保守", "一般", "積極"], index=1)
        style = st.selectbox("操作風格", options=["定期定額", "波段", "趨勢"], index=2)

    profile = Profile(horizon=horizon, risk=risk, style=style)
    run_every = f"{refresh_sec}s"

    @st.fragment(run_every=run_every)
    def live_fragment():
        options = LiveOptions(use_yahoo=use_yahoo, keep_minutes=keep_minutes, exchange=exchange)
        if market == "台股(TWSE)":
            assert stock_id is not None
            tick_key = f"ticks:TW:{stock_id}:{exchange}"
            ticks = st.session_state.get(tick_key, pd.DataFrame(columns=["ts", "price", "cum_volume"]))
            try:
                ctx, ticks = service.get_tw_live_context(stock_id, yahoo_symbol, ticks=ticks, options=options)
                st.session_state[tick_key] = ticks
            except Exception as exc:
                st.error(f"台股資料取得失敗：{exc}")
                return
            quote = ctx.quote
        else:
            assert us_symbol is not None
            if us_symbol.isdigit():
                st.warning("目前為美股模式，請輸入美股代碼（例如 AAPL、TSLA）；若要查台股請切換到台股模式。")
                return
            try:
                ctx = service.get_us_live_context(us_symbol, yahoo_symbol, options)
            except Exception as exc:
                st.error(f"美股資料取得失敗：{exc}")
                return
            quote = ctx.quote

        st.markdown("#### 即時總覽")
        _render_quality_bar(ctx, refresh_sec=refresh_sec)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("名稱", quote.extra.get("name", quote.symbol))
        col2.metric("最新價", _format_price(quote.price))
        col3.metric("漲跌", "—" if quote.change is None else f"{quote.change:+.2f}")
        col4.metric("漲跌幅", "—" if quote.change_pct is None else f"{quote.change_pct:+.2f}%")
        col5, col6 = st.columns(2)
        col5.metric("成交量", _format_int(quote.volume))
        col6.metric("時間", quote.ts.strftime("%Y-%m-%d %H:%M:%S"))
        st.caption("非投資建議；僅供教育/研究。")

        bars_intraday = ctx.intraday.sort_index().dropna(subset=["open", "high", "low", "close"], how="any")
        if bars_intraday.empty:
            st.warning("目前無法取得走勢資料。")
            return

        ind = add_indicators(bars_intraday)

        daily_long = ctx.daily.copy() if isinstance(ctx.daily, pd.DataFrame) else pd.DataFrame()
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
            st.markdown("#### 即時走勢")
            _render_live_chart(ind)
            if not daily_long.empty:
                st.markdown("#### 長期視角（Daily）")
                if daily_split_events:
                    ev_txt = ", ".join([f"{pd.Timestamp(ev.date).date()} x{ev.ratio:.6f}" for ev in daily_split_events])
                    st.caption(f"已套用分割調整（復權）：{ev_txt}")
                daily = add_indicators(daily_long.dropna(subset=["close"], how="any"))
                st.line_chart(daily[["close", "sma_20", "sma_60"]].dropna(how="all"))

        with side_col:
            with st.container(border=True):
                st.markdown("#### 建議")
                if advice_mode.startswith("股癌"):
                    if render_advice_scai_style is None:
                        st.warning("股癌風格載入失敗，已改用一般模式。")
                        st.text(render_advice(ind, profile))
                    else:
                        st.text(render_advice_scai_style(ind, profile, symbol=yahoo_symbol, fundamentals=ctx.fundamentals))
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
                    st.dataframe(ind.iloc[-1][show_cols].to_frame("value").T, use_container_width=True)

            if market == "台股(TWSE)":
                with st.container(border=True):
                    st.markdown("#### 買賣五檔（TW MIS）")
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
                    st.dataframe(ob, use_container_width=True, hide_index=True)
            elif ctx.fundamentals:
                with st.container(border=True):
                    st.markdown("#### 基本面快照（Yahoo）")
                    st.dataframe(pd.DataFrame([ctx.fundamentals]), use_container_width=True, hide_index=True)

            with st.container(border=True):
                st.markdown("#### 外部參考")
                refs = service.get_reference_context("TW" if market == "台股(TWSE)" else "US")
                if refs.empty:
                    st.caption("目前無法取得外部參考資料。")
                else:
                    st.dataframe(refs, use_container_width=True, hide_index=True)

    live_fragment()


def _metrics_to_rows(metrics) -> list[tuple[str, str]]:
    return [
        ("總報酬", f"{metrics.total_return * 100:.2f}%"),
        ("CAGR", f"{metrics.cagr * 100:.2f}%"),
        ("MDD", f"{metrics.max_drawdown * 100:.2f}%"),
        ("Sharpe", f"{metrics.sharpe:.2f}"),
        ("勝率", f"{metrics.win_rate * 100:.2f}%"),
        ("平均獲利", f"{metrics.avg_win:,.0f}"),
        ("平均虧損", f"{metrics.avg_loss:,.0f}"),
        ("交易次數", f"{metrics.trades}"),
    ]


def _render_backtest_view():
    def _parse_symbols(text: str) -> list[str]:
        symbols = [s.strip().upper() for s in text.replace("，", ",").split(",")]
        out = []
        for sym in symbols:
            if sym and sym not in out:
                out.append(sym)
        return out

    def _is_tw_etf(symbol: str) -> bool:
        text = (symbol or "").strip().upper()
        return len(text) == 4 and text.isdigit() and text.startswith("00")

    def _default_cost_params(market_code: str, symbol_list: list[str]) -> tuple[float, float, float]:
        if market_code == "US":
            return 0.0005, 0.0, 0.0010
        tw_tax = 0.001 if symbol_list and all(_is_tw_etf(s) for s in symbol_list) else 0.003
        return 0.001425, tw_tax, 0.0005

    def _series_metrics(series: pd.Series) -> dict[str, float]:
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

    def _benchmark_candidates(market_code: str, choice: str) -> list[str]:
        selected = (choice or "auto").strip().lower()
        if selected == "off":
            return []
        if market_code == "TW":
            mapping = {
                "auto": ["^TWII", "0050", "006208"],
                "twii": ["^TWII"],
                "0050": ["0050"],
                "006208": ["006208"],
            }
            return mapping.get(selected, ["^TWII"])
        mapping = {
            "auto": ["^GSPC", "SPY", "QQQ", "DIA"],
            "gspc": ["^GSPC"],
            "spy": ["SPY"],
            "qqq": ["QQQ"],
            "dia": ["DIA"],
        }
        return mapping.get(selected, ["^GSPC"])

    def _load_benchmark_from_store(
        market_code: str,
        start: datetime,
        end: datetime,
        choice: str,
    ) -> pd.DataFrame:
        candidates = _benchmark_candidates(market_code, choice)
        if not candidates:
            return pd.DataFrame(columns=["close"])

        for bench_symbol in candidates:
            bars = store.load_daily_bars(symbol=bench_symbol, market=market_code, start=start, end=end)
            end_ts = pd.Timestamp(end).tz_convert("UTC")
            needs_sync = bars.empty or pd.Timestamp(bars.index.max()).tz_convert("UTC") < end_ts
            if needs_sync:
                store.sync_symbol_history(symbol=bench_symbol, market=market_code, start=start, end=end)
                bars = store.load_daily_bars(symbol=bench_symbol, market=market_code, start=start, end=end)
            if bars.empty or "close" not in bars.columns:
                continue

            out = bars[["close"]].copy()
            source_text = ""
            if "source" in bars.columns:
                source_vals = sorted(set(bars["source"].dropna().astype(str)))
                if source_vals:
                    source_text = ",".join(source_vals)
            out.attrs["symbol"] = bench_symbol
            out.attrs["source"] = f"sqlite:{source_text}" if source_text else "sqlite"
            return out

        return pd.DataFrame(columns=["close"])

    store = _history_store()
    service = _market_service()

    st.subheader("回測工作台（日K）")
    st.caption("先設定基本條件，再調整策略與成本參數；進階設定可用於 Walk-Forward 與分割調整。")
    st.markdown("#### 基本設定")
    c1, c2, c3, c4 = st.columns(4)
    market = c1.selectbox("市場", ["TW", "US"], index=0, key="bt_market")
    mode = c2.selectbox("模式", ["單一標的", "投組(多標的)"], index=0, key="bt_mode")
    default_symbol = "0052" if market == "TW" else "TSLA"
    symbol_text = c3.text_input(
        "代碼（投組用逗號分隔）",
        value=default_symbol if mode == "單一標的" else ("0052,2330" if market == "TW" else "AAPL,MSFT,TSLA"),
        key="bt_symbol",
    ).strip().upper()
    strategy = c4.selectbox("策略", ["buy_hold", "sma_cross", "ema_cross", "rsi_reversion", "macd_trend"], index=1)
    symbols = _parse_symbols(symbol_text)
    if mode == "單一標的":
        symbols = symbols[:1]
    if not symbols:
        st.warning("請輸入至少一個代碼。")
        return

    auto_cost_key = "bt_auto_cost"
    fee_key = "bt_fee_rate"
    tax_key = "bt_sell_tax"
    slip_key = "bt_slippage"
    cost_profile_key = "bt_cost_profile"
    if auto_cost_key not in st.session_state:
        st.session_state[auto_cost_key] = True
    current_cost_profile = f"{market}:{','.join(symbols)}"
    if st.session_state.get(auto_cost_key) and st.session_state.get(cost_profile_key) != current_cost_profile:
        fee_default, tax_default, slip_default = _default_cost_params(market, symbols)
        st.session_state[fee_key] = float(fee_default)
        st.session_state[tax_key] = float(tax_default)
        st.session_state[slip_key] = float(slip_default)
        st.session_state[cost_profile_key] = current_cost_profile

    d1, d2 = st.columns(2)
    start_date = d1.date_input("起始日期", value=date(date.today().year - 5, 1, 1))
    end_date = d2.date_input("結束日期", value=date.today())
    benchmark_options = (
        [("auto", "Auto（台股加權 ^TWII，失敗時改用 0050/006208）"), ("twii", "^TWII"), ("0050", "0050（ETF代理）"), ("006208", "006208（ETF代理）"), ("off", "關閉 Benchmark")]
        if market == "TW"
        else [("auto", "Auto（S&P500 ^GSPC，失敗時改用 SPY/QQQ/DIA）"), ("gspc", "^GSPC"), ("spy", "SPY（ETF代理）"), ("qqq", "QQQ（ETF代理）"), ("dia", "DIA（ETF代理）"), ("off", "關閉 Benchmark")]
    )
    bench_codes = [x[0] for x in benchmark_options]
    bench_labels = {x[0]: x[1] for x in benchmark_options}
    benchmark_choice = d1.selectbox(
        "Benchmark",
        options=bench_codes,
        format_func=lambda v: bench_labels.get(v, v),
        index=0,
        help="可手動選基準；Auto 會在主來源失敗時自動改用替代基準。",
    )
    invest_mode_key = "bt_invest_start_mode"
    invest_date_key = "bt_invest_start_date"
    invest_k_key = "bt_invest_start_k"

    st.markdown("#### 策略參數")
    param1, param2, _ = st.columns(3)
    if strategy in {"sma_cross", "ema_cross"}:
        fast = param1.slider(
            "Fast",
            min_value=3,
            max_value=80,
            value=10 if strategy == "sma_cross" else 12,
            help="短週期均線。數字越小越敏感，訊號會更快但雜訊也可能變多。",
        )
        slow = param2.slider(
            "Slow",
            min_value=10,
            max_value=240,
            value=30 if strategy == "sma_cross" else 26,
            help="長週期均線。通常要大於 Fast，用來過濾趨勢方向。",
        )
        strategy_params = {"fast": float(fast), "slow": float(slow)}
    elif strategy == "rsi_reversion":
        buy_below = param1.slider(
            "RSI Buy Below",
            min_value=10,
            max_value=40,
            value=30,
            help="RSI 低於此值視為相對偏弱，策略偏向找反彈買點。",
        )
        sell_above = param2.slider(
            "RSI Sell Above",
            min_value=45,
            max_value=80,
            value=55,
            help="RSI 高於此值視為相對偏強，策略偏向減碼/出場。",
        )
        strategy_params = {"buy_below": float(buy_below), "sell_above": float(sell_above)}
    else:
        strategy_params = {}

    auto_cost = st.checkbox(
        "自動套用市場成本參數（台/美股）",
        key=auto_cost_key,
        help="切換市場或標的時，自動帶入對應的手續費、交易稅與滑價預設值。",
    )
    if auto_cost:
        st.caption("US 預設：Fee 0.0005 / Sell Tax 0 / Slippage 0.001；TW 依股票或 ETF 自動調整。")
    elif st.session_state.get(cost_profile_key) != current_cost_profile:
        st.session_state[cost_profile_key] = current_cost_profile

    st.markdown("#### 交易成本")
    cost1, cost2, cost3 = st.columns(3)
    fee_rate = cost1.number_input(
        "Fee Rate",
        min_value=0.0,
        max_value=0.01,
        value=float(st.session_state.get(fee_key, 0.001425)),
        step=0.0001,
        format="%.6f",
        key=fee_key,
        help="每次買進/賣出收取的手續費比例，例如 0.001425 = 0.1425%。",
    )
    sell_tax = cost2.number_input(
        "Sell Tax",
        min_value=0.0,
        max_value=0.01,
        value=float(st.session_state.get(tax_key, 0.0 if market == "US" else 0.003)),
        step=0.0001,
        format="%.6f",
        key=tax_key,
        help="賣出時交易稅比例。台股股票常見 0.3%，ETF 常見 0.1%。",
    )
    slippage = cost3.number_input(
        "Slippage",
        min_value=0.0,
        max_value=0.01,
        value=float(st.session_state.get(slip_key, 0.0005)),
        step=0.0001,
        format="%.6f",
        key=slip_key,
        help="滑價：實際成交價相對理論價的偏移比例，通常用來模擬流動性與摩擦成本。",
    )
    initial_capital = st.number_input(
        "初始資產（回測本金）",
        min_value=10_000.0,
        max_value=1_000_000_000.0,
        value=1_000_000.0,
        step=10_000.0,
        format="%.0f",
    )

    st.markdown("#### 進階設定")
    wf1, wf2, wf3 = st.columns(3)
    enable_wf = wf1.checkbox(
        "啟用 Walk-Forward",
        value=False,
        help="先用 Train 區段挑參數，再到 Test 區段驗證，降低過度擬合風險。",
    )
    if strategy == "buy_hold" and enable_wf:
        st.info("`buy_hold` 不需參數挑選，已自動以一般回測模式執行。")
        enable_wf = False
    train_ratio = wf2.slider(
        "Train 比例",
        min_value=0.50,
        max_value=0.85,
        value=0.70,
        step=0.05,
        help="例如 0.70 代表前 70% 用於挑參數，後 30% 用於驗證。",
    )
    objective = wf3.selectbox(
        "參數挑選目標",
        ["sharpe", "cagr", "total_return", "mdd"],
        index=0,
        help="Walk-Forward 會用這個指標在訓練區挑最佳參數。",
    )
    adj1, adj2 = st.columns(2)
    use_split_adjustment = adj1.checkbox("分割調整（復權）", value=True)
    auto_detect_split = adj2.checkbox("自動偵測分割事件", value=True)
    sp1, sp2, sp3 = st.columns(3)
    if invest_mode_key not in st.session_state:
        st.session_state[invest_mode_key] = "沿用回測起始日期"
    if invest_date_key not in st.session_state:
        st.session_state[invest_date_key] = start_date
    if invest_k_key not in st.session_state:
        st.session_state[invest_k_key] = 0

    invest_start_mode = sp1.selectbox(
        "實際投入起點",
        ["沿用回測起始日期", "指定日期", "指定第幾根K"],
        key=invest_mode_key,
        help="決定本金實際開始投入回測的位置；不是回放進度控制。",
    )
    invest_start_date = None
    invest_start_k = None
    if invest_start_mode == "指定日期":
        current_invest_date = st.session_state.get(invest_date_key, start_date)
        if current_invest_date < start_date:
            current_invest_date = start_date
        if current_invest_date > end_date:
            current_invest_date = end_date
        st.session_state[invest_date_key] = current_invest_date
        invest_start_date = sp2.date_input(
            "投入日期",
            min_value=start_date,
            max_value=end_date,
            key=invest_date_key,
            help="回測會從這個日期（含）開始計算資產與交易。",
        )
    elif invest_start_mode == "指定第幾根K":
        invest_start_k = int(
            sp2.number_input(
                "投入第幾根K（從0開始）",
                min_value=0,
                max_value=200000,
                step=1,
                key=invest_k_key,
                help="0 代表最早一根K；數字越大代表越晚投入。",
            )
        )
        sp3.caption("0 代表最早一根，數字越大代表越晚投入。")

    with st.expander("參數說明（中文）", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- `Slippage（滑價）`單位是比例（小數）。例如：`0.0005 = 0.05%`，`0.001 = 0.1%`。",
                    "- 本系統會在買進與賣出各套用一次滑價（雙邊成本）。",
                    "- `自動套用市場成本參數`：切換台/美股或標的時，自動帶入該市場預設手續費、交易稅、滑價。",
                    "- `Walk-Forward`：先在訓練區（Train）找最佳參數，再到測試區（Test）驗證穩健性。",
                    "- `Train 比例`：決定資料切分比例；例如 `0.70` = 前 70% 訓練、後 30% 測試。",
                    "- `參數挑選目標`（Walk-Forward）：",
                    "  - `sharpe`：風險調整後報酬（越高越好）。",
                    "  - `cagr`：年化報酬率（越高越好）。",
                    "  - `total_return`：區間總報酬（越高越好）。",
                    "  - `mdd`：最大回撤（越小越好，系統會挑回撤較小的參數）。",
                    "- `Fast/Slow`：均線短週期/長週期參數（通常 `Fast < Slow`）。",
                    "- `RSI Buy Below / Sell Above`：RSI 的進出場門檻值。",
                    "- `buy_hold`：買進後持有到回測結束，適合快速檢查「近期投入」表現。",
                    "- `實際投入起點`：決定本金從哪個時間點開始參與回測；",
                    "  `回放位置`只是看圖播放進度，不會改變回測結果。",
                    "- `買賣點顯示`可切換：",
                    "  - `訊號點（價格圖）`：顯示策略由空手/持有切換的時間點。",
                    "  - `實際成交點（資產圖）`：顯示回測成交點（依規則為 T+1 開盤成交）。",
                    "  - `同時顯示（訊號+成交）`：同時顯示訊號點與實際成交點。",
                    "- `Equity`：策略資產曲線，代表回測期間的總資產（現金 + 持倉市值）。",
                    "- `Benchmark Equity`：把基準指數（台股 `^TWII` / 美股 `^GSPC`）縮放到相同初始資產後的曲線。",
                    "- `Benchmark`：可手動選擇；`Auto` 會在主來源失敗時自動 fallback。",
                    "- `Benchmark 比較`：同期間比較策略與基準的總報酬、CAGR、MDD、Sharpe。",
                    "- `策略 vs 買進持有`：比較策略報酬率與買進持有（單檔或等權投組）的報酬率。",
                    "- `指定區間報酬率`：可輸入任意日期區間（如 2026-01-01 ~ 2026-02-11）比較策略與買進持有。",
                    "- `位置顯示`可切換：",
                    "  - `K棒`：用第幾根K（0 起算）定位。",
                    "  - `日期`：用交易日定位回放位置。",
                    "- `回放視窗`：",
                    "  - `固定視窗`：可用 `視窗K數` 與 `生長位置（靠右%）` 調整線圖長出來的位置。",
                    "  - `完整區間`：從第一根一路看到目前回放位置。",
                    "- `Benchmark Base Date`：Benchmark 與策略都會從此基準日（實際交易日）開始重設為 1.0 再比較。",
                    "- `顯示成交連線`：在 Buy/Sell Fill 時刻加上垂直細線，方便對照價格圖與資產圖。",
                ]
            )
        )

    auto_sync = st.checkbox("App 啟動時自動增量同步", value=True)
    sync_key = f"synced:{market}:{','.join(symbols)}:{start_date}:{end_date}"
    sync_start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    sync_end = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    if auto_sync and not st.session_state.get(sync_key):
        sync_rows = []
        for symbol in symbols:
            report = store.sync_symbol_history(symbol=symbol, market=market, start=sync_start, end=sync_end)
            sync_rows.append(
                {
                    "symbol": symbol,
                    "rows": report.rows_upserted,
                    "source": report.source,
                    "fallback": report.fallback_depth,
                    "error": report.error or "",
                }
            )
        st.session_state[sync_key] = True
        sync_df = pd.DataFrame(sync_rows)
        if (sync_df["error"] != "").any():
            st.warning("部分同步失敗，請檢查下方同步結果。")
        st.dataframe(sync_df, use_container_width=True, hide_index=True)

    b1, b2 = st.columns(2)
    if b1.button("同步歷史資料", use_container_width=True):
        sync_rows = []
        for symbol in symbols:
            report = store.sync_symbol_history(symbol=symbol, market=market, start=sync_start, end=sync_end)
            sync_rows.append(
                {
                    "symbol": symbol,
                    "rows": report.rows_upserted,
                    "source": report.source,
                    "fallback": report.fallback_depth,
                    "error": report.error or "",
                }
            )
        sync_df = pd.DataFrame(sync_rows)
        if (sync_df["error"] != "").any():
            st.error("部分同步失敗，請檢查同步結果。")
        else:
            st.success("同步成功。")
        st.dataframe(sync_df, use_container_width=True, hide_index=True)

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    availability_rows = []
    for symbol in symbols:
        bars = store.load_daily_bars(symbol=symbol, market=market, start=sync_start, end=sync_end)
        if bars.empty:
            availability_rows.append({"symbol": symbol, "rows": 0, "sources": "", "status": "EMPTY"})
            continue
        bars = bars.sort_index()
        if use_split_adjustment:
            bars, split_events = apply_split_adjustment(
                bars=bars,
                symbol=symbol,
                market=market,
                use_known=True,
                use_auto_detect=auto_detect_split,
            )
        else:
            split_events = []
        bars_by_symbol[symbol] = bars
        availability_rows.append(
            {
                "symbol": symbol,
                "rows": int(len(bars)),
                "sources": ",".join(sorted(set(bars["source"].dropna().astype(str)))) if "source" in bars.columns else "",
                "status": "OK",
                "splits": ", ".join(
                    [f"{pd.Timestamp(ev.date).date()} x{ev.ratio:.6f}({ev.source})" for ev in split_events]
                )
                if split_events
                else "",
            }
        )
    if availability_rows:
        st.dataframe(pd.DataFrame(availability_rows), use_container_width=True, hide_index=True)
    if not bars_by_symbol:
        st.info("資料庫尚無可用資料，請先同步。")
        return

    bars_by_symbol, invest_df = apply_start_to_bars_map(
        bars_by_symbol=bars_by_symbol,
        mode=invest_start_mode,
        start_date=invest_start_date,
        start_k=invest_start_k,
    )
    if not invest_df.empty:
        st.markdown("**投入起點檢查**")
        st.dataframe(invest_df, use_container_width=True, hide_index=True)
    min_required_bars = 80 if enable_wf else (2 if strategy == "buy_hold" else 40)
    bars_by_symbol = {sym: bars for sym, bars in bars_by_symbol.items() if len(bars) >= min_required_bars}
    if not bars_by_symbol:
        if enable_wf:
            st.warning("投入起點後可用資料太少（Walk-Forward 至少需要 80 根K），請調整起點或日期區間。")
        elif strategy == "buy_hold":
            st.warning("投入起點後可用資料太少（buy_hold 至少需要 2 根K），請調整起點或日期區間。")
        else:
            st.warning("投入起點後可用資料太少（策略回測至少需要 40 根K）。若只想看近期投入，可改用 `buy_hold`。")
        return

    run_key = (
        f"bt_result:{market}:{','.join(symbols)}:{strategy}:{start_date}:{end_date}:"
        f"{int(enable_wf)}:{train_ratio}:{objective}:{int(initial_capital)}:"
        f"{invest_start_mode}:{invest_start_date}:{invest_start_k}"
    )
    if b2.button("執行回測", use_container_width=True):
        cost_model = CostModel(fee_rate=fee_rate, sell_tax_rate=sell_tax, slippage_rate=slippage)
        run_payload = {}
        try:
            if enable_wf:
                if mode == "單一標的":
                    symbol = list(bars_by_symbol.keys())[0]
                    wf_result = walk_forward_single(
                        bars=bars_by_symbol[symbol],
                        strategy_name=strategy,
                        cost_model=cost_model,
                        train_ratio=float(train_ratio),
                        objective=objective,
                        initial_capital=float(initial_capital),
                    )
                    run_payload = {
                        "mode": "single",
                        "walk_forward": True,
                        "initial_capital": float(initial_capital),
                        "symbol": symbol,
                        "bars_by_symbol": bars_by_symbol,
                        "result": wf_result.test_result,
                        "train_result": wf_result.train_result,
                        "split_date": wf_result.split_date,
                        "best_params": wf_result.best_params,
                        "candidates": wf_result.candidates,
                    }
                else:
                    wf_portfolio = walk_forward_portfolio(
                        bars_by_symbol=bars_by_symbol,
                        strategy_name=strategy,
                        cost_model=cost_model,
                        train_ratio=float(train_ratio),
                        objective=objective,
                        initial_capital=float(initial_capital),
                    )
                    run_payload = {
                        "mode": "portfolio",
                        "walk_forward": True,
                        "initial_capital": float(initial_capital),
                        "symbols": list(bars_by_symbol.keys()),
                        "bars_by_symbol": bars_by_symbol,
                        "result": wf_portfolio.test_portfolio,
                        "train_result": wf_portfolio.train_portfolio,
                        "split_date": wf_portfolio.split_date,
                        "best_params": {s: wf.best_params for s, wf in wf_portfolio.symbol_results.items()},
                        "candidates": {s: wf.candidates for s, wf in wf_portfolio.symbol_results.items()},
                    }
            else:
                if mode == "單一標的":
                    symbol = list(bars_by_symbol.keys())[0]
                    result = run_backtest(
                        bars=bars_by_symbol[symbol],
                        strategy_name=strategy,
                        strategy_params=strategy_params,
                        cost_model=cost_model,
                        initial_capital=float(initial_capital),
                    )
                    run_payload = {
                        "mode": "single",
                        "walk_forward": False,
                        "initial_capital": float(initial_capital),
                        "symbol": symbol,
                        "bars_by_symbol": bars_by_symbol,
                        "result": result,
                    }
                else:
                    result = run_portfolio_backtest(
                        bars_by_symbol=bars_by_symbol,
                        strategy_name=strategy,
                        strategy_params=strategy_params,
                        cost_model=cost_model,
                        initial_capital=float(initial_capital),
                    )
                    run_payload = {
                        "mode": "portfolio",
                        "walk_forward": False,
                        "initial_capital": float(initial_capital),
                        "symbols": list(bars_by_symbol.keys()),
                        "bars_by_symbol": bars_by_symbol,
                        "result": result,
                    }
        except Exception as exc:
            st.error(f"回測失敗：{exc}")
            return

        st.session_state[run_key] = run_payload
        primary = ",".join(symbols)
        summary_metrics = run_payload["result"].metrics.__dict__
        store.save_backtest_run(
            symbol=primary,
            market=market,
            strategy=strategy,
            params=run_payload.get("best_params", strategy_params),
            cost={
                "fee_rate": fee_rate,
                "sell_tax_rate": sell_tax,
                "slippage_rate": slippage,
                "initial_capital": float(initial_capital),
            },
            result={
                "metrics": summary_metrics,
                "walk_forward": enable_wf,
                "objective": objective if enable_wf else None,
            },
        )

    payload = st.session_state.get(run_key)
    if not payload:
        st.info("按下「執行回測」後，會顯示績效與回放。")
        return

    result = payload["result"]
    run_initial_capital = float(payload.get("initial_capital", initial_capital))
    bars_by_symbol = payload["bars_by_symbol"]
    selected_symbols = list(bars_by_symbol.keys())
    is_portfolio = payload["mode"] == "portfolio"
    strategy_equity = result.equity_curve["equity"].copy()
    buy_hold_equity = build_buy_hold_equity(
        bars_by_symbol=bars_by_symbol,
        target_index=pd.DatetimeIndex(strategy_equity.index),
        initial_capital=run_initial_capital,
    )
    benchmark_raw = _load_benchmark_from_store(market_code=market, start=sync_start, end=sync_end, choice=benchmark_choice)
    if benchmark_raw.empty:
        benchmark_raw = service.get_benchmark_series(market=market, start=sync_start, end=sync_end, benchmark=benchmark_choice)
    benchmark_split_events = []
    if use_split_adjustment and not benchmark_raw.empty:
        benchmark_symbol = str(benchmark_raw.attrs.get("symbol", "")).strip() if hasattr(benchmark_raw, "attrs") else ""
        if not benchmark_symbol:
            fallback_symbol_map_tw = {"twii": "^TWII", "0050": "0050", "006208": "006208"}
            fallback_symbol_map_us = {"gspc": "^GSPC", "spy": "SPY", "qqq": "QQQ", "dia": "DIA"}
            benchmark_symbol = (
                fallback_symbol_map_tw.get(benchmark_choice, "^TWII")
                if market == "TW"
                else fallback_symbol_map_us.get(benchmark_choice, "^GSPC")
            )
        adjusted, benchmark_split_events = apply_split_adjustment(
            bars=benchmark_raw,
            symbol=benchmark_symbol,
            market=market,
            use_known=True,
            use_auto_detect=auto_detect_split,
        )
        adjusted.attrs = dict(getattr(benchmark_raw, "attrs", {}))
        if benchmark_symbol:
            adjusted.attrs["symbol"] = benchmark_symbol
        benchmark_raw = adjusted
    benchmark_equity = pd.Series(dtype=float)
    if not benchmark_raw.empty and "close" in benchmark_raw.columns:
        bench = benchmark_raw["close"].reindex(strategy_equity.index).ffill()
        bench = bench.dropna()
        if not bench.empty:
            benchmark_equity = (bench / bench.iloc[0]) * run_initial_capital
            benchmark_equity = benchmark_equity.reindex(strategy_equity.index).ffill()

    st.subheader("績效儀表板")
    metric_rows = _metrics_to_rows(result.metrics)
    metric_cols = st.columns(4)
    for idx, (label, val) in enumerate(metric_rows):
        metric_cols[idx % 4].metric(label, val)

    if payload.get("walk_forward"):
        st.subheader("Walk-Forward")
        split_date = payload["split_date"]
        st.caption(f"切分點：{pd.Timestamp(split_date).strftime('%Y-%m-%d')} | 目標：{objective}")
        train = payload["train_result"]
        w1, w2 = st.columns(2)
        with w1:
            st.markdown("**Train(In-Sample)**")
            for label, val in _metrics_to_rows(train.metrics)[:4]:
                st.metric(label, val)
        with w2:
            st.markdown("**Test(Out-of-Sample)**")
            for label, val in _metrics_to_rows(result.metrics)[:4]:
                st.metric(label, val)
        best_params = payload.get("best_params")
        if best_params:
            st.markdown("**最佳參數**")
            if isinstance(best_params, dict) and all(isinstance(v, dict) for v in best_params.values()):
                rows = [{"symbol": s, "params": str(p)} for s, p in best_params.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.code(str(best_params))

    speed_key = f"play_speed:{run_key}"
    play_key = f"play_state:{run_key}"
    idx_key = f"play_idx:{run_key}"
    display_mode_key = f"play_display_mode:{run_key}"
    marker_mode_key = f"play_marker_mode:{run_key}"
    fill_link_key = f"play_fill_link:{run_key}"
    viewport_mode_key = f"play_viewport_mode:{run_key}"
    viewport_window_key = f"play_viewport_window:{run_key}"
    viewport_anchor_key = f"play_viewport_anchor:{run_key}"
    focus_key = f"play_focus:{run_key}"
    if speed_key not in st.session_state:
        st.session_state[speed_key] = "1x"
    if play_key not in st.session_state:
        st.session_state[play_key] = False
    if display_mode_key not in st.session_state:
        st.session_state[display_mode_key] = "K棒"
    if marker_mode_key not in st.session_state:
        st.session_state[marker_mode_key] = "同時顯示（訊號+成交）"
    if fill_link_key not in st.session_state:
        st.session_state[fill_link_key] = True
    if viewport_mode_key not in st.session_state:
        st.session_state[viewport_mode_key] = "固定視窗"
    if viewport_anchor_key not in st.session_state:
        st.session_state[viewport_anchor_key] = 70
    if focus_key not in st.session_state:
        st.session_state[focus_key] = selected_symbols[0]
    focus_symbol = st.selectbox("回放焦點標的", options=selected_symbols, key=focus_key)

    focus_bars = bars_by_symbol[focus_symbol].sort_index().dropna(subset=["open", "high", "low", "close"], how="any")
    if focus_bars.empty:
        st.warning(f"{focus_symbol} 沒有可回放的有效K線資料。")
        return
    focus_result = result.component_results[focus_symbol] if is_portfolio else result
    max_play_idx = len(focus_bars) - 1
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    else:
        st.session_state[idx_key] = min(int(st.session_state[idx_key]), max_play_idx)
    window_min = 20
    window_max = max(window_min, max_play_idx + 1)
    if viewport_window_key not in st.session_state:
        st.session_state[viewport_window_key] = min(180, window_max)
    else:
        st.session_state[viewport_window_key] = int(
            min(max(int(st.session_state[viewport_window_key]), window_min), window_max)
        )
    date_options = [pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in focus_bars.index]
    date_to_idx = {d: i for i, d in enumerate(date_options)}

    st.subheader("回放")
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 2, 2, 3])
    if c1.button("Play", use_container_width=True):
        st.session_state[play_key] = True
    if c2.button("Pause", use_container_width=True):
        st.session_state[play_key] = False
    if c3.button("Reset", use_container_width=True):
        st.session_state[play_key] = False
        st.session_state[idx_key] = 0
    c4.selectbox("速度", options=["0.5x", "1x", "2x", "5x", "10x"], key=speed_key)
    c5.radio("位置顯示", options=["K棒", "日期"], horizontal=True, key=display_mode_key)
    c6.selectbox(
        "買賣點顯示",
        options=["不顯示", "訊號點（價格圖）", "實際成交點（資產圖）", "同時顯示（訊號+成交）"],
        key=marker_mode_key,
    )
    q1, q2, q3 = st.columns([2, 2, 2])
    q1.selectbox("回放視窗", options=["固定視窗", "完整區間"], key=viewport_mode_key)
    q2.slider(
        "視窗K數",
        min_value=window_min,
        max_value=window_max,
        step=5,
        key=viewport_window_key,
        disabled=st.session_state[viewport_mode_key] != "固定視窗",
    )
    q3.slider(
        "生長位置（靠右%）",
        min_value=50,
        max_value=90,
        step=5,
        key=viewport_anchor_key,
        disabled=st.session_state[viewport_mode_key] != "固定視窗",
        help="數值越大，最新K越靠右；例如 70 代表最新K大約落在畫面 70% 的位置。",
    )
    st.checkbox(
        "顯示成交連線（價格圖↔資產圖）",
        key=fill_link_key,
        help="在 Buy/Sell Fill 的時間點畫垂直細線，幫助對照價格與資產變化。",
    )
    st.caption("買賣點模式說明：訊號點=策略切換點；實際成交點=依回測規則 T+1 開盤成交；同時顯示=兩者一起顯示。")
    if st.session_state[display_mode_key] == "K棒":
        st.caption(f"目前以 K 棒序號顯示（0 ~ {max_play_idx}；0 代表最早一根）")
        slider_value = st.slider(
            "回放位置（K棒）",
            min_value=0,
            max_value=max_play_idx,
            value=int(st.session_state[idx_key]),
            disabled=st.session_state[play_key],
        )
        if not st.session_state[play_key]:
            st.session_state[idx_key] = int(slider_value)
    else:
        st.caption(f"目前以交易日期顯示（{date_options[0]} ~ {date_options[-1]}）")
        date_value = st.select_slider(
            "回放位置（日期）",
            options=date_options,
            value=date_options[int(st.session_state[idx_key])],
            disabled=st.session_state[play_key],
        )
        if not st.session_state[play_key]:
            st.session_state[idx_key] = int(date_to_idx[date_value])

    speed_steps = {"0.5x": 1, "1x": 2, "2x": 4, "5x": 8, "10x": 16}

    @st.fragment(run_every="0.5s")
    def playback():
        palette = _ui_palette()
        if st.session_state[play_key]:
            step = speed_steps[st.session_state[speed_key]]
            new_idx = min(len(focus_bars) - 1, st.session_state[idx_key] + step)
            st.session_state[idx_key] = new_idx
            if new_idx >= len(focus_bars) - 1:
                st.session_state[play_key] = False

        idx = st.session_state[idx_key]
        bars_now = focus_bars.iloc[: idx + 1]
        equity_now = result.equity_curve.iloc[: min(idx + 1, len(result.equity_curve))]
        benchmark_now = benchmark_equity.reindex(equity_now.index).ffill() if not benchmark_equity.empty else pd.Series(dtype=float)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28])
        fig.add_trace(
            go.Candlestick(
                x=bars_now.index,
                open=bars_now["open"],
                high=bars_now["high"],
                low=bars_now["low"],
                close=bars_now["close"],
                name="Price",
                increasing_line_color=str(palette["price_up"]),
                increasing_fillcolor=str(palette["price_up"]),
                decreasing_line_color=str(palette["price_down"]),
                decreasing_fillcolor=str(palette["price_down"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=equity_now.index,
                y=equity_now["equity"],
                mode="lines",
                name="Equity",
                line=dict(color=str(palette["equity"]), width=2.0),
            ),
            row=2,
            col=1,
        )
        if not benchmark_now.empty:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_now.index,
                    y=benchmark_now.values,
                    mode="lines",
                    name="Benchmark Equity",
                    line=dict(color=str(palette["benchmark"]), width=1.9),
                ),
                row=2,
                col=1,
            )

        marker_mode = st.session_state[marker_mode_key]
        signal_buy_style = dict(
            color=str(palette["signal_buy"]),
            size=11,
            symbol="triangle-up",
            line=dict(color=str(palette["marker_edge"]), width=1),
        )
        signal_sell_style = dict(
            color=str(palette["signal_sell"]),
            size=11,
            symbol="triangle-down",
            line=dict(color=str(palette["marker_edge"]), width=1),
        )
        fill_buy_style = dict(
            color=str(palette["fill_buy"]),
            size=13,
            symbol="triangle-up",
            line=dict(color=str(palette["marker_edge"]), width=1),
        )
        fill_sell_style = dict(
            color=str(palette["fill_sell"]),
            size=13,
            symbol="triangle-down",
            line=dict(color=str(palette["marker_edge"]), width=1),
        )
        show_signal_markers = marker_mode in {"訊號點（價格圖）", "同時顯示（訊號+成交）"}
        show_fill_markers = marker_mode in {"實際成交點（資產圖）", "同時顯示（訊號+成交）"}
        if show_signal_markers and not focus_result.signals.empty:
            sig_now = focus_result.signals.reindex(bars_now.index).ffill().fillna(0).astype(int)
            buy_idx = sig_now[(sig_now == 1) & (sig_now.shift(1).fillna(0) == 0)].index
            sell_idx = sig_now[(sig_now == 0) & (sig_now.shift(1).fillna(0) == 1)].index
            if len(buy_idx) > 0:
                buy_px = bars_now.loc[buy_idx.intersection(bars_now.index), "close"]
                fig.add_trace(
                    go.Scatter(x=buy_px.index, y=buy_px.values, mode="markers", name="Buy Signal", marker=signal_buy_style),
                    row=1,
                    col=1,
                )
            if len(sell_idx) > 0:
                sell_px = bars_now.loc[sell_idx.intersection(bars_now.index), "close"]
                fig.add_trace(
                    go.Scatter(x=sell_px.index, y=sell_px.values, mode="markers", name="Sell Signal", marker=signal_sell_style),
                    row=1,
                    col=1,
                )
        if show_fill_markers:
            trades = focus_result.trades or []
            if trades:
                cutoff = bars_now.index[-1]
                eq_series = equity_now["equity"]
                buy_x = []
                buy_y = []
                sell_x = []
                sell_y = []
                fill_times: list[pd.Timestamp] = []
                first_line = True
                for tr in trades:
                    entry_dt = pd.Timestamp(tr.entry_date)
                    exit_dt = pd.Timestamp(tr.exit_date)
                    if entry_dt <= cutoff:
                        val = eq_series.reindex([entry_dt], method="ffill")
                        if not val.empty and pd.notna(val.iloc[0]):
                            buy_x.append(entry_dt)
                            buy_y.append(float(val.iloc[0]))
                            fill_times.append(entry_dt)
                    if exit_dt <= cutoff:
                        val = eq_series.reindex([exit_dt], method="ffill")
                        if not val.empty and pd.notna(val.iloc[0]):
                            sell_x.append(exit_dt)
                            sell_y.append(float(val.iloc[0]))
                            fill_times.append(exit_dt)
                    if entry_dt <= cutoff and exit_dt <= cutoff:
                        y1 = eq_series.reindex([entry_dt], method="ffill")
                        y2 = eq_series.reindex([exit_dt], method="ffill")
                        if not y1.empty and not y2.empty and pd.notna(y1.iloc[0]) and pd.notna(y2.iloc[0]):
                            fig.add_trace(
                                go.Scatter(
                                    x=[entry_dt, exit_dt],
                                    y=[float(y1.iloc[0]), float(y2.iloc[0])],
                                    mode="lines",
                                    name="Trade Path",
                                    line=dict(color=str(palette["trade_path"]), width=1),
                                    showlegend=first_line,
                                ),
                                row=2,
                                col=1,
                            )
                            first_line = False
                if st.session_state[fill_link_key] and fill_times:
                    for fill_dt in sorted(set(fill_times)):
                        fig.add_shape(
                            type="line",
                            x0=fill_dt,
                            x1=fill_dt,
                            y0=0.0,
                            y1=1.0,
                            xref="x",
                            yref="paper",
                            line=dict(color=str(palette["fill_link"]), width=1, dash="dot"),
                        )
                if buy_x:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_x,
                            y=buy_y,
                            mode="markers",
                            name="Buy Fill",
                            marker=fill_buy_style,
                        ),
                        row=2,
                        col=1,
                    )
                if sell_x:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_x,
                            y=sell_y,
                            mode="markers",
                            name="Sell Fill",
                            marker=fill_sell_style,
                        ),
                        row=2,
                        col=1,
                    )

        if st.session_state[viewport_mode_key] == "固定視窗":
            view_window = int(st.session_state[viewport_window_key])
            anchor_ratio = float(st.session_state[viewport_anchor_key]) / 100.0
            left_idx = idx - int(round(view_window * anchor_ratio))
            right_idx = left_idx + view_window - 1
            if left_idx < 0:
                right_idx -= left_idx
                left_idx = 0
            if right_idx > max_play_idx:
                left_idx -= right_idx - max_play_idx
                right_idx = max_play_idx
                left_idx = max(0, left_idx)
            x_start = focus_bars.index[left_idx]
            x_end = focus_bars.index[right_idx]
            fig.update_xaxes(range=[x_start, x_end], row=1, col=1)
            fig.update_xaxes(range=[x_start, x_end], row=2, col=1)

        fig.update_xaxes(gridcolor=str(palette["grid"]))
        fig.update_yaxes(gridcolor=str(palette["grid"]))
        fig.update_layout(
            height=680,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            uirevision=f"playback:{run_key}",
            transition=dict(duration=0),
            template=str(palette["plot_template"]),
            paper_bgcolor=str(palette["paper_bg"]),
            plot_bgcolor=str(palette["plot_bg"]),
            font=dict(color=str(palette["text_color"])),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"playback_chart:{run_key}")
        st.caption(f"目前回放到：第 {idx + 1} 根K（{bars_now.index[-1].strftime('%Y-%m-%d')}）")

    playback()

    model_params = payload.get("best_params") if payload.get("walk_forward") else strategy_params
    if isinstance(model_params, dict):
        if all(isinstance(v, dict) for v in model_params.values()):
            strategy_model_desc = f"{strategy} (portfolio optimized params)"
        elif model_params:
            strategy_model_desc = f"{strategy} {model_params}"
        else:
            strategy_model_desc = strategy
    else:
        strategy_model_desc = strategy
    if payload.get("walk_forward"):
        strategy_model_desc = f"{strategy_model_desc} [walk-forward]"

    st.subheader("Benchmark Comparison")
    benchmark = benchmark_raw
    benchmark_symbol = str(benchmark.attrs.get("symbol", "")).strip() if hasattr(benchmark, "attrs") else ""
    benchmark_source = str(benchmark.attrs.get("source", "")).strip() if hasattr(benchmark, "attrs") else ""
    buy_hold_label = (
        "Buy-and-Hold Equity (Equal-Weight Portfolio)"
        if is_portfolio
        else f"Buy-and-Hold Equity ({selected_symbols[0]})"
    )
    if benchmark_choice != "off" and benchmark_symbol:
        st.caption(f"Benchmark 來源：{benchmark_symbol}（{benchmark_source or 'unknown'}）")
    if benchmark_split_events:
        split_text = ", ".join([f"{pd.Timestamp(ev.date).date()} x{ev.ratio:.6f} ({ev.source})" for ev in benchmark_split_events])
        st.caption(f"Benchmark split adjustment: {split_text}")
    if benchmark.empty:
        if benchmark_choice == "off":
            st.caption("你已關閉 Benchmark。")
        else:
            st.caption("目前無法取得 Benchmark（可能是來源限流）；可改選 SPY/QQQ/DIA（美股）或 0050/006208（台股）。")
    else:
        comp = pd.concat(
            [
                result.equity_curve["equity"].rename("strategy"),
                benchmark["close"].rename("benchmark"),
            ],
            axis=1,
        ).dropna()
        if not buy_hold_equity.empty:
            buy_hold_aligned = buy_hold_equity.reindex(comp.index).ffill()
            if not buy_hold_aligned.empty and buy_hold_aligned.notna().any():
                comp = comp.join(buy_hold_aligned.rename("buy_hold"), how="left")
        extra_symbol_lines_key = f"benchmark_symbol_lines:{run_key}"
        if is_portfolio:
            current_extra = st.session_state.get(extra_symbol_lines_key, list(selected_symbols))
            if not isinstance(current_extra, list):
                current_extra = list(selected_symbols)
            current_extra = [s for s in current_extra if s in selected_symbols]
            if not current_extra:
                current_extra = list(selected_symbols)
            st.session_state[extra_symbol_lines_key] = current_extra
        extra_symbol_lines = (
            st.multiselect(
                "Additional Symbol Lines",
                options=selected_symbols,
                key=extra_symbol_lines_key,
                help="在 Benchmark 圖上額外顯示各標的的買進持有線。",
            )
            if is_portfolio
            else []
        )
        if extra_symbol_lines:
            for sym in extra_symbol_lines:
                bars_sym = bars_by_symbol.get(sym, pd.DataFrame())
                if bars_sym.empty or "close" not in bars_sym.columns:
                    continue
                price_series = pd.to_numeric(bars_sym["close"], errors="coerce").reindex(comp.index).ffill()
                if price_series.notna().any():
                    comp[f"asset:{sym}"] = price_series
        if comp.empty:
            st.caption("Benchmark 與策略回測區間沒有重疊交易日，請調整回測日期或投入起點。")
        else:
            base_key = f"benchmark_base_date:{run_key}"
            base_min = comp.index[0].date()
            base_max = comp.index[-1].date()
            if base_key not in st.session_state:
                st.session_state[base_key] = base_min
            if st.session_state[base_key] < base_min:
                st.session_state[base_key] = base_min
            if st.session_state[base_key] > base_max:
                st.session_state[base_key] = base_max
            bc1, bc2 = st.columns([1, 3])
            base_date = bc1.date_input(
                "Benchmark Base Date",
                min_value=base_min,
                max_value=base_max,
                key=base_key,
                help="Strategy 與 Benchmark 會從這個日期（若非交易日則取下一個交易日）開始重設基準。",
            )
            base_dt = datetime.combine(base_date, datetime.min.time())
            if getattr(comp.index, "tz", None) is None:
                base_ts = pd.Timestamp(base_dt)
            else:
                base_ts = pd.Timestamp(base_dt, tz=comp.index.tz)
            base_candidates = comp.index[comp.index >= base_ts]
            if len(base_candidates) == 0:
                st.caption("Benchmark Base Date 之後沒有重疊交易日。")
                comp_view = pd.DataFrame(columns=comp.columns)
            else:
                base_used = pd.Timestamp(base_candidates[0])
                bc2.caption(
                    f"Base Date used: {base_used.strftime('%Y-%m-%d')} | "
                    f"Strategy Model: {strategy_model_desc} | "
                    f"Benchmark Instrument: {benchmark_symbol or 'N/A'}"
                )
                comp_view = comp.loc[comp.index >= base_used]

            if comp_view.empty or len(comp_view) < 2:
                st.caption("基準日之後資料不足，無法比較。")
                norm = pd.DataFrame(columns=comp.columns)
            else:
                norm = pd.DataFrame(index=comp_view.index)
                for col in comp_view.columns:
                    series = pd.to_numeric(comp_view[col], errors="coerce")
                    valid = series.dropna()
                    if valid.empty:
                        continue
                    base_val = float(valid.iloc[0])
                    if base_val == 0:
                        continue
                    norm[col] = series / base_val
                norm = norm.dropna(how="all")

            if not norm.empty:
                palette = _ui_palette()
                fig_cmp = go.Figure()
                fig_cmp.add_trace(
                    go.Scatter(
                        x=norm.index,
                        y=norm["strategy"],
                        mode="lines",
                        name="Strategy Equity",
                        line=dict(color=str(palette["equity"]), width=2.2),
                    )
                )
                fig_cmp.add_trace(
                    go.Scatter(
                        x=norm.index,
                        y=norm["benchmark"],
                        mode="lines",
                        name="Benchmark Equity",
                        line=dict(color=str(palette["benchmark"]), width=2.0),
                    )
                )
                if "buy_hold" in norm.columns and norm["buy_hold"].notna().any():
                    fig_cmp.add_trace(
                        go.Scatter(
                            x=norm.index,
                            y=norm["buy_hold"],
                            mode="lines",
                            name=buy_hold_label,
                            line=dict(color=str(palette["buy_hold"]), width=1.9),
                        )
                    )
                asset_palette = list(palette["asset_palette"])
                asset_color_idx = 0
                for col in [c for c in norm.columns if c.startswith("asset:")]:
                    sym = col.split(":", 1)[1]
                    line_color = asset_palette[asset_color_idx % len(asset_palette)]
                    asset_color_idx += 1
                    fig_cmp.add_trace(
                        go.Scatter(
                            x=norm.index,
                            y=norm[col],
                            mode="lines",
                            name=f"Buy-and-Hold Equity ({sym})",
                            line=dict(color=line_color, width=1.6),
                        )
                    )
                fig_cmp.update_xaxes(gridcolor=str(palette["grid"]))
                fig_cmp.update_yaxes(gridcolor=str(palette["grid"]))
                fig_cmp.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=25, b=10),
                    template=str(palette["plot_template"]),
                    paper_bgcolor=str(palette["paper_bg"]),
                    plot_bgcolor=str(palette["plot_bg"]),
                    font=dict(color=str(palette["text_color"])),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                strategy_perf = _series_metrics(comp_view["strategy"])
                bench_perf = _series_metrics(comp_view["benchmark"])
                cmp_rows = pd.DataFrame(
                    [
                        {
                            "Series": "Strategy Equity",
                            "Definition": "Backtest equity after fees/tax/slippage",
                            "Total Return %": round(strategy_perf["total_return"] * 100.0, 2),
                            "CAGR %": round(strategy_perf["cagr"] * 100.0, 2),
                            "MDD %": round(strategy_perf["max_drawdown"] * 100.0, 2),
                            "Sharpe": round(strategy_perf["sharpe"], 2),
                        },
                        {
                            "Series": "Benchmark Equity",
                            "Definition": f"Buy-and-hold reference ({benchmark_symbol or 'benchmark'})",
                            "Total Return %": round(bench_perf["total_return"] * 100.0, 2),
                            "CAGR %": round(bench_perf["cagr"] * 100.0, 2),
                            "MDD %": round(bench_perf["max_drawdown"] * 100.0, 2),
                            "Sharpe": round(bench_perf["sharpe"], 2),
                        },
                    ]
                )
                if "buy_hold" in comp_view.columns and comp_view["buy_hold"].notna().any():
                    hold_perf = _series_metrics(comp_view["buy_hold"])
                    cmp_rows.loc[len(cmp_rows)] = {
                        "Series": buy_hold_label,
                        "Definition": "Buy-and-hold on backtest symbols",
                        "Total Return %": round(hold_perf["total_return"] * 100.0, 2),
                        "CAGR %": round(hold_perf["cagr"] * 100.0, 2),
                        "MDD %": round(hold_perf["max_drawdown"] * 100.0, 2),
                        "Sharpe": round(hold_perf["sharpe"], 2),
                    }
                for col in [c for c in comp_view.columns if c.startswith("asset:")]:
                    sym = col.split(":", 1)[1]
                    sym_series = pd.to_numeric(comp_view[col], errors="coerce").dropna()
                    if len(sym_series) < 2:
                        continue
                    sym_perf = _series_metrics(sym_series)
                    cmp_rows.loc[len(cmp_rows)] = {
                        "Series": f"Buy-and-Hold Equity ({sym})",
                        "Definition": f"Buy-and-hold on symbol {sym}",
                        "Total Return %": round(sym_perf["total_return"] * 100.0, 2),
                        "CAGR %": round(sym_perf["cagr"] * 100.0, 2),
                        "MDD %": round(sym_perf["max_drawdown"] * 100.0, 2),
                        "Sharpe": round(sym_perf["sharpe"], 2),
                    }
                st.dataframe(cmp_rows, use_container_width=True, hide_index=True)

    st.subheader("策略 vs 買進持有")
    hold_label = "買進持有（等權投組）" if is_portfolio else f"買進持有（{selected_symbols[0]}）"
    st.caption("買進持有以回測標的收盤價計算；若為投組則採等權配置。")

    overall_rows = [
        {
            "比較項目": "策略",
            "區間起日": strategy_equity.index[0].strftime("%Y-%m-%d"),
            "區間迄日": strategy_equity.index[-1].strftime("%Y-%m-%d"),
            "報酬率%": round((strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1.0) * 100.0, 2),
        }
    ]
    if not buy_hold_equity.empty:
        overall_rows.append(
            {
                "比較項目": hold_label,
                "區間起日": buy_hold_equity.index[0].strftime("%Y-%m-%d"),
                "區間迄日": buy_hold_equity.index[-1].strftime("%Y-%m-%d"),
                "報酬率%": round((buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0] - 1.0) * 100.0, 2),
            }
        )
    st.dataframe(pd.DataFrame(overall_rows), use_container_width=True, hide_index=True)

    interval_start_key = f"interval_start:{run_key}"
    interval_end_key = f"interval_end:{run_key}"
    interval_min = strategy_equity.index[0].date()
    interval_max = strategy_equity.index[-1].date()
    if interval_start_key not in st.session_state:
        st.session_state[interval_start_key] = interval_min
    if interval_end_key not in st.session_state:
        st.session_state[interval_end_key] = interval_max
    if st.session_state[interval_start_key] < interval_min:
        st.session_state[interval_start_key] = interval_min
    if st.session_state[interval_start_key] > interval_max:
        st.session_state[interval_start_key] = interval_max
    if st.session_state[interval_end_key] < interval_min:
        st.session_state[interval_end_key] = interval_min
    if st.session_state[interval_end_key] > interval_max:
        st.session_state[interval_end_key] = interval_max

    r1, r2 = st.columns(2)
    interval_start = r1.date_input(
        "指定區間起日",
        min_value=interval_min,
        max_value=interval_max,
        key=interval_start_key,
        help="例如輸入 2026-01-01。",
    )
    interval_end = r2.date_input(
        "指定區間迄日",
        min_value=interval_min,
        max_value=interval_max,
        key=interval_end_key,
        help="例如輸入 2026-02-11。",
    )

    if interval_end < interval_start:
        st.warning("指定區間迄日早於起日，請調整。")
    else:
        strategy_interval = interval_return(strategy_equity, start_date=interval_start, end_date=interval_end)
        hold_interval = interval_return(buy_hold_equity, start_date=interval_start, end_date=interval_end)
        interval_rows = []
        if strategy_interval.get("ok"):
            interval_rows.append(
                {
                    "比較項目": "策略",
                    "起始交易日(實際)": strategy_interval["start_used"].strftime("%Y-%m-%d"),
                    "結束交易日(實際)": strategy_interval["end_used"].strftime("%Y-%m-%d"),
                    "報酬率%": round(float(strategy_interval["return"]) * 100.0, 2),
                }
            )
        if hold_interval.get("ok"):
            interval_rows.append(
                {
                    "比較項目": hold_label,
                    "起始交易日(實際)": hold_interval["start_used"].strftime("%Y-%m-%d"),
                    "結束交易日(實際)": hold_interval["end_used"].strftime("%Y-%m-%d"),
                    "報酬率%": round(float(hold_interval["return"]) * 100.0, 2),
                }
            )
        if interval_rows:
            st.dataframe(pd.DataFrame(interval_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("指定區間內無可計算資料。")

    st.subheader("逐年報酬")
    if result.yearly_returns:
        yr = pd.DataFrame([{"年度": y, "報酬率%": round(v * 100.0, 2)} for y, v in result.yearly_returns.items()])
        st.dataframe(yr.sort_values("年度"), use_container_width=True, hide_index=True)
    else:
        st.caption("樣本不足，無逐年報酬。")

    if is_portfolio:
        st.subheader("投組分項績效")
        rows = []
        for symbol, comp in result.component_results.items():
            rows.append(
                {
                    "symbol": symbol,
                    "total_return%": round(comp.metrics.total_return * 100.0, 2),
                    "cagr%": round(comp.metrics.cagr * 100.0, 2),
                    "mdd%": round(comp.metrics.max_drawdown * 100.0, 2),
                    "sharpe": round(comp.metrics.sharpe, 2),
                    "trades": comp.metrics.trades,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("交易明細")
    if is_portfolio:
        if result.trades is not None and not result.trades.empty:
            trades_df = result.trades.copy()
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.date.astype(str)
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.date.astype(str)
            trades_df["pnl_pct%"] = (trades_df["pnl_pct"] * 100.0).round(2)
            st.dataframe(trades_df.drop(columns=["pnl_pct"]), use_container_width=True, hide_index=True)
        else:
            st.caption("沒有交易紀錄。")
    else:
        if result.trades:
            trades_df = pd.DataFrame(
                [
                    {
                        "entry_date": t.entry_date.date().isoformat(),
                        "entry_price": round(t.entry_price, 4),
                        "exit_date": t.exit_date.date().isoformat(),
                        "exit_price": round(t.exit_price, 4),
                        "qty": round(t.qty, 4),
                        "pnl": round(t.pnl, 2),
                        "pnl_pct%": round(t.pnl_pct * 100.0, 2),
                        "fee": round(t.fee, 2),
                        "tax": round(t.tax, 2),
                        "slippage": round(t.slippage, 4),
                    }
                    for t in result.trades
                ]
            )
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.caption("沒有交易紀錄。")


def _render_tutorial_view():
    st.subheader("新手教學（技術面 + 回測）")
    st.caption("目標：幫你釐清每個參數在做什麼，避免把「看圖控制」和「回測結果」混在一起。")

    st.markdown("### 0) 資料會存在哪裡？")
    storage_df = pd.DataFrame(
        [
            {"項目": "歷史日K（含 Benchmark）", "位置": "SQLite `market_history.sqlite3`", "用途": "同步一次後可重複用於回測與基準比較"},
            {"項目": "回測紀錄", "位置": "SQLite `backtest_runs` 表", "用途": "保存執行過的回測摘要"},
            {"項目": "即時資料", "位置": "Session 記憶體", "用途": "即時看盤暫存，重開 app 會重抓"},
        ]
    )
    st.dataframe(storage_df, use_container_width=True, hide_index=True)

    st.markdown("### 1) 回測流程先看這個")
    flow_df = pd.DataFrame(
        [
            {"步驟": "A. 選回測區間", "說明": "決定要抓哪些歷史資料（例如 2021-01-01 ~ 今天）。"},
            {"步驟": "B. 設定實際投入起點", "說明": "本金從哪一天（或第幾根K）開始投入，這會改變結果。"},
            {"步驟": "C. 選策略與參數", "說明": "buy_hold / 均線 / RSI / MACD。"},
            {"步驟": "D. 設成本", "說明": "手續費、稅、滑價（都會影響績效）。"},
            {"步驟": "E. 執行回測", "說明": "產出績效、交易明細、買進持有比較。"},
            {"步驟": "F. 回放看圖", "說明": "只是在播放結果，不會改變回測數字。"},
        ]
    )
    st.dataframe(flow_df, use_container_width=True, hide_index=True)

    st.markdown("### 2) 安全門檻（你剛剛遇到的）")
    threshold_df = pd.DataFrame(
        [
            {"模式": "buy_hold", "最少K數": "2", "原因": "至少要有起點與終點，才能算報酬率。"},
            {"模式": "一般策略（SMA/EMA/RSI/MACD）", "最少K數": "40", "原因": "需要足夠樣本讓指標與訊號有意義。"},
            {"模式": "Walk-Forward", "最少K數": "80", "原因": "要切 Train/Test，太短會失真。"},
        ]
    )
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)
    st.info("如果你只想看『最近才投入』，建議先選 `buy_hold`，最不容易被資料量門檻擋住。")

    st.markdown("### 3) 訊號點 vs 實際成交點（最常搞混）")
    point_df = pd.DataFrame(
        [
            {
                "類型": "訊號點（價格圖）",
                "代表意思": "策略判斷該進場/出場的時刻",
                "是否等於成交價": "不一定",
                "適合用途": "看策略邏輯何時翻多/翻空",
            },
            {
                "類型": "實際成交點（資產圖）",
                "代表意思": "回測規則真正成交的位置",
                "是否等於成交價": "是（依回測規則）",
                "適合用途": "檢查績效計算是否合理",
            },
        ]
    )
    st.dataframe(point_df, use_container_width=True, hide_index=True)
    st.caption("本系統預設：`T 日收盤產生訊號，T+1 開盤成交`。所以訊號點與成交點通常不在同一根K。")

    st.markdown("### 4) 參數白話解釋")
    params_df = pd.DataFrame(
        [
            {"參數": "Fast", "白話意思": "短天期均線", "單位/格式": "天數（整數）", "新手建議": "先用 10~20"},
            {"參數": "Slow", "白話意思": "長天期均線", "單位/格式": "天數（整數）", "新手建議": "先用 30~120，且大於 Fast"},
            {"參數": "RSI Buy Below", "白話意思": "RSI 低於此值才考慮買入", "單位/格式": "0~100 指標值", "新手建議": "先用 30"},
            {"參數": "RSI Sell Above", "白話意思": "RSI 高於此值才考慮賣出", "單位/格式": "0~100 指標值", "新手建議": "先用 55~65"},
            {"參數": "Fee Rate", "白話意思": "券商手續費比例", "單位/格式": "小數比例", "新手建議": "台股常見 0.001425"},
            {"參數": "Sell Tax", "白話意思": "賣出交易稅比例", "單位/格式": "小數比例", "新手建議": "台股股票常見 0.003、ETF 常見 0.001"},
            {"參數": "Slippage", "白話意思": "理論價與實際成交價偏差", "單位/格式": "小數比例", "新手建議": "先用 0.0005（0.05%）"},
            {"參數": "Train 比例", "白話意思": "Walk-Forward 的訓練區占比", "單位/格式": "0~1", "新手建議": "先用 0.70"},
            {"參數": "參數挑選目標", "白話意思": "Train 區用哪個績效挑最佳參數", "單位/格式": "sharpe/cagr/total_return/mdd", "新手建議": "先用 sharpe"},
        ]
    )
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("### 5) 三個容易誤會的控制")
    confusion_df = pd.DataFrame(
        [
            {"控制項": "起始日期 / 結束日期", "會不會改回測結果": "會", "重點": "決定樣本資料範圍"},
            {"控制項": "實際投入起點（日期或第幾根K）", "會不會改回測結果": "會", "重點": "決定本金何時開始進場"},
            {"控制項": "回放位置（K棒/日期）", "會不會改回測結果": "不會", "重點": "只控制圖表播放到哪裡"},
            {"控制項": "回放視窗 / 生長位置（靠右%）", "會不會改回測結果": "不會", "重點": "只控制線圖長出來的位置與視覺效果"},
            {"控制項": "顯示成交連線", "會不會改回測結果": "不會", "重點": "只幫你對照價格圖和資產圖的同一成交時刻"},
        ]
    )
    st.dataframe(confusion_df, use_container_width=True, hide_index=True)

    st.markdown("### 6) 新手建議操作（先簡單再進階）")
    st.markdown(
        "\n".join(
            [
                "1. 先用 `buy_hold` 熟悉投入起點與區間報酬。",
                "2. 確認成本參數（手續費/稅/滑價）後，再換 `sma_cross`。",
                "3. 最後再開 Walk-Forward 比較參數穩健性。",
            ]
        )
    )


def main():
    st.set_page_config(page_title="即時看盤 + 回測平台", layout="wide")
    _inject_ui_styles()
    st.title("即時走勢 / 多來源資料 / 回測平台")
    live_tab, bt_tab, guide_tab = st.tabs(["即時看盤", "回測工作台", "新手教學"])
    with live_tab:
        _render_live_view()
    with bt_tab:
        _render_backtest_view()
    with guide_tab:
        _render_tutorial_view()


if __name__ == "__main__":
    main()
