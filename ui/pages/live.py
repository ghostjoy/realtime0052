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


def _render_live_view(*, ctx: object):
    _bind_ctx(ctx)
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
        keep_minutes = st.slider("保留即時資料（分鐘）", min_value=30, max_value=360, value=180, step=30)
        use_yahoo = st.checkbox("允許 Yahoo 補K", value=True)
        use_fugle_ws = False
        if market == "台股(TWSE)":
            use_fugle_ws = st.checkbox(
                "啟用 Fugle WebSocket（需 API Key）",
                value=True,
                help="有設定 FUGLE_MARKETDATA_API_KEY，或已放置 key 檔（iCloud 預設路徑）時，台股即時會優先走 Fugle；失敗會自動回退。",
            )

        st.subheader("建議偏好")
        advice_mode = st.selectbox("建議模式", options=["一般(技術面)", "股癌風格(心法/檢核)"], index=1)
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
            ticks = st.session_state.get(tick_key, pd.DataFrame(columns=["ts", "price", "cum_volume"]))
            try:
                ctx, ticks = service.get_tw_live_context(stock_id, yahoo_symbol, ticks=ticks, options=options)
                st.session_state[tick_key] = ticks
            except Exception as exc:
                st.error(f"台股資料取得失敗：{exc}")
                return
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
                st.warning("目前為美股模式，請輸入美股代碼（例如 AAPL、TSLA）；若要查台股請切換到台股模式。")
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
                    display_name = str(service.get_tw_symbol_names([stock_id]).get(stock_id, "") or "").strip()
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
                        st.caption(f"即時走勢：K數偏少，已用本地快取補齊（{before} -> {len(bars_intraday)}）。")
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
            ev_txt = ", ".join([f"{pd.Timestamp(ev.date).date()} x{ev.ratio:.6f}" for ev in intraday_split_events])
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
                live_watermark_symbol = stock_id if market == "台股(TWSE)" else (us_symbol or quote.symbol or "")
                live_watermark_market = "TW" if market == "台股(TWSE)" else "US"
                live_watermark_text = _symbol_watermark_text(
                    symbol=live_watermark_symbol,
                    market=live_watermark_market,
                    service=service,
                )
                _render_live_chart(ind, watermark_text=live_watermark_text)
                live_symbol_key = (stock_id if market == "台股(TWSE)" else (us_symbol or quote.symbol or "US")).strip().upper()
                live_indicator_toggle_key = f"live_indicator_panel:{market}:{live_symbol_key}"
                if live_indicator_toggle_key not in st.session_state:
                    st.session_state[live_indicator_toggle_key] = True
                st.checkbox(
                    "顯示技術指標副圖（RSI / MACD / 布林 / KD）",
                    key=live_indicator_toggle_key,
                )
                live_indicator_compact_key = f"live_indicator_compact:{market}:{live_symbol_key}"
                if live_indicator_compact_key not in st.session_state:
                    st.session_state[live_indicator_compact_key] = True
                st.checkbox(
                    "緊湊指標副圖（筆電建議）",
                    key=live_indicator_compact_key,
                )
                if st.session_state.get(live_indicator_toggle_key):
                    _render_indicator_panels(
                        ind,
                        chart_key=f"live_indicator_chart:{market}:{live_symbol_key}",
                        height=340 if st.session_state.get(live_indicator_compact_key) else 430,
                        watermark_text=live_watermark_text,
                    )
                if not daily_long.empty:
                    st.markdown("#### 長期視角（Daily）")
                    if daily_split_events:
                        ev_txt = ", ".join([f"{pd.Timestamp(ev.date).date()} x{ev.ratio:.6f}" for ev in daily_split_events])
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
