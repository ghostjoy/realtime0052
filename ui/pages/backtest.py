from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest import (
    CostModel,
    apply_split_adjustment,
    apply_start_to_bars_map,
    build_buy_hold_equity,
    build_dca_benchmark_equity,
    build_dca_contribution_plan,
    build_dca_equity,
    dca_summary_metrics,
    get_strategy_min_bars,
    interval_return,
    required_walkforward_bars,
)
from indicators import add_indicators
from services.backtest_cache import (
    build_backtest_run_key,
    build_backtest_run_params_base,
    build_replay_params_with_signature,
)
from services.backtest_runner import (
    BacktestExecutionInput,
    execute_backtest_run,
    load_and_prepare_symbol_bars,
)
from services.backtest_runner import (
    default_cost_params as runner_default_cost_params,
)
from services.backtest_runner import (
    load_benchmark_from_store as runner_load_benchmark_from_store,
)
from services.backtest_runner import (
    parse_symbols as runner_parse_symbols,
)
from services.backtest_runner import (
    queue_benchmark_writeback as runner_queue_benchmark_writeback,
)
from services.backtest_runner import (
    series_metrics as runner_series_metrics,
)
from services.sync_orchestrator import sync_symbols_if_needed
from state_keys import BT_KEYS
from ui.charts import render_lightweight_kline_equity_chart
from ui.core.charts import _render_benchmark_lines_chart, _render_indicator_panels
from ui.shared.perf import PerfTimer, perf_debug_enabled
from ui.shared.runtime import configure_module_runtime
from ui.shared.session_utils import ensure_defaults

REQUIRED_RUNTIME_NAMES = (
    "BACKTEST_AUTORUN_PENDING_KEY",
    "BACKTEST_REPLAY_SCHEMA_VERSION",
    "BACKTEST_RUN_REQUEST_KEY",
    "DAILY_STRATEGY_OPTIONS",
    "STRATEGY_DESC",
    "_apply_plotly_watermark",
    "_apply_total_return_adjustment",
    "_apply_unified_benchmark_hover",
    "_benchmark_line_style",
    "_build_data_health",
    "_build_sync_rows",
    "_collect_tw_symbol_codes",
    "_decorate_tw_symbol_columns",
    "_enable_plotly_draw_tools",
    "_format_price",
    "_format_tw_symbol_with_name",
    "_history_store",
    "_infer_market_target_from_symbols",
    "_load_cached_backtest_payload",
    "_market_service",
    "_metrics_to_rows",
    "_render_card_section_header",
    "_render_crisp_table",
    "_render_data_health_caption",
    "_render_plotly_chart",
    "_replay_kline_renderer",
    "_safe_float",
    "_serialize_backtest_run_payload",
    "_strategy_label",
    "_to_rgba",
    "_ui_palette",
)

BACKTEST_AUTORUN_PENDING_KEY: Any = None
BACKTEST_REPLAY_SCHEMA_VERSION: Any = None
BACKTEST_RUN_REQUEST_KEY: Any = None
DAILY_STRATEGY_OPTIONS: Any = None
STRATEGY_DESC: Any = None
_apply_plotly_watermark: Any = None
_apply_total_return_adjustment: Any = None
_apply_unified_benchmark_hover: Any = None
_benchmark_line_style: Any = None
_build_data_health: Any = None
_build_sync_rows: Any = None
_collect_tw_symbol_codes: Any = None
_decorate_tw_symbol_columns: Any = None
_enable_plotly_draw_tools: Any = None
_format_price: Any = None
_format_tw_symbol_with_name: Any = None
_history_store: Any = None
_infer_market_target_from_symbols: Any = None
_load_cached_backtest_payload: Any = None
_market_service: Any = None
_metrics_to_rows: Any = None
_render_card_section_header: Any = None
_render_crisp_table: Any = None
_render_data_health_caption: Any = None
_render_plotly_chart: Any = None
_replay_kline_renderer: Any = None
_safe_float: Any = None
_serialize_backtest_run_payload: Any = None
_strategy_label: Any = None
_to_rgba: Any = None
_ui_palette: Any = None


def configure_runtime(values: Mapping[str, Any]) -> None:
    configure_module_runtime(globals(), REQUIRED_RUNTIME_NAMES, values, module_name=__name__)


def _render_backtest_view():
    perf_timer = PerfTimer(enabled=perf_debug_enabled())
    _parse_symbols = runner_parse_symbols
    _default_cost_params = runner_default_cost_params
    store = _history_store()
    service = _market_service()

    def _load_benchmark_from_store(
        market_code: str,
        start: datetime,
        end: datetime,
        choice: str,
    ) -> pd.DataFrame:
        return runner_load_benchmark_from_store(
            store=store,
            market_code=market_code,
            start=start,
            end=end,
            choice=choice,
        )

    market_auto_note_key = "bt_market_auto_inferred"

    symbol_prefill_text = str(st.session_state.get(BT_KEYS.symbol, "")).strip().upper()
    prefill_symbols = _parse_symbols(symbol_prefill_text)
    prefill_target = _infer_market_target_from_symbols(prefill_symbols)
    if (
        prefill_target in {"TW", "OTC", "US"}
        and st.session_state.get(BT_KEYS.market) != prefill_target
    ):
        st.session_state[BT_KEYS.market] = prefill_target
    if prefill_target in {"TW", "OTC", "US"}:
        st.session_state[market_auto_note_key] = prefill_target
    else:
        st.session_state.pop(market_auto_note_key, None)
    if len(prefill_symbols) > 1 and st.session_state.get(BT_KEYS.mode) != "投組(多標的)":
        st.session_state[BT_KEYS.mode] = "投組(多標的)"
    elif len(prefill_symbols) == 1 and st.session_state.get(BT_KEYS.mode) != "單一標的":
        st.session_state[BT_KEYS.mode] = "單一標的"

    def _on_bt_symbol_change():
        current_text = str(st.session_state.get(BT_KEYS.symbol, "")).strip().upper()
        current_symbols = _parse_symbols(current_text)
        if len(current_symbols) > 1 and st.session_state.get(BT_KEYS.mode) != "投組(多標的)":
            st.session_state[BT_KEYS.mode] = "投組(多標的)"
        elif len(current_symbols) == 1 and st.session_state.get(BT_KEYS.mode) != "單一標的":
            st.session_state[BT_KEYS.mode] = "單一標的"
        inferred = _infer_market_target_from_symbols(current_symbols)
        if inferred in {"TW", "OTC", "US"}:
            st.session_state[market_auto_note_key] = inferred
            if st.session_state.get(BT_KEYS.market) != inferred:
                st.session_state[BT_KEYS.market] = inferred
        else:
            st.session_state.pop(market_auto_note_key, None)

    _render_card_section_header("回測設定卡", "先設定基本條件，再調整策略/成本與進階回放選項。")
    st.markdown("#### 基本設定")
    c1, c2, c3, c4 = st.columns(4)
    market_options = ["TW", "OTC", "US"]
    market_labels = {"TW": "台股上市(TWSE)", "OTC": "台股上櫃(OTC)", "US": "美股(US)"}
    market_selector = c1.selectbox(
        "市場",
        market_options,
        index=0,
        format_func=lambda v: market_labels.get(str(v), str(v)),
        key=BT_KEYS.market,
    )
    is_tw_market = market_selector in {"TW", "OTC"}
    market_code = "TW" if is_tw_market else "US"
    tw_symbol_label_enabled = market_code == "TW"
    mode = c2.selectbox("模式", ["單一標的", "投組(多標的)"], index=0, key=BT_KEYS.mode)
    default_symbol = (
        "0052" if market_selector == "TW" else ("8069" if market_selector == "OTC" else "TSLA")
    )
    default_symbol_multi = (
        "0052,2330"
        if market_selector == "TW"
        else ("8069,4123" if market_selector == "OTC" else "AAPL,MSFT,TSLA")
    )
    symbol_input_value = symbol_prefill_text or (
        default_symbol if mode == "單一標的" else default_symbol_multi
    )
    symbol_text = (
        c3.text_input(
            "代碼（投組用逗號分隔）",
            value=symbol_input_value,
            key=BT_KEYS.symbol,
            on_change=_on_bt_symbol_change,
        )
        .strip()
        .upper()
    )
    strategy = c4.selectbox(
        "策略",
        options=DAILY_STRATEGY_OPTIONS,
        index=0,
        format_func=_strategy_label,
        key=BT_KEYS.strategy,
    )
    symbols = _parse_symbols(symbol_text)
    if len(symbols) > 1 and mode != "投組(多標的)":
        mode = "投組(多標的)"
    elif len(symbols) == 1 and mode != "單一標的":
        mode = "單一標的"
    if mode == "單一標的":
        symbols = symbols[:1]
    if not symbols:
        st.warning("請輸入至少一個代碼。")
        return
    if len(symbols) > 1:
        st.caption("已偵測多組標的，模式自動切換為投組(多標的)。")
    inferred_target = _infer_market_target_from_symbols(symbols)
    if inferred_target in {"TW", "OTC", "US"}:
        st.session_state[market_auto_note_key] = inferred_target
    else:
        st.session_state.pop(market_auto_note_key, None)
    inferred_note = str(st.session_state.get(market_auto_note_key, "")).strip().upper()
    if inferred_note == "OTC":
        st.caption("代碼已自動判斷為台股上櫃（OTC）。")
    elif inferred_note == "TW":
        st.caption("代碼已自動判斷為台股上市（TWSE）。")
    elif inferred_note == "US":
        st.caption("代碼已自動判斷為美股（US）。")

    auto_cost_key = BT_KEYS.auto_cost
    fee_key = BT_KEYS.fee_rate
    tax_key = BT_KEYS.sell_tax
    slip_key = BT_KEYS.slippage
    cost_profile_key = BT_KEYS.cost_profile
    if auto_cost_key not in st.session_state:
        st.session_state[auto_cost_key] = True
    current_cost_profile = f"{market_selector}:{','.join(symbols)}"
    if (
        st.session_state.get(auto_cost_key)
        and st.session_state.get(cost_profile_key) != current_cost_profile
    ):
        fee_default, tax_default, slip_default = _default_cost_params(market_code, symbols)
        st.session_state[fee_key] = float(fee_default)
        st.session_state[tax_key] = float(tax_default)
        st.session_state[slip_key] = float(slip_default)
        st.session_state[cost_profile_key] = current_cost_profile

    d1, d2 = st.columns(2)
    start_date = d1.date_input(
        "起始日期", value=date(date.today().year - 5, 1, 1), key=BT_KEYS.start_date
    )
    end_date = d2.date_input("結束日期", value=date.today(), key=BT_KEYS.end_date)
    benchmark_options = (
        [
            ("auto", "Auto（台股加權 ^TWII，失敗時改用 0050/006208）"),
            ("twii", "^TWII"),
            ("0050", "0050（ETF代理）"),
            ("006208", "006208（ETF代理）"),
            ("off", "關閉 Benchmark"),
        ]
        if is_tw_market
        else [
            ("auto", "Auto（S&P500 ^GSPC，失敗時改用 SPY/QQQ/DIA）"),
            ("gspc", "^GSPC"),
            ("spy", "SPY（ETF代理）"),
            ("qqq", "QQQ（ETF代理）"),
            ("dia", "DIA（ETF代理）"),
            ("off", "關閉 Benchmark"),
        ]
    )
    bench_codes = [x[0] for x in benchmark_options]
    bench_labels = {x[0]: x[1] for x in benchmark_options}
    benchmark_choice = d1.selectbox(
        "Benchmark",
        options=bench_codes,
        format_func=lambda v: bench_labels.get(v, v),
        index=0,
        key=BT_KEYS.benchmark_choice,
        help="可手動選基準；Auto 會在主來源失敗時自動改用替代基準。",
    )
    invest_mode_key = BT_KEYS.invest_start_mode
    invest_date_key = BT_KEYS.invest_start_date
    invest_k_key = BT_KEYS.invest_start_k

    st.markdown("#### 策略參數")
    st.caption(f"目前策略：{_strategy_label(strategy)} | {STRATEGY_DESC.get(strategy, '')}")
    param1, param2, param3 = st.columns(3)
    if strategy in {"sma_cross", "ema_cross"}:
        fast = param1.slider(
            "Fast",
            min_value=3,
            max_value=80,
            value=20 if strategy == "sma_cross" else 12,
            help="短週期均線。數字越小越敏感，訊號會更快但雜訊也可能變多。",
        )
        slow = param2.slider(
            "Slow",
            min_value=10,
            max_value=240,
            value=60 if strategy == "sma_cross" else 26,
            help="長週期均線。通常要大於 Fast，用來過濾趨勢方向。",
        )
        strategy_params = {"fast": float(fast), "slow": float(slow)}
    elif strategy == "sma_trend_filter":
        fast = param1.slider(
            "Fast",
            min_value=5,
            max_value=80,
            value=20,
            help="短期均線（反應速度）。",
        )
        slow = param2.slider(
            "Slow",
            min_value=20,
            max_value=240,
            value=60,
            help="中期均線（趨勢確認）。",
        )
        trend = param3.slider(
            "Trend Filter",
            min_value=60,
            max_value=300,
            value=120,
            help="長期濾網均線；收盤要在其上方才允許持有。",
        )
        strategy_params = {"fast": float(fast), "slow": float(slow), "trend": float(trend)}
    elif strategy == "donchian_breakout":
        entry_n = param1.slider(
            "Breakout Lookback",
            min_value=20,
            max_value=120,
            value=55,
            help="突破判定視窗（N日新高）。",
        )
        exit_max = max(10, int(entry_n) - 1)
        exit_n = param2.slider(
            "Exit Lookback",
            min_value=5,
            max_value=exit_max,
            value=min(20, exit_max),
            help="出場判定視窗（M日新低）；通常 M < N。",
        )
        trend = param3.slider(
            "Trend Filter",
            min_value=60,
            max_value=300,
            value=120,
            help="長期均線濾網，避免逆大方向追突破。",
        )
        strategy_params = {
            "entry_n": float(entry_n),
            "exit_n": float(exit_n),
            "trend": float(trend),
        }
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

    def _coerce_session_float(key: str, default: float):
        try:
            st.session_state[key] = float(st.session_state.get(key, default))
        except Exception:
            st.session_state[key] = float(default)

    _coerce_session_float(fee_key, 0.001425)
    _coerce_session_float(tax_key, 0.0 if market_code == "US" else 0.003)
    _coerce_session_float(slip_key, 0.0005)

    st.markdown("#### 交易成本")
    cost1, cost2, cost3 = st.columns(3)
    fee_rate = cost1.number_input(
        "Fee Rate",
        min_value=0.0,
        max_value=0.01,
        step=0.0001,
        format="%.6f",
        key=fee_key,
        help="每次買進/賣出收取的手續費比例，例如 0.001425 = 0.1425%。",
    )
    sell_tax = cost2.number_input(
        "Sell Tax",
        min_value=0.0,
        max_value=0.01,
        step=0.0001,
        format="%.6f",
        key=tax_key,
        help="賣出時交易稅比例。台股股票常見 0.3%，ETF 常見 0.1%。",
    )
    slippage = cost3.number_input(
        "Slippage",
        min_value=0.0,
        max_value=0.01,
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
        key=BT_KEYS.enable_wf,
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
        key=BT_KEYS.train_ratio,
        help="例如 0.70 代表前 70% 用於挑參數，後 30% 用於驗證。",
    )
    objective = wf3.selectbox(
        "參數挑選目標",
        ["sharpe", "cagr", "total_return", "mdd"],
        index=0,
        key=BT_KEYS.objective,
        help="Walk-Forward 會用這個指標在訓練區挑最佳參數。",
    )
    adj1, adj2, adj3 = st.columns(3)
    use_split_adjustment = adj1.checkbox(
        "分割調整（復權）", value=True, key=BT_KEYS.use_split_adjustment
    )
    auto_detect_split = adj2.checkbox("自動偵測分割事件", value=True, key=BT_KEYS.auto_detect_split)
    use_total_return_adjustment = adj3.checkbox(
        "還原權息計算（Adj Close）",
        value=False,
        key=BT_KEYS.use_total_return_adjustment,
        help="有 `adj_close` 時，會將 OHLC 轉成還原權息價格。若已還原，會略過額外分割調整避免重複計算。",
    )
    if use_total_return_adjustment and use_split_adjustment:
        st.caption(
            "已啟用還原權息：若標的 `adj_close` 覆蓋率足夠，將優先使用還原權息並略過分割調整。"
        )
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
                    "- Walk-Forward 所需最少K數會依策略自動提高（長週期策略通常需要更多資料）。",
                    "- `參數挑選目標`（Walk-Forward）：",
                    "  - `sharpe`：風險調整後報酬（越高越好）。",
                    "  - `cagr`：年化報酬率（越高越好）。",
                    "  - `total_return`：區間總報酬（越高越好）。",
                    "  - `mdd`：最大回撤（越小越好，系統會挑回撤較小的參數）。",
                    "- `Fast/Slow`：均線短週期/長週期參數（通常 `Fast < Slow`）。",
                    "- `Trend Filter`：長期趨勢濾網（如 120 日均線），可降低日K雜訊。",
                    "- `Breakout Lookback / Exit Lookback`：突破/出場視窗（通常 Exit < Breakout）。",
                    "- `RSI Buy Below / Sell Above`：RSI 的進出場門檻值。",
                    "- `buy_hold`：買進後持有到回測結束，適合快速檢查「近期投入」表現。",
                    "- `sma_trend_filter`：均線交叉 + 長期濾網，偏中期日K趨勢。",
                    "- `donchian_breakout`：日K通道突破追蹤，適合波段趨勢市場。",
                    "- `還原權息計算（Adj Close）`：若資料含 `adj_close`，會把價格改為含配息/分割影響的還原序列。",
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
                    "- `策略 vs 買進持有`：比較策略報酬率與買進持有（理論無摩擦，單檔或等權投組）的報酬率。",
                    "- `指定區間報酬率`：可輸入任意日期區間（如 2026-01-01 ~ 2026-02-11）比較策略與買進持有。",
                    "- `位置顯示`可切換：",
                    "  - `K棒`：用第幾根K（0 起算）定位。",
                    "  - `日期`：用交易日定位回放位置。",
                    "- `回放視窗`：",
                    "  - `固定視窗`：可用 `視窗K數` 與 `生長位置（靠右%）` 調整線圖長出來的位置。",
                    "  - `完整區間`：從第一根一路看到目前回放位置。",
                    "- `Benchmark Base Date`：Benchmark 與策略都會從此基準日（實際交易日）開始重設為 1.0 再比較。",
                    "- 回放圖預設顯示成交連線與技術指標副圖，方便同時檢視訊號、成交與動能。",
                ]
            )
        )

    run_key = build_backtest_run_key(
        market=market_selector,
        symbols=list(symbols),
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        enable_wf=bool(enable_wf),
        train_ratio=float(train_ratio),
        objective=objective,
        initial_capital=float(initial_capital),
        strategy_params=strategy_params if isinstance(strategy_params, dict) else {},
        fee_rate=float(fee_rate),
        sell_tax=float(sell_tax),
        slippage=float(slippage),
        use_split_adjustment=bool(use_split_adjustment),
        auto_detect_split=bool(auto_detect_split),
        use_total_return_adjustment=bool(use_total_return_adjustment),
        invest_start_mode=str(invest_start_mode),
        invest_start_date=invest_start_date,
        invest_start_k=int(invest_start_k) if invest_start_k is not None else -1,
    )
    auto_run_payload = st.session_state.get(BACKTEST_AUTORUN_PENDING_KEY)
    if isinstance(auto_run_payload, dict):
        req_symbol = str(auto_run_payload.get("symbol", "")).strip().upper()
        req_market = str(auto_run_payload.get("market", "")).strip().upper()
        if (
            len(symbols) == 1
            and req_symbol
            and symbols[0] == req_symbol
            and (not req_market or req_market == market_selector)
        ):
            st.session_state[BACKTEST_RUN_REQUEST_KEY] = run_key
            st.session_state.pop(BACKTEST_AUTORUN_PENDING_KEY, None)
            st.caption(f"已從表格帶入 `{req_symbol}`，自動執行回測。")
    with st.container(border=True):
        _render_card_section_header("回測執行", "按下按鈕後才會執行；同條件會優先讀取本地快取。")
        run_clicked = st.button("執行回測", type="primary", width="stretch", key="bt_execute_run")
    if run_clicked:
        st.session_state[BACKTEST_RUN_REQUEST_KEY] = run_key
    run_requested = st.session_state.get(BACKTEST_RUN_REQUEST_KEY) == run_key

    sync_parallel = st.checkbox(
        "多標的同步平行處理",
        value=True,
        key=BT_KEYS.sync_parallel,
        help="多檔標的時通常更快；若網路不穩可關閉改為逐檔同步。",
    )
    auto_sync = st.checkbox(
        "App 啟動時自動增量同步（較慢）",
        value=False,
        key=BT_KEYS.auto_sync,
        help="關閉可先直接使用本地資料庫；需要時再手動同步。",
    )
    auto_fill_gaps = st.checkbox(
        "回測前自動補資料缺口（推薦）",
        value=True,
        key=BT_KEYS.auto_fill_gaps,
        help="若發現本地資料起訖缺口，會只同步缺口標的；可減少「回測天數不夠」問題。",
    )
    sync_key = f"synced:{market_selector}:{','.join(symbols)}:{start_date}:{end_date}"
    gapfill_key = f"gapfill:{market_selector}:{','.join(symbols)}:{start_date}:{end_date}"
    sync_start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    sync_end = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    if run_requested and auto_sync and not st.session_state.get(sync_key):
        reports, plan = sync_symbols_if_needed(
            store=store,
            market=market_code,
            symbols=symbols,
            start=sync_start,
            end=sync_end,
            parallel=sync_parallel,
            mode="all",
        )
        sync_rows = _build_sync_rows(list(symbols), reports)
        st.session_state[sync_key] = True
        sync_df = pd.DataFrame(sync_rows)
        if plan.issues:
            st.warning("部分同步失敗，請檢查下方同步結果。")
        st.dataframe(
            _decorate_tw_symbol_columns(
                sync_df,
                service=service,
                enabled=tw_symbol_label_enabled,
                columns=["symbol"],
            ),
            width="stretch",
            hide_index=True,
        )

    if run_requested and auto_fill_gaps and not st.session_state.get(gapfill_key):
        reports, plan = sync_symbols_if_needed(
            store=store,
            market=market_code,
            symbols=symbols,
            start=sync_start,
            end=sync_end,
            parallel=sync_parallel,
            mode="backfill",
        )
        if plan.synced_symbols:
            sync_rows = _build_sync_rows(plan.synced_symbols, reports)
            if plan.issues:
                st.warning("缺口補齊時有部分同步失敗，請檢查下方結果。")
            else:
                st.caption("已自動補齊回測區間資料缺口。")
            st.dataframe(
                _decorate_tw_symbol_columns(
                    pd.DataFrame(sync_rows),
                    service=service,
                    enabled=tw_symbol_label_enabled,
                    columns=["symbol"],
                ),
                width="stretch",
                hide_index=True,
            )
        st.session_state[gapfill_key] = True

    if st.button("同步歷史資料", width="stretch"):
        reports, plan = sync_symbols_if_needed(
            store=store,
            market=market_code,
            symbols=symbols,
            start=sync_start,
            end=sync_end,
            parallel=sync_parallel,
            mode="all",
        )
        sync_rows = _build_sync_rows(list(symbols), reports)
        sync_df = pd.DataFrame(sync_rows)
        if plan.issues:
            st.error("部分同步失敗，請檢查同步結果。")
        else:
            st.success("同步成功。")
        st.dataframe(
            _decorate_tw_symbol_columns(
                sync_df,
                service=service,
                enabled=tw_symbol_label_enabled,
                columns=["symbol"],
            ),
            width="stretch",
            hide_index=True,
        )

    if not run_requested:
        st.info("已就緒。按下「執行回測」後才會開始載入資料並執行回測。")
        return

    prepared_bars = load_and_prepare_symbol_bars(
        store=store,
        market_code=market_code,
        symbols=list(symbols),
        start=sync_start,
        end=sync_end,
        use_total_return_adjustment=bool(use_total_return_adjustment),
        use_split_adjustment=bool(use_split_adjustment),
        auto_detect_split=bool(auto_detect_split),
        apply_total_return_adjustment=_apply_total_return_adjustment,
    )
    bars_by_symbol = dict(prepared_bars.bars_by_symbol)
    availability_rows = list(prepared_bars.availability_rows)
    perf_timer.mark("bars_prepared")
    if availability_rows:
        st.dataframe(
            _decorate_tw_symbol_columns(
                pd.DataFrame(availability_rows),
                service=service,
                enabled=tw_symbol_label_enabled,
                columns=["symbol"],
            ),
            width="stretch",
            hide_index=True,
        )
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
        st.dataframe(
            _decorate_tw_symbol_columns(
                invest_df,
                service=service,
                enabled=tw_symbol_label_enabled,
                columns=["symbol"],
            ),
            width="stretch",
            hide_index=True,
        )
    base_min_bars = get_strategy_min_bars(strategy)
    min_required_bars = (
        required_walkforward_bars(strategy_name=strategy, train_ratio=float(train_ratio))
        if enable_wf
        else base_min_bars
    )
    bars_by_symbol = {
        sym: bars for sym, bars in bars_by_symbol.items() if len(bars) >= min_required_bars
    }
    if not bars_by_symbol:
        if enable_wf:
            st.warning(
                f"投入起點後可用資料太少（{strategy} + Walk-Forward 至少需要 {min_required_bars} 根K），"
                "請調整起點或日期區間。"
            )
        elif strategy == "buy_hold":
            st.warning("投入起點後可用資料太少（buy_hold 至少需要 2 根K），請調整起點或日期區間。")
        else:
            st.warning(
                f"投入起點後可用資料太少（{strategy} 至少需要 {min_required_bars} 根K）。"
                "若只想看近期投入，可改用 `buy_hold`。"
            )
        return

    run_params_base = build_backtest_run_params_base(
        market=market_selector,
        mode=mode,
        symbols=list(symbols),
        strategy=strategy,
        strategy_params=strategy_params if isinstance(strategy_params, dict) else {},
        start_date=start_date,
        end_date=end_date,
        enable_walk_forward=bool(enable_wf),
        train_ratio=float(train_ratio),
        objective=objective,
        initial_capital=float(initial_capital),
        fee_rate=float(fee_rate),
        sell_tax_rate=float(sell_tax),
        slippage_rate=float(slippage),
        use_split_adjustment=bool(use_split_adjustment),
        auto_detect_split=bool(auto_detect_split),
        use_total_return_adjustment=bool(use_total_return_adjustment),
        invest_start_mode=str(invest_start_mode),
        invest_start_date=invest_start_date,
        invest_start_k=int(invest_start_k) if invest_start_k is not None else -1,
    )
    run_params, run_params_hash = build_replay_params_with_signature(
        params_base=run_params_base,
        schema_version=BACKTEST_REPLAY_SCHEMA_VERSION,
    )
    run_cache_order_key = "bt_result_session_order"
    max_session_payloads = 3

    def _cache_run_payload(cache_key: str, cache_payload: dict[str, object]) -> None:
        st.session_state[cache_key] = cache_payload
        raw_order = st.session_state.get(run_cache_order_key, [])
        order = raw_order if isinstance(raw_order, list) else []
        order = [str(item) for item in order if str(item)]
        order = [item for item in order if item != cache_key]
        order.append(cache_key)
        while len(order) > max_session_payloads:
            evicted = order.pop(0)
            st.session_state.pop(evicted, None)
        st.session_state[run_cache_order_key] = order

    payload = st.session_state.get(run_key)
    if payload is None:
        cached_payload, cache_message = _load_cached_backtest_payload(
            store=store,
            run_key=run_key,
            expected_schema=BACKTEST_REPLAY_SCHEMA_VERSION,
            expected_hash=run_params_hash,
        )
        if cache_message:
            st.caption(cache_message)
        if isinstance(cached_payload, dict):
            payload = cached_payload
            _cache_run_payload(run_key, cached_payload)

    if payload is None:
        cost_model = CostModel(fee_rate=fee_rate, sell_tax_rate=sell_tax, slippage_rate=slippage)
        try:
            with st.spinner("回測計算中..."):
                run_payload = execute_backtest_run(
                    bars_by_symbol=bars_by_symbol,
                    config=BacktestExecutionInput(
                        mode=mode,
                        strategy=strategy,
                        strategy_params=strategy_params
                        if isinstance(strategy_params, dict)
                        else {},
                        enable_walk_forward=bool(enable_wf),
                        train_ratio=float(train_ratio),
                        objective=objective,
                        initial_capital=float(initial_capital),
                        cost_model=cost_model,
                    ),
                )
                perf_timer.mark("backtest_executed")
        except Exception as exc:
            st.error(f"回測失敗：{exc}")
            return

        _cache_run_payload(run_key, run_payload)
        payload = run_payload
        primary = ",".join(symbols)
        summary_metrics = run_payload["result"].metrics.__dict__
        store.save_backtest_run(
            symbol=primary,
            market=market_selector,
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
        replay_payload = _serialize_backtest_run_payload(run_payload)
        store.save_backtest_replay_run(
            run_key=run_key,
            params=run_params,
            payload=replay_payload,
        )
        perf_timer.mark("backtest_cached")

    if not payload:
        st.info("尚未有可用回測結果。")
        return

    result = payload["result"]
    run_initial_capital = float(payload.get("initial_capital", initial_capital))
    bars_by_symbol = payload["bars_by_symbol"]
    selected_symbols = list(bars_by_symbol.keys())
    multi_symbol_compare = len(selected_symbols) > 1
    tw_name_map: dict[str, str] = {}
    if tw_symbol_label_enabled and selected_symbols:
        tw_name_map = service.get_tw_symbol_names(_collect_tw_symbol_codes(list(selected_symbols)))

    def _tw_label(value: object) -> str:
        return _format_tw_symbol_with_name(value, tw_name_map)

    is_portfolio = payload["mode"] == "portfolio"
    strategy_equity = result.equity_curve["equity"].copy()
    buy_hold_equity = build_buy_hold_equity(
        bars_by_symbol=bars_by_symbol,
        target_index=pd.DatetimeIndex(strategy_equity.index),
        initial_capital=run_initial_capital,
    )
    benchmark_raw = _load_benchmark_from_store(
        market_code=market_code, start=sync_start, end=sync_end, choice=benchmark_choice
    )
    if benchmark_raw.empty:
        benchmark_raw = service.get_benchmark_series(
            market=market_code, start=sync_start, end=sync_end, benchmark=benchmark_choice
        )
        runner_queue_benchmark_writeback(
            store=store, market_code=market_code, benchmark=benchmark_raw
        )
    perf_timer.mark("benchmark_loaded")
    benchmark_adj_info: dict[str, object] = {"applied": False}
    if use_total_return_adjustment and not benchmark_raw.empty:
        benchmark_raw, benchmark_adj_info = _apply_total_return_adjustment(benchmark_raw)
    benchmark_split_events = []
    if use_split_adjustment and not benchmark_raw.empty:
        if not bool(benchmark_adj_info.get("applied")):
            benchmark_symbol = (
                str(benchmark_raw.attrs.get("symbol", "")).strip()
                if hasattr(benchmark_raw, "attrs")
                else ""
            )
            if not benchmark_symbol:
                fallback_symbol_map_tw = {"twii": "^TWII", "0050": "0050", "006208": "006208"}
                fallback_symbol_map_us = {"gspc": "^GSPC", "spy": "SPY", "qqq": "QQQ", "dia": "DIA"}
                benchmark_symbol = (
                    fallback_symbol_map_tw.get(benchmark_choice, "^TWII")
                    if market_code == "TW"
                    else fallback_symbol_map_us.get(benchmark_choice, "^GSPC")
                )
            adjusted, benchmark_split_events = apply_split_adjustment(
                bars=benchmark_raw,
                symbol=benchmark_symbol,
                market=market_code,
                use_known=True,
                use_auto_detect=auto_detect_split,
            )
            adjusted.attrs = dict(getattr(benchmark_raw, "attrs", {}))
            if benchmark_symbol:
                adjusted.attrs["symbol"] = benchmark_symbol
            benchmark_raw = adjusted

    benchmark_symbol_now = (
        str(benchmark_raw.attrs.get("symbol", "")).strip()
        if hasattr(benchmark_raw, "attrs")
        else ""
    )
    if tw_symbol_label_enabled and benchmark_symbol_now:
        extra_names = service.get_tw_symbol_names(_collect_tw_symbol_codes([benchmark_symbol_now]))
        if isinstance(extra_names, dict) and extra_names:
            tw_name_map.update(
                {str(k).strip().upper(): str(v).strip() for k, v in extra_names.items()}
            )

    benchmark_equity = pd.Series(dtype=float)
    if not benchmark_raw.empty and "close" in benchmark_raw.columns:
        bench = benchmark_raw["close"].reindex(strategy_equity.index).ffill()
        bench = bench.dropna()
        if not bench.empty:
            benchmark_equity = (bench / bench.iloc[0]) * run_initial_capital
            benchmark_equity = benchmark_equity.reindex(strategy_equity.index).ffill()

    benchmark_rel = pd.DataFrame()
    strategy_ret: float | None = None
    benchmark_ret: float | None = None
    diff_pct: float | None = None
    verdict = ""
    if benchmark_choice != "off" and not benchmark_equity.empty:
        benchmark_rel = pd.concat(
            [
                strategy_equity.rename("strategy"),
                benchmark_equity.rename("benchmark"),
            ],
            axis=1,
        ).dropna()
        if len(benchmark_rel) >= 2:
            strategy_ret = float(
                benchmark_rel["strategy"].iloc[-1] / benchmark_rel["strategy"].iloc[0] - 1.0
            )
            benchmark_ret = float(
                benchmark_rel["benchmark"].iloc[-1] / benchmark_rel["benchmark"].iloc[0] - 1.0
            )
            diff_pct = (strategy_ret - benchmark_ret) * 100.0
            verdict = "贏過大盤" if diff_pct > 0 else ("輸給大盤" if diff_pct < 0 else "與大盤持平")

    component_vs_benchmark_rows: list[dict[str, object]] = []
    component_winner_count = 0
    component_comparable_count = 0
    if is_portfolio and benchmark_choice != "off" and not benchmark_equity.empty:
        for symbol, comp in result.component_results.items():
            symbol_equity = comp.equity_curve.get("equity", pd.Series(dtype=float))
            if not isinstance(symbol_equity, pd.Series) or symbol_equity.empty:
                component_vs_benchmark_rows.append(
                    {
                        "代碼": _tw_label(symbol),
                        "策略總報酬(%)": np.nan,
                        "Benchmark總報酬(%)": np.nan,
                        "相對Benchmark(%)": np.nan,
                        "結論": "資料不足",
                    }
                )
                continue
            rel_sym = pd.concat(
                [
                    symbol_equity.rename("strategy"),
                    benchmark_equity.reindex(symbol_equity.index).ffill().rename("benchmark"),
                ],
                axis=1,
            ).dropna()
            if len(rel_sym) < 2:
                component_vs_benchmark_rows.append(
                    {
                        "代碼": _tw_label(symbol),
                        "策略總報酬(%)": np.nan,
                        "Benchmark總報酬(%)": np.nan,
                        "相對Benchmark(%)": np.nan,
                        "結論": "資料不足",
                    }
                )
                continue
            symbol_ret = float(rel_sym["strategy"].iloc[-1] / rel_sym["strategy"].iloc[0] - 1.0)
            symbol_bench_ret = float(
                rel_sym["benchmark"].iloc[-1] / rel_sym["benchmark"].iloc[0] - 1.0
            )
            symbol_diff_pct = (symbol_ret - symbol_bench_ret) * 100.0
            component_comparable_count += 1
            if symbol_diff_pct > 0:
                component_winner_count += 1
            symbol_verdict = (
                "勝過Benchmark"
                if symbol_diff_pct > 0
                else ("輸給Benchmark" if symbol_diff_pct < 0 else "與Benchmark持平")
            )
            component_vs_benchmark_rows.append(
                {
                    "代碼": _tw_label(symbol),
                    "策略總報酬(%)": round(symbol_ret * 100.0, 2),
                    "Benchmark總報酬(%)": round(symbol_bench_ret * 100.0, 2),
                    "相對Benchmark(%)": round(symbol_diff_pct, 2),
                    "結論": symbol_verdict,
                }
            )

    if not multi_symbol_compare:
        with st.container(border=True):
            _render_card_section_header("績效卡", "總報酬、CAGR、MDD 與相對大盤結果。")
            if is_portfolio:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("投組總報酬", f"{result.metrics.total_return * 100:.2f}%")
                k2.metric(
                    "Benchmark總報酬",
                    "—" if benchmark_ret is None else f"{benchmark_ret * 100:.2f}%",
                )
                k3.metric("超額報酬", "—" if diff_pct is None else f"{diff_pct:+.2f}%")
                winner_text = (
                    "—"
                    if component_comparable_count <= 0
                    else f"{component_winner_count}/{component_comparable_count}"
                )
                k4.metric("勝過Benchmark檔數", winner_text)

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("投組CAGR", f"{result.metrics.cagr * 100:.2f}%")
                s2.metric("投組MDD", f"{result.metrics.max_drawdown * 100:.2f}%")
                s3.metric("投組Sharpe", f"{result.metrics.sharpe:.2f}")
                s4.metric("交易次數", str(int(result.metrics.trades)))

                if diff_pct is not None and strategy_ret is not None and benchmark_ret is not None:
                    st.caption(
                        "計算基準：同一重疊區間的 Total Return；"
                        f"Strategy={strategy_ret * 100:.2f}% vs Benchmark={benchmark_ret * 100:.2f}%（{verdict}）"
                    )
                elif benchmark_choice != "off" and benchmark_equity.empty:
                    st.caption("目前沒有可用的 Benchmark 資料，無法計算超額報酬。")
                elif benchmark_choice != "off":
                    st.caption("Benchmark 可用資料不足，暫時無法判斷投組是否勝過大盤。")
            else:
                metric_rows = _metrics_to_rows(result.metrics)
                metric_cols = st.columns(4)
                for idx, (label, val) in enumerate(metric_rows):
                    metric_cols[idx % 4].metric(label, val)
                if diff_pct is not None and strategy_ret is not None and benchmark_ret is not None:
                    r1, r2 = st.columns(2)
                    r1.metric("相對大盤結果", verdict)
                    r2.metric("贏/輸大盤（百分比）", f"{diff_pct:+.2f}%")
                    st.caption(
                        "計算基準：同一重疊區間的 Total Return；"
                        f"Strategy={strategy_ret * 100:.2f}% vs Benchmark={benchmark_ret * 100:.2f}%"
                    )
                elif benchmark_choice != "off" and benchmark_equity.empty:
                    st.caption("目前沒有可用的 Benchmark 資料，無法計算是否贏過大盤。")
                elif benchmark_choice != "off":
                    st.caption("Benchmark 可用資料不足，暫時無法判斷是否贏過大盤。")

    if is_portfolio and not multi_symbol_compare:
        with st.container(border=True):
            _render_card_section_header(
                "投組分項相對Benchmark", "多標的時優先看這張：每檔相對基準的超額報酬。"
            )
            if component_vs_benchmark_rows:
                comp_rel_df = pd.DataFrame(component_vs_benchmark_rows).sort_values(
                    ["相對Benchmark(%)", "代碼"],
                    ascending=[False, True],
                    na_position="last",
                )
                st.dataframe(comp_rel_df, width="stretch", hide_index=True)
            else:
                st.caption("目前沒有可顯示的分項相對 Benchmark 資料。")

    if payload.get("walk_forward") and not multi_symbol_compare:
        with st.container(border=True):
            _render_card_section_header("Walk-Forward 卡", "Train/Test 分段結果與最佳參數。")
            split_date = payload["split_date"]
            st.caption(
                f"切分點：{pd.Timestamp(split_date).strftime('%Y-%m-%d')} | 目標：{objective}"
            )
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
                if isinstance(best_params, dict) and all(
                    isinstance(v, dict) for v in best_params.values()
                ):
                    rows = [
                        {"symbol": _tw_label(s), "params": str(p)} for s, p in best_params.items()
                    ]
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                else:
                    st.code(str(best_params))

    speed_key = f"play_speed:{run_key}"
    play_key = f"play_state:{run_key}"
    idx_key = f"play_idx:{run_key}"
    display_mode_key = f"play_display_mode:{run_key}"
    marker_mode_key = f"play_marker_mode:{run_key}"
    viewport_mode_key = f"play_viewport_mode:{run_key}"
    viewport_window_key = f"play_viewport_window:{run_key}"
    viewport_anchor_key = f"play_viewport_anchor:{run_key}"
    focus_key = f"play_focus:{run_key}"
    ensure_defaults(
        st.session_state,
        {
            speed_key: "1x",
            play_key: False,
            display_mode_key: "K棒",
            marker_mode_key: "同時顯示（訊號+成交）",
            viewport_mode_key: "完整區間",
            viewport_anchor_key: 70,
            focus_key: selected_symbols[0],
        },
    )
    if multi_symbol_compare:
        focus_symbol = str(selected_symbols[0])
    else:
        focus_symbol = st.selectbox(
            "回放焦點標的",
            options=selected_symbols,
            key=focus_key,
            format_func=lambda sym: _tw_label(sym),
        )

    focus_bars = (
        bars_by_symbol[focus_symbol]
        .sort_index()
        .dropna(subset=["open", "high", "low", "close"], how="any")
    )
    if focus_bars.empty:
        if not multi_symbol_compare:
            st.warning(f"{focus_symbol} 沒有可回放的有效K線資料。")
            return
    focus_ind = add_indicators(focus_bars)
    focus_result = result.component_results[focus_symbol] if is_portfolio else result
    max_play_idx = len(focus_bars) - 1
    default_play_idx = max_play_idx
    replay_reset_idx = 0
    if idx_key not in st.session_state:
        st.session_state[idx_key] = default_play_idx
    else:
        st.session_state[idx_key] = min(int(st.session_state[idx_key]), max_play_idx)
    window_min = 20
    window_max = max(window_min, max_play_idx + 1)
    window_has_range = window_max > window_min
    if viewport_window_key not in st.session_state:
        st.session_state[viewport_window_key] = min(180, window_max)
    else:
        st.session_state[viewport_window_key] = int(
            min(max(int(st.session_state[viewport_window_key]), window_min), window_max)
        )
    if not window_has_range:
        st.session_state[viewport_window_key] = window_min
    date_options = [pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in focus_bars.index]
    date_to_idx = {d: i for i, d in enumerate(date_options)}

    if not multi_symbol_compare:
        with st.container(border=True):
            _render_card_section_header("回放控制卡", "播放速度、標記模式與視窗控制。")
            c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 2, 2, 3])
            if c1.button("Play", width="stretch"):
                st.session_state[play_key] = True
            if c2.button("Pause", width="stretch"):
                st.session_state[play_key] = False
            if c3.button("Reset", width="stretch"):
                st.session_state[play_key] = False
                st.session_state[idx_key] = replay_reset_idx
            c4.selectbox("速度", options=["0.5x", "1x", "2x", "5x", "10x"], key=speed_key)
            c5.radio("位置顯示", options=["K棒", "日期"], horizontal=True, key=display_mode_key)
            c6.selectbox(
                "買賣點顯示",
                options=[
                    "不顯示",
                    "訊號點（價格圖）",
                    "實際成交點（資產圖）",
                    "同時顯示（訊號+成交）",
                ],
                key=marker_mode_key,
            )
            q1, q2, q3 = st.columns([2, 2, 2])
            q1.selectbox("回放視窗", options=["固定視窗", "完整區間"], key=viewport_mode_key)
            if window_has_range:
                q2.slider(
                    "視窗K數",
                    min_value=window_min,
                    max_value=window_max,
                    step=5,
                    key=viewport_window_key,
                    disabled=st.session_state[viewport_mode_key] != "固定視窗",
                )
            else:
                q2.caption(f"視窗K數：{window_min}（固定）")
            q3.slider(
                "生長位置（靠右%）",
                min_value=50,
                max_value=90,
                step=5,
                key=viewport_anchor_key,
                disabled=st.session_state[viewport_mode_key] != "固定視窗",
                help="數值越大，最新K越靠右；例如 70 代表最新K大約落在畫面 70% 的位置。",
            )
            st.caption("成交連線與技術指標副圖預設固定顯示。")
            st.caption(
                "買賣點模式說明：訊號點=策略切換點；實際成交點=依回測規則 T+1 開盤成交；同時顯示=兩者一起顯示。"
            )
            st.caption(
                f"回放預設位置：第 {default_play_idx} 根K（完整區間末端）。"
                f"若要重播動畫可按 Reset 回到第 {replay_reset_idx} 根。"
            )

    speed_steps = {"0.5x": 1, "1x": 2, "2x": 4, "5x": 8, "10x": 16}
    rsi_oversold = 30.0
    rsi_weak = 45.0
    rsi_neutral_strong = 60.0
    rsi_strong = 70.0
    kd_oversold = 20.0
    kd_weak = 40.0
    kd_sweet = 60.0
    kd_overheat_level = 80.0
    mfi_oversold = 20.0
    mfi_weak = 50.0
    mfi_strong = 80.0
    bias_overheat_pct = 0.06
    bias_weak_pct = -0.03
    stop_loss_atr_mult = 1.5
    stop_loss_atr_conservative_mult = 2.0
    vwap_hot_gap_pct = 0.03
    vwap_near_gap_pct = 0.01
    atr_high_vol_ratio = 0.035
    indicator_snapshot_cols = [
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

    def _to_float(value: object) -> float | None:
        fv = pd.to_numeric(value, errors="coerce")
        if pd.isna(fv):
            return None
        return float(fv)

    def _fmt_num(value: float | None, digits: int = 2) -> str:
        if value is None:
            return "—"
        return f"{float(value):.{int(digits)}f}"

    def _fmt_pct(value: float | None, digits: int = 2) -> str:
        if value is None:
            return "—"
        return f"{float(value) * 100:.{int(digits)}f}%"

    def _tf(flag: bool | None) -> str:
        if flag is None:
            return "N/A"
        return "True" if bool(flag) else "False"

    def _clamp_score(value: float) -> int:
        return int(max(0, min(100, round(float(value)))))

    def _render_replay_indicator_snapshot_table(ind_now: pd.DataFrame, idx_now: int) -> None:
        if not isinstance(ind_now, pd.DataFrame) or ind_now.empty:
            st.caption("技術快照完整指標：資料不足。")
            return
        row_pos = min(max(int(idx_now), 0), len(ind_now.index) - 1)
        row = ind_now.iloc[row_pos]
        row_dt = pd.Timestamp(ind_now.index[row_pos]).strftime("%Y-%m-%d")
        snapshot_data: dict[str, object] = {}
        for col in indicator_snapshot_cols:
            value = pd.to_numeric(row.get(col), errors="coerce")
            snapshot_data[col] = float(value) if pd.notna(value) else np.nan
        snapshot_df = pd.DataFrame([snapshot_data])

        close = _to_float(snapshot_data.get("close"))
        sma_5 = _to_float(snapshot_data.get("sma_5"))
        sma_20 = _to_float(snapshot_data.get("sma_20"))
        sma_60 = _to_float(snapshot_data.get("sma_60"))
        rsi_14 = _to_float(snapshot_data.get("rsi_14"))
        stoch_k = _to_float(snapshot_data.get("stoch_k"))
        stoch_d = _to_float(snapshot_data.get("stoch_d"))
        mfi_14 = _to_float(snapshot_data.get("mfi_14"))
        macd = _to_float(snapshot_data.get("macd"))
        macd_signal = _to_float(snapshot_data.get("macd_signal"))
        macd_hist = _to_float(snapshot_data.get("macd_hist"))
        bb_mid = _to_float(snapshot_data.get("bb_mid"))
        bb_upper = _to_float(snapshot_data.get("bb_upper"))
        bb_lower = _to_float(snapshot_data.get("bb_lower"))
        vwap = _to_float(snapshot_data.get("vwap"))
        atr_14 = _to_float(snapshot_data.get("atr_14"))

        bias_pct = (
            (float(close) - float(sma_20)) / float(sma_20)
            if close is not None and sma_20 not in {None, 0}
            else None
        )
        vwap_gap_pct = (
            (float(close) - float(vwap)) / float(vwap)
            if close is not None and vwap not in {None, 0}
            else None
        )
        atr_ratio = (
            float(atr_14) / float(close) if atr_14 is not None and close not in {None, 0} else None
        )

        bull_align = bool(
            sma_5 is not None
            and sma_20 is not None
            and sma_60 is not None
            and close is not None
            and sma_5 > sma_20 > sma_60
            and close > sma_20
        )
        trend_weaken = bool(
            (close is not None and sma_20 is not None and close < sma_20)
            or (sma_20 is not None and sma_60 is not None and sma_20 < sma_60)
        )
        if bull_align:
            trend_state = "多"
        elif trend_weaken and close is not None and sma_20 is not None and close < sma_20:
            trend_state = "空"
        else:
            trend_state = "盤整"

        if rsi_14 is None:
            rsi_text = "資料不足"
        elif rsi_14 < rsi_oversold:
            rsi_text = "超賣（<30）"
        elif rsi_14 < rsi_weak:
            rsi_text = "偏弱（30-45）"
        elif rsi_14 < rsi_neutral_strong:
            rsi_text = "中性偏強（45-60）"
        elif rsi_14 <= rsi_strong:
            rsi_text = "強勢（60-70）"
        else:
            rsi_text = "過熱（>70）"

        prev_k = None
        prev_d = None
        if row_pos >= 1:
            prev_row = ind_now.iloc[row_pos - 1]
            prev_k = _to_float(prev_row.get("stoch_k"))
            prev_d = _to_float(prev_row.get("stoch_d"))
        kd_golden_cross = bool(
            prev_k is not None
            and prev_d is not None
            and stoch_k is not None
            and stoch_d is not None
            and prev_k <= prev_d
            and stoch_k > stoch_d
        )
        kd_death_cross = bool(
            prev_k is not None
            and prev_d is not None
            and stoch_k is not None
            and stoch_d is not None
            and prev_k >= prev_d
            and stoch_k < stoch_d
        )
        kd_sweet_cross = bool(
            kd_golden_cross
            and stoch_k is not None
            and stoch_d is not None
            and kd_weak <= stoch_k <= kd_sweet
            and kd_weak <= stoch_d <= kd_sweet
        )
        kd_overheat = bool(
            stoch_k is not None
            and stoch_d is not None
            and stoch_k > kd_overheat_level
            and stoch_d > kd_overheat_level
        )
        if stoch_k is None or stoch_d is None:
            kd_zone_text = "資料不足"
        elif kd_overheat:
            kd_zone_text = "高檔鈍化/過熱（K、D > 80）"
        elif stoch_k < kd_oversold and stoch_d < kd_oversold:
            kd_zone_text = "低檔超賣（K、D < 20）"
        elif kd_sweet_cross:
            kd_zone_text = "40-60 黃金交叉甜蜜區"
        elif kd_golden_cross:
            kd_zone_text = "黃金交叉"
        elif kd_death_cross:
            kd_zone_text = "死亡交叉"
        else:
            kd_zone_text = "中性區間震盪"

        if mfi_14 is None:
            mfi_text = "資料不足"
        elif mfi_14 < mfi_oversold:
            mfi_text = "超賣（<20）"
        elif mfi_14 < mfi_weak:
            mfi_text = "中性偏弱（20-50）"
        elif mfi_14 <= mfi_strong:
            mfi_text = "中性偏強（50-80）"
        else:
            mfi_text = "過熱（>80）"

        if macd_hist is None:
            macd_hist_text = "資料不足"
        elif macd_hist > 0:
            macd_hist_text = "動能偏多（hist > 0）"
        elif macd_hist < 0:
            macd_hist_text = "動能轉弱（hist < 0）"
        else:
            macd_hist_text = "動能中性（hist = 0）"
        macd_bias_bull = bool(macd is not None and macd_signal is not None and macd > macd_signal)
        macd_bias_text = (
            "多方較強（macd > signal）" if macd_bias_bull else "多方較弱（macd <= signal）"
        )

        if close is None or bb_upper is None or bb_mid is None or bb_lower is None:
            bb_zone_text = "資料不足"
        elif close > bb_upper:
            bb_zone_text = "close > bb_upper（強勢突破/短線過熱）"
        elif bb_upper >= close >= bb_mid:
            bb_zone_text = "upper >= close >= mid（偏強區）"
        elif bb_mid > close >= bb_lower:
            bb_zone_text = "mid > close >= lower（回檔整理區）"
        else:
            bb_zone_text = "close < bb_lower（恐慌/超賣風險）"

        if vwap_gap_pct is None:
            vwap_text = "資料不足"
        elif abs(vwap_gap_pct) <= vwap_near_gap_pct:
            vwap_text = "close 回到 VWAP 附近（相對安全成本區）"
        elif vwap_gap_pct > vwap_hot_gap_pct:
            vwap_text = "close 明顯高於 VWAP（偏熱）"
        elif vwap_gap_pct > 0:
            vwap_text = "close 高於 VWAP（短線偏強）"
        elif vwap_gap_pct < -vwap_hot_gap_pct:
            vwap_text = "close 明顯低於 VWAP（偏弱）"
        else:
            vwap_text = "close 低於 VWAP（短線偏弱）"

        if atr_ratio is None:
            atr_text = "資料不足"
        elif atr_ratio > atr_high_vol_ratio:
            atr_text = "波動過大（ATR/Close 偏高）"
        elif atr_ratio > 0.02:
            atr_text = "波動偏高"
        else:
            atr_text = "波動可控"

        overheated = bool(
            (bias_pct is not None and bias_pct > bias_overheat_pct)
            or (rsi_14 is not None and rsi_14 > rsi_strong)
            or (mfi_14 is not None and mfi_14 > mfi_strong)
            or (close is not None and bb_upper is not None and close > bb_upper)
            or (vwap_gap_pct is not None and vwap_gap_pct > vwap_hot_gap_pct)
        )
        false_breakout_risk = bool(
            close is not None
            and bb_upper is not None
            and close > bb_upper
            and ((macd_hist is not None and macd_hist <= 0) or (kd_overheat and kd_death_cross))
        )

        trend_score = 50.0
        if bull_align:
            trend_score += 30.0
        if close is not None and sma_20 is not None and close > sma_20:
            trend_score += 10.0
        else:
            trend_score -= 15.0
        if sma_20 is not None and sma_60 is not None and sma_20 > sma_60:
            trend_score += 10.0
        else:
            trend_score -= 10.0
        if bias_pct is not None and bias_pct > bias_overheat_pct:
            trend_score -= 15.0
        if bias_pct is not None and bias_pct < bias_weak_pct:
            trend_score -= 10.0
        if trend_state == "空":
            trend_score -= 20.0
        trend_score_int = _clamp_score(trend_score)

        momentum_score = 50.0
        if rsi_14 is not None:
            if rsi_weak <= rsi_14 < rsi_neutral_strong:
                momentum_score += 12.0
            elif rsi_neutral_strong <= rsi_14 <= rsi_strong:
                momentum_score += 17.0
            elif rsi_14 > rsi_strong:
                momentum_score -= 8.0
            elif rsi_14 < rsi_oversold:
                momentum_score -= 12.0
        if kd_sweet_cross:
            momentum_score += 18.0
        elif kd_golden_cross:
            momentum_score += 10.0
        elif kd_death_cross:
            momentum_score -= 10.0
        if macd_hist is not None:
            momentum_score += 12.0 if macd_hist > 0 else -10.0
        if macd is not None and macd_signal is not None:
            momentum_score += 8.0 if macd > macd_signal else -6.0
        if mfi_14 is not None:
            if mfi_14 > mfi_strong or mfi_14 < mfi_oversold:
                momentum_score -= 8.0
            elif mfi_weak <= mfi_14 <= mfi_strong:
                momentum_score += 6.0
        momentum_score_int = _clamp_score(momentum_score)

        risk_score = 20.0
        if trend_state == "空":
            risk_score += 20.0
        if bias_pct is not None and bias_pct > bias_overheat_pct:
            risk_score += 15.0
        if bias_pct is not None and bias_pct < bias_weak_pct:
            risk_score += 10.0
        if rsi_14 is not None and rsi_14 > rsi_strong:
            risk_score += 12.0
        if close is not None and bb_upper is not None and close > bb_upper:
            risk_score += 10.0
        if close is not None and bb_lower is not None and close < bb_lower:
            risk_score += 12.0
        if vwap_gap_pct is not None and abs(vwap_gap_pct) > vwap_hot_gap_pct:
            risk_score += 10.0
        if atr_ratio is not None and atr_ratio > atr_high_vol_ratio:
            risk_score += 20.0
        if false_breakout_risk:
            risk_score += 12.0
        if (
            macd_hist is not None
            and macd_hist < 0
            and close is not None
            and sma_20 is not None
            and close < sma_20
        ):
            risk_score += 10.0
        risk_score_int = _clamp_score(risk_score)

        if trend_state == "多" and (not overheated) and risk_score_int <= 45:
            status_label = "可佈局"
        elif trend_state == "多" and overheated:
            status_label = "避免追高"
        elif trend_state == "空" or risk_score_int >= 70:
            status_label = "風險偏高"
        else:
            status_label = "觀望"

        if trend_state == "多" and (not overheated) and momentum_score_int >= 55:
            action_label = "拉回佈局"
        elif (
            trend_state == "多"
            and close is not None
            and bb_upper is not None
            and close > bb_upper
            and momentum_score_int >= 60
            and risk_score_int < 65
        ):
            action_label = "突破追價（少量）"
        else:
            action_label = "觀望"

        entry_anchor_name = ""
        entry_anchor = None
        for col_name, col_val in (
            ("sma_20", sma_20),
            ("bb_mid", bb_mid),
            ("vwap", vwap),
        ):
            if col_val is not None:
                entry_anchor_name = col_name
                entry_anchor = col_val
                break
        if entry_anchor is not None and atr_14 is not None:
            entry_low = float(entry_anchor) - 0.5 * float(atr_14)
            entry_high = float(entry_anchor) + 0.5 * float(atr_14)
            entry_zone_text = (
                f"{_fmt_num(entry_low)} ~ {_fmt_num(entry_high)}（以 {entry_anchor_name} 為中心）"
            )
            stop_main = float(entry_anchor) - stop_loss_atr_mult * float(atr_14)
            stop_conservative = float(entry_anchor) - stop_loss_atr_conservative_mult * float(
                atr_14
            )
            stop_text = (
                f"{_fmt_num(stop_main)}（1.5*ATR） / {_fmt_num(stop_conservative)}（2*ATR，保守）"
            )
        elif entry_anchor is not None:
            entry_zone_text = (
                f"{_fmt_num(float(entry_anchor) * 0.995)} ~ {_fmt_num(float(entry_anchor) * 1.005)}"
            )
            stop_text = f"{_fmt_num(float(entry_anchor) * 0.97)}（暫用 -3%）"
        else:
            entry_zone_text = "—"
            stop_text = "—"

        hold_exit_parts: list[str] = []
        if sma_20 is not None:
            hold_exit_parts.append("close >= sma_20 可續抱")
            hold_exit_parts.append("跌破 sma_20 視為減碼/出場訊號")
        if macd_hist is not None:
            hold_exit_parts.append("macd_hist >= 0 視為動能維持")
            macd_hist_series = pd.to_numeric(ind_now.get("macd_hist"), errors="coerce").dropna()
            if len(macd_hist_series) >= 2:
                latest_two_neg = bool(
                    macd_hist_series.iloc[-1] < 0 and macd_hist_series.iloc[-2] < 0
                )
                if latest_two_neg:
                    hold_exit_parts.append("macd_hist 已連2根負值，偏向防守")
                else:
                    hold_exit_parts.append("若 macd_hist 連2根轉負，偏向減碼")
            else:
                hold_exit_parts.append("若 macd_hist 連2根轉負，偏向減碼")
        hold_exit_text = "；".join(hold_exit_parts) if hold_exit_parts else "—"

        risk_warnings: list[str] = []
        if bias_pct is not None and bias_pct > bias_overheat_pct:
            risk_warnings.append("乖離過大（close 相對 sma_20 > +6%）")
        if rsi_14 is not None and rsi_14 > rsi_strong:
            risk_warnings.append("RSI > 70，動能過熱")
        if close is not None and bb_upper is not None and close > bb_upper:
            risk_warnings.append("close 位於 bb_upper 之外，短線追價風險")
        if false_breakout_risk:
            risk_warnings.append("疑似假突破（站上上軌但動能未同步）")
        if atr_ratio is not None and atr_ratio > atr_high_vol_ratio:
            risk_warnings.append("ATR/Close 偏高，波動過大")
        if not risk_warnings:
            risk_warnings.append("目前未觸發主要過熱警示，仍需控倉")

        decision_rows: list[tuple[str, str, str, str]] = [
            (
                "sma_5 > sma_20 > sma_60 ?",
                _tf(
                    sma_5 is not None
                    and sma_20 is not None
                    and sma_60 is not None
                    and sma_5 > sma_20 > sma_60
                ),
                "均線多頭排列判斷",
                "加分（趨勢）" if bull_align else "提醒（未形成多頭排列）",
            ),
            (
                "close > sma_20 ?",
                _tf(close is not None and sma_20 is not None and close > sma_20),
                f"close={_fmt_num(close)}, sma_20={_fmt_num(sma_20)}",
                "加分（守住中期趨勢）"
                if close is not None and sma_20 is not None and close > sma_20
                else "扣分（跌破中期均線）",
            ),
            (
                "乖離 > +6% ?",
                _tf(bias_pct is not None and bias_pct > bias_overheat_pct),
                f"bias={_fmt_pct(bias_pct)}",
                "提醒（過熱）" if bias_pct is not None and bias_pct > bias_overheat_pct else "中性",
            ),
            (
                "乖離 < -3% ?",
                _tf(bias_pct is not None and bias_pct < bias_weak_pct),
                f"bias={_fmt_pct(bias_pct)}",
                "扣分（偏弱/回測）"
                if bias_pct is not None and bias_pct < bias_weak_pct
                else "中性",
            ),
            (
                "RSI > 70 ?",
                _tf(rsi_14 is not None and rsi_14 > rsi_strong),
                f"RSI={_fmt_num(rsi_14)}",
                "提醒（過熱）" if rsi_14 is not None and rsi_14 > rsi_strong else "中性",
            ),
            (
                "KD 黃金交叉且位於 40-60 ?",
                _tf(kd_sweet_cross),
                f"K={_fmt_num(stoch_k)}, D={_fmt_num(stoch_d)}",
                "加分（波段甜蜜區）" if kd_sweet_cross else "中性",
            ),
            (
                "macd_hist > 0 ?",
                _tf(macd_hist is not None and macd_hist > 0),
                f"macd_hist={_fmt_num(macd_hist, 4)}",
                "加分（動能偏多）"
                if macd_hist is not None and macd_hist > 0
                else "扣分（動能偏弱）",
            ),
            (
                "macd > macd_signal ?",
                _tf(macd is not None and macd_signal is not None and macd > macd_signal),
                f"macd={_fmt_num(macd, 4)}, signal={_fmt_num(macd_signal, 4)}",
                "加分（多方較強）" if macd_bias_bull else "提醒（多方較弱）",
            ),
            (
                "close > bb_upper ?",
                _tf(close is not None and bb_upper is not None and close > bb_upper),
                f"close={_fmt_num(close)}, upper={_fmt_num(bb_upper)}",
                "提醒（突破或過熱）"
                if close is not None and bb_upper is not None and close > bb_upper
                else "中性",
            ),
            (
                "close < bb_lower ?",
                _tf(close is not None and bb_lower is not None and close < bb_lower),
                f"close={_fmt_num(close)}, lower={_fmt_num(bb_lower)}",
                "提醒（恐慌/超賣）"
                if close is not None and bb_lower is not None and close < bb_lower
                else "中性",
            ),
            (
                "|close-vwap|/vwap > 3% ?",
                _tf(vwap_gap_pct is not None and abs(vwap_gap_pct) > vwap_hot_gap_pct),
                f"gap={_fmt_pct(vwap_gap_pct)}",
                "提醒（溢價/折價偏大）"
                if vwap_gap_pct is not None and abs(vwap_gap_pct) > vwap_hot_gap_pct
                else "中性",
            ),
            (
                "ATR/close > 3.5% ?",
                _tf(atr_ratio is not None and atr_ratio > atr_high_vol_ratio),
                f"ATR/close={_fmt_pct(atr_ratio)}",
                "扣分（波動過大）"
                if atr_ratio is not None and atr_ratio > atr_high_vol_ratio
                else "中性",
            ),
        ]
        decision_md_lines = [
            "| 條件 | 目前結果 | 解釋 | 對波段意義 |",
            "|---|---|---|---|",
        ]
        decision_md_lines.extend([f"| {c} | {r} | {e} | {m} |" for c, r, e, m in decision_rows])

        symbol_title = str(focus_symbol or "").strip().upper() or "UNKNOWN"
        trend_line = (
            f"close={_fmt_num(close)} / sma_5={_fmt_num(sma_5)} / sma_20={_fmt_num(sma_20)} / "
            f"sma_60={_fmt_num(sma_60)}，乖離={_fmt_pct(bias_pct)}"
        )
        kd_relation = (
            "K>D（偏強）"
            if stoch_k is not None and stoch_d is not None and stoch_k > stoch_d
            else "K<=D（偏弱）"
        )
        macd_line = (
            f"macd={_fmt_num(macd, 4)}, signal={_fmt_num(macd_signal, 4)}, hist={_fmt_num(macd_hist, 4)}；"
            f"{macd_hist_text}，{macd_bias_text}"
        )
        if trend_state == "多":
            trend_advice = "多"
        elif trend_state == "空":
            trend_advice = "空"
        else:
            trend_advice = "盤整"

        gail_style = (
            f"盤面現在是 `{trend_advice}` 結構，動能分數 {momentum_score_int}/100，風險分數 {risk_score_int}/100。"
            f"策略上偏向 `{action_label}`，不是叫你看到紅K就梭哈。"
            f" 先看 `{entry_zone_text}` 這段能不能站穩，再用 `{stop_text}` 做風險上限，"
            "守紀律比猜高點低點重要。"
        )

        st.markdown("#### 技術快照完整指標表格")
        st.caption(
            f"資料日期：{row_dt} | 焦點標的：{_tw_label(focus_symbol) or str(focus_symbol)}（隨回放位置同步）"
        )
        st.dataframe(snapshot_df.round(4), width="stretch", hide_index=True)
        st.markdown(f"# {symbol_title} {row_dt}")
        st.markdown("## 指標判讀")
        st.markdown(
            "\n".join(
                [
                    f"- 趨勢/均線：{trend_line}。目前判定 `{trend_state}`。",
                    f"- RSI(14)：{_fmt_num(rsi_14)}（{rsi_text}）。",
                    f"- KD：K={_fmt_num(stoch_k)} / D={_fmt_num(stoch_d)}，{kd_relation}，{kd_zone_text}。",
                    f"- MFI(14)：{_fmt_num(mfi_14)}（{mfi_text}）。",
                    f"- MACD：{macd_line}",
                    f"- 布林：{bb_zone_text}（mid={_fmt_num(bb_mid)}, upper={_fmt_num(bb_upper)}, lower={_fmt_num(bb_lower)}）。",
                    f"- VWAP：{vwap_text}（vwap={_fmt_num(vwap)}, gap={_fmt_pct(vwap_gap_pct)}）。",
                    f"- ATR：atr_14={_fmt_num(atr_14)}，{atr_text}（ATR/close={_fmt_pct(atr_ratio)}）。",
                ]
            )
        )
        st.markdown("## 波段結論（只給可執行建議，不空泛）")
        st.markdown(
            "\n".join(
                [
                    f"- 趨勢狀態：{trend_state}",
                    f"- 建議動作：{action_label}",
                    f"- 進場參考區：{entry_zone_text}",
                    f"- 停損參考：{stop_text}",
                    f"- 續抱/出場條件：{hold_exit_text}",
                    f"- 風險提醒：{'；'.join(risk_warnings)}",
                ]
            )
        )
        st.markdown("## 判斷表（Decision Table）")
        st.markdown("\n".join(decision_md_lines))
        st.markdown("## 打分數（0-100）")
        st.markdown(
            "\n".join(
                [
                    f"- Trend：{trend_score_int}/100",
                    f"- Momentum：{momentum_score_int}/100",
                    f"- Risk：{risk_score_int}/100（分數越高代表風險越高）",
                    f"- 總結狀態：**{status_label}**",
                ]
            )
        )
        st.markdown("## 股癌風格敘述")
        st.markdown(gail_style)

    @st.fragment(run_every="0.5s")
    def playback():
        palette = _ui_palette()
        playback_symbol_label = _tw_label(focus_symbol) or str(focus_symbol)
        price_up_line = str(palette.get("price_up_line", palette["price_up"]))
        price_down_line = str(palette.get("price_down_line", palette["price_down"]))
        price_up_fill = str(palette.get("price_up_fill", _to_rgba(price_up_line, 0.42)))
        price_down_fill = str(palette.get("price_down_fill", _to_rgba(price_down_line, 0.42)))
        if st.session_state[play_key]:
            step = speed_steps[st.session_state[speed_key]]
            new_idx = min(len(focus_bars) - 1, st.session_state[idx_key] + step)
            st.session_state[idx_key] = new_idx
            if new_idx >= len(focus_bars) - 1:
                st.session_state[play_key] = False

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

        idx = st.session_state[idx_key]
        bars_now = focus_bars.iloc[: idx + 1]
        ind_now = focus_ind.reindex(bars_now.index)
        equity_now = result.equity_curve.iloc[: min(idx + 1, len(result.equity_curve))]
        benchmark_now = (
            benchmark_equity.reindex(equity_now.index).ffill()
            if not benchmark_equity.empty
            else pd.Series(dtype=float)
        )
        panel_x_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
        if _replay_kline_renderer() == "lightweight":
            strategy_series = (
                equity_now["equity"] if "equity" in equity_now.columns else pd.Series(dtype=float)
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
                panel_x_range = (
                    pd.Timestamp(focus_bars.index[left_idx]),
                    pd.Timestamp(focus_bars.index[right_idx]),
                )
            elif len(bars_now.index) >= 2:
                panel_x_range = (pd.Timestamp(bars_now.index[0]), pd.Timestamp(bars_now.index[-1]))

            ok = render_lightweight_kline_equity_chart(
                bars=bars_now,
                strategy=strategy_series,
                benchmark=benchmark_now,
                palette=palette,
                key=f"playback_lw:{run_key}",
            )
            if ok:
                _render_indicator_panels(
                    ind_now,
                    chart_key=f"play_indicator_chart:{run_key}:{focus_symbol}",
                    height=440,
                    x_range=panel_x_range,
                    watermark_text=playback_symbol_label,
                )
                _render_replay_indicator_snapshot_table(ind_now, idx)
                st.caption(
                    "lightweight 模式目前專注 K線+Equity+Benchmark；"
                    "訊號點/成交點僅在 Plotly 模式顯示。"
                )
                st.caption(
                    f"目前回放到：第 {idx + 1} 根K（{bars_now.index[-1].strftime('%Y-%m-%d')}）"
                )
                return
            st.caption("lightweight-charts 渲染失敗，已自動回退 Plotly。")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.018, row_heights=[0.68, 0.32]
        )
        fig.add_trace(
            go.Candlestick(
                x=bars_now.index,
                open=bars_now["open"],
                high=bars_now["high"],
                low=bars_now["low"],
                close=bars_now["close"],
                name="Price",
                increasing_line_color=price_up_line,
                increasing_fillcolor=price_up_fill,
                decreasing_line_color=price_down_line,
                decreasing_fillcolor=price_down_fill,
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
                    line=_benchmark_line_style(palette, width=2.0),
                ),
                row=2,
                col=1,
            )

        high_series = pd.to_numeric(bars_now.get("high", pd.Series(dtype=float)), errors="coerce")
        low_series = pd.to_numeric(bars_now.get("low", pd.Series(dtype=float)), errors="coerce")
        high_marker_color = str(palette.get("signal_sell", "#D6465A"))
        low_marker_color = str(palette.get("signal_buy", "#2F9E6B"))
        if (not multi_symbol_compare) and high_series.notna().any():
            highest_idx = high_series.idxmax()
            highest_val = float(high_series.loc[highest_idx])
            highest_date = pd.Timestamp(highest_idx).strftime("%Y-%m-%d")
            fig.add_annotation(
                x=highest_idx,
                y=highest_val,
                text=f"最高價 {_format_price(highest_val)}<br>日期 {highest_date}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowwidth=1.8,
                arrowcolor=high_marker_color,
                ax=0,
                ay=-42,
                font=dict(color=high_marker_color, size=11),
                bgcolor=_to_rgba(str(palette["paper_bg"]), 0.78),
                bordercolor=high_marker_color,
                borderwidth=1,
                row=1,
                col=1,
            )
        if (not multi_symbol_compare) and low_series.notna().any():
            lowest_idx = low_series.idxmin()
            lowest_val = float(low_series.loc[lowest_idx])
            lowest_date = pd.Timestamp(lowest_idx).strftime("%Y-%m-%d")
            fig.add_annotation(
                x=lowest_idx,
                y=lowest_val,
                text=f"最低價 {_format_price(lowest_val)}<br>日期 {lowest_date}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowwidth=1.8,
                arrowcolor=low_marker_color,
                ax=0,
                ay=42,
                font=dict(color=low_marker_color, size=11),
                bgcolor=_to_rgba(str(palette["paper_bg"]), 0.78),
                bordercolor=low_marker_color,
                borderwidth=1,
                row=1,
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
                    go.Scatter(
                        x=buy_px.index,
                        y=buy_px.values,
                        mode="markers",
                        name="Buy Signal",
                        marker=signal_buy_style,
                    ),
                    row=1,
                    col=1,
                )
            if len(sell_idx) > 0:
                sell_px = bars_now.loc[sell_idx.intersection(bars_now.index), "close"]
                fig.add_trace(
                    go.Scatter(
                        x=sell_px.index,
                        y=sell_px.values,
                        mode="markers",
                        name="Sell Signal",
                        marker=signal_sell_style,
                    ),
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
                        if (
                            not y1.empty
                            and not y2.empty
                            and pd.notna(y1.iloc[0])
                            and pd.notna(y2.iloc[0])
                        ):
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
                if fill_times:
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
            panel_x_range = (pd.Timestamp(x_start), pd.Timestamp(x_end))
        elif len(bars_now.index) >= 2:
            panel_x_range = (pd.Timestamp(bars_now.index[0]), pd.Timestamp(bars_now.index[-1]))

        fig.update_xaxes(gridcolor=str(palette["grid"]))
        fig.update_yaxes(gridcolor=str(palette["grid"]))
        fig.update_layout(
            height=560,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=68, b=10),
            uirevision=f"playback:{run_key}",
            transition=dict(duration=0),
            template=str(palette["plot_template"]),
            paper_bgcolor=str(palette["paper_bg"]),
            plot_bgcolor=str(palette["plot_bg"]),
            font=dict(color=str(palette["text_color"])),
        )
        _enable_plotly_draw_tools(fig)
        _apply_plotly_watermark(fig, text=playback_symbol_label, palette=palette)
        _apply_unified_benchmark_hover(fig, palette)
        playback_file = str(run_key).replace(":", "_").replace("/", "_")
        _render_plotly_chart(
            fig,
            chart_key=f"playback_chart:{run_key}",
            filename=f"playback_{playback_file}",
            scale=2,
            width="stretch",
            watermark_text=str(playback_symbol_label),
            palette=palette,
        )
        _render_indicator_panels(
            ind_now,
            chart_key=f"play_indicator_chart:{run_key}:{focus_symbol}",
            height=440,
            x_range=panel_x_range,
            watermark_text=playback_symbol_label,
        )
        _render_replay_indicator_snapshot_table(ind_now, idx)
        st.caption(f"目前回放到：第 {idx + 1} 根K（{bars_now.index[-1].strftime('%Y-%m-%d')}）")

    if not multi_symbol_compare:
        with st.container(border=True):
            _render_card_section_header("回放圖卡", "K線 + Equity + Benchmark 動態回放。")
            playback()
        replay_symbol_label = _tw_label(focus_symbol)
        benchmark_between_label = ""
        if hasattr(benchmark_raw, "attrs"):
            benchmark_between_label = _tw_label(str(benchmark_raw.attrs.get("symbol", "")).strip())
        if replay_symbol_label and benchmark_between_label:
            st.caption(f"回放標的：{replay_symbol_label} | Benchmark：{benchmark_between_label}")
        elif replay_symbol_label:
            st.caption(f"回放標的：{replay_symbol_label}")

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

    _render_card_section_header(
        "Benchmark 對照卡", "策略曲線、基準曲線與買進持有（理論無摩擦）同圖比較。"
    )
    benchmark = benchmark_raw
    benchmark_symbol = (
        str(benchmark.attrs.get("symbol", "")).strip() if hasattr(benchmark, "attrs") else ""
    )
    benchmark_source = (
        str(benchmark.attrs.get("source", "")).strip() if hasattr(benchmark, "attrs") else ""
    )
    hide_strategy_line = str(strategy).strip().lower() == "buy_hold"
    benchmark_line_label = (
        f"Benchmark ({_tw_label(benchmark_symbol)})" if benchmark_symbol else "Benchmark Equity"
    )
    single_symbol_label = _tw_label(selected_symbols[0]) if selected_symbols else ""
    buy_hold_label = (
        "Buy and Hold (EW Portfolio)"
        if is_portfolio
        else f"Buy and Hold ({single_symbol_label or selected_symbols[0]})"
    )
    benchmark_as_of = result.equity_curve.index.max() if not result.equity_curve.empty else ""
    if not benchmark.empty and len(benchmark.index):
        benchmark_as_of = benchmark.index.max()
    health_notes = []
    if bool(benchmark_adj_info.get("applied")):
        health_notes.append(f"adj_close={benchmark_adj_info.get('coverage_pct', 0)}%")
    if benchmark_split_events:
        health_notes.append(f"split_events={len(benchmark_split_events)}")
    benchmark_health = _build_data_health(
        as_of=benchmark_as_of,
        data_sources=[benchmark_source or "benchmark_unknown"],
        source_chain=[benchmark_symbol] if benchmark_symbol else [],
        degraded=bool(benchmark_choice != "off" and benchmark.empty),
        fallback_depth=1 if benchmark_choice != "off" and benchmark.empty else 0,
        notes=" | ".join(health_notes),
    )
    _render_data_health_caption("Benchmark 資料健康度", benchmark_health)
    if benchmark_choice != "off" and benchmark_symbol:
        st.caption(
            f"Benchmark 來源：{_tw_label(benchmark_symbol)}（{benchmark_source or 'unknown'}）"
        )
    if bool(benchmark_adj_info.get("applied")):
        st.caption(
            f"Benchmark 已套用還原權息（Adj Close 覆蓋率 {benchmark_adj_info.get('coverage_pct', 0)}%）。"
        )
    if benchmark_split_events:
        split_text = ", ".join(
            [
                f"{pd.Timestamp(ev.date).date()} x{ev.ratio:.6f} ({ev.source})"
                for ev in benchmark_split_events
            ]
        )
        st.caption(f"Benchmark split adjustment: {split_text}")
    if benchmark.empty:
        if benchmark_choice == "off":
            st.caption("你已關閉 Benchmark。")
        else:
            st.caption(
                "目前無法取得 Benchmark（可能是來源限流）；可改選 SPY/QQQ/DIA（美股）或 0050/006208（台股）。"
            )
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
                format_func=lambda sym: _tw_label(sym),
            )
            if is_portfolio
            else []
        )
        if extra_symbol_lines:
            for sym in extra_symbol_lines:
                bars_sym = bars_by_symbol.get(sym, pd.DataFrame())
                if bars_sym.empty or "close" not in bars_sym.columns:
                    continue
                price_series = (
                    pd.to_numeric(bars_sym["close"], errors="coerce").reindex(comp.index).ffill()
                )
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
                bench_style = _benchmark_line_style(palette, width=2.0)
                chart_lines: list[dict[str, Any]] = []
                if not hide_strategy_line:
                    chart_lines.append(
                        {
                            "name": "Strategy Equity",
                            "series": norm["strategy"],
                            "color": str(palette["equity"]),
                            "width": 2.2,
                            "dash": "solid",
                            "hover_code": "STRATEGY",
                            "value_label": "Normalized",
                            "y_format": ".4f",
                        }
                    )
                chart_lines.append(
                    {
                        "name": benchmark_line_label,
                        "series": norm["benchmark"],
                        "color": str(bench_style["color"]),
                        "width": float(bench_style["width"]),
                        "dash": str(bench_style["dash"]),
                        "hover_code": benchmark_symbol or "BENCHMARK",
                        "value_label": "Normalized",
                        "y_format": ".4f",
                    }
                )
                if "buy_hold" in norm.columns and norm["buy_hold"].notna().any():
                    buy_hold_code = "EW_POOL" if is_portfolio else selected_symbols[0]
                    chart_lines.append(
                        {
                            "name": buy_hold_label,
                            "series": norm["buy_hold"],
                            "color": str(palette["buy_hold"]),
                            "width": 1.9,
                            "dash": "solid",
                            "hover_code": buy_hold_code,
                            "value_label": "Normalized",
                            "y_format": ".4f",
                        }
                    )
                asset_palette = list(palette["asset_palette"])
                asset_color_idx = 0
                for col in [c for c in norm.columns if c.startswith("asset:")]:
                    sym = col.split(":", 1)[1]
                    sym_label = _tw_label(sym)
                    line_color = asset_palette[asset_color_idx % len(asset_palette)]
                    asset_color_idx += 1
                    chart_lines.append(
                        {
                            "name": f"Buy and Hold ({sym_label})",
                            "series": norm[col],
                            "color": str(line_color),
                            "width": 1.6,
                            "dash": "solid",
                            "hover_code": sym_label,
                            "value_label": "Normalized",
                            "y_format": ".4f",
                        }
                    )
                extrema_target = "Strategy Equity"
                if hide_strategy_line:
                    extrema_target = (
                        buy_hold_label
                        if any(str(line.get("name")) == buy_hold_label for line in chart_lines)
                        else benchmark_line_label
                    )
                benchmark_watermark_text = (
                    _tw_label(selected_symbols[0])
                    if (not is_portfolio and selected_symbols)
                    else "EW Portfolio"
                )
                _render_benchmark_lines_chart(
                    lines=chart_lines,
                    height=360,
                    chart_key=f"backtest_benchmark:{run_key}",
                    annotate_extrema=not multi_symbol_compare,
                    extrema_series_name=extrema_target,
                    watermark_text=benchmark_watermark_text,
                )

                bench_perf = runner_series_metrics(comp_view["benchmark"])
                cmp_rows_data: list[dict[str, object]] = []
                if not hide_strategy_line:
                    strategy_perf = runner_series_metrics(comp_view["strategy"])
                    cmp_rows_data.append(
                        {
                            "Series": "Strategy Equity",
                            "Definition": "Backtest equity after fees/tax/slippage",
                            "Total Return %": round(strategy_perf["total_return"] * 100.0, 2),
                            "CAGR %": round(strategy_perf["cagr"] * 100.0, 2),
                            "MDD %": round(strategy_perf["max_drawdown"] * 100.0, 2),
                            "Sharpe": round(strategy_perf["sharpe"], 2),
                        }
                    )
                cmp_rows_data.append(
                    {
                        "Series": benchmark_line_label,
                        "Definition": f"Buy-and-hold reference ({benchmark_symbol or 'benchmark'})",
                        "Total Return %": round(bench_perf["total_return"] * 100.0, 2),
                        "CAGR %": round(bench_perf["cagr"] * 100.0, 2),
                        "MDD %": round(bench_perf["max_drawdown"] * 100.0, 2),
                        "Sharpe": round(bench_perf["sharpe"], 2),
                    }
                )
                cmp_rows = pd.DataFrame(cmp_rows_data)
                if "buy_hold" in comp_view.columns and comp_view["buy_hold"].notna().any():
                    hold_perf = runner_series_metrics(comp_view["buy_hold"])
                    cmp_rows.loc[len(cmp_rows)] = {
                        "Series": buy_hold_label,
                        "Definition": "Buy-and-hold (frictionless close-price baseline on backtest symbols)",
                        "Total Return %": round(hold_perf["total_return"] * 100.0, 2),
                        "CAGR %": round(hold_perf["cagr"] * 100.0, 2),
                        "MDD %": round(hold_perf["max_drawdown"] * 100.0, 2),
                        "Sharpe": round(hold_perf["sharpe"], 2),
                    }
                for col in [c for c in comp_view.columns if c.startswith("asset:")]:
                    sym = col.split(":", 1)[1]
                    sym_label = _tw_label(sym)
                    sym_series = pd.to_numeric(comp_view[col], errors="coerce").dropna()
                    if len(sym_series) < 2:
                        continue
                    sym_perf = runner_series_metrics(sym_series)
                    cmp_rows.loc[len(cmp_rows)] = {
                        "Series": f"Buy and Hold ({sym_label})",
                        "Definition": f"Buy-and-hold frictionless close-price baseline on symbol {sym_label}",
                        "Total Return %": round(sym_perf["total_return"] * 100.0, 2),
                        "CAGR %": round(sym_perf["cagr"] * 100.0, 2),
                        "MDD %": round(sym_perf["max_drawdown"] * 100.0, 2),
                        "Sharpe": round(sym_perf["sharpe"], 2),
                    }
                st.dataframe(
                    _decorate_tw_symbol_columns(
                        cmp_rows,
                        service=service,
                        enabled=tw_symbol_label_enabled,
                        columns=["Series", "Definition"],
                    ),
                    width="stretch",
                    hide_index=True,
                )

    def _render_component_and_trade_cards() -> None:
        if is_portfolio:
            _render_card_section_header(
                "投組分項策略績效卡", "此卡顯示各檔策略本身績效（不含相對Benchmark比較）。"
            )
            rows = []
            for symbol, comp in result.component_results.items():
                rows.append(
                    {
                        "symbol": _tw_label(symbol),
                        "total_return%": round(comp.metrics.total_return * 100.0, 2),
                        "cagr%": round(comp.metrics.cagr * 100.0, 2),
                        "mdd%": round(comp.metrics.max_drawdown * 100.0, 2),
                        "sharpe": round(comp.metrics.sharpe, 2),
                        "trades": comp.metrics.trades,
                    }
                )
            _render_crisp_table(pd.DataFrame(rows), max_height=360)

        _render_card_section_header("交易明細卡")
        if is_portfolio:
            if result.trades is not None and not result.trades.empty:
                trades_df = result.trades.copy()
                trades_df = _decorate_tw_symbol_columns(
                    trades_df,
                    service=service,
                    enabled=tw_symbol_label_enabled,
                    columns=["symbol"],
                )
                trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.date.astype(
                    str
                )
                trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.date.astype(str)
                trades_df["pnl_pct%"] = (trades_df["pnl_pct"] * 100.0).round(2)
                _render_crisp_table(trades_df.drop(columns=["pnl_pct"]), max_height=460)
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
                _render_crisp_table(trades_df, max_height=460)
            else:
                st.caption("沒有交易紀錄。")

    if multi_symbol_compare:
        _render_component_and_trade_cards()
        perf_timer.mark("render_complete")
        if perf_timer.enabled:
            st.caption(perf_timer.summary_text(prefix="perf/backtest"))
        return

    _render_card_section_header(
        "策略 vs 買進持有卡", "看整段與指定區間的報酬率差異（買進持有為理論無摩擦基準）。"
    )
    is_buy_hold_strategy = str(strategy).strip().lower() == "buy_hold"
    strategy_label = "Buy and Hold（策略）" if is_buy_hold_strategy else "策略"
    hold_label = (
        "買進持有（理論無摩擦，等權投組）"
        if is_portfolio
        else f"買進持有（理論無摩擦，{single_symbol_label or selected_symbols[0]}）"
    )
    theoretical_label = (
        "Theoretical No Friction（買進持有，等權投組）"
        if is_portfolio
        else f"Theoretical No Friction（買進持有，{single_symbol_label or selected_symbols[0]}）"
    )
    st.caption(
        "買進持有為理論無摩擦基準：以回測標的收盤價計算，不含手續費、交易稅、滑價；若為投組則採等權配置。"
    )

    comparison_rows = [
        {
            "比較區間": "整段",
            "比較項目": strategy_label,
            "起始交易日(實際)": strategy_equity.index[0].strftime("%Y-%m-%d"),
            "結束交易日(實際)": strategy_equity.index[-1].strftime("%Y-%m-%d"),
            "報酬率%": round((strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1.0) * 100.0, 2),
        }
    ]
    if not buy_hold_equity.empty:
        comparison_rows.append(
            {
                "比較區間": "整段",
                "比較項目": theoretical_label if is_buy_hold_strategy else hold_label,
                "起始交易日(實際)": buy_hold_equity.index[0].strftime("%Y-%m-%d"),
                "結束交易日(實際)": buy_hold_equity.index[-1].strftime("%Y-%m-%d"),
                "報酬率%": round(
                    (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0] - 1.0) * 100.0, 2
                ),
            }
        )

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
        strategy_interval = interval_return(
            strategy_equity, start_date=interval_start, end_date=interval_end
        )
        hold_interval = interval_return(
            buy_hold_equity, start_date=interval_start, end_date=interval_end
        )
        if strategy_interval.get("ok"):
            comparison_rows.append(
                {
                    "比較區間": "指定區間",
                    "比較項目": strategy_label,
                    "起始交易日(實際)": strategy_interval["start_used"].strftime("%Y-%m-%d"),
                    "結束交易日(實際)": strategy_interval["end_used"].strftime("%Y-%m-%d"),
                    "報酬率%": round(float(strategy_interval["return"]) * 100.0, 2),
                }
            )
        if hold_interval.get("ok"):
            comparison_rows.append(
                {
                    "比較區間": "指定區間",
                    "比較項目": theoretical_label if is_buy_hold_strategy else hold_label,
                    "起始交易日(實際)": hold_interval["start_used"].strftime("%Y-%m-%d"),
                    "結束交易日(實際)": hold_interval["end_used"].strftime("%Y-%m-%d"),
                    "報酬率%": round(float(hold_interval["return"]) * 100.0, 2),
                }
            )
    if comparison_rows:
        out_df = pd.DataFrame(comparison_rows)
        st.dataframe(
            _decorate_tw_symbol_columns(
                out_df,
                service=service,
                enabled=tw_symbol_label_enabled,
                columns=["比較項目"],
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("目前沒有可計算的比較資料。")

    with st.container(border=True):
        _render_card_section_header(
            "DCA 比較卡", "期初投入 + 每月定期定額（每月首個交易日收盤；等權分配）。"
        )
        dca_initial_key = f"dca_initial_lump:{run_key}"
        dca_monthly_key = f"dca_monthly_contribution:{run_key}"
        dca_c1, dca_c2 = st.columns(2)
        dca_initial_lump = float(
            dca_c1.number_input(
                "期初投入金額",
                min_value=0.0,
                value=float(run_initial_capital),
                step=10_000.0,
                format="%.0f",
                key=dca_initial_key,
            )
        )
        dca_monthly_contribution = float(
            dca_c2.number_input(
                "每月定期定額金額",
                min_value=0.0,
                value=20_000.0,
                step=1_000.0,
                format="%.0f",
                key=dca_monthly_key,
            )
        )

        dca_target_index = pd.DatetimeIndex(strategy_equity.index)
        dca_plan = build_dca_contribution_plan(dca_target_index)
        monthly_dates_raw = dca_plan.get("monthly_dates", [])
        monthly_dates = list(monthly_dates_raw) if isinstance(monthly_dates_raw, list) else []
        monthly_contribution_count = int(len(monthly_dates))
        total_dca_contribution = float(
            dca_initial_lump + dca_monthly_contribution * monthly_contribution_count
        )
        dca_benchmark_label = ""
        if hasattr(benchmark_raw, "attrs"):
            dca_benchmark_label = str(benchmark_raw.attrs.get("symbol", "")).strip()
        if not dca_benchmark_label:
            fallback_symbol_map_tw = {"twii": "^TWII", "0050": "0050", "006208": "006208"}
            fallback_symbol_map_us = {"gspc": "^GSPC", "spy": "SPY", "qqq": "QQQ", "dia": "DIA"}
            dca_benchmark_label = (
                fallback_symbol_map_tw.get(benchmark_choice, "^TWII")
                if market_code == "TW"
                else fallback_symbol_map_us.get(benchmark_choice, "^GSPC")
            )
        dca_benchmark_label = _tw_label(dca_benchmark_label)

        dca_equity = build_dca_equity(
            bars_by_symbol=bars_by_symbol,
            target_index=dca_target_index,
            initial_lump_sum=dca_initial_lump,
            monthly_contribution=dca_monthly_contribution,
            fee_rate=float(fee_rate),
            sell_tax_rate=float(sell_tax),
            slippage_rate=float(slippage),
        )
        dca_metrics = dca_summary_metrics(
            dca_equity,
            total_contribution=total_dca_contribution,
        )

        dca_benchmark_metrics: dict[str, float] = {}
        if (
            benchmark_choice != "off"
            and not benchmark_raw.empty
            and "close" in benchmark_raw.columns
        ):
            bench_close_series = pd.to_numeric(benchmark_raw["close"], errors="coerce").dropna()
            if not bench_close_series.empty:
                dca_benchmark_equity = build_dca_benchmark_equity(
                    benchmark_close=bench_close_series,
                    target_index=dca_target_index,
                    initial_lump_sum=dca_initial_lump,
                    monthly_contribution=dca_monthly_contribution,
                    fee_rate=float(fee_rate),
                    sell_tax_rate=float(sell_tax),
                    slippage_rate=float(slippage),
                )
                dca_benchmark_metrics = dca_summary_metrics(
                    dca_benchmark_equity,
                    total_contribution=total_dca_contribution,
                )

        dca_ret = _safe_float(dca_metrics.get("total_return"))
        dca_bench_ret = (
            _safe_float(dca_benchmark_metrics.get("total_return"))
            if dca_benchmark_metrics
            else None
        )
        dca_excess_pct = (
            (float(dca_ret) - float(dca_bench_ret)) * 100.0
            if dca_ret is not None and dca_bench_ret is not None
            else None
        )

        def _fmt_money(v: float | None) -> str:
            if v is None or not math.isfinite(float(v)):
                return "—"
            return f"{float(v):,.0f}"

        def _fmt_pct(v: float | None, *, scale: float = 100.0) -> str:
            if v is None or not math.isfinite(float(v)):
                return "—"
            return f"{float(v) * scale:+.2f}%"

        dca_m1, dca_m2, dca_m3, dca_m4 = st.columns(4)
        dca_m1.metric("DCA 期末淨值", _fmt_money(_safe_float(dca_metrics.get("end_value"))))
        dca_m2.metric("DCA 總投入", _fmt_money(total_dca_contribution))
        dca_m3.metric("DCA 報酬率", _fmt_pct(dca_ret, scale=100.0))
        dca_m4.metric(
            "相對 Benchmark DCA", "—" if dca_excess_pct is None else f"{dca_excess_pct:+.2f}%"
        )

        st.caption(
            f"投入規則：期初投入 1 次；每月定投採每月首個交易日收盤，且從次月開始。"
            f"本次每月投入次數：{monthly_contribution_count} 次。"
        )

        dca_rows = [
            {
                "比較項目": "DCA策略",
                "總投入": _fmt_money(total_dca_contribution),
                "期末淨值": _fmt_money(_safe_float(dca_metrics.get("end_value"))),
                "損益": _fmt_money(_safe_float(dca_metrics.get("pnl"))),
                "報酬率(%)": "—" if dca_ret is None else f"{dca_ret * 100.0:+.2f}%",
            }
        ]
        if dca_benchmark_metrics:
            dca_rows.append(
                {
                    "比較項目": f"Benchmark DCA（{dca_benchmark_label or 'Benchmark'}）",
                    "總投入": _fmt_money(total_dca_contribution),
                    "期末淨值": _fmt_money(_safe_float(dca_benchmark_metrics.get("end_value"))),
                    "損益": _fmt_money(_safe_float(dca_benchmark_metrics.get("pnl"))),
                    "報酬率(%)": "—" if dca_bench_ret is None else f"{dca_bench_ret * 100.0:+.2f}%",
                }
            )
        st.dataframe(pd.DataFrame(dca_rows), width="stretch", hide_index=True)
        if benchmark_choice == "off":
            st.caption("你已關閉 Benchmark，因此本卡僅顯示 DCA策略結果。")
        elif not dca_benchmark_metrics:
            st.caption("Benchmark 可用資料不足，暫時無法建立 Benchmark DCA 比較。")

    _render_card_section_header("逐年報酬卡")
    if result.yearly_returns:
        yr = pd.DataFrame(
            [{"年度": y, "報酬率%": round(v * 100.0, 2)} for y, v in result.yearly_returns.items()]
        )
        st.dataframe(yr.sort_values("年度"), width="stretch", hide_index=True)
    else:
        st.caption("樣本不足，無逐年報酬。")

    _render_component_and_trade_cards()

    perf_timer.mark("render_complete")
    if perf_timer.enabled:
        st.caption(perf_timer.summary_text(prefix="perf/backtest"))


__all__ = ["_render_backtest_view"]
