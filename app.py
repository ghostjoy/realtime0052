from __future__ import annotations

import csv
import json
import logging
import math
import re
import sqlite3
from collections import Counter
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlencode
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest import (
    ROTATION_DEFAULT_UNIVERSE,
    ROTATION_MIN_BARS,
    BacktestMetrics,
    BacktestResult,
    CostModel,
    PortfolioBacktestResult,
    Trade,
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
    run_backtest,
    run_tw_etf_rotation_backtest,
)
from backtest.adjustments import known_split_events
from config_loader import cfg_or_env, cfg_or_env_bool, cfg_or_env_str, get_config_source
from di import get_history_store as di_get_history_store
from di import get_market_service as di_get_market_service
from indicators import add_indicators
from market_data_types import DataHealth
from providers import TwMisProvider
from services import LiveOptions, MarketDataService
from services.backtest_cache import (
    build_backtest_run_key,
    build_backtest_run_params_base,
    build_replay_params_with_signature,
    build_source_hash,
    stable_json_dumps,
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
from services.benchmark_loader import (
    benchmark_candidates_tw,
    load_tw_benchmark_bars,
    load_tw_benchmark_close,
)
from services.bootstrap_loader import run_incremental_refresh, run_market_data_bootstrap
from services.heatmap_runner import compute_heatmap_rows, prepare_heatmap_bars
from services.rotation_runner import (
    build_rotation_holding_rank as runner_build_rotation_holding_rank,
)
from services.rotation_runner import (
    build_rotation_payload as runner_build_rotation_payload,
)
from services.rotation_runner import (
    prepare_rotation_bars,
)
from services.sync_orchestrator import (
    bars_need_backfill as orchestrated_bars_need_backfill,
)
from services.sync_orchestrator import (
    sync_symbols_history as orchestrated_sync_symbols_history,
)
from services.sync_orchestrator import sync_symbols_if_needed
from state_keys import BT_KEYS
from storage import HistoryStore
from ui.charts import (
    render_lightweight_kline_equity_chart,
    render_lightweight_live_chart,
    render_lightweight_multi_line_chart,
)
from ui.helpers import (
    build_backtest_drill_url,
    build_heatmap_drill_url,
    looks_like_tw_symbol,
    looks_like_us_symbol,
    parse_drill_symbol,
    strip_symbol_label_token,
)
from ui.shared.perf import PerfTimer, perf_debug_enabled
from ui.shared.session_utils import ensure_defaults
from utils import normalize_ohlcv_frame

# Compatibility alias for ui/pages/* ctx-injection bridge.
_normalize_ohlcv_frame = normalize_ohlcv_frame

# Streamlit may emit repetitive INFO logs when a scheduled fragment rerun arrives
# after a full-app rerun removed that fragment. This is harmless but noisy.
logging.getLogger("streamlit.runtime.app_session").setLevel(logging.WARNING)

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]

try:
    from advice import Profile, render_advice, render_advice_scai_style
except ImportError:
    render_advice_scai_style = None  # type: ignore[assignment]


@st.cache_resource
def _market_service() -> MarketDataService:
    return di_get_market_service()


@st.cache_resource
def _history_store() -> HistoryStore:
    return di_get_history_store()


def _auto_run_daily_incremental_refresh(store: HistoryStore):
    auto_enabled = cfg_or_env_bool(
        "sync.auto_incremental_on_startup",
        "REALTIME0052_AUTO_INCREMENTAL_ON_STARTUP",
        default=False,
    )
    if not auto_enabled:
        return
    today_token = date.today().isoformat()
    session_key = f"daily_incremental_done:{today_token}"
    if st.session_state.get(session_key):
        return
    latest = store.load_latest_bootstrap_run()
    if latest is not None:
        latest_date = latest.started_at.astimezone(timezone.utc).date()
        if latest.scope == "daily_incremental" and latest_date == date.today():
            st.session_state[session_key] = True
            return
    if not store.list_symbols("TW", limit=1) and not store.list_symbols("US", limit=1):
        return
    st.session_state[session_key] = True
    try:
        summary = run_incremental_refresh(
            store=store,
            years=int(
                cfg_or_env("sync.years", "REALTIME0052_SYNC_YEARS", default=5, cast=int) or 5
            ),
            parallel=True,
            max_workers=int(
                cfg_or_env(
                    "sync.parallel_workers", "REALTIME0052_SYNC_WORKERS", default=4, cast=int
                )
                or 4
            ),
            tw_limit=120,
            us_limit=50,
        )
        st.session_state["daily_incremental_summary"] = summary
    except Exception as exc:
        st.session_state["daily_incremental_error"] = str(exc)


def _renderer_choice(path: str, env_var: str, default: str = "plotly") -> str:
    token = cfg_or_env_str(path, env_var, default).strip().lower()
    return token if token in {"plotly", "lightweight"} else default


def _live_kline_renderer() -> str:
    return _renderer_choice(
        "features.kline_renderer_live", "REALTIME0052_KLINE_RENDERER_LIVE", "plotly"
    )


def _replay_kline_renderer() -> str:
    return _renderer_choice(
        "features.kline_renderer_replay", "REALTIME0052_KLINE_RENDERER_REPLAY", "plotly"
    )


def _benchmark_renderer() -> str:
    return _renderer_choice(
        "features.benchmark_renderer", "REALTIME0052_BENCHMARK_RENDERER", "plotly"
    )


def _path_scope_label(path_like: object) -> str:
    text = str(path_like or "").strip()
    icloud_root = str(Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs")
    return "iCloud" if text.startswith(icloud_root) else "local"


def _runtime_stack_caption() -> str:
    store = _history_store()
    backend = str(getattr(store, "backend_name", "duckdb") or "duckdb").strip().lower()
    db_path = str(getattr(store, "db_path", "") or "")
    parts = [
        f"config={get_config_source()}",
        f"storage={backend}",
        f"db={db_path} ({_path_scope_label(db_path)})",
        f"live_chart={_live_kline_renderer()}",
        f"replay_chart={_replay_kline_renderer()}",
        f"benchmark_chart={_benchmark_renderer()}",
    ]
    if backend == "duckdb":
        parquet_root = str(getattr(store, "parquet_root", "") or "")
        if parquet_root:
            parts.append(f"parquet={parquet_root} ({_path_scope_label(parquet_root)})")
    return "技術線：" + " | ".join(parts)


def _store_data_source(store: object, dataset: str) -> str:
    backend = str(getattr(store, "backend_name", "duckdb") or "duckdb").strip().lower()
    return f"{backend}:{dataset}"


BACKTEST_DRILL_CODE_COLUMNS = {
    "代碼",
    "ETF代碼",
    "台股代號",
    "symbol",
    "Symbol",
    "代號",
    "股票代號",
    "證券代號",
    "標的",
}
BACKTEST_DRILL_MARKET_COLUMNS = {"市場", "market", "Market", "交易所", "exchange", "Exchange"}
BACKTEST_DRILL_QUERY_KEYS = ("bt_symbol", "bt_market", "bt_autorun", "bt_src")
BACKTEST_AUTORUN_PENDING_KEY = "bt_autorun_pending"
HEATMAP_DRILL_QUERY_KEYS = ("hm_etf", "hm_name", "hm_label", "hm_open", "hm_src")
HEATMAP_HUB_SESSION_ACTIVE_KEY = "heatmap_hub_active_etf"
HEATMAP_DYNAMIC_CARD_PREFIX = "ETF熱力圖:"
HEATMAP_CARD_BLOCKLIST = {"00993A"}
HEATMAP_STATIC_CARD_CODES = {"00910", "00935", "00735", "0050", "0052"}
HEATMAP_STATIC_PAGE_BY_CODE = {
    "00910": "00910 熱力圖",
    "00935": "00935 熱力圖",
    "00735": "00735 熱力圖",
    "0050": "0050 熱力圖",
    "0052": "0052 熱力圖",
}


def _heatmap_page_key_for_code(code: str) -> str:
    code_text = str(code or "").strip().upper()
    return HEATMAP_STATIC_PAGE_BY_CODE.get(code_text, f"{HEATMAP_DYNAMIC_CARD_PREFIX}{code_text}")


def _normalize_heatmap_etf_code(value: object) -> str:
    code = str(value or "").strip().upper()
    if not code:
        return ""
    return code if re.fullmatch(r"\d{4,6}[A-Z]?", code) else ""


def _clean_heatmap_name_for_query(value: object) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    return text.replace("&", "＆").replace("?", "").replace("#", "")


def _normalize_market_tag_for_drill(value: object) -> str:
    token = str(value or "").strip().upper()
    if not token:
        return ""
    if ("OTC" in token) or ("TWO" in token) or ("上櫃" in token):
        return "OTC"
    if ("US" in token) or ("NYSE" in token) or ("NASDAQ" in token) or ("美股" in token):
        return "US"
    if (
        ("TW" in token)
        or ("TWSE" in token)
        or ("TSE" in token)
        or ("上市" in token)
        or ("台股" in token)
    ):
        return "TW"
    return ""


def _decorate_tw_etf_name_heatmap_links(
    df: pd.DataFrame,
    *,
    code_col: str = "代碼",
    name_col: str = "ETF",
    src: str = "all_types_table",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df, {}
    if (name_col not in df.columns) or (code_col not in df.columns):
        return df, {}
    out = df.copy()
    links: list[str | None] = []
    linked_count = 0
    for _, row in out.iterrows():
        code = _normalize_heatmap_etf_code(row.get(code_col))
        name = str(row.get(name_col, "")).strip()
        if not code:
            links.append(None)
            continue
        links.append(build_heatmap_drill_url(code, name, src=src))
        linked_count += 1
    if linked_count <= 0:
        return out, {}
    out[name_col] = links
    return out, {
        name_col: st.column_config.LinkColumn(
            label=str(name_col),
            help="點擊 ETF 名稱可在新分頁開啟對應熱力圖（內容比照 00935 熱力圖）",
            display_text=r"hm_label=([^&]+)",
        )
    }


def _decorate_dataframe_backtest_links(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df, {}
    code_cols = [col for col in df.columns if str(col).strip() in BACKTEST_DRILL_CODE_COLUMNS]
    if not code_cols:
        return df, {}
    market_col = next(
        (col for col in df.columns if str(col).strip() in BACKTEST_DRILL_MARKET_COLUMNS), None
    )
    market_series = (
        df[market_col] if market_col in df.columns else pd.Series(index=df.index, dtype=str)
    )
    out = df.copy()
    link_config: dict[str, object] = {}
    for col in code_cols:
        links: list[str | None] = []
        linked_count = 0
        col_series = out[col]
        for row_idx, value in col_series.items():
            symbol, default_market = parse_drill_symbol(value)
            if not symbol or symbol.startswith("^"):
                links.append(None)
                continue
            row_market = _normalize_market_tag_for_drill(
                market_series.get(row_idx, "") if hasattr(market_series, "get") else ""
            )
            market = row_market or default_market
            if market not in {"TW", "OTC", "US"}:
                inferred = _infer_market_target_from_symbols([symbol]) if symbol else None
                market = (
                    inferred
                    if inferred in {"TW", "OTC", "US"}
                    else (default_market if default_market in {"TW", "OTC", "US"} else "TW")
                )
            links.append(build_backtest_drill_url(symbol=symbol, market=market))
            linked_count += 1
        if linked_count <= 0:
            continue
        out[col] = links
        link_config[str(col)] = st.column_config.LinkColumn(
            label=str(col),
            help="點擊代碼可帶入回測工作台並自動執行回測",
            display_text=r"bt_symbol=([^&]+)",
            max_chars=20,
        )
    return out, link_config


_ORIGINAL_ST_DATAFRAME = st.dataframe


def _tw_etf_precision_column_config(frame: pd.DataFrame) -> dict[str, object]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    cfg: dict[str, object] = {}
    if "管理費(%)" in frame.columns:
        cfg["管理費(%)"] = st.column_config.NumberColumn("管理費(%)", format="%.2f")
    if "ETF規模(億)" in frame.columns:
        cfg["ETF規模(億)"] = st.column_config.NumberColumn("ETF規模(億)", format="%d")
    if "2025績效(%)" in frame.columns:
        cfg["2025績效(%)"] = st.column_config.NumberColumn("2025績效(%)", format="%.2f")
    if "輸贏大盤2025(%)" in frame.columns:
        cfg["輸贏大盤2025(%)"] = st.column_config.NumberColumn("輸贏大盤2025(%)", format="%.2f")
    if "2026YTD績效(%)" in frame.columns:
        cfg["2026YTD績效(%)"] = st.column_config.NumberColumn("2026YTD績效(%)", format="%.2f")
    if "輸贏大盤2026YTD(%)" in frame.columns:
        cfg["輸贏大盤2026YTD(%)"] = st.column_config.NumberColumn(
            "輸贏大盤2026YTD(%)", format="%.2f"
        )
    return cfg


def _dataframe_with_backtest_drilldown(data: Any = None, *args, **kwargs):
    opts = dict(kwargs)
    disable_drilldown = bool(opts.pop("disable_backtest_drilldown", False))
    frame = data
    if isinstance(frame, pd.DataFrame) and (not disable_drilldown):
        precision_config = _tw_etf_precision_column_config(frame)
        frame, auto_config = _decorate_dataframe_backtest_links(frame)
        merged_config: dict[str, object] = {}
        existing = opts.get("column_config")
        if isinstance(existing, dict):
            merged_config.update(existing)
        if precision_config:
            for col_name, cfg in precision_config.items():
                if col_name not in merged_config:
                    merged_config[col_name] = cfg
        if auto_config:
            for col_name, cfg in auto_config.items():
                if col_name not in merged_config:
                    merged_config[col_name] = cfg
        if merged_config:
            opts["column_config"] = merged_config
    return _ORIGINAL_ST_DATAFRAME(frame, *args, **opts)


st.dataframe = _dataframe_with_backtest_drilldown  # type: ignore[assignment]


def _query_param_first(name: str) -> str:
    try:
        params = st.query_params
        value = params.get(name)
        if isinstance(value, list):
            return str(value[0]).strip() if value else ""
        return str(value or "").strip()
    except Exception:
        try:
            params_legacy = st.experimental_get_query_params()
            values = params_legacy.get(name, [])
            if isinstance(values, list):
                return str(values[0]).strip() if values else ""
            return str(values or "").strip()
        except Exception:
            return ""


def _clear_query_params(keys: tuple[str, ...]) -> None:
    try:
        params = st.query_params
        for key in keys:
            if key in params:
                del params[key]
        return
    except Exception:
        pass
    try:
        params_legacy = dict(st.experimental_get_query_params())
        changed = False
        for key in keys:
            if key in params_legacy:
                params_legacy.pop(key, None)
                changed = True
        if changed:
            st.experimental_set_query_params(**params_legacy)
    except Exception:
        return


def _consume_backtest_drilldown_query() -> None:
    symbol_raw = _query_param_first("bt_symbol")
    if not symbol_raw:
        return
    symbol, default_market = parse_drill_symbol(symbol_raw)
    if not symbol:
        _clear_query_params(BACKTEST_DRILL_QUERY_KEYS)
        return
    market_hint = _normalize_market_tag_for_drill(_query_param_first("bt_market"))
    market = market_hint or default_market
    if market not in {"TW", "OTC", "US"}:
        inferred = _infer_market_target_from_symbols([symbol]) if symbol else None
        market = inferred if inferred in {"TW", "OTC", "US"} else "TW"
    st.session_state["active_page"] = "回測工作台"
    st.session_state[BT_KEYS.symbol] = str(symbol).strip().upper()
    st.session_state[BT_KEYS.mode] = "單一標的"
    st.session_state[BT_KEYS.market] = market
    autorun_flag = _query_param_first("bt_autorun").strip().lower()
    if autorun_flag in {"1", "true", "yes", "y", "on", "run"}:
        st.session_state[BACKTEST_AUTORUN_PENDING_KEY] = {
            "symbol": str(symbol).strip().upper(),
            "market": market,
        }
    _clear_query_params(BACKTEST_DRILL_QUERY_KEYS)


# JS|@st.cache_data(ttl=21600, show_spinner=False)
def _load_heatmap_hub_entries(*, pinned_only: bool = False) -> list[Any]:
    store = _history_store()
    loader = getattr(store, "list_heatmap_hub_entries", None)
    if not callable(loader):
        return []
    try:
        rows = loader(pinned_only=bool(pinned_only))
    except Exception:
        return []
    return list(rows) if isinstance(rows, list) else []


def _upsert_heatmap_hub_entry(
    *, etf_code: str, etf_name: str, opened: bool = False, pin_as_card: bool | None = None
) -> None:
    code = _normalize_heatmap_etf_code(etf_code)
    if not code:
        return
    if code in HEATMAP_CARD_BLOCKLIST:
        return
    store = _history_store()
    writer = getattr(store, "upsert_heatmap_hub_entry", None)
    if not callable(writer):
        return
    try:
        writer(
            etf_code=code,
            etf_name=str(etf_name or code).strip(),
            opened=bool(opened),
            pin_as_card=pin_as_card,
        )
    except Exception:
        return


def _set_heatmap_hub_pin(*, etf_code: str, pin_as_card: bool) -> bool:
    code = _normalize_heatmap_etf_code(etf_code)
    if not code:
        return False
    if code in HEATMAP_CARD_BLOCKLIST:
        return False
    store = _history_store()
    setter = getattr(store, "set_heatmap_hub_pin", None)
    if not callable(setter):
        return False
    try:
        return bool(setter(etf_code=code, pin_as_card=bool(pin_as_card)))
    except Exception:
        return False


def _session_active_heatmap() -> tuple[str, str]:
    payload = st.session_state.get(HEATMAP_HUB_SESSION_ACTIVE_KEY)
    if not isinstance(payload, dict):
        return "", ""
    code = _normalize_heatmap_etf_code(payload.get("code", ""))
    if not code:
        return "", ""
    name = _clean_heatmap_name_for_query(payload.get("name", "")) or code
    return code, name


def _dynamic_heatmap_cards() -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    for row in _load_heatmap_hub_entries(pinned_only=True):
        code = _normalize_heatmap_etf_code(getattr(row, "etf_code", ""))
        if not code or code in HEATMAP_CARD_BLOCKLIST or code in HEATMAP_STATIC_CARD_CODES:
            continue
        seen_codes.add(code)
        name = str(getattr(row, "etf_name", "") or code).strip()
        cards.append(
            {
                "key": f"{HEATMAP_DYNAMIC_CARD_PREFIX}{code}",
                "desc": f"{code} {name} 成分股熱力圖（動態釘選）",
            }
        )
    active_code, active_name = _session_active_heatmap()
    if (
        active_code
        and active_code not in seen_codes
        and active_code not in HEATMAP_CARD_BLOCKLIST
        and active_code not in HEATMAP_STATIC_CARD_CODES
    ):
        cards.append(
            {
                "key": f"{HEATMAP_DYNAMIC_CARD_PREFIX}{active_code}",
                "desc": f"{active_code} {active_name} 成分股熱力圖（當前開啟）",
            }
        )
    return cards


def _dynamic_heatmap_page_renderers() -> dict[str, Any]:
    renderers: dict[str, Any] = {}
    rows = list(_load_heatmap_hub_entries(pinned_only=True))
    active_code, active_name = _session_active_heatmap()
    if active_code:
        rows.append(
            type(
                "_ActiveHeatmapEntry",
                (),
                {"etf_code": active_code, "etf_name": active_name},
            )()
        )
    built_keys: set[str] = set()
    for row in rows:
        code = _normalize_heatmap_etf_code(getattr(row, "etf_code", ""))
        if not code or code in HEATMAP_CARD_BLOCKLIST or code in HEATMAP_STATIC_CARD_CODES:
            continue
        name = str(getattr(row, "etf_name", "") or code).strip() or code
        key = f"{HEATMAP_DYNAMIC_CARD_PREFIX}{code}"
        if key in built_keys:
            continue
        built_keys.add(key)

        def _render_dynamic_heatmap(_code: str = code, _name: str = name):
            _render_tw_etf_heatmap_view(_code, page_desc=_name, auto_run_if_missing=True)

        renderers[key] = _render_dynamic_heatmap
    return renderers


def _consume_heatmap_drilldown_query() -> None:
    code_raw = _query_param_first("hm_etf")
    if not code_raw:
        return
    code = _normalize_heatmap_etf_code(code_raw)
    if not code:
        _clear_query_params(HEATMAP_DRILL_QUERY_KEYS)
        return
    if code in HEATMAP_CARD_BLOCKLIST:
        _clear_query_params(HEATMAP_DRILL_QUERY_KEYS)
        st.warning(f"{code} 熱力圖卡片已停用。")
        return
    name = _query_param_first("hm_name")
    if name:
        name = unquote(name)
    if not name:
        name = _query_param_first("hm_label")
    name = _clean_heatmap_name_for_query(name) or code
    st.session_state["active_page"] = _heatmap_page_key_for_code(code)
    st.session_state[HEATMAP_HUB_SESSION_ACTIVE_KEY] = {"code": code, "name": name}
    opened_flag = _query_param_first("hm_open").strip().lower()
    opened = opened_flag in {"1", "true", "yes", "y", "on", "open"}
    _upsert_heatmap_hub_entry(etf_code=code, etf_name=name, opened=opened)
    _clear_query_params(HEATMAP_DRILL_QUERY_KEYS)


BACKTEST_REPLAY_SCHEMA_VERSION = 3


def _sync_symbols_history(
    store: HistoryStore,
    *,
    market: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
    parallel: bool = True,
    max_workers: int = 6,
) -> tuple[dict[str, object], list[str]]:
    return orchestrated_sync_symbols_history(
        store=store,
        market=market,
        symbols=symbols,
        start=start,
        end=end,
        parallel=parallel,
        max_workers=max_workers,
    )


def _bars_need_backfill(bars: pd.DataFrame, *, start: datetime, end: datetime) -> bool:
    return orchestrated_bars_need_backfill(bars, start=start, end=end)


@st.cache_data(ttl=21600, show_spinner=False)
def _fetch_tw_exchange_symbol_lists() -> tuple[list[str], list[str]]:
    import requests

    twse_codes: set[str] = set()
    tpex_codes: set[str] = set()

    try:
        resp = requests.get(
            "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=12
        )
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("Code", "")).strip().upper()
                if looks_like_tw_symbol(code):
                    twse_codes.add(code)
    except Exception:
        pass

    try:
        resp = requests.get(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes", timeout=12
        )
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("SecuritiesCompanyCode", "")).strip().upper()
                if looks_like_tw_symbol(code):
                    tpex_codes.add(code)
    except Exception:
        pass

    return sorted(twse_codes), sorted(tpex_codes)


def _infer_tw_symbol_exchanges(symbols: list[str]) -> dict[str, str]:
    twse_list, tpex_list = _fetch_tw_exchange_symbol_lists()
    twse_set = set(twse_list)
    tpex_set = set(tpex_list)
    out: dict[str, str] = {}
    for symbol in symbols:
        code = str(symbol or "").strip().upper()
        if not looks_like_tw_symbol(code):
            continue
        in_twse = code in twse_set
        in_tpex = code in tpex_set
        if in_twse and not in_tpex:
            out[code] = "TW"
        elif in_tpex and not in_twse:
            out[code] = "OTC"
        elif in_twse and in_tpex:
            out[code] = "TW"
        elif code.startswith("00"):
            # Most Taiwan ETF codes are listed on TWSE.
            out[code] = "TW"
    return out


def _infer_market_target_from_symbols(symbols: list[str]) -> str | None:
    if not symbols:
        return None
    tokens = [str(sym or "").strip().upper() for sym in symbols if str(sym or "").strip()]
    if not tokens:
        return None

    if all(token.endswith(".TW") for token in tokens):
        return "TW"
    if all(token.endswith(".TWO") for token in tokens):
        return "OTC"

    tw_like_symbols = [token for token in tokens if looks_like_tw_symbol(token)]
    if tw_like_symbols:
        if len(tw_like_symbols) != len(tokens):
            return None
        inferred_map = _infer_tw_symbol_exchanges(tw_like_symbols)
        inferred_tags = [
            inferred_map.get(sym)
            for sym in tw_like_symbols
            if inferred_map.get(sym) in {"TW", "OTC"}
        ]
        if not inferred_tags:
            return None
        return "OTC" if all(tag == "OTC" for tag in inferred_tags) else "TW"

    if all(looks_like_us_symbol(token) for token in tokens):
        return "US"
    return None


def _persist_live_tick_buffer(
    *,
    store: HistoryStore,
    symbol: str,
    market: str,
    quote,
    buffer_key: str,
    flush_key: str,
    batch_size: int = 24,
    flush_interval_sec: int = 5,
):
    if quote is None or quote.price is None:
        return
    ts = pd.Timestamp(quote.ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    row = {
        "ts": ts.isoformat(),
        "price": float(quote.price),
        "cum_volume": float(quote.volume or 0.0),
        "source": str(getattr(quote, "source", "") or "unknown"),
    }

    buffer_raw = st.session_state.get(buffer_key, [])
    buffer = list(buffer_raw) if isinstance(buffer_raw, list) else []
    if not buffer or any(buffer[-1].get(k) != row.get(k) for k in ("ts", "price", "cum_volume")):
        buffer.append(row)

    now_ts = datetime.now(tz=timezone.utc).timestamp()
    last_flush = float(st.session_state.get(flush_key, 0.0) or 0.0)
    should_flush = bool(buffer) and (
        len(buffer) >= max(1, int(batch_size)) or now_ts - last_flush >= float(flush_interval_sec)
    )
    if should_flush:
        try:
            store.save_intraday_ticks(symbol=symbol, market=market, ticks=buffer)
            buffer = []
            st.session_state[flush_key] = now_ts
        except Exception as exc:
            st.warning(f"即時資料寫入失敗（已略過本輪）：{exc}")
    st.session_state[buffer_key] = buffer


def _load_intraday_bars_from_sqlite(
    *,
    store: HistoryStore,
    symbol: str,
    market: str,
    keep_minutes: int,
) -> pd.DataFrame:
    end = datetime.now(tz=timezone.utc)
    start = end - pd.Timedelta(minutes=max(30, int(keep_minutes)))
    try:
        raw = store.load_intraday_ticks(symbol=symbol, market=market, start=start, end=end)
    except Exception:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    ticks = raw.reset_index().rename(columns={"ts_utc": "ts"})
    ticks["ts"] = pd.to_datetime(ticks.get("ts"), utc=True, errors="coerce")
    ticks["price"] = pd.to_numeric(ticks.get("price"), errors="coerce")
    ticks["cum_volume"] = pd.to_numeric(ticks.get("cum_volume"), errors="coerce").fillna(0.0)
    ticks = ticks.dropna(subset=["ts", "price"]).sort_values("ts")
    if ticks.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return TwMisProvider.build_bars_from_ticks(ticks[["ts", "price", "cum_volume"]])


DAILY_STRATEGY_OPTIONS = [
    "buy_hold",
    "sma_trend_filter",
    "donchian_breakout",
    "sma_cross",
    "ema_cross",
    "rsi_reversion",
    "macd_trend",
]

STRATEGY_LABELS = {
    "buy_hold": "buy_hold（買進持有）",
    "sma_trend_filter": "sma_trend_filter（日K 趨勢濾網）",
    "donchian_breakout": "donchian_breakout（日K 通道突破）",
    "sma_cross": "sma_cross（短中期均線交叉）",
    "ema_cross": "ema_cross（EMA 交叉）",
    "rsi_reversion": "rsi_reversion（RSI 反轉）",
    "macd_trend": "macd_trend（MACD 趨勢）",
}

STRATEGY_DESC = {
    "buy_hold": "不做擇時，直接持有到區間結束，適合當基準線。",
    "sma_trend_filter": "以 Fast/Slow + 長期趨勢均線過濾，降低日K短期雜訊。",
    "donchian_breakout": "突破近 N 日高點進場、跌破近 M 日低點出場，偏趨勢追蹤。",
    "sma_cross": "短中期 SMA 交叉，反應較快但也較容易來回洗。",
    "ema_cross": "EMA 交叉，對近期價格更敏感。",
    "rsi_reversion": "RSI 低檔買、高檔賣，偏均值回歸。",
    "macd_trend": "MACD 快慢線方向判斷趨勢。",
}

ETF_00910_COMPANY_BRIEFS: dict[str, str] = {
    "099320.KS": "韓國衛星製造商，主要做地球觀測衛星系統與整合服務。",
    "PL.US": "Planet 提供地球觀測影像資料，核心是高頻率衛星遙測與數據平台服務。",
    "ASTS.US": "AST SpaceMobile 目標用低軌衛星直接連手機，提供太空行動通訊網路。",
    "GSAT.US": "Globalstar 經營衛星通訊網路，提供語音/數據連線與物聯網服務。",
    "9412.JP": "SKY Perfect JSAT 經營衛星通訊與付費影視，具日本大型衛星營運規模。",
    "SATS.US": "EchoStar 聚焦衛星通訊與相關電信基礎設施，服務企業與消費市場。",
    "047810.KS": "Korea Aerospace 為韓國航太與國防製造商，涵蓋軍機與航太系統。",
    "RKLB.US": "Rocket Lab 提供小型衛星發射與太空系統製造，屬商業航太成長公司。",
    "VSAT.US": "Viasat 提供衛星寬頻與機上網路服務，面向民航、企業與政府客戶。",
    "ESE.US": "ESCO Technologies 主要做航太/工業測試與工程設備，偏關鍵零組件供應。",
    "6271.TW": "同欣電以感測與車用電子封裝模組為主，供應高階影像與通訊應用。",
    "6271": "同欣電以感測與車用電子封裝模組為主，供應高階影像與通訊應用。",
    "PKE.US": "Park Aerospace 提供航太用複合材料與結構材料，客戶多在航空與國防。",
    "6285.TW": "啟碁科技專注網通與無線通訊設備，產品涵蓋基地台與衛星通訊終端。",
    "6285": "啟碁科技專注網通與無線通訊設備，產品涵蓋基地台與衛星通訊終端。",
    "6486.JP": "Eagle Industry 為精密密封與機械零件廠，應用於汽車與航太工業。",
    "7011.JP": "三菱重工涵蓋能源、航太與國防，是日本大型重工與航太體系公司。",
    "2455.TW": "全新為砷化鎵射頻元件供應商，產品常見於通訊與高頻應用。",
    "2455": "全新為砷化鎵射頻元件供應商，產品常見於通訊與高頻應用。",
    "RDW.US": "Redwire 提供太空基礎設施與在軌技術，服務商業與政府太空任務。",
    "IRDM.US": "Iridium 經營全球低軌衛星通訊網路，主打語音/數據與物聯網連線。",
    "GOGO.US": "Gogo 主要提供航空機上網路與連線服務，客戶以商務航空為主。",
}


def _company_brief_for_00910(symbol: str, name: str, market_tag: str) -> str:
    key = str(symbol or "").strip().upper()
    if key in ETF_00910_COMPANY_BRIEFS:
        return ETF_00910_COMPANY_BRIEFS[key]
    if "." in key:
        base = key.split(".")[0]
        if base in ETF_00910_COMPANY_BRIEFS:
            return ETF_00910_COMPANY_BRIEFS[base]
    market = str(market_tag or "").strip().upper()
    if market == "TW":
        return "台灣供應鏈公司，主要布局通訊、半導體或航太相關電子零組件。"
    if market == "US":
        return "美國航太/衛星相關公司，重點在衛星通訊、太空系統或資料服務。"
    if market == "JP":
        return "日本航太或工業技術公司，主要參與衛星/航太與高階製造供應鏈。"
    if market == "KS":
        return "韓國航太/衛星供應鏈公司，涵蓋衛星製造與相關系統整合能力。"
    return f"{name} 為 00910 成分股，主要屬於航太衛星主題供應鏈。"


ETF_INDEX_METHOD_SUMMARY: dict[str, dict[str, object]] = {
    "0050": {
        "title": "0050 指數編製標準（官方摘要）",
        "index_name": "臺灣50指數（FTSE TWSE Taiwan 50 Index）",
        "provider": "臺灣證券交易所與 FTSE 合作編製",
        "rules": [
            "標的指數由臺灣證交所上市股票中，挑選總市值最大的 50 家公司作為成分股。",
            "成分股審核與調整生效時點：3、6、9、12 月第 3 個星期五後的下一個交易日。",
        ],
        "sources": [
            (
                "元大投信 0050 基本資訊（Index Profile / Index Methodology）",
                "https://www.yuantaetfs.com/product/detail/0050/Basic_information",
            ),
        ],
    },
    "0052": {
        "title": "0052 指數編製標準（官方摘要）",
        "index_name": "臺灣資訊科技指數（FTSE TWSE Taiwan Technology Index）",
        "provider": "臺灣證券交易所與 FTSE Russell 合作編製",
        "rules": [
            "成分股母體來自 FTSE 臺灣 50 指數與 FTSE 臺灣中型 100 指數。",
            "依 FTSE ICB 產業分類，納入資訊科技（Technology）產業群公司。",
            "採自由流通市值加權，成分股檔數不固定（無固定上限）。",
            "定期審核頻率為每年 3、6、9、12 月（季度調整）。",
        ],
        "sources": [
            (
                "臺灣指數公司：FTSE TWSE Taiwan Technology Index（TWIT）",
                "https://taiwanindex.com.tw/en/indexes/TWIT",
            ),
            (
                "富邦投信 0052 專頁（追蹤指數與調整頻率）",
                "https://www.fubon.com/asset-management/ph/0052/index.html",
            ),
        ],
    },
    "00993A": {
        "title": "00993A 編製/管理規則（官方摘要）",
        "index_name": "主動式 ETF（非被動複製單一指數）",
        "provider": "安聯投信（主動管理）；資訊揭露於 TWSE ETF e添富",
        "rules": [
            "00993A 屬主動式 ETF，由基金經理團隊主動選股與調整部位，不採被動複製追蹤方式。",
            "TWSE 新上市資訊中，追蹤指數/編製機構/定審頻率欄位揭露為「-」，屬主動管理商品。",
            "ETF e添富商品頁另揭露「標的指數：臺灣證券交易所發行量加權股價指數」，可作市場比較參考。",
            "實際投資策略與持股調整仍以基金公開說明書與基金公司最新公告為準。",
        ],
        "sources": [
            (
                "TWSE ETF e添富：00993A 新上市資訊",
                "https://www.twse.com.tw/zh/ETFortune/newsDetail/8fbe8fd8-53ec-11f0-b0e4-0242ac110003",
            ),
            (
                "TWSE ETF e添富：00993A 商品資訊",
                "https://www.twse.com.tw/zh/ETFortune/etfInfo/00993A",
            ),
        ],
    },
    "00935": {
        "title": "00935 指數編製標準（官方摘要）",
        "index_name": "臺灣指數公司特選臺灣上市上櫃 FactSet 創新科技 50 指數",
        "provider": "臺灣指數公司（TWSE e添富新上市資訊揭露）",
        "rules": [
            "成分股條件包含市值須達新台幣 50 億元以上。",
            "需符合 FactSet RBICS Level 6 創新科技次產業，且相關營收占比須大於 50%。",
            "要求研發費用投入門檻，並排除研發投入比例排名最差 25% 企業。",
            "定審頻率為半年，成分股檔數 50 檔。",
        ],
        "sources": [
            (
                "TWSE ETF e添富：00935 新上市資訊",
                "https://www.twse.com.tw/zh/ETFortune/newsDetail/ff8080818b7e232e018b83ab7de9002b",
            ),
        ],
    },
    "00735": {
        "title": "00735 指數編製標準（官方摘要）",
        "index_name": "臺韓資訊科技指數",
        "provider": "臺灣指數公司（ETF e添富商品頁揭露）",
        "rules": [
            "追蹤臺灣與韓國資訊科技主題股票，跨市場布局台韓科技供應鏈。",
            "成分股調整與檔數以臺灣指數公司與基金公司最新公告為準。",
            "熱力圖頁會將完整成分股（含海外）與台股可回測子集合分開呈現。",
        ],
        "sources": [
            (
                "TWSE ETF e添富：00735 商品資訊",
                "https://www.twse.com.tw/zh/ETFortune/etfInfo/00735",
            ),
            (
                "TWSE ETF e添富：ETF 商品總覽（含 00735 指數名稱）",
                "https://www.twse.com.tw/zh/ETFortune/products",
            ),
            (
                "臺灣指數公司：臺韓資訊科技指數定期審核公告（00735）",
                "https://taiwanindex.com.tw/news/366",
            ),
        ],
    },
    "00910": {
        "title": "00910 指數編製標準（官方摘要）",
        "index_name": "Solactive 太空衛星指數（Solactive Aerospace and Satellite Index）",
        "provider": "Solactive（由第一金投信官網揭露編製方法）",
        "rules": [
            "依 Solactive 國家分類，先鎖定已開發國家加上台灣、韓國，再篩選市值與流動性佳標的。",
            "透過 NLP 系統 ARTIS 評分與排序，篩選與太空及衛星產業高度投入的企業。",
            "精選全球 30 檔企業，涵蓋太空旅遊、太空技術、衛星通訊、衛星零組件。",
            "每年 2 月、8 月調整（官網註記指數調整日為 2 月與 8 月最後一個營業日）。",
        ],
        "sources": [
            (
                "第一金投信 00910 專頁（標的指數編製方法）",
                "https://www.fsitc.com.tw/act/202206_asetf/",
            ),
        ],
    },
}

ETF_FORCE_GLOBAL_HEATMAP = {"00910", "00735"}


def _render_etf_index_method_summary(etf_code: str):
    info = ETF_INDEX_METHOD_SUMMARY.get(str(etf_code or "").strip().upper())
    if not info:
        return
    st.markdown(f"#### {str(info.get('title', '指數編製標準（官方摘要）'))}")

    lines: list[str] = []
    index_name = str(info.get("index_name", "")).strip()
    provider = str(info.get("provider", "")).strip()
    if index_name:
        lines.append(f"- 標的指數：{index_name}")
    if provider:
        lines.append(f"- 編製機構：{provider}")
    rules = info.get("rules", [])
    if isinstance(rules, list):
        lines.extend([f"- {str(item)}" for item in rules if str(item).strip()])
    if lines:
        st.markdown("\n".join(lines))

    sources = info.get("sources", [])
    if isinstance(sources, list) and sources:
        source_lines = [
            f"- [{str(label)}]({str(url)})"
            for label, url in sources
            if str(label).strip() and str(url).strip()
        ]
        if source_lines:
            st.caption("官方來源：")
            st.markdown("\n".join(source_lines))
    st.caption("以上為官方公開資訊摘要，實際請以最新公開說明書與指數編製機構公告為準。")


TW_COMPANY_BRIEF_OVERRIDES: dict[str, str] = {
    "2330": "台積電是晶圓代工龍頭，主要為全球 IC 設計公司提供先進製程製造服務。",
    "2454": "聯發科是 IC 設計公司，核心在手機與通訊晶片、AIoT 與車用平台。",
    "2317": "鴻海是電子代工與製造服務大廠，業務涵蓋消費電子、伺服器與電動車。",
    "2308": "台達電聚焦電源管理與節能解決方案，布局工控、自動化與資料中心。",
    "2382": "廣達是筆電與伺服器代工大廠，近年在 AI 伺服器供應鏈角色提升。",
    "2881": "富邦金為金控公司，主要業務包含銀行、保險、證券與資產管理。",
    "2882": "國泰金為金控公司，核心涵蓋壽險、銀行、證券與整合金融服務。",
    "2891": "中信金為金控公司，重點在銀行、信用卡與跨境金融服務。",
    "2886": "兆豐金為金控公司，銀行與外匯業務比重高，具企業金融優勢。",
    "2412": "中華電為台灣電信商，提供行動通訊、寬頻網路與企業數位服務。",
    "3034": "聯詠是顯示與影像相關 IC 設計公司，產品聚焦驅動 IC 與 SoC。",
    "2303": "聯電是晶圓代工廠，主要服務成熟製程與多元應用晶片生產。",
    "3017": "奇鋐主力為散熱模組與熱管理方案，受惠伺服器與 AI 基礎建設需求。",
    "3653": "健策提供散熱與精密金屬零組件，產品應用於伺服器與高階運算平台。",
    "6669": "緯穎聚焦雲端伺服器與資料中心解決方案，是大型雲端客戶供應商。",
    "1216": "統一以食品、通路與民生消費品牌為主，屬內需型大型企業。",
    "1301": "台塑為石化與材料公司，業務涵蓋塑化產品與相關工業原料。",
    "1303": "南亞主力在化工材料、電子材料與塑膠加工，應用範圍廣。",
    "2002": "中鋼是台灣主要鋼鐵公司，提供基礎工業與製造業所需鋼材。",
    "2603": "長榮是貨櫃航運公司，主要收入來自全球海運運輸業務。",
    "6505": "台塑化以煉油與石化產品為核心，提供燃料與石化中下游原料。",
    "6510": "精測主要做晶圓測試介面板與探針卡等測試方案，服務高階晶片測試需求。",
}


TW_INDUSTRY_INFO: dict[str, tuple[str, str]] = {
    "01": ("水泥工業", "主要生產水泥與建材，營運與基礎建設及景氣循環相關。"),
    "02": ("食品工業", "主要經營食品與民生消費品，需求相對穩定。"),
    "03": ("塑膠工業", "主要提供塑化材料與製品，受原料價格與景氣影響。"),
    "04": ("紡織纖維", "主要生產紡織纖維與布料，應用在成衣與工業材料。"),
    "05": ("電機機械", "主要提供機械設備與自動化產品，受資本支出循環影響。"),
    "06": ("電器電纜", "主要經營電線電纜與電力相關零組件。"),
    "08": ("玻璃陶瓷", "主要生產玻璃與陶瓷材料，應用於建材與工業。"),
    "09": ("造紙工業", "主要生產紙漿與紙類產品，受原料與需求循環影響。"),
    "10": ("鋼鐵工業", "主要提供鋼材與金屬材料，與製造業與建設需求連動。"),
    "11": ("橡膠工業", "主要生產橡膠材料與輪胎等產品。"),
    "12": ("汽車工業", "主要經營汽車整車或零組件供應鏈。"),
    "14": ("建材營造", "主要從事建設、營造與不動產相關業務。"),
    "15": ("航運業", "主要提供海運、空運或物流服務，受運價與景氣波動影響。"),
    "16": ("觀光餐旅", "主要經營旅遊、飯店與餐飲服務。"),
    "17": ("金融保險", "主要提供銀行、保險、證券與資產管理等金融服務。"),
    "18": ("貿易百貨", "主要經營通路零售、百貨與貿易業務。"),
    "20": ("其他", "主要業務多元，涵蓋不同產業與應用場景。"),
    "21": ("化學工業", "主要生產化學品與材料，應用在工業與消費市場。"),
    "22": ("生技醫療", "主要從事製藥、醫材或醫療相關服務。"),
    "23": ("油電燃氣", "主要提供能源、燃料或公用事業服務。"),
    "24": ("半導體業", "主要從事 IC 設計、晶圓製造、封測或測試相關業務。"),
    "25": ("電腦及週邊設備業", "主要提供伺服器、電腦與相關硬體設備。"),
    "26": ("光電業", "主要從事面板、光學元件與光電模組相關產品。"),
    "27": ("通信網路業", "主要提供通訊設備、網通產品與連網解決方案。"),
    "28": ("電子零組件業", "主要供應電子零件、模組與關鍵材料。"),
    "29": ("電子通路業", "主要經營電子零組件代理與通路服務。"),
    "30": ("資訊服務業", "主要提供軟體、資訊系統與數位服務。"),
    "31": ("其他電子業", "主要為電子產業其他細分領域與應用。"),
    "32": ("文化創意業", "主要從事內容、設計與文化創意相關業務。"),
    "33": ("農業科技", "主要經營農業、食品原料與相關科技應用。"),
}


def _company_brief_for_tw(symbol: str, name: str, etf_code: str, industry_code: str = "") -> str:
    code = str(symbol or "").strip().upper()
    title = str(name or code).strip() or code
    if code in TW_COMPANY_BRIEF_OVERRIDES:
        return TW_COMPANY_BRIEF_OVERRIDES[code]

    ind = str(industry_code or "").strip()
    if ind in TW_INDUSTRY_INFO:
        industry_name, industry_desc = TW_INDUSTRY_INFO[ind]
        return f"{title} 屬{industry_name}，{industry_desc}"

    name_text = title
    if any(token in name_text for token in ("金控", "銀行", "證券", "保險")):
        return f"{title} 主要屬金融產業，核心在銀行、保險、證券或資產管理等金融服務。"
    if any(token in name_text for token in ("電信",)):
        return f"{title} 主要提供通訊與網路服務，屬電信與數位基礎建設相關公司。"
    if any(token in name_text for token in ("航運", "海運", "航空")):
        return f"{title} 主要從事運輸與物流業務，營運與景氣循環及運價變動高度相關。"
    if any(token in name_text for token in ("塑", "鋼", "化", "水泥")):
        return f"{title} 屬傳產材料或基礎工業公司，營運與原物料及景氣循環關聯較高。"
    if any(token in name_text for token in ("電", "科", "晶", "半導體", "光", "網")):
        return f"{title} 屬電子科技供應鏈公司，主要提供晶片、零組件或系統產品。"
    return f"{title} 為 {etf_code} 成分股，主要為台股大型或中大型企業。"


def _split_brief_to_points(text: str) -> tuple[str, str]:
    raw = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not raw:
        return "", ""
    parts = [p.strip() for p in re.split(r"[，。；;]", raw) if p.strip()]
    if not parts:
        return raw, ""
    if len(parts) == 1:
        return parts[0], ""
    core = parts[0]
    product = "；".join(parts[1:3])
    return core, product


def _full_table_height(df: pd.DataFrame, *, min_height: int = 220, max_height: int = 3200) -> int:
    row_count = int(len(df.index)) if isinstance(df, pd.DataFrame) else 0
    estimated = 40 + row_count * 36
    return max(min_height, min(max_height, estimated))


def _render_tw_constituent_intro_table(
    *,
    etf_code: str,
    symbols: list[str],
    service: MarketDataService,
):
    etf_code_upper = str(etf_code or "").strip().upper()
    ordered_symbols: list[str] = []
    for symbol in symbols:
        code = str(symbol or "").strip().upper()
        if not code or code in ordered_symbols:
            continue
        ordered_symbols.append(code)
    if not ordered_symbols:
        return

    full_rows, _ = service.get_etf_constituents_full(etf_code, limit=None, force_refresh=False)
    weight_map: dict[str, float] = {}
    for row in full_rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("tw_code", "")).strip().upper()
        if not code:
            raw_symbol = str(row.get("symbol", "")).strip().upper()
            m = re.search(r"(\d{4})", raw_symbol)
            if m:
                code = m.group(1)
        if not code or code in weight_map:
            continue
        try:
            weight = float(row.get("weight_pct"))  # type: ignore[arg-type]
        except Exception:
            continue
        if math.isfinite(weight):
            weight_map[code] = round(weight, 2)

    ordered_for_intro = list(ordered_symbols)
    if etf_code_upper == "0050":
        original_pos = {code: idx for idx, code in enumerate(ordered_symbols)}
        ordered_for_intro = sorted(
            ordered_symbols,
            key=lambda code: (
                1 if code not in weight_map else 0,
                -float(weight_map.get(code, 0.0)),
                original_pos.get(code, 0),
            ),
        )

    name_map = service.get_tw_symbol_names(ordered_for_intro)
    industry_map = service.get_tw_symbol_industries(ordered_for_intro)
    rows: list[dict[str, object]] = []
    for idx, code in enumerate(ordered_for_intro, start=1):
        name = str(name_map.get(code, code)).strip() or code
        industry_code = str(industry_map.get(code, "")).strip()
        industry_name = TW_INDUSTRY_INFO.get(industry_code, ("", ""))[0]
        fallback = _company_brief_for_tw(code, name, etf_code_upper, industry_code)
        core, product = _split_brief_to_points(fallback)
        rows.append(
            {
                "排名": idx,
                "代號": code,
                "名稱": name,
                "權重(%)": weight_map.get(code, "—"),
                "產業": industry_name or ("其他" if industry_code else ""),
                "市場": "TW",
                "核心業務": core,
                "產品面向": product,
                "來源": "規則摘要（產業/內建）",
            }
        )
    out_df = pd.DataFrame(rows)
    st.dataframe(
        out_df,
        width="stretch",
        hide_index=True,
        height=_full_table_height(out_df),
    )


def _extract_tw_code_from_row(row: dict[str, object]) -> str:
    code = str(row.get("tw_code", "")).strip().upper()
    if re.fullmatch(r"\d{4}", code):
        return code
    raw_symbol = str(row.get("symbol", "")).strip().upper()
    if not raw_symbol:
        return ""
    if re.fullmatch(r"\d{4}", raw_symbol):
        return raw_symbol
    match = re.fullmatch(r"(\d{4})\.(TW|TWO)", raw_symbol)
    if not match:
        return ""
    return str(match.group(1)).strip().upper()


def _enrich_rows_with_tw_names(
    *,
    rows: list[dict[str, object]],
    service: MarketDataService,
) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    tw_codes: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _extract_tw_code_from_row(row)
        if code and code not in tw_codes:
            tw_codes.append(code)
    name_map = service.get_tw_symbol_names(tw_codes) if tw_codes else {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        row_out = dict(row)
        tw_code = _extract_tw_code_from_row(row_out)
        if tw_code and not str(row_out.get("tw_code", "")).strip():
            row_out["tw_code"] = tw_code

        raw_symbol = str(row_out.get("symbol", "")).strip()
        raw_name = str(row_out.get("name", "")).strip()
        mapped_name = str(name_map.get(tw_code, "")).strip() if tw_code else ""
        if tw_code and mapped_name:
            if (
                (not raw_name)
                or raw_name.upper() == raw_symbol.upper()
                or raw_name.upper() == tw_code
            ):
                row_out["name"] = mapped_name
        out_rows.append(row_out)
    return out_rows


def _render_full_constituents_if_has_overseas(
    *,
    etf_code: str,
    full_rows: list[dict[str, object]],
    source: str,
) -> bool:
    if not full_rows:
        return False
    tw_subset_count = sum(
        1 for row in full_rows if isinstance(row, dict) and bool(_extract_tw_code_from_row(row))
    )
    total_count = len(full_rows)
    overseas_count = max(0, total_count - tw_subset_count)
    if overseas_count <= 0:
        return False

    etf_text = str(etf_code or "").strip().upper()
    st.caption(
        f"{etf_text} 完整成分股（含海外）共 {total_count} 檔 | 來源：{source} | "
        f"其中台股可回測 {tw_subset_count} 檔。"
    )
    with st.expander(f"查看完整成分股（含海外，共 {total_count} 檔）", expanded=False):
        out_df = pd.DataFrame(full_rows).rename(
            columns={
                "rank": "排名",
                "symbol": "代號",
                "name": "名稱",
                "market": "市場",
                "weight_pct": "權重(%)",
                "shares": "持有股數",
                "tw_code": "台股代碼(可回測)",
            }
        )
        if "權重(%)" in out_df.columns:
            out_df["權重(%)"] = pd.to_numeric(out_df["權重(%)"], errors="coerce").round(2)
        st.dataframe(out_df, width="stretch", hide_index=True)
    return True


def _is_unresolved_symbol_name(symbol: str, name: str) -> bool:
    sym = str(symbol or "").strip().upper()
    nm = str(name or "").strip().upper()
    if not nm:
        return True
    return nm == sym


def _extract_tw_symbol_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    suffix_match = re.fullmatch(r".+\.([A-Z]+)", text)
    if suffix_match and str(suffix_match.group(1)) not in {"TW", "TWO"}:
        return ""
    text = re.sub(r"\.(TW|TWO)$", "", text)
    if re.fullmatch(r"\d{4,6}[A-Z]?", text):
        return text
    matched = re.search(r"(?<!\d)(\d{4,6}[A-Z]?)(?!\d)", text)
    return str(matched.group(1)).upper() if matched else ""


def _collect_tw_symbol_codes(values: list[object]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        code = _extract_tw_symbol_code(value)
        if not code:
            continue
        if code not in ordered:
            ordered.append(code)
    return ordered


def _format_tw_symbol_with_name(value: object, name_map: dict[str, str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    code = _extract_tw_symbol_code(raw)
    if not code:
        return raw
    name = str(name_map.get(code, "")).strip()
    if not name or _is_unresolved_symbol_name(code, name):
        return raw
    if name in raw:
        return raw
    if raw.upper() == code:
        return f"{code} {name}"
    pattern = re.compile(re.escape(code), flags=re.IGNORECASE)
    return pattern.sub(f"{code} {name}", raw, count=1)


def _symbol_watermark_text(
    *,
    symbol: object,
    market: str,
    service: MarketDataService | None = None,
) -> str:
    raw_symbol = str(symbol or "").strip()
    token = raw_symbol.upper()
    if not raw_symbol:
        return ""
    market_tag = str(market or "").strip().upper()
    if market_tag in {"TW", "OTC"}:
        code = _extract_tw_symbol_code(token)
        if not code:
            return token
        alias = re.sub(re.escape(code), "", raw_symbol, count=1, flags=re.IGNORECASE).strip()
        alias = re.sub(r"^[\s:：\-_/]+|[\s:：\-_/]+$", "", alias)
        if service is None:
            return f"{code} {alias}".strip() if alias else code
        name_map = service.get_tw_symbol_names([code])
        mapped = str(name_map.get(code, "")).strip() if isinstance(name_map, dict) else ""
        parts: list[str] = [code]
        if alias and alias.upper() != code:
            parts.append(alias)
        if mapped and not _is_unresolved_symbol_name(code, mapped):
            if all(str(mapped) != str(existing) for existing in parts):
                parts.append(mapped)
        return " ".join(parts).strip()
    return token


def _decorate_tw_symbol_columns(
    frame: pd.DataFrame,
    *,
    service: MarketDataService,
    enabled: bool,
    columns: list[str],
) -> pd.DataFrame:
    if not enabled or frame is None or frame.empty:
        return frame
    target_cols = [col for col in columns if col in frame.columns]
    if not target_cols:
        return frame
    values: list[object] = []
    for col in target_cols:
        values.extend(frame[col].tolist())
    codes = _collect_tw_symbol_codes(values)
    if not codes:
        return frame
    name_map = service.get_tw_symbol_names(codes)
    if not isinstance(name_map, dict) or not name_map:
        return frame
    out = frame.copy()
    for col in target_cols:
        out[col] = out[col].map(lambda v: _format_tw_symbol_with_name(v, name_map))
    return out


def _build_tw_name_map_from_full_rows(rows: list[dict[str, object]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _extract_tw_code_from_row(row)
        if not code:
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        if _is_unresolved_symbol_name(code, name):
            continue
        if code not in out:
            out[code] = name
    return out


def _resolve_tw_symbol_names(
    *,
    service: MarketDataService,
    symbols: list[str],
    full_rows: list[dict[str, object]] | None = None,
) -> dict[str, str]:
    ordered: list[str] = []
    for symbol in symbols:
        code = str(symbol or "").strip().upper()
        if not re.fullmatch(r"\d{4}", code):
            continue
        if code not in ordered:
            ordered.append(code)
    if not ordered:
        return {}

    api_map = service.get_tw_symbol_names(ordered)
    full_map = _build_tw_name_map_from_full_rows(full_rows or [])
    out: dict[str, str] = {}
    for code in ordered:
        resolved = str(api_map.get(code, code)).strip() or code
        if _is_unresolved_symbol_name(code, resolved):
            from_full = str(full_map.get(code, "")).strip()
            if from_full and not _is_unresolved_symbol_name(code, from_full):
                resolved = from_full
        out[code] = resolved
    return out


def _fill_unresolved_tw_names(
    *,
    frame: pd.DataFrame,
    service: MarketDataService,
    full_rows: list[dict[str, object]] | None = None,
    symbol_col: str = "symbol",
    name_col: str = "name",
) -> pd.DataFrame:
    if frame is None or frame.empty or symbol_col not in frame.columns:
        return frame
    out = frame.copy()
    if name_col not in out.columns:
        out[name_col] = out[symbol_col].astype(str)
    out[symbol_col] = out[symbol_col].astype(str).str.strip().str.upper()
    out[name_col] = out[name_col].astype(str).str.strip()

    tw_symbols: list[str] = []
    for symbol in out[symbol_col].tolist():
        code = str(symbol or "").strip().upper()
        if re.fullmatch(r"\d{4}", code) and code not in tw_symbols:
            tw_symbols.append(code)
    if not tw_symbols:
        return out

    name_map = _resolve_tw_symbol_names(service=service, symbols=tw_symbols, full_rows=full_rows)
    unresolved_mask = out.apply(
        lambda row: _is_unresolved_symbol_name(
            str(row.get(symbol_col, "")), str(row.get(name_col, ""))
        ),
        axis=1,
    )
    for idx, is_unresolved in unresolved_mask.items():
        if not bool(is_unresolved):
            continue
        code = str(out.at[idx, symbol_col] or "").strip().upper()
        mapped = str(name_map.get(code, "")).strip()
        if mapped and not _is_unresolved_symbol_name(code, mapped):
            out.at[idx, name_col] = mapped
    return out


def _render_00910_constituent_intro_table(
    *,
    service: MarketDataService,
    full_rows: list[dict[str, object]],
):
    rows_full_for_intro = list(full_rows)
    if not rows_full_for_intro:
        rows_full_for_intro, _ = service.get_etf_constituents_full(
            "00910", limit=None, force_refresh=False
        )
    if not rows_full_for_intro:
        return

    rows_full_for_intro = _enrich_rows_with_tw_names(rows=rows_full_for_intro, service=service)
    intro_df = pd.DataFrame(rows_full_for_intro)
    if "rank" in intro_df.columns:
        intro_df["rank"] = pd.to_numeric(intro_df["rank"], errors="coerce")
        intro_df = intro_df.sort_values("rank", ascending=True, na_position="last")
    intro_df["symbol"] = intro_df.get("symbol", pd.Series(dtype=str)).astype(str)
    intro_df["tw_code"] = intro_df.get("tw_code", pd.Series(dtype=str)).astype(str).str.upper()
    intro_df["name"] = intro_df.get("name", pd.Series(dtype=str)).astype(str)
    intro_df["market"] = intro_df.get("market", pd.Series(dtype=str)).astype(str).str.upper()
    tw_mask = intro_df["tw_code"].str.fullmatch(r"\d{4}", na=False)
    if tw_mask.any():
        intro_df.loc[tw_mask, "symbol"] = intro_df.loc[tw_mask, "tw_code"]
        intro_df.loc[tw_mask & (intro_df["market"] == ""), "market"] = "TW"
    intro_df["weight_pct"] = pd.to_numeric(intro_df.get("weight_pct"), errors="coerce").round(2)
    briefs: list[str] = []
    for _, row in intro_df.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        name = str(row.get("name", "")).strip()
        market = str(row.get("market", "")).strip().upper()
        brief = _company_brief_for_00910(symbol=symbol, name=name, market_tag=market)
        briefs.append(brief)
    intro_df["brief"] = briefs
    show_cols = [
        c
        for c in ["rank", "symbol", "name", "market", "weight_pct", "brief"]
        if c in intro_df.columns
    ]
    out_intro = intro_df[show_cols].rename(
        columns={
            "rank": "排名",
            "symbol": "代號",
            "name": "名稱",
            "market": "市場",
            "weight_pct": "權重(%)",
            "brief": "公司簡介",
        }
    )
    st.dataframe(
        out_intro,
        width="stretch",
        hide_index=True,
        height=_full_table_height(out_intro),
    )


def _rows_have_overseas_constituents(rows: list[dict[str, object]]) -> bool:
    for row in rows:
        if not isinstance(row, dict):
            continue
        market = str(row.get("market", "")).strip().upper()
        if market and market not in {"TW", "TWO"}:
            return True
        symbol = str(row.get("symbol", "")).strip().upper()
        tw_code = _extract_tw_code_from_row(row)
        if "." in symbol and not tw_code:
            return True
    return False


def _company_brief_for_global_generic(symbol: str, name: str, market_tag: str) -> str:
    market = str(market_tag or "").strip().upper()
    if market == "TW":
        return "台灣科技供應鏈公司，主要布局半導體、電子零組件、網通或系統整合。"
    if market == "KS":
        return "韓國科技產業公司，常見於記憶體、半導體、平台服務與高階製造供應鏈。"
    if market == "US":
        return "美國科技公司，重點涵蓋雲端平台、軟體、晶片設計或數位服務。"
    if market == "JP":
        return "日本科技公司，常見於電子零組件、精密製造與工業自動化供應鏈。"
    return f"{name or symbol} 為海外科技成分股，實際業務請以公司最新公開資訊為準。"


def _render_global_constituent_intro_table(
    *,
    etf_code: str,
    service: MarketDataService,
    full_rows: list[dict[str, object]],
):
    rows_full_for_intro = list(full_rows)
    if not rows_full_for_intro:
        rows_full_for_intro, _ = service.get_etf_constituents_full(
            str(etf_code or "").strip().upper(), limit=None, force_refresh=False
        )
    if not rows_full_for_intro:
        return

    rows_full_for_intro = _enrich_rows_with_tw_names(rows=rows_full_for_intro, service=service)
    intro_df = pd.DataFrame(rows_full_for_intro)
    if intro_df.empty:
        return
    if "rank" in intro_df.columns:
        intro_df["rank"] = pd.to_numeric(intro_df["rank"], errors="coerce")
        intro_df = intro_df.sort_values("rank", ascending=True, na_position="last")
    intro_df["symbol"] = intro_df.get("symbol", pd.Series(dtype=str)).astype(str)
    intro_df["tw_code"] = intro_df.get("tw_code", pd.Series(dtype=str)).astype(str).str.upper()
    intro_df["name"] = intro_df.get("name", pd.Series(dtype=str)).astype(str)
    intro_df["market"] = intro_df.get("market", pd.Series(dtype=str)).astype(str).str.upper()
    tw_mask = intro_df["tw_code"].str.fullmatch(r"\d{4}", na=False)
    if tw_mask.any():
        intro_df.loc[tw_mask, "symbol"] = intro_df.loc[tw_mask, "tw_code"]
        intro_df.loc[tw_mask & (intro_df["market"] == ""), "market"] = "TW"
    if "market" in intro_df.columns:
        intro_df = intro_df[intro_df["market"] != ""]
        # Filter to overseas list for global introduction.
        intro_df = intro_df[intro_df["market"] != "TW"].copy()
    if intro_df.empty:
        return
    intro_df["weight_pct"] = pd.to_numeric(intro_df.get("weight_pct"), errors="coerce").round(2)
    etf_text = str(etf_code or "").strip().upper()
    briefs: list[str] = []
    for _, row in intro_df.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        name = str(row.get("name", "")).strip()
        market = str(row.get("market", "")).strip().upper()
        if etf_text == "00910":
            brief = _company_brief_for_00910(symbol=symbol, name=name, market_tag=market)
        else:
            brief = _company_brief_for_global_generic(symbol=symbol, name=name, market_tag=market)
        briefs.append(brief)
    intro_df["brief"] = briefs
    show_cols = [
        c
        for c in ["rank", "symbol", "name", "market", "weight_pct", "brief"]
        if c in intro_df.columns
    ]
    out_intro = intro_df[show_cols].rename(
        columns={
            "rank": "排名",
            "symbol": "代號",
            "name": "名稱",
            "market": "市場",
            "weight_pct": "權重(%)",
            "brief": "公司簡介",
        }
    )
    st.dataframe(
        out_intro,
        width="stretch",
        hide_index=True,
        height=_full_table_height(out_intro),
    )


def _render_heatmap_constituent_intro_sections(
    *,
    etf_code: str,
    snapshot_symbols: list[str],
    service: MarketDataService,
    full_rows_00910: list[dict[str, object]],
):
    symbols = [
        str(symbol or "").strip().upper()
        for symbol in snapshot_symbols
        if str(symbol or "").strip()
    ]
    if not symbols:
        return

    etf_text = str(etf_code or "").strip().upper()
    st.markdown("---")
    has_global_intro = _rows_have_overseas_constituents(full_rows_00910)
    if has_global_intro:
        st.markdown(f"#### {etf_text} 全球成分股公司簡介")
        if etf_text == "00910":
            _render_00910_constituent_intro_table(
                service=service,
                full_rows=full_rows_00910,
            )
        else:
            _render_global_constituent_intro_table(
                etf_code=etf_text,
                service=service,
                full_rows=full_rows_00910,
            )
        st.markdown("#### 成分股公司簡介")
    else:
        st.markdown("#### 成分股公司簡介")

    _render_tw_constituent_intro_table(
        etf_code=etf_text,
        symbols=symbols,
        service=service,
    )


def _render_00735_heatmap_intro_tabs(
    *,
    snapshot_symbols: list[str],
    service: MarketDataService,
    full_rows: list[dict[str, object]],
):
    symbols = [
        str(symbol or "").strip().upper()
        for symbol in snapshot_symbols
        if str(symbol or "").strip()
    ]
    if not symbols:
        return

    st.markdown("---")
    st.markdown("#### 00735 成分股資訊")
    tab_global, tab_tw = st.tabs(["全球成分股", "台股可回測"])
    with tab_global:
        st.caption("全球成分股公司簡介（已合併重複區塊）。")
        _render_global_constituent_intro_table(
            etf_code="00735",
            service=service,
            full_rows=full_rows,
        )
    with tab_tw:
        st.caption("台股可回測成分股公司簡介。")
        _render_tw_constituent_intro_table(
            etf_code="00735",
            symbols=symbols,
            service=service,
        )


PAGE_CARDS = [
    {"key": "即時看盤", "desc": "台股/美股即時報價、即時走勢與技術快照。"},
    {"key": "回測工作台", "desc": "日K同步、策略回測、回放與績效比較。"},
    {
        "key": "2026 YTD 前十大股利型、配息型 ETF",
        "desc": "2026 年截至今日的台股股利/配息型 ETF 報酬率前十名。",
    },
    {"key": "2026 YTD 前十大 ETF", "desc": "2026 年截至今日的台股 ETF 報酬率前十名。"},
    {
        "key": "台股 ETF 全類型總表",
        "desc": "台股 ETF 全名單（含類型）與 2025/2026 YTD/大盤勝負比較。",
    },
    {"key": "2025 後20大最差勁 ETF", "desc": "2025 全年區間台股 ETF 報酬率後20名。"},
    {"key": "共識代表 ETF", "desc": "以前10 ETF 成分股交集，找出最具代表性的單一 ETF。"},
    {"key": "兩檔 ETF 推薦", "desc": "以共識代表+低重疊動能，收斂為兩檔建議組合。"},
    {"key": "2026 YTD 主動式 ETF", "desc": "台股主動式 ETF 在 2026 年截至今日的 Buy & Hold 績效。"},
    {"key": "ETF 輪動策略", "desc": "6檔台股ETF月頻輪動與基準對照。"},
    {"key": "熱力圖總表", "desc": "集中管理你開啟過的 ETF 熱力圖與釘選卡片。"},
    {"key": "00910 熱力圖", "desc": "00910 成分股回測的相對大盤熱力圖。"},
    {"key": "00935 熱力圖", "desc": "00935 成分股回測的相對大盤熱力圖。"},
    {"key": "00735 熱力圖", "desc": "00735 成分股回測的相對大盤熱力圖。"},
    {"key": "0050 熱力圖", "desc": "0050 成分股回測的相對大盤熱力圖。"},
    {"key": "0052 熱力圖", "desc": "0052 成分股回測的相對大盤熱力圖。"},
    {"key": "資料庫檢視", "desc": "直接查看 SQLite / DuckDB 各表筆數、欄位與內容。"},
    {"key": "新手教學", "desc": "參數白話解釋與常見回測誤區。"},
]
DEFAULT_ACTIVE_PAGE = "回測工作台"
DEFAULT_UI_THEME = "灰白專業（Soft Gray）"
BACKTEST_RUN_REQUEST_KEY = "bt_requested_run_key"


def _runtime_page_cards() -> list[dict[str, str]]:
    from ui.core.page_registry import runtime_page_cards

    return runtime_page_cards(PAGE_CARDS, _dynamic_heatmap_cards)


def _strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(str(name), str(name))


def _format_price(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.2f}"


def _format_int(v: int | None) -> str:
    if v is None:
        return "—"
    return f"{v:,}"


def _safe_float(value: object) -> float | None:
    try:
        fv = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def _format_as_of_token(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "N/A"
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return text
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d")


def _build_data_health(
    *,
    as_of: object,
    data_sources: list[str],
    source_chain: list[str] | None = None,
    degraded: bool = False,
    fallback_depth: int = 0,
    freshness_sec: int | None = None,
    notes: str = "",
) -> DataHealth:
    from ui.core.health import build_data_health

    return build_data_health(
        as_of=as_of,
        data_sources=data_sources,
        source_chain=source_chain,
        degraded=degraded,
        fallback_depth=fallback_depth,
        freshness_sec=freshness_sec,
        notes=notes,
    )


def _render_data_health_caption(title: str, health: DataHealth):
    from ui.core.health import render_data_health_caption

    st.caption(render_data_health_caption(title, health))


def _classify_issue_level(message: str) -> str:
    text = str(message or "").strip()
    lowered = text.lower()
    if "部分" in text and ("失敗" in text or "錯誤" in text):
        return "warning"
    error_tokens = [
        "traceback",
        "arrowtypeerror",
        "conversion failed",
        "schema",
        "parse",
        "valueerror",
        "keyerror",
        "無法",
        "fatal",
    ]
    warning_tokens = [
        "timeout",
        "timed out",
        "connection",
        "max retries",
        "temporarily",
        "rate limit",
        "stale",
        "fallback",
        "同步失敗",
    ]
    if any(token in lowered for token in error_tokens) or ("失敗" in text and "部分" not in text):
        return "error"
    if any(token in lowered for token in warning_tokens) or "錯誤" in text:
        return "warning"
    return "info"


def _emit_issue_message(message: str):
    level = _classify_issue_level(message)
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)


def _render_sync_issues(prefix: str, issues: list[str], *, preview_limit: int = 3):
    from ui.core.health import render_sync_issues

    msg = render_sync_issues(prefix, issues, preview_limit=preview_limit)
    if msg:
        _emit_issue_message(msg)


def _compact_sync_issue_messages(issues: list[str]) -> list[str]:
    if not isinstance(issues, list) or not issues:
        return []
    out: list[str] = []
    seen: set[str] = set()
    noisy_tokens = [
        "fugle api http 404",
        "returned empty ohlcv",
        "unsupported tw symbol",
        "only provides latest daily snapshot",
    ]
    for item in issues:
        text = " ".join(str(item).split())
        if not text:
            continue
        symbol, sep, detail = text.partition(":")
        symbol_token = str(symbol or "").strip().upper()
        detail_lower = str(detail or "").lower()
        if (
            sep
            and symbol_token.endswith("A")
            and any(token in detail_lower for token in noisy_tokens)
        ):
            text = f"{symbol_token}: 外部來源暫無完整日K（可能新上市或資料源尚未覆蓋）"
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _snapshot_fallback_depth(target_yyyymmdd: str, used_yyyymmdd: str) -> int:
    try:
        target_date = datetime.strptime(str(target_yyyymmdd), "%Y%m%d").date()
        used_date = datetime.strptime(str(used_yyyymmdd), "%Y%m%d").date()
        return max(0, (target_date - used_date).days)
    except Exception:
        return 0


def _build_snapshot_health(*, start_used: str, end_used: str, target_yyyymmdd: str) -> DataHealth:
    fallback_depth = _snapshot_fallback_depth(
        target_yyyymmdd=target_yyyymmdd, used_yyyymmdd=end_used
    )
    return _build_data_health(
        as_of=end_used,
        data_sources=["twse_mi_index"],
        source_chain=[f"start:{start_used}", f"end:{end_used}"],
        degraded=fallback_depth > 0,
        fallback_depth=fallback_depth,
        notes=f"target={_format_as_of_token(target_yyyymmdd)}",
    )


def _stable_json_dumps(payload: object) -> str:
    return stable_json_dumps(payload)


def _build_replay_source_hash(payload: dict[str, object]) -> str:
    return build_source_hash(payload)


def _resolve_live_change_metrics(
    quote,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
) -> tuple[float | None, float | None, str]:
    def _market_tz(market: str):
        text = str(market or "").strip().upper()
        if text == "TW":
            return ZoneInfo("Asia/Taipei")
        if text == "US":
            return ZoneInfo("America/New_York")
        return timezone.utc

    price = _safe_float(getattr(quote, "price", None))
    prev_close = _safe_float(getattr(quote, "prev_close", None))
    prev_close_basis = "provider"
    quote_interval = str(getattr(quote, "interval", "") or "").strip().lower()
    quote_ts = pd.Timestamp(getattr(quote, "ts", datetime.now(tz=timezone.utc)))
    if quote_ts.tzinfo is None:
        quote_ts = quote_ts.tz_localize("UTC")
    else:
        quote_ts = quote_ts.tz_convert("UTC")
    market_tz = _market_tz(str(getattr(quote, "market", "")))
    quote_local_date = quote_ts.tz_convert(market_tz).date()

    daily_norm = normalize_ohlcv_frame(daily)
    if (
        (prev_close is None or abs(prev_close) < 1e-12)
        and not daily_norm.empty
        and "close" in daily_norm.columns
    ):
        closes = pd.to_numeric(daily_norm["close"], errors="coerce").dropna().sort_index()
        # For daily snapshot quotes (e.g. Stooq / delayed daily feeds), price often equals
        # the latest daily close. In that case prev close should be the prior bar.
        if quote_interval == "1d" and price is not None and len(closes) >= 2:
            latest_close = _safe_float(closes.iloc[-1])
            if latest_close is not None:
                tol = max(1e-6, abs(latest_close) * 1e-6)
                if abs(latest_close - price) <= tol:
                    prev_close = _safe_float(closes.iloc[-2])
                    prev_close_basis = "daily_prev_bar_for_1d_quote"

        local_dates = closes.index.tz_convert(market_tz).date
        if prev_close is None or abs(prev_close) < 1e-12:
            prior_mask = local_dates < quote_local_date
            prior = closes[prior_mask]
            if not prior.empty:
                prev_close = _safe_float(prior.iloc[-1])
                prev_close_basis = "daily_prev_close"
            elif not closes.empty:
                prev_close = _safe_float(closes.iloc[-1])
                prev_close_basis = "daily_latest"

    intraday_norm = normalize_ohlcv_frame(intraday)
    if (
        (prev_close is None or abs(prev_close) < 1e-12)
        and not intraday_norm.empty
        and "close" in intraday_norm.columns
    ):
        closes = pd.to_numeric(intraday_norm["close"], errors="coerce").dropna().sort_index()
        prior = closes[closes.index < quote_ts]
        if not prior.empty:
            prev_close = _safe_float(prior.iloc[-1])
            prev_close_basis = "intraday_prev_bar"
        elif len(closes) >= 2:
            prev_close = _safe_float(closes.iloc[-2])
            prev_close_basis = "intraday_prev_bar"
        elif len(closes) == 1:
            prev_close = _safe_float(closes.iloc[-1])
            prev_close_basis = "intraday_latest"

    change = _safe_float(getattr(quote, "change", None))
    if change is None and price is not None and prev_close is not None:
        change = price - prev_close

    change_pct = _safe_float(getattr(quote, "change_pct", None))
    if (
        change_pct is None
        and change is not None
        and prev_close is not None
        and abs(prev_close) >= 1e-12
    ):
        change_pct = change / prev_close * 100.0

    source_name = str(getattr(quote, "source", "unknown") or "unknown")
    basis_text = f"source={source_name}, prev_close={prev_close_basis}"
    return change, change_pct, basis_text


def _apply_total_return_adjustment(
    bars: pd.DataFrame,
    *,
    min_coverage_ratio: float = 0.6,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return bars.copy(), {"applied": False, "coverage_pct": 0.0, "reason": "empty"}
    if "adj_close" not in bars.columns or "close" not in bars.columns:
        return bars.copy(), {"applied": False, "coverage_pct": 0.0, "reason": "no_adj_close"}

    close = pd.to_numeric(bars["close"], errors="coerce")
    adj = pd.to_numeric(bars["adj_close"], errors="coerce")
    valid = close.notna() & adj.notna() & (close > 0) & (adj > 0)
    total_rows = int(len(bars))
    valid_rows = int(valid.sum())
    coverage = float(valid_rows / total_rows) if total_rows > 0 else 0.0
    if valid_rows < 2 or coverage < float(min_coverage_ratio):
        return bars.copy(), {
            "applied": False,
            "coverage_pct": round(coverage * 100.0, 1),
            "reason": "insufficient_adj_close",
            "valid_rows": valid_rows,
            "total_rows": total_rows,
        }

    factor = pd.Series(np.nan, index=bars.index, dtype=float)
    factor.loc[valid] = adj.loc[valid] / close.loc[valid]
    factor = factor.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1.0)

    out = bars.copy()
    for col in ["open", "high", "low", "close"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * factor
    if "adj_close" in out.columns:
        out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")

    max_delta = float((factor - 1.0).abs().max()) if not factor.empty else 0.0
    return out, {
        "applied": True,
        "coverage_pct": round(coverage * 100.0, 1),
        "valid_rows": valid_rows,
        "total_rows": total_rows,
        "factor_delta_max": round(max_delta, 6),
    }


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


def _hovertemplate_with_code(
    code: str,
    *,
    value_label: str = "Equity",
    y_format: str = ",.2f",
) -> str:
    token = str(code or "").strip() or "N/A"
    return f"Date=%{{x|%Y-%m-%d}}<br>代碼={token}<br>{value_label}=%{{y:{y_format}}}<extra></extra>"


def _benchmark_line_style(
    palette: dict[str, object],
    *,
    width: float = 2.0,
    dash: str | None = None,
) -> dict[str, object]:
    return {
        "color": str(palette["benchmark"]),
        "width": float(width),
        "dash": str(dash or palette.get("benchmark_dash", "dash")),
    }


def _plot_hoverlabel_style(palette: dict[str, object]) -> dict[str, object]:
    is_dark = bool(palette.get("is_dark", False))
    if is_dark:
        return {
            "bgcolor": "rgba(17,24,39,0.95)",
            "bordercolor": "rgba(148,163,184,0.38)",
            "font": {"color": "#E5E7EB"},
        }
    return {
        "bgcolor": "rgba(255,255,255,0.94)",
        "bordercolor": "rgba(15,23,42,0.16)",
        "font": {"color": "#0F172A"},
    }


def _apply_unified_benchmark_hover(fig: go.Figure, palette: dict[str, object]):
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=_plot_hoverlabel_style(palette),
        hoverdistance=140,
        spikedistance=-1,
    )
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor=str(palette.get("benchmark", palette.get("text_muted", "#64748b"))),
        spikethickness=1,
    )


_PLOTLY_DRAW_TOOL_COLOR = "#FF2D55"
_PLOTLY_DRAW_MODEBAR_BUTTONS = ["drawline", "drawopenpath", "drawrect", "drawcircle"]


def _enable_plotly_draw_tools(
    fig: go.Figure,
    *,
    color: str = _PLOTLY_DRAW_TOOL_COLOR,
) -> None:
    if not isinstance(fig, go.Figure):
        return
    fig.update_layout(
        dragmode="pan",
        newshape=dict(
            line=dict(color=str(color), width=2),
            fillcolor=_to_rgba(str(color), 0.18),
            opacity=0.9,
        ),
    )


def _apply_plotly_watermark(
    fig: go.Figure,
    *,
    text: str,
    palette: dict[str, object] | None = None,
    force: bool = False,
) -> None:
    _ = (fig, text, palette, force)
    return


def _safe_export_filename(value: str, *, fallback: str = "chart") -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    token = token.strip("._-")
    return token or fallback


def _plotly_chart_config(*, filename: str = "", scale: int = 2) -> dict[str, object]:
    cfg: dict[str, object] = {
        "displaylogo": False,
        "modeBarButtonsToAdd": list(_PLOTLY_DRAW_MODEBAR_BUTTONS),
        "edits": {"shapePosition": True},
    }
    token = str(filename or "").strip()
    if token:
        cfg["toImageButtonOptions"] = {
            "format": "png",
            "filename": token,
            "scale": int(scale),
        }
    return cfg


def _render_plotly_chart(
    fig: go.Figure,
    *,
    chart_key: str = "",
    filename: str = "",
    scale: int = 2,
    width: str = "stretch",
    watermark_text: str = "",
    palette: dict[str, object] | None = None,
) -> None:
    _ = (watermark_text, palette)
    plot_kwargs: dict[str, object] = {}
    if str(chart_key).strip():
        plot_kwargs["key"] = str(chart_key).strip()
    resolved_name = _safe_export_filename(
        filename, fallback=_safe_export_filename(chart_key, fallback="chart")
    )
    st.plotly_chart(
        fig,
        width=width,
        config=_plotly_chart_config(filename=resolved_name, scale=int(scale)),
        **plot_kwargs,
    )


HEATMAP_EXCESS_COLORSCALE: list[list[object]] = [
    [0.00, "rgba(239,68,68,0.78)"],
    [0.24, "rgba(248,113,113,0.58)"],
    [0.50, "rgba(148,163,184,0.18)"],
    [0.76, "rgba(52,211,153,0.58)"],
    [1.00, "rgba(5,150,105,0.78)"],
]
HEATMAP_TEXT_COLOR = "#0F172A"
ACTIVE_ETF_LINE_COLORS: list[str] = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#7f7f7f",
    "#0ea5e9",
    "#dc2626",
    "#16a34a",
    "#f59e0b",
    "#4f46e5",
]
ACTIVE_ETF_LINE_DASHES: list[str] = ["solid", "dot", "dash", "longdash", "dashdot"]


def _heatmap_max_abs(z: np.ndarray, *, min_floor: float = 0.15) -> float:
    finite = np.abs(z[np.isfinite(z)])
    if finite.size == 0:
        return 1.0
    raw_max = float(np.nanmax(finite))
    p85 = float(np.nanpercentile(finite, 85))
    # Cap by high percentile so one outlier does not wash out most tiles.
    capped = min(raw_max, p85 * 1.35)
    return max(0.01, max(float(min_floor), capped))


def _series_metrics_basic(series: pd.Series) -> dict[str, float]:
    if series is None or series.empty:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(clean) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    returns = clean.pct_change().fillna(0.0)
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    years = max((clean.index[-1] - clean.index[0]).days / 365.25, 1 / 365.25)
    total_return = clean.iloc[-1] / clean.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    sharpe = (returns.mean() / returns.std() * np.sqrt(252.0)) if returns.std() > 0 else 0.0
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(sharpe),
    }


def _build_symbol_line_styles(symbols: list[str]) -> dict[str, dict[str, str]]:
    ordered = sorted({str(sym or "").strip().upper() for sym in symbols if str(sym or "").strip()})
    out: dict[str, dict[str, str]] = {}
    if not ordered:
        return out
    color_count = len(ACTIVE_ETF_LINE_COLORS)
    dash_count = len(ACTIVE_ETF_LINE_DASHES)
    for idx, sym in enumerate(ordered):
        out[sym] = {
            "color": ACTIVE_ETF_LINE_COLORS[idx % color_count],
            "dash": ACTIVE_ETF_LINE_DASHES[(idx // color_count) % dash_count],
        }
    return out


def _benchmark_candidates_tw(choice: str, *, allow_twii_fallback: bool = False) -> list[str]:
    return benchmark_candidates_tw(choice, allow_twii_fallback=allow_twii_fallback)


def _load_tw_benchmark_bars(
    *,
    store: HistoryStore,
    choice: str,
    start_dt: datetime,
    end_dt: datetime,
    sync_first: bool,
    allow_twii_fallback: bool = False,
    min_rows: int = 2,
) -> tuple[pd.DataFrame, str, list[str]]:
    loaded = load_tw_benchmark_bars(
        store=store,
        choice=choice,
        start_dt=start_dt,
        end_dt=end_dt,
        sync_first=sync_first,
        allow_twii_fallback=allow_twii_fallback,
        min_rows=min_rows,
    )
    return loaded.bars, loaded.symbol_used, loaded.sync_issues


def _load_tw_benchmark_close(
    *,
    store: HistoryStore,
    choice: str,
    start_dt: datetime,
    end_dt: datetime,
    sync_first: bool,
    allow_twii_fallback: bool = False,
) -> tuple[pd.Series, str, list[str]]:
    loaded = load_tw_benchmark_close(
        store=store,
        choice=choice,
        start_dt=start_dt,
        end_dt=end_dt,
        sync_first=sync_first,
        allow_twii_fallback=allow_twii_fallback,
    )
    return loaded.close, loaded.symbol_used, loaded.sync_issues


def _compute_tw_equal_weight_compare_payload(
    *,
    symbols: list[str],
    start_dt: datetime,
    end_dt: datetime,
    benchmark_choice: str,
    sync_before_run: bool,
    insufficient_msg: str,
    initial_capital: float = 1_000_000.0,
) -> dict[str, object]:
    store = _history_store()
    symbol_sync_issues: list[str] = []
    benchmark_sync_issues: list[str] = []

    if sync_before_run:
        _, symbol_sync_issues = _sync_symbols_history(
            store,
            market="TW",
            symbols=symbols,
            start=start_dt,
            end=end_dt,
            parallel=True,
        )
        symbol_sync_issues = _compact_sync_issue_messages(symbol_sync_issues)

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    skipped_symbols: list[str] = []
    for symbol in symbols:
        bars = normalize_ohlcv_frame(
            store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
        )
        if len(bars) < 2 and not sync_before_run:
            report = store.sync_symbol_history(
                symbol=symbol, market="TW", start=start_dt, end=end_dt
            )
            if report.error:
                symbol_sync_issues.append(f"{symbol}: {report.error}")
            bars = normalize_ohlcv_frame(
                store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            )
        if len(bars) < 2:
            skipped_symbols.append(symbol)
            continue
        bars, _ = apply_split_adjustment(
            bars=bars,
            symbol=symbol,
            market="TW",
            use_known=True,
            use_auto_detect=True,
        )
        if len(bars) < 2:
            skipped_symbols.append(symbol)
            continue
        bars_by_symbol[symbol] = bars

    if not bars_by_symbol:
        return {
            "error": insufficient_msg,
            "symbol_sync_issues": _compact_sync_issue_messages(symbol_sync_issues),
            "benchmark_sync_issues": benchmark_sync_issues,
            "used_symbols": [],
            "skipped_symbols": sorted(skipped_symbols),
        }

    target_index = pd.DatetimeIndex([])
    for bars in bars_by_symbol.values():
        target_index = target_index.union(pd.DatetimeIndex(bars.index))
    target_index = target_index.sort_values()

    strategy_equity = build_buy_hold_equity(
        bars_by_symbol=bars_by_symbol,
        target_index=target_index,
        initial_capital=float(initial_capital),
    ).dropna()

    per_symbol_equity: dict[str, pd.Series] = {}
    for symbol, bars in bars_by_symbol.items():
        eq_sym = build_buy_hold_equity(
            bars_by_symbol={symbol: bars},
            target_index=target_index,
            initial_capital=float(initial_capital),
        ).dropna()
        if not eq_sym.empty:
            per_symbol_equity[symbol] = eq_sym

    bench_bars, benchmark_symbol_used, benchmark_sync_issues = _load_tw_benchmark_bars(
        store=store,
        choice=benchmark_choice,
        start_dt=start_dt,
        end_dt=end_dt,
        sync_first=sync_before_run,
        allow_twii_fallback=True,
        min_rows=2,
    )
    benchmark_equity = pd.Series(dtype=float)
    if not bench_bars.empty and "close" in bench_bars.columns:
        bench_close = pd.to_numeric(bench_bars["close"], errors="coerce").dropna().sort_index()
        aligned = bench_close.reindex(target_index).ffill()
        valid = aligned.dropna()
        if len(valid) >= 2:
            base_val = float(valid.iloc[0])
            if math.isfinite(base_val) and base_val > 0:
                benchmark_equity = (aligned.loc[valid.index[0] :] / base_val) * float(
                    initial_capital
                )
                benchmark_equity = benchmark_equity.dropna()

    common_index = pd.DatetimeIndex(strategy_equity.index)
    if not benchmark_equity.empty:
        common_index = common_index.intersection(pd.DatetimeIndex(benchmark_equity.index))
    common_index = common_index.sort_values()
    if len(common_index) < 2:
        return {
            "error": "Strategy 與 Benchmark 缺少足夠重疊交易日，無法建立對照圖。",
            "symbol_sync_issues": _compact_sync_issue_messages(symbol_sync_issues),
            "benchmark_sync_issues": benchmark_sync_issues,
            "used_symbols": sorted(list(bars_by_symbol.keys())),
            "skipped_symbols": sorted(skipped_symbols),
        }

    strategy_plot = strategy_equity.reindex(common_index).ffill().dropna()
    benchmark_plot = (
        benchmark_equity.reindex(common_index).ffill().dropna()
        if not benchmark_equity.empty
        else pd.Series(dtype=float)
    )
    per_symbol_plot: dict[str, pd.Series] = {}
    for symbol, series in per_symbol_equity.items():
        aligned = series.reindex(common_index).ffill().dropna()
        if len(aligned) >= 2:
            per_symbol_plot[symbol] = aligned

    return {
        "error": "",
        "benchmark_symbol": benchmark_symbol_used,
        "strategy_equity": strategy_plot,
        "benchmark_equity": benchmark_plot,
        "per_symbol_equity": per_symbol_plot,
        "used_symbols": sorted(list(bars_by_symbol.keys())),
        "skipped_symbols": sorted(skipped_symbols),
        "symbol_sync_issues": _compact_sync_issue_messages(symbol_sync_issues),
        "benchmark_sync_issues": benchmark_sync_issues,
    }


def _palette_with(base: dict[str, object], **overrides: object) -> dict[str, object]:
    out = dict(base)
    out.update(overrides)
    return out


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
    "price_up": "#5FA783",
    "price_down": "#D78C95",
    "sma20": "#0284c7",
    "sma60": "#d97706",
    "vwap": "#059669",
    "rsi_line": "#0284c7",
    "kd_k": "#2563eb",
    "kd_d": "#ea580c",
    "macd_line": "#16a34a",
    "macd_signal": "#d97706",
    "bb_upper": "rgba(220,38,38,0.28)",
    "bb_lower": "rgba(37,99,235,0.24)",
    "volume_up": "rgba(22,163,74,0.35)",
    "volume_down": "rgba(220,38,38,0.28)",
    "equity": "#16a34a",
    "benchmark": "#64748b",
    "benchmark_dash": "dash",
    "buy_hold": "#d97706",
    "asset_palette": ["#0284c7", "#be123c", "#0f766e", "#7c2d12", "#4f46e5", "#0369a1"],
    "signal_buy": "#0284c7",
    "signal_sell": "#ea580c",
    "fill_buy": "#16a34a",
    "fill_sell": "#dc2626",
    "marker_edge": "#0f172a",
    "trade_path": "rgba(71,85,105,0.45)",
    "fill_link": "rgba(100,116,139,0.30)",
    "tag_bg": "#ffffff",
    "tag_border": "rgba(15,23,42,0.30)",
    "tag_text": "#0F172A",
    "tag_bg_hover": "#ffffff",
    "card_shadow": "0 4px 16px rgba(15, 23, 42, 0.04)",
}

_THEME_PALETTES: dict[str, dict[str, object]] = {
    "日光白（Paper Light）": _palette_with(_BASE_LIGHT_PALETTE),
    "灰白專業（Soft Gray）": _palette_with(
        _BASE_LIGHT_PALETTE,
        background="#F3F4F6",
        sidebar_bg="#ECEFF3",
        paper_bg="#F7F8FA",
        plot_bg="#F7F8FA",
        card_bg="rgba(55, 65, 81, 0.06)",
        card_border="rgba(75, 85, 99, 0.20)",
        control_bg="#F9FAFB",
        control_border="rgba(75, 85, 99, 0.24)",
        tab_bg="rgba(107, 114, 128, 0.16)",
        tab_text="#1F2937",
        text_color="#1F2937",
        text_muted="#4B5563",
        accent="#4B5563",
        grid="rgba(107,114,128,0.16)",
        price_up="#5F9F84",
        price_down="#CF8F98",
        sma20="#3B82F6",
        sma60="#D97706",
        vwap="#0F766E",
        rsi_line="#2563EB",
        kd_k="#1D4ED8",
        kd_d="#C2410C",
        macd_line="#059669",
        macd_signal="#B45309",
        benchmark="#6B7280",
        asset_palette=["#4B5563", "#0EA5E9", "#10B981", "#F59E0B", "#6366F1", "#E11D48"],
        signal_buy="#2563EB",
        signal_sell="#D97706",
        fill_buy="#15803D",
        fill_sell="#DC2626",
    ),
    "深色專業（Data Dark）": _palette_with(
        _BASE_LIGHT_PALETTE,
        is_dark=True,
        background="#0F1419",
        sidebar_bg="#121922",
        text_color="#E7E9EA",
        text_muted="#8B98A5",
        card_bg="rgba(30, 39, 50, 0.86)",
        card_border="rgba(56, 68, 82, 0.62)",
        control_bg="rgba(22, 30, 40, 0.90)",
        control_border="rgba(56, 68, 82, 0.82)",
        tab_bg="rgba(30, 39, 50, 0.78)",
        tab_text="#E7E9EA",
        accent="#F59E0B",
        plot_template="plotly_dark",
        paper_bg="#0F1419",
        plot_bg="#111821",
        grid="rgba(139,152,165,0.22)",
        price_up="#5FA783",
        price_down="#D78C95",
        sma20="#38BDF8",
        sma60="#F59E0B",
        vwap="#34D399",
        rsi_line="#60A5FA",
        kd_k="#7DD3FC",
        kd_d="#FBBF24",
        macd_line="#4ADE80",
        macd_signal="#FB923C",
        bb_upper="rgba(248,113,113,0.34)",
        bb_lower="rgba(96,165,250,0.30)",
        volume_up="rgba(52,211,153,0.34)",
        volume_down="rgba(248,113,113,0.30)",
        equity="#22C55E",
        benchmark="#8B98A5",
        buy_hold="#F59E0B",
        asset_palette=["#38BDF8", "#34D399", "#F59E0B", "#A78BFA", "#F87171", "#60A5FA"],
        signal_buy="#60A5FA",
        signal_sell="#FB923C",
        fill_buy="#22C55E",
        fill_sell="#F87171",
        marker_edge="#E5E7EB",
        trade_path="rgba(148,163,184,0.42)",
        fill_link="rgba(148,163,184,0.34)",
        tag_bg="rgba(30,39,50,0.92)",
        tag_border="rgba(139,152,165,0.46)",
        tag_text="#E7E9EA",
        tag_bg_hover="rgba(38,49,63,0.96)",
        card_shadow="0 8px 24px rgba(0, 0, 0, 0.24)",
    ),
}


def _theme_options() -> list[str]:
    return [_current_theme_name()]


def _current_theme_name() -> str:
    theme = cfg_or_env_str("ui.default_theme", "REALTIME0052_UI_THEME", DEFAULT_UI_THEME)
    if theme not in _THEME_PALETTES:
        theme = DEFAULT_UI_THEME
    if st.session_state.get("ui_theme") != theme:
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
    paper_bg = str(palette.get("paper_bg", background))
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
    tag_bg = str(palette.get("tag_bg", "#ffffff"))
    tag_border = str(palette.get("tag_border", "rgba(15, 23, 42, 0.30)"))
    tag_text = str(palette.get("tag_text", "#0F172A"))
    tag_bg_hover = str(palette.get("tag_bg_hover", tag_bg))
    card_shadow = str(palette.get("card_shadow", "0 4px 16px rgba(15, 23, 42, 0.04)"))
    active_border = _to_rgba(accent, 0.55)
    active_shadow = f"0 0 0 1px {_to_rgba(accent, 0.22)}"

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
        div[class*="st-key-page-card"] button[kind="primary"] {{
            background: #16a34a !important;
            color: #ffffff !important;
            border: 1px solid #15803d !important;
            box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.16) !important;
        }}
        div[class*="st-key-page-card"] button[kind="primary"]:hover {{
            border-color: #166534 !important;
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
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {{
            background: {tag_bg} !important;
            border: 1px solid {tag_border} !important;
            border-radius: 9px !important;
        }}
        [data-testid="stMultiSelect"] [data-baseweb="tag"] *,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] span {{
            color: {tag_text} !important;
            fill: {tag_text} !important;
        }}
        [data-testid="stMultiSelect"] [data-baseweb="tag"]:hover {{
            border-color: {tag_border} !important;
            background: {tag_bg_hover} !important;
        }}
        div[data-testid="stMetric"] {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 10px;
            padding: 8px 12px;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 14px;
            box-shadow: {card_shadow};
        }}
        .page-nav-card {{
            border: 1px solid {card_border};
            background: {card_bg};
            border-radius: 10px;
            padding: 7px 9px 6px;
            margin-bottom: 0.30rem;
            min-height: 78px;
        }}
        .page-nav-card.active {{
            border-color: {active_border};
            box-shadow: {active_shadow};
        }}
        .page-card-title {{
            font-size: 0.90rem;
            font-weight: 700;
            line-height: 1.18;
            margin-bottom: 0.08rem;
        }}
        .page-card-desc {{
            font-size: 0.76rem;
            line-height: 1.2;
            color: {text_muted};
            min-height: 1.75rem;
            margin-bottom: 0.18rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .card-section-title {{
            font-size: 1.04rem;
            font-weight: 700;
            font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", "Segoe UI", sans-serif;
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            margin-bottom: 0.12rem;
        }}
        .card-section-sub {{
            font-size: 0.82rem;
            color: {text_muted};
            font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", "Segoe UI", sans-serif;
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            margin-bottom: 0.5rem;
        }}
        .crisp-table-wrap {{
            border: 1px solid {card_border};
            border-radius: 10px;
            background: {control_bg};
            overflow: auto;
            max-height: 460px;
        }}
        .crisp-table-wrap table {{
            width: 100%;
            border-collapse: collapse;
            font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", "Segoe UI", sans-serif;
            font-size: 0.86rem;
            line-height: 1.38;
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        .crisp-table-wrap thead th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: {paper_bg};
            color: {text_color};
            text-align: left;
            font-weight: 700;
            padding: 8px 10px;
            border-bottom: 1px solid {card_border};
            white-space: nowrap;
        }}
        .crisp-table-wrap tbody td {{
            color: {text_color};
            padding: 7px 10px;
            border-bottom: 1px solid {card_border};
            white-space: nowrap;
        }}
        .crisp-table-wrap tbody tr:nth-child(2n) {{
            background: {tab_bg};
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


def _render_card_section_header(title: str, subtitle: str = ""):
    st.markdown(f"<div class='card-section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='card-section-sub'>{subtitle}</div>", unsafe_allow_html=True)


def _render_crisp_table(
    df: pd.DataFrame,
    *,
    max_height: int = 460,
    max_html_rows: int = 1500,
) -> None:
    if df is None or df.empty:
        st.caption("目前沒有可顯示的資料。")
        return
    if len(df) > max(50, int(max_html_rows)):
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption("資料筆數較多，已切換為高效表格模式。")
        return
    safe_height = max(220, int(max_height))
    table_html = df.to_html(index=False, border=0)
    st.markdown(
        f"<div class='crisp-table-wrap' style='max-height:{safe_height}px'>{table_html}</div>",
        unsafe_allow_html=True,
    )


def _design_tokens_payload() -> str:
    palette = _ui_palette()
    tokens = {
        "theme_name": _current_theme_name(),
        "surface": {
            "background": palette["background"],
            "paper": palette["paper_bg"],
            "card": palette["card_bg"],
            "border": palette["card_border"],
        },
        "text": {"primary": palette["text_color"], "secondary": palette["text_muted"]},
        "accent": {"primary": palette["accent"]},
        "chart": {
            "price_up": palette["price_up"],
            "price_down": palette["price_down"],
            "equity": palette["equity"],
            "benchmark": palette["benchmark"],
        },
    }
    return json.dumps(tokens, ensure_ascii=False, indent=2)


def _render_design_toolbox():
    with st.expander("設計協作（Figma / Pencil）", expanded=False):
        st.caption("可下載 design tokens JSON，拿去對齊 Figma Variables 或 Pencil 色票。")
        st.download_button(
            "下載 design-tokens.json",
            data=_design_tokens_payload(),
            file_name="realtime0052-design-tokens.json",
            mime="application/json",
            width="stretch",
        )
        c1, c2 = st.columns(2)
        c1.link_button("Figma", "https://www.figma.com", width="stretch")
        c2.link_button("Pencil", "https://pencil.evolus.vn", width="stretch")


def _render_page_cards_nav() -> str:
    from ui.core.page_registry import render_page_cards_nav

    cards = _runtime_page_cards()
    return render_page_cards_nav(cards=cards, default_active_page=DEFAULT_ACTIVE_PAGE)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_twse_snapshot_with_fallback(
    target_yyyymmdd: str, lookback_days: int = 14
) -> tuple[str, pd.DataFrame]:
    import requests

    target_dt = datetime.strptime(target_yyyymmdd, "%Y%m%d").date()
    errors: list[str] = []
    for offset in range(max(1, int(lookback_days))):
        query_dt = target_dt - pd.Timedelta(days=offset)
        date_token = query_dt.strftime("%Y%m%d")
        try:
            resp = requests.get(
                "https://www.twse.com.tw/exchangeReport/MI_INDEX",
                params={"response": "json", "date": date_token, "type": "ALLBUT0999"},
                timeout=18,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            errors.append(f"{date_token}:{exc}")
            continue

        if str(payload.get("stat", "")).strip().upper() != "OK":
            errors.append(f"{date_token}:{payload.get('stat', 'not ok')}")
            continue

        tables = payload.get("tables", [])
        if not isinstance(tables, list):
            errors.append(f"{date_token}:tables missing")
            continue
        target_fields = {"證券代號", "證券名稱", "收盤價"}
        selected: dict | None = None
        for table in tables:
            if not isinstance(table, dict):
                continue
            fields = table.get("fields", [])
            if not isinstance(fields, list):
                continue
            if target_fields.issubset(set(fields)):
                selected = table
                break
        if selected is None:
            errors.append(f"{date_token}:price table missing")
            continue

        fields = [str(x) for x in selected.get("fields", [])]
        rows = selected.get("data", [])
        if not isinstance(rows, list) or not rows:
            errors.append(f"{date_token}:empty rows")
            continue

        idx_code = fields.index("證券代號")
        idx_name = fields.index("證券名稱")
        idx_close = fields.index("收盤價")
        idx_open = fields.index("開盤價") if "開盤價" in fields else -1
        out_rows: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, list):
                continue
            if max(idx_code, idx_name, idx_close) >= len(row):
                continue
            code = str(row[idx_code] or "").strip().upper()
            name = str(row[idx_name] or "").strip()
            close_raw = str(row[idx_close] or "").strip().replace(",", "")
            if not code or close_raw in {"", "--", "-"}:
                continue
            try:
                close = float(close_raw)
            except Exception:
                continue
            open_price: float | None = None
            if idx_open >= 0 and idx_open < len(row):
                open_raw = str(row[idx_open] or "").strip().replace(",", "")
                if open_raw not in {"", "--", "-"}:
                    try:
                        open_candidate = float(open_raw)
                        if math.isfinite(open_candidate) and open_candidate > 0:
                            open_price = float(open_candidate)
                    except Exception:
                        pass
            out_rows.append({"code": code, "name": name, "open": open_price, "close": close})
        if out_rows:
            return date_token, pd.DataFrame(out_rows)
        errors.append(f"{date_token}:no parsable rows")

    raise RuntimeError("TWSE snapshot fetch failed: " + " | ".join(errors[-5:]))


TW_ETF_TYPE_WHITELIST: dict[str, str] = {
    # 市值型
    "0050": "市值型",
    "0051": "市值型",
    "0057": "市值型",
    "006203": "市值型",
    "006204": "市值型",
    "006208": "市值型",
    "00690": "市值型",
    "00894": "市值型",
    "00905": "市值型",
    "00912": "市值型",
    "00921": "市值型",
    "00922": "市值型",
    "00938": "市值型",
    "009802": "市值型",
    "009803": "市值型",
    "009804": "市值型",
    "009808": "市值型",
    "009813": "市值型",
    "009816": "市值型",
    # 股利型
    "0056": "股利型",
    "00701": "股利型",
    "00713": "股利型",
    "00730": "股利型",
    "00731": "股利型",
    "00878": "股利型",
    "00900": "股利型",
    "00907": "股利型",
    "00915": "股利型",
    "00918": "股利型",
    "00919": "股利型",
    "00927": "股利型",
    "00929": "股利型",
    "00932": "股利型",
    "00934": "股利型",
    "00939": "股利型",
    "00940": "股利型",
    "00944": "股利型",
    "00946": "股利型",
    "00964": "股利型",
    # 科技型
    "0052": "科技型",
    "0053": "科技型",
    "00735": "科技型",
    "00737": "科技型",
    "00830": "科技型",
    "00875": "科技型",
    "00881": "科技型",
    "00891": "科技型",
    "00892": "科技型",
    "00904": "科技型",
    "00911": "科技型",
    "00913": "科技型",
    "00935": "科技型",
    "00941": "科技型",
    "00943": "科技型",
    "00947": "科技型",
    "00952": "科技型",
    "00962": "科技型",
    # 金融型
    "0055": "金融型",
    "00917": "金融型",
    # 永續ESG型
    "00692": "永續ESG型",
    "00850": "永續ESG型",
    "00896": "永續ESG型",
    "00899": "永續ESG型",
    "00920": "永續ESG型",
    "00923": "永續ESG型",
    "00930": "永續ESG型",
    "00936": "永續ESG型",
    "00961": "永續ESG型",
    "009809": "永續ESG型",
    # 產業主題型
    "00678": "產業主題型",
    "00728": "產業主題型",
    "00893": "產業主題型",
    "00895": "產業主題型",
    "00897": "產業主題型",
    "00898": "產業主題型",
    "00901": "產業主題型",
    "00902": "產業主題型",
    "00903": "產業主題型",
    "00909": "產業主題型",
    "00910": "產業主題型",
    "00965": "產業主題型",
    # 海外市場型
    "0061": "海外市場型",
    "006205": "海外市場型",
    "006206": "海外市場型",
    "006207": "海外市場型",
    "00639": "海外市場型",
    "00643": "海外市場型",
    "00660": "海外市場型",
    "00700": "海外市場型",
    "00702": "海外市場型",
    "00709": "海外市場型",
    "00736": "海外市場型",
    "00739": "海外市場型",
    "00757": "海外市場型",
    "00770": "海外市場型",
    "00783": "海外市場型",
    # 不動產收益型
    "00712": "不動產收益型",
    "00908": "不動產收益型",
    # 債券收益型
    "00711B": "債券收益型",
    "00771": "債券收益型",
    "00865B": "債券收益型",
    # 平衡收益型
    "00981T": "平衡收益型",
    "00982T": "平衡收益型",
    # 主動式
    "00980A": "主動式",
    "00981A": "主動式",
    "00982A": "主動式",
    "00982D": "主動式",
    "00983A": "主動式",
    "00983D": "主動式",
    "00984A": "主動式",
    "00985A": "主動式",
    "00986A": "主動式",
    "00987A": "主動式",
    "00990A": "主動式",
    "00991A": "主動式",
    "00992A": "主動式",
    "00993A": "主動式",
    "00994A": "主動式",
    "00995A": "主動式",
}


TW_ETF_MANAGEMENT_FEE_FALLBACK: dict[str, str] = {
    # NOTE:
    # - 管理費為公開資訊（基金公開說明書/公告）。
    # - 若為級距費率，顯示為「x.xx%起」。
    "0050": "0.15%起",
    "0052": "0.15%起",
    "0056": "0.40%起",
    "006208": "0.15%起",
    "00935": "0.40%起",
    "00993A": "0.80%",
    "00995A": "0.75%起",
}


def _normalize_tw_etf_management_fee_label(value: object) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    token = token.replace("％", "%").replace("﹪", "%")
    has_from = "起" in token
    if token.endswith("%") or token.endswith("%起"):
        return token
    try:
        base = float(token)
    except Exception:
        m = re.search(r"(\d+(?:\.\d+)?)", token)
        if not m:
            return ""
        base = float(m.group(1))
    out = f"{base:.2f}%"
    if has_from:
        out += "起"
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _load_tw_etf_aum_billion_map(target_yyyymmdd: str = "") -> dict[str, float]:
    import requests

    out: dict[str, float] = {}
    try:
        params: dict[str, str] = {}
        token = str(target_yyyymmdd or "").strip()
        if re.fullmatch(r"\d{8}", token):
            params["date"] = token
        resp = requests.get(
            "https://www.twse.com.tw/zh/ETFortune/etfExcel", params=params, timeout=20
        )
        resp.raise_for_status()
        text = resp.content.decode("cp950", errors="ignore")
        rows = csv.reader(StringIO(text))
        for row in rows:
            if not isinstance(row, list) or len(row) < 5:
                continue
            raw_code = str(row[0] or "").strip().upper()
            m = re.search(r"(\d{4,6}[A-Z]?)", raw_code)
            if not m:
                continue
            code = m.group(1)
            raw_aum = str(row[4] or "").strip().replace(",", "")
            if raw_aum in {"", "--", "-"}:
                continue
            try:
                aum = float(raw_aum)
            except Exception:
                continue
            if not math.isfinite(aum) or aum < 0:
                continue
            out[code] = float(aum)
    except Exception:
        return {}
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _is_twse_trading_day(target_yyyymmdd: str) -> bool:
    import requests

    token = str(target_yyyymmdd or "").strip()
    if not re.fullmatch(r"\d{8}", token):
        return False
    try:
        resp = requests.get(
            "https://www.twse.com.tw/exchangeReport/MI_INDEX",
            params={"response": "json", "date": token, "type": "ALLBUT0999"},
            timeout=12,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return False
    return str(payload.get("stat", "")).strip().upper() == "OK"


def _recent_twse_trading_days(
    *,
    anchor_yyyymmdd: str,
    count: int = 22,
    max_scan_days: int = 90,
) -> list[str]:
    token = str(anchor_yyyymmdd or "").strip()
    if not re.fullmatch(r"\d{8}", token):
        token = datetime.now().strftime("%Y%m%d")
    target_dt = datetime.strptime(token, "%Y%m%d").date()
    out: list[str] = []
    for offset in range(max(1, int(max_scan_days))):
        query_dt = target_dt - pd.Timedelta(days=offset)
        day_token = query_dt.strftime("%Y%m%d")
        if _is_twse_trading_day(day_token):
            out.append(day_token)
        if len(out) >= max(1, int(count)):
            break
    out = sorted(set(out))
    if len(out) > max(1, int(count)):
        out = out[-int(count) :]
    return out


def _lookup_tw_etf_aum_billion(code: object) -> float | None:
    token = str(code or "").strip().upper()
    if not token or token.startswith("^"):
        return None
    value = _load_tw_etf_aum_billion_map().get(token)
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if not math.isfinite(val) or val < 0:
        return None
    return val


def _format_tw_etf_aum_billion(code: object) -> str:
    value = _lookup_tw_etf_aum_billion(code)
    if value is None:
        return "—"
    text = f"{value:,.2f}"
    if text.endswith(".00"):
        return text[:-3]
    return text.rstrip("0").rstrip(".")


def _attach_tw_etf_aum_column(
    frame: pd.DataFrame,
    *,
    code_col_candidates: tuple[str, ...] = ("代碼", "ETF代碼"),
    column_name: str = "ETF規模(億)",
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    code_col = next((col for col in code_col_candidates if col in frame.columns), "")
    if not code_col:
        return frame

    out = frame.copy()
    out[column_name] = _truncate_integer_series(out[code_col].map(_lookup_tw_etf_aum_billion))

    anchor_col = next(
        (
            col
            for col in ("管理費(%)", "管理費", "ETF", "ETF名稱", "代碼", "ETF代碼")
            if col in out.columns
        ),
        "",
    )
    if not anchor_col:
        return out
    cols = list(out.columns)
    if column_name in cols:
        cols.remove(column_name)
        insert_at = cols.index(anchor_col) + 1
        cols.insert(insert_at, column_name)
        out = out[cols]
    return out


def _build_tw_etf_aum_snapshot_rows(
    frame: pd.DataFrame,
    *,
    aum_map: dict[str, float],
) -> list[dict[str, object]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    code_col = next((col for col in ("代碼", "ETF代碼", "台股代號") if col in frame.columns), "")
    name_col = next((col for col in ("ETF", "ETF名稱", "etf_name") if col in frame.columns), "")
    if not code_col:
        return []

    rows: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        code_token = str(row.get(code_col, "")).strip().upper()
        code_match = re.fullmatch(r"\d{4,6}[A-Z]?", code_token)
        if not code_match:
            continue
        code = code_match.group(0)
        raw_value = aum_map.get(code)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except Exception:
            continue
        if (not math.isfinite(value)) or value < 0:
            continue
        name = str(row.get(name_col, "")).strip() if name_col else ""
        rows[code] = {
            "etf_code": code,
            "etf_name": name or code,
            "aum_billion": value,
        }
    return list(rows.values())


_TW_ETF_AUM_TRACK_ANCHOR_CODE = "__AUM_TRACK_ANCHOR__"
_TW_ETF_AUM_TRACK_ANCHOR_NAME = "AUM_TRACK_ANCHOR"


def _resolve_latest_tw_trade_day_token(anchor_yyyymmdd: str | None = None) -> str:
    token = re.sub(r"\D", "", str(anchor_yyyymmdd or "").strip())
    if not token:
        token = datetime.now().strftime("%Y%m%d")
    trade_days = _recent_twse_trading_days(anchor_yyyymmdd=token, count=1, max_scan_days=7)
    return trade_days[-1] if trade_days else token


def _resolve_latest_tw_trade_date_iso(anchor_yyyymmdd: str | None = None) -> str:
    trade_token = _resolve_latest_tw_trade_day_token(anchor_yyyymmdd=anchor_yyyymmdd)
    try:
        return pd.Timestamp(trade_token).date().isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def _load_tw_etf_aum_track_anchor_date(store) -> str | None:
    try:
        meta_df = store.load_tw_etf_aum_history(
            etf_codes=[_TW_ETF_AUM_TRACK_ANCHOR_CODE],
            keep_days=1,
        )
    except Exception:
        return None
    if not isinstance(meta_df, pd.DataFrame) or meta_df.empty:
        return None
    token = str(meta_df.iloc[-1].get("trade_date", "")).strip()
    try:
        return pd.Timestamp(token).date().isoformat()
    except Exception:
        return None


def _set_tw_etf_aum_track_anchor_date(store, *, trade_date: str) -> bool:
    try:
        trade_date_iso = pd.Timestamp(trade_date).date().isoformat()
    except Exception:
        return False
    try:
        store.save_tw_etf_aum_snapshot(
            rows=[
                {
                    "etf_code": _TW_ETF_AUM_TRACK_ANCHOR_CODE,
                    "etf_name": _TW_ETF_AUM_TRACK_ANCHOR_NAME,
                    "aum_billion": 0.0,
                }
            ],
            trade_date=trade_date_iso,
            keep_days=1,
        )
    except Exception:
        return False
    return True


def _build_tw_etf_aum_history_wide(
    history_df: pd.DataFrame,
    *,
    start_date: str | None = None,
    max_date_cols: int = 10,
) -> pd.DataFrame:
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return pd.DataFrame(columns=["編號", "台股代號", "ETF名稱"])
    required = {"etf_code", "etf_name", "trade_date", "aum_billion"}
    if not required.issubset(set(history_df.columns)):
        return pd.DataFrame(columns=["編號", "台股代號", "ETF名稱"])

    work = history_df.copy()
    work["etf_code"] = work["etf_code"].astype(str).str.strip().str.upper()
    work["etf_name"] = work["etf_name"].astype(str).str.strip()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["aum_billion"] = pd.to_numeric(work["aum_billion"], errors="coerce")
    work = work.dropna(subset=["trade_date", "aum_billion"])
    if start_date:
        start_ts = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_ts):
            work = work[work["trade_date"] >= pd.Timestamp(start_ts).normalize()]
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work = work.dropna(subset=["trade_date"])
    if work.empty:
        return pd.DataFrame(columns=["編號", "台股代號", "ETF名稱"])

    pivot = (
        work.pivot_table(
            index=["etf_code", "etf_name"],
            columns="trade_date",
            values="aum_billion",
            aggfunc="last",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    pivot = pivot.rename(columns={"etf_code": "台股代號", "etf_name": "ETF名稱"})
    date_cols = [col for col in pivot.columns if col not in {"台股代號", "ETF名稱"}]

    def _date_sort_key(value: object) -> tuple[int, str]:
        dt = pd.to_datetime(str(value), errors="coerce")
        if pd.isna(dt):
            return (1, str(value))
        return (0, pd.Timestamp(dt).strftime("%Y-%m-%d"))

    date_cols = sorted(date_cols, key=_date_sort_key)
    try:
        max_cols = int(max_date_cols)
    except Exception:
        max_cols = 10
    if max_cols > 0 and len(date_cols) > max_cols:
        date_cols = date_cols[-max_cols:]
    out = pivot[["台股代號", "ETF名稱", *date_cols]].copy()
    date_header_map = {col: f"{col}(億)" for col in date_cols}
    out = out.rename(columns=date_header_map)
    out = out.sort_values(["台股代號", "ETF名稱"]).reset_index(drop=True)
    out.insert(0, "編號", range(1, len(out) + 1))
    return out


def _extract_aum_history_trade_date(label: object) -> pd.Timestamp | None:
    text = str(label or "").strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if not m:
        return None
    ts = pd.to_datetime(m.group(1), errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _aum_history_date_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[tuple[pd.Timestamp, str]] = []
    for col in frame.columns:
        if str(col) in {"編號", "台股代號", "ETF名稱"}:
            continue
        dt = _extract_aum_history_trade_date(col)
        if dt is None:
            continue
        cols.append((dt, str(col)))
    cols.sort(key=lambda item: item[0])
    return [col for _, col in cols]


def _compute_tw_etf_aum_alert_mask(
    frame: pd.DataFrame,
    *,
    up_threshold: float = 0.10,
    down_threshold: float | None = None,
) -> dict[tuple[int, str], str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    date_cols = _aum_history_date_columns(frame)
    if len(date_cols) < 2:
        return {}

    out: dict[tuple[int, str], str] = {}
    for row_idx in range(len(frame)):
        for idx in range(1, len(date_cols)):
            prev_col = date_cols[idx - 1]
            curr_col = date_cols[idx]
            prev_raw = pd.to_numeric(frame.iloc[row_idx][prev_col], errors="coerce")
            curr_raw = pd.to_numeric(frame.iloc[row_idx][curr_col], errors="coerce")
            if not pd.notna(prev_raw) or not pd.notna(curr_raw):
                continue
            prev_val = float(prev_raw)
            curr_val = float(curr_raw)
            if prev_val <= 0:
                continue
            pct = (curr_val - prev_val) / prev_val
            if pct > float(up_threshold):
                out[(row_idx, curr_col)] = "#ffd1dc"
            elif (down_threshold is not None) and (pct < float(down_threshold)):
                out[(row_idx, curr_col)] = "#d8ecff"
    return out


def _style_tw_etf_aum_history_table(frame: pd.DataFrame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    work = frame.copy()
    date_cols = _aum_history_date_columns(work)
    for col in date_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    alert_mask = _compute_tw_etf_aum_alert_mask(work, up_threshold=0.10, down_threshold=None)

    def _apply_row(row: pd.Series) -> list[str]:
        ridx = int(row.name)
        styles: list[str] = []
        for col in work.columns:
            color = alert_mask.get((ridx, str(col)))
            styles.append(f"background-color: {color}" if color else "")
        return styles

    styler = work.style.apply(_apply_row, axis=1)
    if date_cols:
        styler = styler.format(
            {
                col: (lambda val: "—" if pd.isna(val) else f"{int(round(float(val))):,}")
                for col in date_cols
            }
        )
    return styler


def _decorate_tw_etf_aum_history_links(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df, {}
    if ("台股代號" not in df.columns) or ("ETF名稱" not in df.columns):
        return df, {}
    out = df.copy()

    code_links: list[str | None] = []
    name_links: list[str | None] = []
    linked_code_count = 0
    linked_name_count = 0

    for _, row in out.iterrows():
        code = _normalize_heatmap_etf_code(row.get("台股代號"))
        name = str(row.get("ETF名稱", "")).strip()
        if not code:
            code_links.append(None)
            name_links.append(None)
            continue
        code_links.append(build_backtest_drill_url(symbol=code, market="TW"))
        linked_code_count += 1
        name_links.append(build_heatmap_drill_url(code, name, src="aum_history_table"))
        linked_name_count += 1

    cfg: dict[str, object] = {}
    if linked_code_count > 0:
        out["台股代號"] = code_links
        cfg["台股代號"] = st.column_config.LinkColumn(
            label="台股代號",
            help="點擊代號可帶入回測工作台並自動執行回測",
            display_text=r"bt_symbol=([^&]+)",
            max_chars=20,
        )
    if linked_name_count > 0:
        out["ETF名稱"] = name_links
        cfg["ETF名稱"] = st.column_config.LinkColumn(
            label="ETF名稱",
            help="點擊 ETF 名稱可在新分頁開啟對應熱力圖（內容比照 00935 熱力圖）",
            display_text=r"hm_label=([^&]+)",
        )
    return out, cfg


def _tw_etf_management_fee_config_path() -> Path:
    return Path(__file__).resolve().parent / "conf" / "tw_etf_management_fees.json"


def _tw_etf_management_fee_config_signature() -> str:
    cfg_path = _tw_etf_management_fee_config_path()
    try:
        stat = cfg_path.stat()
    except Exception:
        return "missing"
    return f"{int(stat.st_mtime_ns)}:{int(stat.st_size)}"


@st.cache_data(ttl=120, show_spinner=False)
def _load_tw_etf_management_fee_whitelist(config_signature: str) -> dict[str, str]:
    _ = config_signature
    out = dict(TW_ETF_MANAGEMENT_FEE_FALLBACK)
    cfg_path = _tw_etf_management_fee_config_path()
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    raw = payload.get("fees") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        code = str(key or "").strip().upper()
        if not code:
            continue
        label = _normalize_tw_etf_management_fee_label(value)
        if not label:
            continue
        out[code] = label
    return out


def _get_tw_etf_management_fee_whitelist() -> dict[str, str]:
    return _load_tw_etf_management_fee_whitelist(_tw_etf_management_fee_config_signature())


def _lookup_tw_etf_management_fee_label(code: object) -> str:
    token = str(code or "").strip().upper()
    if not token or token.startswith("^"):
        return ""
    value = _get_tw_etf_management_fee_whitelist().get(token)
    if value is None:
        return ""
    return _normalize_tw_etf_management_fee_label(value)


def _lookup_tw_etf_management_fee_pct(code: object) -> float | None:
    label = _lookup_tw_etf_management_fee_label(code)
    if not label:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", label)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _truncate_series(values: pd.Series, *, digits: int = 2) -> pd.Series:
    factor = 10.0 ** max(0, int(digits))
    numeric = pd.to_numeric(values, errors="coerce")
    out = np.trunc(numeric * factor) / factor
    return pd.Series(out, index=numeric.index, dtype="float64")


def _truncate_integer_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    truncated = pd.Series(np.trunc(numeric), index=numeric.index)
    return truncated.where(pd.notna(numeric)).astype("Int64")


def _truncate_value(value: object, *, digits: int = 2) -> float | None:
    number = _safe_float(value)
    if number is None or (not math.isfinite(number)):
        return None
    factor = 10.0 ** max(0, int(digits))
    return math.trunc(float(number) * factor) / factor


def _format_tw_etf_management_fee(code: object) -> str:
    label = _lookup_tw_etf_management_fee_label(code)
    if not label:
        return "—"
    return label


def _attach_tw_etf_management_fee_column(
    frame: pd.DataFrame,
    *,
    code_col_candidates: tuple[str, ...] = ("代碼", "ETF代碼"),
    column_name: str = "管理費(%)",
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    code_col = next((col for col in code_col_candidates if col in frame.columns), "")
    if not code_col:
        return frame

    out = frame.copy()
    out[column_name] = _truncate_series(
        out[code_col].map(_lookup_tw_etf_management_fee_pct), digits=2
    )

    anchor_col = next(
        (col for col in ("ETF", "ETF名稱", "代碼", "ETF代碼") if col in out.columns), ""
    )
    if not anchor_col:
        return out
    cols = list(out.columns)
    if column_name in cols:
        cols.remove(column_name)
        insert_at = cols.index(anchor_col) + 1
        cols.insert(insert_at, column_name)
        out = out[cols]
    return _attach_tw_etf_aum_column(out, code_col_candidates=code_col_candidates)


def _classify_tw_etf(name: str, code: str = "") -> str:
    text = str(name or "").strip()
    code_text = str(code or "").strip().upper()
    if code_text and code_text in TW_ETF_TYPE_WHITELIST:
        return str(TW_ETF_TYPE_WHITELIST[code_text])
    if code_text.endswith("A"):
        return "主動式"
    if not text:
        return "其他"
    text_upper = text.upper()
    dividend_keywords = ("高股息", "股利", "股息", "收益", "配息", "月配", "季配", "年配")
    market_cap_keywords = ("台灣50", "臺灣50", "台灣中型100", "臺灣中型100", "市值", "台灣領袖")
    active_keywords = ("主動", "ACTIVE")
    esg_keywords = ("ESG", "永續", "低碳", "減碳", "碳中和", "碳權", "綠能", "潔淨能源")
    tech_keywords = ("科技", "半導體", "電子", "晶片", "AI", "人工智慧", "5G", "雲端", "電動車")
    finance_keywords = ("金融", "銀行", "證券", "保險")
    theme_keywords = ("生技", "醫療", "健康", "航太", "軍工", "內需", "消費", "高端製造", "供應鏈")
    if any(key in text for key in dividend_keywords):
        return "股利型"
    if any(key in text for key in market_cap_keywords):
        return "市值型"
    if any((key in text) or (key in text_upper) for key in active_keywords):
        return "主動式"
    if any((key in text) or (key in text_upper) for key in esg_keywords):
        return "永續ESG型"
    if any((key in text) or (key in text_upper) for key in tech_keywords):
        return "科技型"
    if any(key in text for key in finance_keywords):
        return "金融型"
    if any(key in text for key in theme_keywords):
        return "產業主題型"
    return "其他"


def _is_tw_equity_etf(code: str, name: str) -> bool:
    code_text = str(code or "").strip().upper()
    name_text = str(name or "").strip()
    if not code_text.startswith("00"):
        return False
    if code_text.endswith(("L", "R")):
        return False
    if not name_text:
        return False

    exclude_keywords = (
        "原油",
        "黃金",
        "白銀",
        "布蘭特",
        "日圓",
        "美元",
        "美債",
        "公司債",
        "投等債",
        "非投等債",
        "中國",
        "日本",
        "日經",
        "越南",
        "印度",
        "美國",
        "全球",
        "道瓊",
        "NASDAQ",
        "S&P",
        "反1",
        "正2",
    )
    if any(token in name_text for token in exclude_keywords):
        return False
    return True


def _is_tw_any_etf(code: str, name: str) -> bool:
    code_text = str(code or "").strip().upper()
    name_text = str(name or "").strip()
    if not code_text.startswith("00"):
        return False
    if not name_text:
        return False
    return True


def _is_tw_active_etf(code: str, name: str) -> bool:
    code_text = str(code or "").strip().upper()
    name_text = str(name or "").strip()
    if not _is_tw_equity_etf(code_text, name_text):
        return False
    if not code_text.endswith("A"):
        return False
    if "主動" not in name_text:
        return False
    return True


def _split_factor_and_events_between(
    symbol: str, start_used: str, end_used: str
) -> tuple[float, str]:
    factor = 1.0
    tags: list[str] = []
    start_used_ts = pd.Timestamp(start_used, tz="UTC")
    end_used_ts = pd.Timestamp(end_used, tz="UTC")
    for ev in known_split_events(symbol=str(symbol), market="TW"):
        ev_ts = pd.Timestamp(ev.date)
        if ev_ts.tzinfo is None:
            ev_ts = ev_ts.tz_localize("UTC")
        else:
            ev_ts = ev_ts.tz_convert("UTC")
        if start_used_ts < ev_ts <= end_used_ts:
            ratio = float(ev.ratio)
            if ratio > 0:
                factor *= ratio
                tags.append(f"{ev_ts.date()} x{ratio:.6f}")
    return float(factor), ", ".join(tags)


@st.cache_data(ttl=3600, show_spinner=False)
def _build_tw_etf_top10_between(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    type_filter: str | None = None,
    top_n: int = 10,
    sort_ascending: bool = False,
    exclude_split_event: bool = False,
    include_all_etf: bool = False,
) -> tuple[pd.DataFrame, str, str, int]:
    start_used, start_df = _fetch_twse_snapshot_with_fallback(start_yyyymmdd)
    end_used, end_df = _fetch_twse_snapshot_with_fallback(end_yyyymmdd)

    # 預設為台股股票型 ETF；可切換為納入所有 ETF 類型。
    etf_filter_fn = _is_tw_any_etf if bool(include_all_etf) else _is_tw_equity_etf
    start_df = start_df[
        start_df.apply(
            lambda r: etf_filter_fn(str(r.get("code", "")), str(r.get("name", ""))), axis=1
        )
    ].copy()
    end_df = end_df[
        end_df.apply(
            lambda r: etf_filter_fn(str(r.get("code", "")), str(r.get("name", ""))), axis=1
        )
    ].copy()
    if end_df.empty:
        return pd.DataFrame(), start_used, end_used, 0

    start_close_map: dict[str, float] = {}
    if not start_df.empty:
        start_close_series = pd.to_numeric(start_df["close"], errors="coerce")
        for code, close in zip(start_df["code"].astype(str), start_close_series, strict=False):
            code_text = str(code).strip().upper()
            close_value = float(close) if pd.notna(close) else float("nan")
            if code_text and math.isfinite(close_value) and close_value > 0:
                start_close_map[code_text] = close_value

    start_trade_date_map: dict[str, str] = {}
    if start_close_map:
        start_trade_date_map = dict.fromkeys(start_close_map, start_used)

    end_dt = datetime.combine(
        datetime.strptime(end_used, "%Y%m%d").date(), datetime.min.time()
    ).replace(tzinfo=timezone.utc)
    start_dt = datetime.combine(
        datetime.strptime(start_used, "%Y%m%d").date(), datetime.min.time()
    ).replace(tzinfo=timezone.utc)
    missing_symbols: list[str] = []
    for code in end_df["code"].astype(str).tolist():
        symbol = str(code).strip().upper()
        if not symbol or symbol in start_close_map or symbol in missing_symbols:
            continue
        missing_symbols.append(symbol)
    if missing_symbols:
        store = _history_store()
        for symbol in missing_symbols:
            bars = normalize_ohlcv_frame(
                store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            )
            need_sync = True
            if isinstance(bars, pd.DataFrame) and not bars.empty:
                idx = pd.to_datetime(bars.index, utc=True, errors="coerce").dropna()
                if not idx.empty:
                    last_ts = pd.Timestamp(idx.max()).to_pydatetime().replace(tzinfo=timezone.utc)
                    need_sync = last_ts < end_dt
            if need_sync:
                store.sync_symbol_history(symbol=symbol, market="TW", start=start_dt, end=end_dt)
                bars = normalize_ohlcv_frame(
                    store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
                )
            if bars.empty or "close" not in bars.columns:
                continue
            closes = pd.to_numeric(bars["close"], errors="coerce")
            closes.index = pd.to_datetime(closes.index, utc=True, errors="coerce")
            closes = closes.dropna().sort_index()
            closes = closes[(closes.index >= start_dt) & (closes.index <= end_dt)]
            if closes.empty:
                continue
            first_close = float(closes.iloc[0])
            if not math.isfinite(first_close) or first_close <= 0:
                continue
            start_close_map[symbol] = first_close
            start_trade_date_map[symbol] = pd.Timestamp(closes.index[0]).strftime("%Y%m%d")

    merged = end_df.copy()
    merged["code"] = merged["code"].astype(str).str.strip().str.upper()
    merged["end_close"] = pd.to_numeric(merged["close"], errors="coerce")
    merged["start_close"] = merged["code"].map(start_close_map)
    merged["start_trade_date"] = merged["code"].map(start_trade_date_map)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["start_close", "end_close", "start_trade_date"]
    )
    if merged.empty:
        return pd.DataFrame(), start_used, end_used, 0
    merged["split_info"] = merged.apply(
        lambda r: _split_factor_and_events_between(
            symbol=str(r.get("code", "")),
            start_used=str(r.get("start_trade_date", "")),
            end_used=end_used,
        ),
        axis=1,
    )
    merged["split_factor"] = merged["split_info"].map(lambda x: float(x[0]))
    merged["split_events"] = merged["split_info"].map(lambda x: str(x[1]))
    merged = merged.drop(columns=["split_info"])
    if bool(exclude_split_event):
        merged = merged[merged["split_events"].astype(str).str.strip() == ""].copy()
        if merged.empty:
            return pd.DataFrame(), start_used, end_used, 0
    merged["adj_start_close"] = pd.to_numeric(
        merged["start_close"], errors="coerce"
    ) * pd.to_numeric(merged["split_factor"], errors="coerce")
    merged["return_pct"] = (
        pd.to_numeric(merged["end_close"], errors="coerce")
        / pd.to_numeric(merged["adj_start_close"], errors="coerce")
        - 1.0
    ) * 100.0
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["return_pct", "start_close", "end_close", "adj_start_close"]
    )
    if merged.empty:
        return pd.DataFrame(), start_used, end_used, 0

    merged["type"] = merged.apply(
        lambda r: _classify_tw_etf(
            str(r.get("name", "")),
            code=str(r.get("code", "")),
        ),
        axis=1,
    )
    type_filter_text = str(type_filter or "").strip()
    if type_filter_text:
        merged = merged[merged["type"] == type_filter_text].copy()
    if merged.empty:
        return pd.DataFrame(), start_used, end_used, 0
    universe_count = int(len(merged))

    top_n_value = int(top_n) if int(top_n) > 0 else 10
    merged = (
        merged.sort_values("return_pct", ascending=bool(sort_ascending)).head(top_n_value).copy()
    )
    merged["rank"] = range(1, len(merged) + 1)
    out = merged[
        [
            "rank",
            "code",
            "name",
            "type",
            "adj_start_close",
            "end_close",
            "return_pct",
        ]
    ].copy()
    out = out.rename(
        columns={
            "rank": "排名",
            "code": "代碼",
            "name": "ETF",
            "type": "類型",
            "adj_start_close": "開盤",
            "end_close": "收盤",
            "return_pct": "區間報酬(%)",
        }
    )
    out["區間報酬(%)"] = pd.to_numeric(out["區間報酬(%)"], errors="coerce").round(2)
    out["開盤"] = pd.to_numeric(out["開盤"], errors="coerce").round(2)
    out["收盤"] = pd.to_numeric(out["收盤"], errors="coerce").round(2)
    out = _attach_tw_etf_management_fee_column(out, code_col_candidates=("代碼",))
    return out, start_used, end_used, universe_count


@st.cache_data(ttl=3600, show_spinner=False)
def _build_tw_active_etf_ytd_between(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    symbols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, str, str]:
    start_used, start_df = _fetch_twse_snapshot_with_fallback(start_yyyymmdd)
    end_used, end_df = _fetch_twse_snapshot_with_fallback(end_yyyymmdd)

    end_active_df = end_df[
        end_df.apply(
            lambda r: _is_tw_active_etf(str(r.get("code", "")), str(r.get("name", ""))), axis=1
        )
    ].copy()
    if end_active_df.empty and not symbols:
        return pd.DataFrame(), start_used, end_used

    requested_symbols: list[str] = []
    for token in symbols:
        sym = str(token or "").strip().upper()
        if sym and sym not in requested_symbols:
            requested_symbols.append(sym)
    if requested_symbols:
        universe_symbols = requested_symbols
    else:
        universe_symbols = [
            str(v).strip().upper()
            for v in end_active_df["code"].astype(str).tolist()
            if str(v).strip()
        ]

    if not universe_symbols:
        return pd.DataFrame(), start_used, end_used

    end_name_map = {
        str(row["code"]).strip().upper(): str(row["name"]).strip()
        for _, row in end_active_df.iterrows()
        if str(row.get("code", "")).strip()
    }

    start_dt = datetime.combine(
        datetime.strptime(start_yyyymmdd, "%Y%m%d").date(), datetime.min.time()
    ).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(
        datetime.strptime(end_used, "%Y%m%d").date(), datetime.min.time()
    ).replace(tzinfo=timezone.utc)
    store = _history_store()

    rows: list[dict[str, object]] = []
    for symbol in universe_symbols:
        bars = normalize_ohlcv_frame(
            store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
        )
        need_sync = True
        if isinstance(bars, pd.DataFrame) and not bars.empty:
            idx = pd.to_datetime(bars.index, utc=True, errors="coerce").dropna()
            if not idx.empty:
                last_ts = pd.Timestamp(idx.max()).to_pydatetime().replace(tzinfo=timezone.utc)
                need_sync = last_ts < end_dt
        if need_sync:
            store.sync_symbol_history(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            bars = normalize_ohlcv_frame(
                store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            )
        if bars.empty or "close" not in bars.columns:
            continue

        closes = pd.to_numeric(bars["close"], errors="coerce")
        closes.index = pd.to_datetime(closes.index, utc=True, errors="coerce")
        closes = closes.dropna().sort_index()
        closes = closes[(closes.index >= start_dt) & (closes.index <= end_dt)]
        if closes.empty:
            continue

        first_ts = pd.Timestamp(closes.index[0])
        last_ts = pd.Timestamp(closes.index[-1])
        start_close = float(closes.iloc[0])
        end_close = float(closes.iloc[-1])
        if not math.isfinite(start_close) or not math.isfinite(end_close) or start_close <= 0:
            continue

        first_token = first_ts.strftime("%Y%m%d")
        last_token = last_ts.strftime("%Y%m%d")
        split_factor, split_events = _split_factor_and_events_between(
            symbol=symbol, start_used=first_token, end_used=last_token
        )
        adj_start_close = start_close * float(split_factor)
        if not math.isfinite(adj_start_close) or adj_start_close <= 0:
            continue
        ret_pct = (end_close / adj_start_close - 1.0) * 100.0
        if not math.isfinite(ret_pct):
            continue

        rows.append(
            {
                "code": symbol,
                "name": end_name_map.get(symbol, symbol),
                "start_close": start_close,
                "adj_start_close": adj_start_close,
                "end_close": end_close,
                "start_trade_date": first_token,
                "end_trade_date": last_token,
                "split_events": split_events,
                "return_pct": ret_pct,
            }
        )

    if not rows:
        return pd.DataFrame(), start_used, end_used

    merged = pd.DataFrame(rows).sort_values("return_pct", ascending=False).copy()
    merged["rank"] = range(1, len(merged) + 1)
    out = merged[
        [
            "rank",
            "code",
            "name",
            "adj_start_close",
            "end_close",
            "start_trade_date",
            "end_trade_date",
            "return_pct",
        ]
    ].copy()
    out = out.rename(
        columns={
            "rank": "排名",
            "code": "代碼",
            "name": "ETF",
            "adj_start_close": "開盤",
            "end_close": "收盤",
            "start_trade_date": "績效起算日",
            "end_trade_date": "績效終點日",
            "return_pct": "YTD報酬(%)",
        }
    )
    out["YTD報酬(%)"] = pd.to_numeric(out["YTD報酬(%)"], errors="coerce").round(2)
    out["開盤"] = pd.to_numeric(out["開盤"], errors="coerce").round(2)
    out["收盤"] = pd.to_numeric(out["收盤"], errors="coerce").round(2)
    out = _attach_tw_etf_management_fee_column(out, code_col_candidates=("代碼",))
    return out, start_used, end_used


@st.cache_data(ttl=600, show_spinner=False)
def _load_twii_twse_month_close_map(
    month_anchor_yyyymmdd: str,
) -> tuple[dict[str, float], list[str]]:
    import requests

    token = re.sub(r"\D", "", str(month_anchor_yyyymmdd or "").strip())
    if not re.fullmatch(r"\d{8}", token):
        return {}, [f"invalid month anchor: {month_anchor_yyyymmdd}"]
    query_token = f"{token[:6]}01"
    try:
        resp = requests.get(
            "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST",
            params={"response": "json", "date": query_token},
            timeout=18,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return {}, [f"{query_token}: {exc}"]

    stat = str(payload.get("stat", "")).strip().upper()
    if stat != "OK":
        return {}, [f"{query_token}: {payload.get('stat', 'not ok')}"]
    fields = payload.get("fields", [])
    rows = payload.get("data", [])
    if not isinstance(fields, list) or not isinstance(rows, list):
        return {}, [f"{query_token}: malformed payload"]
    if "日期" not in fields or "收盤指數" not in fields:
        return {}, [f"{query_token}: fields missing"]
    idx_date = fields.index("日期")
    idx_close = fields.index("收盤指數")

    out: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, list):
            continue
        if max(idx_date, idx_close) >= len(row):
            continue
        roc_date = str(row[idx_date] or "").strip()
        close_raw = str(row[idx_close] or "").strip().replace(",", "")
        m = re.fullmatch(r"(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})", roc_date)
        if not m:
            continue
        try:
            year = int(m.group(1)) + 1911
            month = int(m.group(2))
            day = int(m.group(3))
            date_token = datetime(year, month, day).strftime("%Y%m%d")
            close_val = float(close_raw)
        except Exception:
            continue
        if not math.isfinite(close_val) or close_val <= 0:
            continue
        out[date_token] = float(close_val)
    if not out:
        return {}, [f"{query_token}: no parsable rows"]
    return out, []


def _month_anchor_tokens_between(start_yyyymmdd: str, end_yyyymmdd: str) -> list[str]:
    start_dt = datetime.strptime(start_yyyymmdd, "%Y%m%d").date().replace(day=1)
    end_dt = datetime.strptime(end_yyyymmdd, "%Y%m%d").date().replace(day=1)
    out: list[str] = []
    cursor = start_dt
    while cursor <= end_dt:
        out.append(cursor.strftime("%Y%m01"))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return out


@st.cache_data(ttl=600, show_spinner=False)
def _load_twii_twse_return_between(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> tuple[float | None, list[str]]:
    start_token = re.sub(r"\D", "", str(start_yyyymmdd or "").strip())
    end_token = re.sub(r"\D", "", str(end_yyyymmdd or "").strip())
    if not re.fullmatch(r"\d{8}", start_token) or not re.fullmatch(r"\d{8}", end_token):
        return None, [f"invalid date range: {start_yyyymmdd}->{end_yyyymmdd}"]
    if end_token <= start_token:
        return None, [f"non-positive range: {start_token}->{end_token}"]

    issues: list[str] = []
    close_map: dict[str, float] = {}
    for month_token in _month_anchor_tokens_between(start_token, end_token):
        month_map, month_issues = _load_twii_twse_month_close_map(month_token)
        if month_map:
            close_map.update(month_map)
        if month_issues:
            issues.extend(month_issues)
    if not close_map:
        return None, issues

    eligible = [
        (token, value)
        for token, value in close_map.items()
        if start_token <= str(token) <= end_token
        and math.isfinite(float(value))
        and float(value) > 0
    ]
    eligible = sorted(eligible, key=lambda x: x[0])
    if len(eligible) < 2:
        issues.append(f"{start_token}->{end_token}: insufficient_rows({len(eligible)})")
        return None, issues
    base_val = float(eligible[0][1])
    end_val = float(eligible[-1][1])
    if not math.isfinite(base_val) or not math.isfinite(end_val) or base_val <= 0:
        issues.append(f"{start_token}->{end_token}: invalid close values")
        return None, issues
    return (end_val / base_val - 1.0) * 100.0, issues


@st.cache_data(ttl=600, show_spinner=False)
def _load_tw_market_return_between(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    *,
    force_sync: bool = False,
) -> tuple[float | None, str, list[str]]:
    start_dt = datetime.combine(
        datetime.strptime(start_yyyymmdd, "%Y%m%d").date(), datetime.min.time()
    ).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(
        datetime.strptime(end_yyyymmdd, "%Y%m%d").date(), datetime.min.time()
    ).replace(tzinfo=timezone.utc)
    if end_dt <= start_dt:
        return None, "", []

    issues: list[str] = []
    twii_return, twii_issues = _load_twii_twse_return_between(
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
    )
    if twii_issues:
        issues.extend([f"TWSE:^TWII {msg}" for msg in twii_issues])
    if twii_return is not None and math.isfinite(float(twii_return)):
        return float(twii_return), "^TWII", issues

    store = _history_store()
    candidates = ["^TWII", "0050", "006208"]
    for symbol in candidates:
        if force_sync:
            report = store.sync_symbol_history(
                symbol=symbol, market="TW", start=start_dt, end=end_dt
            )
            err = str(getattr(report, "error", "") or "").strip()
            if err:
                issues.append(f"{symbol}: {err}")

        bars = normalize_ohlcv_frame(
            store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
        )
        if _bars_need_backfill(bars, start=start_dt, end=end_dt):
            report = store.sync_symbol_history(
                symbol=symbol, market="TW", start=start_dt, end=end_dt
            )
            err = str(getattr(report, "error", "") or "").strip()
            if err:
                issues.append(f"{symbol}: {err}")
            bars = normalize_ohlcv_frame(
                store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            )
        if len(bars) < 2 or "close" not in bars.columns:
            continue

        if symbol in {"0050", "006208"}:
            bars, _ = apply_split_adjustment(
                bars=bars,
                symbol=symbol,
                market="TW",
                use_known=True,
                use_auto_detect=True,
            )
        if len(bars) < 2:
            continue
        closes = pd.to_numeric(bars["close"], errors="coerce")
        closes.index = pd.to_datetime(closes.index, utc=True, errors="coerce")
        closes = closes.dropna().sort_index()
        closes = closes[(closes.index >= start_dt) & (closes.index <= end_dt)]
        if len(closes) < 2:
            continue
        base_val = float(closes.iloc[0])
        end_val = float(closes.iloc[-1])
        if not math.isfinite(base_val) or not math.isfinite(end_val) or base_val <= 0:
            continue
        return (end_val / base_val - 1.0) * 100.0, symbol, issues

    return None, "", issues


@st.cache_data(ttl=600, show_spinner=False)
def _load_tw_snapshot_close_map(target_yyyymmdd: str) -> tuple[str, dict[str, float]]:
    used, frame = _fetch_twse_snapshot_with_fallback(target_yyyymmdd)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return used, {}
    out: dict[str, float] = {}
    for _, row in frame.iterrows():
        code = str(row.get("code", "")).strip().upper()
        if not code:
            continue
        close_val = _safe_float(row.get("close"))
        if close_val is None or (not math.isfinite(close_val)) or close_val <= 0:
            continue
        out[code] = float(close_val)
    return used, out


@st.cache_data(ttl=600, show_spinner=False)
def _load_tw_snapshot_open_map(target_yyyymmdd: str) -> tuple[str, dict[str, float]]:
    used, frame = _fetch_twse_snapshot_with_fallback(target_yyyymmdd)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return used, {}
    out: dict[str, float] = {}
    for _, row in frame.iterrows():
        code = str(row.get("code", "")).strip().upper()
        if not code:
            continue
        open_val = _safe_float(row.get("open"))
        if open_val is None or (not math.isfinite(open_val)) or open_val <= 0:
            continue
        out[code] = float(open_val)
    return used, out


@st.cache_data(ttl=600, show_spinner=False)
def _load_tw_etf_daily_change_map(
    anchor_yyyymmdd: str,
) -> tuple[dict[str, float], str, str]:
    end_used, end_close_map = _load_tw_snapshot_close_map(anchor_yyyymmdd)
    end_token = re.sub(r"\D", "", str(end_used or "").strip()) or str(anchor_yyyymmdd or "").strip()
    trade_days = _recent_twse_trading_days(anchor_yyyymmdd=end_token, count=2, max_scan_days=20)
    if len(trade_days) < 2:
        return {}, end_used, ""
    prev_token = trade_days[-2]
    prev_used, prev_close_map = _load_tw_snapshot_close_map(prev_token)
    out: dict[str, float] = {}
    for code, end_close in end_close_map.items():
        prev_close = prev_close_map.get(code)
        if prev_close is None or prev_close <= 0:
            continue
        pct = (float(end_close) / float(prev_close) - 1.0) * 100.0
        if not math.isfinite(pct):
            continue
        out[code] = float(pct)
    return out, end_used, prev_used


@st.cache_data(ttl=600, show_spinner=False)
def _load_tw_market_daily_return(
    anchor_yyyymmdd: str,
    *,
    force_sync: bool = False,
) -> tuple[float | None, str, str, str, list[str]]:
    token = re.sub(r"\D", "", str(anchor_yyyymmdd or "").strip())
    if not token:
        token = datetime.now().strftime("%Y%m%d")
    trade_days = _recent_twse_trading_days(anchor_yyyymmdd=token, count=2, max_scan_days=20)
    if len(trade_days) < 2:
        return None, "", "", "", []
    prev_token = trade_days[-2]
    end_token = trade_days[-1]
    value, symbol, issues = _load_tw_market_return_between(
        start_yyyymmdd=prev_token,
        end_yyyymmdd=end_token,
        force_sync=force_sync,
    )
    return value, symbol, prev_token, end_token, issues


def _with_tw_today_fields(
    frame: pd.DataFrame,
    *,
    daily_change_map: dict[str, float] | None = None,
    daily_open_map: dict[str, float] | None = None,
    market_daily_return_pct: float | None = None,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    out = frame.copy()
    code_series = out.get("代碼", pd.Series(dtype=str)).astype(str).str.strip().str.upper()

    open_map = daily_open_map if isinstance(daily_open_map, dict) else {}
    if open_map:
        out["開盤"] = pd.to_numeric(code_series.map(open_map), errors="coerce").round(2)
    elif "開盤" in out.columns:
        out["開盤"] = pd.to_numeric(out["開盤"], errors="coerce").round(2)
    elif "復權期初" in out.columns:
        out["開盤"] = pd.to_numeric(out["復權期初"], errors="coerce").round(2)
    elif "期初收盤" in out.columns:
        out["開盤"] = pd.to_numeric(out["期初收盤"], errors="coerce").round(2)
    if "收盤" in out.columns:
        out["收盤"] = pd.to_numeric(out["收盤"], errors="coerce").round(2)
    elif "期末收盤" in out.columns:
        out["收盤"] = pd.to_numeric(out["期末收盤"], errors="coerce").round(2)

    daily_map = daily_change_map or {}
    out["今日漲幅"] = pd.to_numeric(code_series.map(daily_map), errors="coerce").round(2)

    market_daily = _safe_float(market_daily_return_pct)
    if market_daily is not None and math.isfinite(float(market_daily)):
        benchmark_mask = code_series.str.startswith("^")
        out.loc[benchmark_mask, "今日漲幅"] = float(market_daily)
        out["今日贏大盤%"] = (
            pd.to_numeric(out["今日漲幅"], errors="coerce") - float(market_daily)
        ).round(2)
        out.loc[benchmark_mask, "今日贏大盤%"] = 0.0
    else:
        out["今日贏大盤%"] = np.nan

    drop_cols = [
        "期初收盤",
        "復權期初",
        "期末收盤",
        "復權事件",
        "start_close",
        "adj_start_close",
        "end_close",
    ]
    return out.drop(columns=[col for col in drop_cols if col in out.columns], errors="ignore")


def _attach_rank_movement_columns(
    frame: pd.DataFrame,
    *,
    previous_rank_map: dict[str, int] | None = None,
    code_col: str = "代碼",
    rank_col: str = "排名",
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    out = frame.copy()
    prev_rank_lookup = {
        str(code).strip().upper(): int(rank)
        for code, rank in (previous_rank_map or {}).items()
        if str(code).strip() and int(rank) > 0
    }
    code_series = out.get(code_col, pd.Series(dtype=str)).astype(str).str.strip().str.upper()
    current_rank_series = pd.to_numeric(out.get(rank_col, pd.Series(dtype=float)), errors="coerce")
    previous_rank_series = (
        code_series.map(prev_rank_lookup)
        if prev_rank_lookup
        else pd.Series([np.nan] * len(out), index=out.index)
    )
    rank_display: list[str] = []
    for idx in out.index:
        code = str(code_series.get(idx, "")).strip().upper()
        current_rank_raw = current_rank_series.get(idx)
        previous_rank_raw = previous_rank_series.get(idx)
        if code.startswith("^"):
            rank_display.append(str(out.at[idx, rank_col]))
            continue
        if pd.isna(current_rank_raw):
            rank_display.append(str(out.at[idx, rank_col]))
            continue
        current_rank = int(float(current_rank_raw))
        if not prev_rank_lookup:
            movement = "—"
        elif pd.isna(previous_rank_raw):
            movement = "新進榜"
        else:
            prev_rank = int(float(previous_rank_raw))
            diff = prev_rank - current_rank
            if diff > 0:
                movement = f"↑{diff}"
            elif diff < 0:
                movement = f"↓{abs(diff)}"
            else:
                movement = "持平"
        rank_display.append(f"{current_rank}  ({movement})")
    out[rank_col] = rank_display
    return out


def _style_tw_today_move_table(frame: pd.DataFrame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    work = frame.copy()
    for col in ("今日漲幅", "今日贏大盤%"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ("開盤", "收盤"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    def _apply_row(row: pd.Series) -> list[str]:
        styles: list[str] = []
        today_move = _safe_float(row.get("今日漲幅")) if "今日漲幅" in work.columns else None
        today_vs_market = (
            _safe_float(row.get("今日贏大盤%")) if "今日贏大盤%" in work.columns else None
        )
        for col in work.columns:
            style_parts: list[str] = []
            if col == "今日漲幅" and today_move is not None:
                if today_move > 0:
                    style_parts.append("background-color: #e8f7e8")
                elif today_move < 0:
                    style_parts.append("background-color: #fdecec")
            if col == "今日贏大盤%" and today_vs_market is not None:
                if today_vs_market > 0:
                    style_parts.append("color: #1f7a1f")
                elif today_vs_market < 0:
                    style_parts.append("color: #b42318")
            styles.append(";".join(style_parts))
        return styles

    styler = work.style.apply(_apply_row, axis=1)
    formatters: dict[str, Any] = {}
    if "今日漲幅" in work.columns:
        formatters["今日漲幅"] = lambda v: "—" if pd.isna(v) else f"{float(v):.2f}%"
    if "今日贏大盤%" in work.columns:
        formatters["今日贏大盤%"] = lambda v: "—" if pd.isna(v) else f"{float(v):.2f}%"
    if "開盤" in work.columns:
        formatters["開盤"] = lambda v: "—" if pd.isna(v) else f"{float(v):.2f}"
    if "收盤" in work.columns:
        formatters["收盤"] = lambda v: "—" if pd.isna(v) else f"{float(v):.2f}"
    if "管理費(%)" in work.columns:
        formatters["管理費(%)"] = lambda v: "—" if pd.isna(v) else f"{float(v):.2f}"
    if "ETF規模(億)" in work.columns:
        formatters["ETF規模(億)"] = lambda v: "—" if pd.isna(v) else f"{int(float(v))}"
    if "2025績效(%)" in work.columns:
        formatters["2025績效(%)"] = lambda v: "—" if pd.isna(v) else f"{float(v):.2f}"
    if "輸贏大盤2025(%)" in work.columns:
        formatters["輸贏大盤2025(%)"] = lambda v: "—" if pd.isna(v) else f"{float(v):.2f}"
    if "2026YTD績效(%)" in work.columns:
        formatters["2026YTD績效(%)"] = lambda v: "—" if pd.isna(v) else f"{float(v):.2f}"
    if "輸贏大盤2026YTD(%)" in work.columns:
        formatters["輸贏大盤2026YTD(%)"] = lambda v: "—" if pd.isna(v) else f"{float(v):.2f}"
    if formatters:
        styler = styler.format(formatters)
    return styler


def _decorate_tw_etf_top10_ytd_table(
    top10_df: pd.DataFrame,
    *,
    compare_return_map: dict[str, float],
    market_return_pct: float | None,
    market_compare_return_pct: float | None,
    benchmark_code: str,
    end_used: str,
    compare_col_label: str = "2025績效(%)",
    performance_col_label: str = "YTD報酬(%)",
    underperform_col_label: str | None = None,
) -> pd.DataFrame:
    etf_df = top10_df.copy()
    if "區間報酬(%)" in etf_df.columns and performance_col_label not in etf_df.columns:
        etf_df = etf_df.rename(columns={"區間報酬(%)": performance_col_label})
    if performance_col_label not in etf_df.columns:
        etf_df[performance_col_label] = np.nan

    code_series = etf_df.get("代碼", pd.Series(dtype=str)).astype(str).str.strip().str.upper()
    etf_df[compare_col_label] = code_series.map(compare_return_map)
    etf_df[compare_col_label] = _truncate_series(etf_df[compare_col_label], digits=2)
    etf_df["贏輸台股大盤(%)"] = np.nan
    if market_return_pct is not None and math.isfinite(float(market_return_pct)):
        etf_df["贏輸台股大盤(%)"] = _truncate_series(
            pd.to_numeric(etf_df[performance_col_label], errors="coerce")
            - float(market_return_pct),
            digits=2,
        )
    underperform_label = str(underperform_col_label or "").strip()
    if underperform_label:
        etf_df[underperform_label] = np.nan
        if market_return_pct is not None and math.isfinite(float(market_return_pct)):
            underperform_series = float(market_return_pct) - pd.to_numeric(
                etf_df[performance_col_label], errors="coerce"
            )
            etf_df[underperform_label] = _truncate_series(
                underperform_series.clip(lower=0.0),
                digits=2,
            )

    benchmark_row = {
        "排名": "—",
        "代碼": str(benchmark_code or "^TWII"),
        "ETF": "台股大盤",
        "類型": "大盤",
        "開盤": np.nan,
        "收盤": np.nan,
        compare_col_label: (
            _truncate_value(market_compare_return_pct, digits=2)
            if market_compare_return_pct is not None
            else np.nan
        ),
        performance_col_label: (
            _truncate_value(market_return_pct, digits=2)
            if market_return_pct is not None
            else np.nan
        ),
        "贏輸台股大盤(%)": 0.0 if market_return_pct is not None else np.nan,
        "績效終點日": str(end_used),
    }
    if underperform_label:
        benchmark_row[underperform_label] = 0.0 if market_return_pct is not None else np.nan
    table_df = pd.concat([pd.DataFrame([benchmark_row]), etf_df], ignore_index=True)
    table_df = _attach_tw_etf_management_fee_column(table_df, code_col_candidates=("代碼",))
    if "排名" in table_df.columns:
        table_df["排名"] = table_df["排名"].map(lambda v: str(v) if pd.notna(v) else "")
    columns_order = [
        "排名",
        "代碼",
        "ETF",
        "管理費(%)",
        "ETF規模(億)",
        "類型",
        "開盤",
        "收盤",
        compare_col_label,
        performance_col_label,
        "贏輸台股大盤(%)",
    ]
    if underperform_label:
        columns_order.append(underperform_label)
    return table_df[[col for col in columns_order if col in table_df.columns]]


@st.cache_data(ttl=1800, show_spinner=False)
def _build_tw_etf_all_types_performance_table(
    *,
    ytd_start_yyyymmdd: str,
    ytd_end_yyyymmdd: str,
    compare_start_yyyymmdd: str = "20250101",
    compare_end_yyyymmdd: str = "20251231",
) -> tuple[pd.DataFrame, dict[str, object]]:
    ytd_df, ytd_start_used, ytd_end_used, universe_count = _build_tw_etf_top10_between(
        start_yyyymmdd=ytd_start_yyyymmdd,
        end_yyyymmdd=ytd_end_yyyymmdd,
        top_n=99999,
        sort_ascending=False,
        include_all_etf=True,
    )
    if ytd_df.empty:
        return pd.DataFrame(), {
            "ytd_start_used": ytd_start_used,
            "ytd_end_used": ytd_end_used,
            "compare_start_used": compare_start_yyyymmdd,
            "compare_end_used": compare_end_yyyymmdd,
            "universe_count": int(universe_count),
            "issues": [],
        }

    compare_df, compare_start_used, compare_end_used, _ = _build_tw_etf_top10_between(
        start_yyyymmdd=compare_start_yyyymmdd,
        end_yyyymmdd=compare_end_yyyymmdd,
        top_n=99999,
        sort_ascending=False,
        include_all_etf=True,
    )
    compare_map: dict[str, float] = {}
    if isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
        for _, row in compare_df.iterrows():
            code = str(row.get("代碼", "")).strip().upper()
            value = _safe_float(row.get("區間報酬(%)"))
            if code and value is not None:
                compare_map[code] = float(value)

    market_ytd_return, market_ytd_symbol, market_ytd_issues = _load_tw_market_return_between(
        start_yyyymmdd=ytd_start_used,
        end_yyyymmdd=ytd_end_used,
        force_sync=False,
    )
    market_2025_return, market_2025_symbol, market_2025_issues = _load_tw_market_return_between(
        start_yyyymmdd=compare_start_used,
        end_yyyymmdd=compare_end_used,
        force_sync=False,
    )
    daily_change_map, daily_end_used, daily_prev_used = _load_tw_etf_daily_change_map(ytd_end_used)
    _, daily_open_map = _load_tw_snapshot_open_map(daily_end_used or ytd_end_used)
    market_daily_return, market_daily_symbol, _, _, market_daily_issues = (
        _load_tw_market_daily_return(ytd_end_used, force_sync=False)
    )

    table_df = ytd_df.rename(columns={"區間報酬(%)": "2026YTD績效(%)"}).copy()
    code_series = table_df.get("代碼", pd.Series(dtype=str)).astype(str).str.strip().str.upper()
    table_df["2025績效(%)"] = code_series.map(compare_map)
    table_df["2025績效(%)"] = _truncate_series(table_df["2025績效(%)"], digits=2)
    table_df["2026YTD績效(%)"] = _truncate_series(table_df["2026YTD績效(%)"], digits=2)
    table_df["輸贏大盤2025(%)"] = np.nan
    table_df["輸贏大盤2026YTD(%)"] = np.nan
    if market_2025_return is not None and math.isfinite(float(market_2025_return)):
        table_df["輸贏大盤2025(%)"] = _truncate_series(
            table_df["2025績效(%)"] - float(market_2025_return),
            digits=2,
        )
    if market_ytd_return is not None and math.isfinite(float(market_ytd_return)):
        table_df["輸贏大盤2026YTD(%)"] = _truncate_series(
            table_df["2026YTD績效(%)"] - float(market_ytd_return),
            digits=2,
        )

    table_df = _with_tw_today_fields(
        table_df,
        daily_change_map=daily_change_map,
        daily_open_map=daily_open_map,
        market_daily_return_pct=market_daily_return,
    )
    table_df = _attach_tw_etf_management_fee_column(table_df, code_col_candidates=("代碼",))
    table_df = table_df.sort_values(
        ["類型", "2026YTD績效(%)"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)
    table_df["編號"] = range(1, len(table_df) + 1)
    columns_order = [
        "編號",
        "代碼",
        "ETF",
        "管理費(%)",
        "ETF規模(億)",
        "類型",
        "2025績效(%)",
        "輸贏大盤2025(%)",
        "2026YTD績效(%)",
        "輸贏大盤2026YTD(%)",
        "開盤",
        "收盤",
        "今日漲幅",
        "今日贏大盤%",
    ]
    table_df = table_df[[col for col in columns_order if col in table_df.columns]]
    return table_df, {
        "ytd_start_used": ytd_start_used,
        "ytd_end_used": ytd_end_used,
        "daily_prev_used": daily_prev_used,
        "daily_end_used": daily_end_used,
        "compare_start_used": compare_start_used,
        "compare_end_used": compare_end_used,
        "universe_count": int(universe_count),
        "market_ytd_return": market_ytd_return,
        "market_ytd_symbol": market_ytd_symbol,
        "market_2025_return": market_2025_return,
        "market_2025_symbol": market_2025_symbol,
        "market_daily_return": market_daily_return,
        "market_daily_symbol": market_daily_symbol,
        "issues": [*market_ytd_issues, *market_2025_issues, *market_daily_issues],
    }


def _normalize_constituent_symbol(symbol: object, tw_code: object) -> str:
    code_token = str(tw_code or "").strip().upper()
    if re.fullmatch(r"\d{4,6}[A-Z]?", code_token):
        return code_token
    symbol_token = str(symbol or "").strip().upper()
    if not symbol_token:
        return ""
    tw_match = re.fullmatch(r"(\d{4,6}[A-Z]?)\.TW", symbol_token, flags=re.IGNORECASE)
    if tw_match:
        return tw_match.group(1).upper()
    return symbol_token


def _safe_weight_pct(value: object) -> float | None:
    text = str(value or "").strip().replace(",", "").replace("%", "")
    if not text:
        return None
    number = _safe_float(text)
    if number is None or number < 0:
        return None
    return float(number)


def _format_weight_pct_label(value: object) -> str:
    weight = _safe_weight_pct(value)
    if weight is None:
        return "—"
    return f"{weight:.2f}%"


def _consensus_threshold_candidates(etf_count: int) -> list[int]:
    n = max(0, int(etf_count))
    if n <= 0:
        return []
    if n >= 10:
        out = [n]
        if n > 8:
            out.append(8)
        if n > 7:
            out.append(7)
        return out
    out = [n]
    if n - 1 >= 2:
        out.append(n - 1)
    if n - 2 >= 2:
        out.append(n - 2)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _build_consensus_representative_between(
    *,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    force_refresh_constituents: bool = False,
) -> dict[str, object]:
    top10_df, start_used, end_used, universe_count = _build_tw_etf_top10_between(
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
    )
    if top10_df.empty:
        return {
            "error": "目前無法建立前10 ETF 清單，請稍後重試。",
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
        }

    top10_codes = [
        str(x).strip().upper()
        for x in top10_df.get("代碼", pd.Series(dtype=str)).astype(str).tolist()
        if str(x).strip()
    ]
    top10_names = {
        str(row.get("代碼", "")).strip().upper(): str(row.get("ETF", "")).strip()
        for _, row in top10_df.iterrows()
        if str(row.get("代碼", "")).strip()
    }
    if not top10_codes:
        return {
            "error": "前10 ETF 代碼清單為空，無法計算共識代表。",
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
        }

    service = _market_service()
    issues: list[str] = []
    holdings_by_etf: dict[str, dict[str, dict[str, object]]] = {}
    source_map: dict[str, str] = {}

    for etf_code in top10_codes:
        rows_full, source = service.get_etf_constituents_full(
            etf_code,
            limit=None,
            force_refresh=bool(force_refresh_constituents),
        )
        parsed_rows = list(rows_full) if isinstance(rows_full, list) else []

        if not parsed_rows:
            fallback_symbols, fallback_source = service.get_tw_etf_constituents(
                etf_code, limit=None
            )
            if fallback_symbols:
                parsed_rows = [
                    {
                        "symbol": f"{str(sym).strip().upper()}.TW",
                        "tw_code": str(sym).strip().upper(),
                        "name": str(sym).strip().upper(),
                        "market": "TW",
                        "weight_pct": None,
                    }
                    for sym in fallback_symbols
                    if str(sym).strip()
                ]
                source = f"{fallback_source}:no_weight"

        if not parsed_rows:
            issues.append(f"{etf_code}: 無法取得成分股資料")
            continue

        holding_map: dict[str, dict[str, object]] = {}
        for row in parsed_rows:
            if not isinstance(row, dict):
                continue
            symbol = _normalize_constituent_symbol(row.get("symbol"), row.get("tw_code"))
            if not symbol:
                continue
            name = str(row.get("name", "")).strip()
            weight_pct = _safe_weight_pct(row.get("weight_pct"))
            existed = holding_map.get(symbol)
            if existed is None:
                holding_map[symbol] = {
                    "name": name,
                    "weight_pct": weight_pct,
                }
                continue
            old_weight = existed.get("weight_pct")
            if old_weight is None and weight_pct is not None:
                existed["weight_pct"] = weight_pct
            elif old_weight is not None and weight_pct is not None:
                existed["weight_pct"] = max(float(old_weight), float(weight_pct))
            if not str(existed.get("name", "")).strip() and name:
                existed["name"] = name

        if not holding_map:
            issues.append(f"{etf_code}: 成分股欄位解析失敗")
            continue
        holdings_by_etf[etf_code] = holding_map
        source_map[etf_code] = str(source or "unknown")

    usable_codes = [code for code in top10_codes if code in holdings_by_etf]
    if len(usable_codes) < 2:
        return {
            "error": "可用 ETF 成分股不足（少於 2 檔），無法建立共識代表。",
            "issues": issues,
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
            "top10_count": int(len(top10_codes)),
            "usable_count": int(len(usable_codes)),
        }

    symbol_presence: Counter[str] = Counter()
    for etf_code in usable_codes:
        symbol_presence.update(holdings_by_etf[etf_code].keys())

    threshold_used = 0
    consensus_symbols: list[str] = []
    threshold_candidates = _consensus_threshold_candidates(len(usable_codes))
    for threshold in threshold_candidates:
        picked = [sym for sym, cnt in symbol_presence.items() if int(cnt) >= int(threshold)]
        if picked:
            threshold_used = int(threshold)
            consensus_symbols = sorted(picked)
            break

    if not consensus_symbols:
        return {
            "error": "目前無法形成穩定交集（含降級門檻），請改期或重抓成分股後再試。",
            "issues": issues,
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
            "top10_count": int(len(top10_codes)),
            "usable_count": int(len(usable_codes)),
        }

    consensus_rows: list[dict[str, object]] = []
    for symbol in consensus_symbols:
        held_codes: list[str] = []
        names: list[str] = []
        weights: list[float] = []
        for etf_code in usable_codes:
            rec = holdings_by_etf[etf_code].get(symbol)
            if rec is None:
                continue
            held_codes.append(etf_code)
            name_text = str(rec.get("name", "")).strip()
            if name_text:
                names.append(name_text)
            weight = rec.get("weight_pct")
            if weight is not None:
                weights.append(float(weight))
        picked_name = Counter(names).most_common(1)[0][0] if names else symbol
        avg_weight = float(np.mean(weights)) if weights else np.nan
        std_weight = (
            float(np.std(weights, ddof=0))
            if len(weights) >= 2
            else 0.0
            if len(weights) == 1
            else np.nan
        )
        consensus_rows.append(
            {
                "代號": symbol,
                "名稱": picked_name,
                "被持有檔數": int(len(held_codes)),
                "平均權重(%)": round(avg_weight, 3) if math.isfinite(avg_weight) else np.nan,
                "權重離散度(%)": round(std_weight, 3) if math.isfinite(std_weight) else np.nan,
                "持有ETF": " / ".join(held_codes),
            }
        )
    consensus_df = pd.DataFrame(consensus_rows)
    if not consensus_df.empty:
        consensus_df = consensus_df.sort_values(
            ["被持有檔數", "平均權重(%)", "代號"],
            ascending=[False, False, True],
            na_position="last",
        )

    representative_rows: list[dict[str, object]] = []
    for etf_code in usable_codes:
        holding_map = holdings_by_etf[etf_code]
        total_weight_available = float(
            sum(
                float(v.get("weight_pct"))
                for v in holding_map.values()
                if v.get("weight_pct") is not None
            )
        )
        intersection_count = 0
        intersection_weight_sum = 0.0
        for symbol in consensus_symbols:
            rec = holding_map.get(symbol)
            if rec is None:
                continue
            intersection_count += 1
            weight = rec.get("weight_pct")
            if weight is not None:
                intersection_weight_sum += float(weight)
        coverage_pct = np.nan
        if total_weight_available > 0:
            coverage_pct = (intersection_weight_sum / total_weight_available) * 100.0
        representative_rows.append(
            {
                "ETF代碼": etf_code,
                "ETF名稱": str(top10_names.get(etf_code, etf_code)),
                "交集股數": int(intersection_count),
                "交集權重總和(%)": round(float(intersection_weight_sum), 3),
                "可用權重總和(%)": round(float(total_weight_available), 3),
                "代表性分數(覆蓋率%)": round(float(coverage_pct), 3)
                if math.isfinite(float(coverage_pct))
                else np.nan,
                "資料來源": source_map.get(etf_code, "unknown"),
            }
        )
    representative_df = pd.DataFrame(representative_rows)
    if representative_df.empty:
        return {
            "error": "無法建立代表性排名資料。",
            "issues": issues,
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
            "top10_count": int(len(top10_codes)),
            "usable_count": int(len(usable_codes)),
        }
    representative_df = representative_df.sort_values(
        ["代表性分數(覆蓋率%)", "交集股數", "交集權重總和(%)", "ETF代碼"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    representative_df.insert(0, "排名", range(1, len(representative_df) + 1))

    top_pick = representative_df.iloc[0].to_dict() if not representative_df.empty else {}
    alt_rows = representative_df.head(3).copy()
    fallback_applied = bool(threshold_used < len(usable_codes))

    return {
        "error": "",
        "issues": issues,
        "start_used": start_used,
        "end_used": end_used,
        "universe_count": int(universe_count),
        "top10_count": int(len(top10_codes)),
        "usable_count": int(len(usable_codes)),
        "threshold_used": int(threshold_used),
        "threshold_label": f">={threshold_used}/{len(usable_codes)}",
        "fallback_applied": fallback_applied,
        "consensus_count": int(len(consensus_symbols)),
        "consensus_df": consensus_df,
        "representative_df": representative_df,
        "top_pick": top_pick,
        "alternatives_df": alt_rows,
    }


def _infer_constituent_market(*, symbol: object, tw_code: object, market: object) -> str:
    market_token = str(market or "").strip().upper()
    if market_token:
        return market_token
    code_token = str(tw_code or "").strip().upper()
    if re.fullmatch(r"\d{4,6}[A-Z]?", code_token):
        return "TW"
    symbol_token = str(symbol or "").strip().upper()
    if not symbol_token:
        return ""
    if re.fullmatch(r"\d{4,6}[A-Z]?", symbol_token):
        return "TW"
    if symbol_token.endswith(".TW"):
        return "TW"
    if "." in symbol_token:
        suffix = symbol_token.rsplit(".", 1)[-1].strip().upper()
        if suffix:
            return suffix
    return ""


def _load_etf_constituents_rows(
    *,
    service: MarketDataService,
    etf_code: str,
    force_refresh_constituents: bool = False,
) -> tuple[list[dict[str, object]], str, str]:
    rows_full, source = service.get_etf_constituents_full(
        etf_code,
        limit=None,
        force_refresh=bool(force_refresh_constituents),
    )
    parsed_rows = list(rows_full) if isinstance(rows_full, list) else []

    if not parsed_rows:
        fallback_symbols, fallback_source = service.get_tw_etf_constituents(etf_code, limit=None)
        if fallback_symbols:
            parsed_rows = [
                {
                    "symbol": f"{str(sym).strip().upper()}.TW",
                    "tw_code": str(sym).strip().upper(),
                    "name": str(sym).strip().upper(),
                    "market": "TW",
                    "weight_pct": None,
                }
                for sym in fallback_symbols
                if str(sym).strip()
            ]
            source = f"{fallback_source}:no_weight"

    if not parsed_rows:
        return [], str(source or "unknown"), f"{etf_code}: 無法取得成分股資料"

    dedup_rows: dict[str, dict[str, object]] = {}
    for row in parsed_rows:
        if not isinstance(row, dict):
            continue
        tw_code = str(row.get("tw_code", "")).strip().upper()
        symbol = _normalize_constituent_symbol(row.get("symbol"), row.get("tw_code"))
        if not symbol:
            continue
        name = str(row.get("name", "")).strip()
        weight_pct = _safe_weight_pct(row.get("weight_pct"))
        market_token = _infer_constituent_market(
            symbol=symbol,
            tw_code=tw_code,
            market=row.get("market"),
        )
        rec = dedup_rows.get(symbol)
        if rec is None:
            dedup_rows[symbol] = {
                "symbol": symbol,
                "tw_code": tw_code,
                "name": name,
                "weight_pct": weight_pct,
                "market": market_token,
            }
            continue
        old_weight = rec.get("weight_pct")
        if old_weight is None and weight_pct is not None:
            rec["weight_pct"] = weight_pct
        elif old_weight is not None and weight_pct is not None:
            rec["weight_pct"] = max(float(old_weight), float(weight_pct))
        if not str(rec.get("name", "")).strip() and name:
            rec["name"] = name
        if not str(rec.get("market", "")).strip() and market_token:
            rec["market"] = market_token
        if not str(rec.get("tw_code", "")).strip() and tw_code:
            rec["tw_code"] = tw_code

    if not dedup_rows:
        return [], str(source or "unknown"), f"{etf_code}: 成分股欄位解析失敗"

    out_rows = list(dedup_rows.values())
    return out_rows, str(source or "unknown"), ""


def _build_etf_constituent_sets(
    *,
    etf_codes: list[str],
    force_refresh_constituents: bool = False,
) -> tuple[dict[str, set[str]], dict[str, list[dict[str, object]]], dict[str, str], list[str]]:
    service = _market_service()
    constituent_sets: dict[str, set[str]] = {}
    rows_by_etf: dict[str, list[dict[str, object]]] = {}
    source_map: dict[str, str] = {}
    issues: list[str] = []

    for etf_code in etf_codes:
        rows, source, issue = _load_etf_constituents_rows(
            service=service,
            etf_code=etf_code,
            force_refresh_constituents=bool(force_refresh_constituents),
        )
        if issue:
            issues.append(issue)
            continue
        symbol_set = {
            str(row.get("symbol", "")).strip().upper()
            for row in rows
            if str(row.get("symbol", "")).strip()
        }
        if not symbol_set:
            issues.append(f"{etf_code}: 成分股清單為空")
            continue
        rows_by_etf[etf_code] = rows
        constituent_sets[etf_code] = symbol_set
        source_map[etf_code] = source
    return constituent_sets, rows_by_etf, source_map, issues


def _compute_jaccard_pct(a: set[str], b: set[str]) -> float:
    union = set(a) | set(b)
    if not union:
        return 0.0
    intersection = set(a) & set(b)
    return (len(intersection) / len(union)) * 100.0


def _is_overseas_tilt(etf_code: str, constituents_rows: list[dict[str, object]]) -> bool:
    _ = etf_code
    foreign_count = 0
    for row in constituents_rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        market_token = _infer_constituent_market(
            symbol=symbol,
            tw_code=row.get("tw_code"),
            market=row.get("market"),
        )
        if market_token and market_token != "TW":
            foreign_count += 1
            continue
        if not market_token:
            if re.fullmatch(r"\d{4,6}[A-Z]?", symbol):
                continue
            if symbol.endswith(".TW"):
                continue
            if "." in symbol:
                suffix = symbol.rsplit(".", 1)[-1].strip().upper()
                if suffix and suffix != "TW":
                    foreign_count += 1
    return foreign_count > 0


@st.cache_data(ttl=3600, show_spinner=False)
def _build_two_etf_aggressive_picks(
    *,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    allow_overseas: bool = False,
    overlap_cap_pct: float = 10.0,
    force_refresh_constituents: bool = False,
) -> dict[str, object]:
    top10_df, start_used, end_used, universe_count = _build_tw_etf_top10_between(
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
    )
    if top10_df.empty:
        return {
            "error": "目前無法建立前10 ETF 清單，請稍後重試。",
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
        }

    top10_codes = [
        str(x).strip().upper()
        for x in top10_df.get("代碼", pd.Series(dtype=str)).astype(str).tolist()
        if str(x).strip()
    ]
    if not top10_codes:
        return {
            "error": "前10 ETF 代碼清單為空，無法建立兩檔推薦。",
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
        }

    top10_names = {
        str(row.get("代碼", "")).strip().upper(): str(row.get("ETF", "")).strip()
        or str(row.get("代碼", "")).strip().upper()
        for _, row in top10_df.iterrows()
        if str(row.get("代碼", "")).strip()
    }
    top10_types = {
        str(row.get("代碼", "")).strip().upper(): str(row.get("類型", "")).strip() or "其他"
        for _, row in top10_df.iterrows()
        if str(row.get("代碼", "")).strip()
    }
    ytd_map = {
        str(row.get("代碼", "")).strip().upper(): _safe_float(row.get("區間報酬(%)"))
        for _, row in top10_df.iterrows()
        if str(row.get("代碼", "")).strip()
    }
    rank_map = {code: idx for idx, code in enumerate(top10_codes, start=1)}

    consensus_payload = _build_consensus_representative_between(
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
        force_refresh_constituents=bool(force_refresh_constituents),
    )
    consensus_error = (
        str(consensus_payload.get("error", "")).strip()
        if isinstance(consensus_payload, dict)
        else ""
    )
    top_pick = consensus_payload.get("top_pick", {}) if isinstance(consensus_payload, dict) else {}
    top_pick = top_pick if isinstance(top_pick, dict) else {}

    pick_1_code = str(top_pick.get("ETF代碼", "")).strip().upper()
    pick_1_reason = ""
    consensus_score = _safe_float(top_pick.get("代表性分數(覆蓋率%)"))
    if pick_1_code:
        score_text = "—" if consensus_score is None else f"{consensus_score:.2f}%"
        pick_1_reason = f"共識代表第一名（代表性分數 {score_text}）"
    if consensus_error or not pick_1_code or pick_1_code not in top10_codes:
        pick_1_code = top10_codes[0]
        pick_1_reason = "共識代表資料不可用，回退為前十大報酬第1名。"

    constituent_sets, rows_by_etf, source_map, constituent_issues = _build_etf_constituent_sets(
        etf_codes=top10_codes,
        force_refresh_constituents=bool(force_refresh_constituents),
    )
    issues: list[str] = []
    if consensus_error:
        issues.append(f"consensus: {consensus_error}")
    issues.extend(constituent_issues)

    if pick_1_code not in constituent_sets:
        fallback_code = next((code for code in top10_codes if code in constituent_sets), "")
        if fallback_code:
            pick_1_code = fallback_code
            pick_1_reason = "核心ETF成分股資料不足，回退為首檔可用成分股ETF。"

    pick_1_set = constituent_sets.get(pick_1_code, set())
    if not pick_1_set:
        return {
            "error": "核心ETF成分股資料不足，無法計算兩檔重疊度。",
            "issues": issues,
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
            "top10_count": int(len(top10_codes)),
        }

    overlap_cap_value = max(0.0, float(overlap_cap_pct))
    excluded_overseas_codes: list[str] = []
    candidates_raw: list[dict[str, object]] = []

    for code in top10_codes:
        if code == pick_1_code:
            continue
        symbol_set = constituent_sets.get(code, set())
        overlap_pct = _compute_jaccard_pct(pick_1_set, symbol_set)
        is_overseas = _is_overseas_tilt(code, rows_by_etf.get(code, []))
        includable = bool(symbol_set)
        exclusion_reason = ""
        if not symbol_set:
            exclusion_reason = "成分股資料不足"
        if not bool(allow_overseas) and is_overseas:
            includable = False
            exclusion_reason = "含海外成分，不納入本次候選"
            excluded_overseas_codes.append(code)
        candidates_raw.append(
            {
                "前10排名": int(rank_map.get(code, 0)),
                "ETF代碼": code,
                "ETF名稱": top10_names.get(code, code),
                "ETF類型": top10_types.get(code, "其他"),
                "YTD報酬(%)": ytd_map.get(code, np.nan),
                "與核心重疊度(%)": round(float(overlap_pct), 2),
                "成分股數": int(len(symbol_set)),
                "是否海外成分": "是" if is_overseas else "否",
                "可納入候選": "是" if includable else "否",
                "排除原因": exclusion_reason or "—",
                "資料來源": source_map.get(code, "unknown"),
            }
        )

    def _candidate_sort_key(row: dict[str, object]) -> tuple[float, float, int]:
        ytd_val = _safe_float(row.get("YTD報酬(%)"))
        overlap_val = _safe_float(row.get("與核心重疊度(%)"))
        rank_val = int(row.get("前10排名", 999) or 999)
        return (
            -(ytd_val if ytd_val is not None else -1e9),
            overlap_val if overlap_val is not None else 1e9,
            rank_val,
        )

    eligible_rows = [
        row for row in candidates_raw if str(row.get("可納入候選", "")).strip() == "是"
    ]
    if not eligible_rows:
        return {
            "error": "在目前限制條件下沒有可用的第2檔候選 ETF。",
            "issues": issues,
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
            "top10_count": int(len(top10_codes)),
            "excluded_overseas_codes": excluded_overseas_codes,
            "candidate_df": pd.DataFrame(candidates_raw),
        }

    strict_rows = [
        row
        for row in eligible_rows
        if (_safe_float(row.get("與核心重疊度(%)")) or 0.0) <= overlap_cap_value
    ]
    selected_row: dict[str, object] | None = None
    fallback_mode = "strict_overlap"
    overlap_cap_used = overlap_cap_value
    if strict_rows:
        selected_row = sorted(strict_rows, key=_candidate_sort_key)[0]
    else:
        relaxed_cap = max(20.0, overlap_cap_value)
        relaxed_rows = [
            row
            for row in eligible_rows
            if (_safe_float(row.get("與核心重疊度(%)")) or 0.0) <= relaxed_cap
        ]
        if relaxed_rows:
            selected_row = sorted(relaxed_rows, key=_candidate_sort_key)[0]
            fallback_mode = "relaxed_overlap_20"
            overlap_cap_used = relaxed_cap
        else:
            selected_row = sorted(eligible_rows, key=_candidate_sort_key)[0]
            fallback_mode = "top_return_fallback"
            overlap_cap_used = relaxed_cap

    pick_2_code = (
        str(selected_row.get("ETF代碼", "")).strip().upper()
        if isinstance(selected_row, dict)
        else ""
    )
    if not pick_2_code:
        return {
            "error": "無法選出第2檔 ETF。",
            "issues": issues,
            "start_used": start_used,
            "end_used": end_used,
            "universe_count": int(universe_count),
            "top10_count": int(len(top10_codes)),
            "excluded_overseas_codes": excluded_overseas_codes,
        }

    pick_2_overlap = (
        _safe_float(selected_row.get("與核心重疊度(%)")) if isinstance(selected_row, dict) else None
    )
    if fallback_mode == "strict_overlap":
        pick_2_reason = f"在重疊門檻 <= {overlap_cap_value:.1f}% 下，YTD報酬最高。"
    elif fallback_mode == "relaxed_overlap_20":
        pick_2_reason = "原始重疊門檻無候選，放寬到 <= 20% 後選 YTD報酬最高。"
    else:
        pick_2_reason = "重疊門檻放寬後仍無候選，回退為可投資候選中 YTD報酬最高。"

    recommendation_rows = [
        {
            "角色": "核心",
            "ETF代碼": pick_1_code,
            "ETF名稱": top10_names.get(pick_1_code, pick_1_code),
            "ETF類型": top10_types.get(pick_1_code, "其他"),
            "YTD報酬(%)": ytd_map.get(pick_1_code, np.nan),
            "與核心重疊度(%)": 100.0,
            "說明": pick_1_reason,
        },
        {
            "角色": "衛星",
            "ETF代碼": pick_2_code,
            "ETF名稱": top10_names.get(pick_2_code, pick_2_code),
            "ETF類型": top10_types.get(pick_2_code, "其他"),
            "YTD報酬(%)": ytd_map.get(pick_2_code, np.nan),
            "與核心重疊度(%)": pick_2_overlap if pick_2_overlap is not None else np.nan,
            "說明": pick_2_reason,
        },
    ]
    recommendation_df = pd.DataFrame(recommendation_rows)

    candidate_df = pd.DataFrame(candidates_raw)
    if not candidate_df.empty:
        candidate_df["_includable_sort"] = candidate_df["可納入候選"].map(
            lambda v: 1 if str(v).strip() == "是" else 0
        )
        candidate_df = (
            candidate_df.sort_values(
                ["_includable_sort", "YTD報酬(%)", "與核心重疊度(%)", "前10排名"],
                ascending=[False, False, True, True],
                na_position="last",
            )
            .drop(columns=["_includable_sort"])
            .reset_index(drop=True)
        )

    return {
        "error": "",
        "issues": issues,
        "start_used": start_used,
        "end_used": end_used,
        "universe_count": int(universe_count),
        "top10_count": int(len(top10_codes)),
        "allow_overseas": bool(allow_overseas),
        "selection_style": "進攻型",
        "rebalance_hint": "每月",
        "overlap_cap_pct": round(float(overlap_cap_value), 2),
        "overlap_cap_used": round(float(overlap_cap_used), 2),
        "fallback_mode": fallback_mode,
        "pick_1": recommendation_rows[0],
        "pick_2": recommendation_rows[1],
        "excluded_overseas_codes": excluded_overseas_codes,
        "recommendation_df": recommendation_df,
        "candidate_df": candidate_df,
    }


def _render_tw_etf_top10_page(
    title: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    *,
    top_n: int = 10,
    sort_ascending: bool = False,
    count_label: str = "前10檔數",
    ratio_label: str = "前10占比",
    empty_warning_text: str = "目前沒有可顯示的 ETF 排行資料。",
):
    st.subheader(title)
    with st.container(border=True):
        _render_card_section_header("排行卡", "依 TWSE 全市場日收盤快照計算區間報酬率。")
        try:
            top10, start_used, end_used, universe_count = _build_tw_etf_top10_between(
                start_yyyymmdd=start_yyyymmdd,
                end_yyyymmdd=end_yyyymmdd,
                top_n=top_n,
                sort_ascending=sort_ascending,
            )
        except Exception as exc:
            st.error(f"無法建立 ETF 排行：{exc}")
            return
        if top10.empty:
            st.warning(empty_warning_text)
            return
        daily_change_map, daily_end_used, daily_prev_used = _load_tw_etf_daily_change_map(end_used)
        _, daily_open_map = _load_tw_snapshot_open_map(daily_end_used or end_used)
        market_daily_return, market_daily_symbol, _, _, market_daily_issues = (
            _load_tw_market_daily_return(end_used, force_sync=False)
        )
        top10_display = _with_tw_today_fields(
            top10,
            daily_change_map=daily_change_map,
            daily_open_map=daily_open_map,
            market_daily_return_pct=market_daily_return,
        )

        top10_ratio_text = (
            "—" if universe_count <= 0 else f"{(len(top10) / universe_count) * 100.0:.1f}%"
        )
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(count_label, str(len(top10)))
        m2.metric("母體檔數（可比較）", str(universe_count))
        m3.metric(ratio_label, top10_ratio_text)
        m4.metric("市值型", str(int((top10_display["類型"] == "市值型").sum())))
        m5.metric("股利型", str(int((top10_display["類型"] == "股利型").sum())))
        m6.metric("今日上漲檔數", str(int((top10_display["今日漲幅"] > 0).sum())))
        snapshot_health = _build_snapshot_health(
            start_used=start_used,
            end_used=end_used,
            target_yyyymmdd=end_yyyymmdd,
        )
        st.caption(f"計算區間（實際交易日）：{start_used} -> {end_used}")
        _render_data_health_caption("快照資料健康度", snapshot_health)
        st.caption("資料來源：TWSE MI_INDEX（上市全市場快照）；已排除槓反/期貨/海外與債券商品。")
        st.caption("母體檔數採起訖快照交集（經股票型 ETF 過濾）。")
        if (
            market_daily_return is not None
            and market_daily_symbol
            and daily_prev_used
            and daily_end_used
        ):
            st.caption(
                f"今日大盤漲幅：{market_daily_symbol} {market_daily_return:.2f}%（{daily_prev_used} -> {daily_end_used}）"
            )
        if market_daily_issues:
            _render_sync_issues(
                "更新今日大盤漲幅時有部分同步錯誤", market_daily_issues, preview_limit=2
            )
        st.dataframe(top10_display, width="stretch", hide_index=True)

        with st.expander("分類說明", expanded=False):
            st.markdown(
                "\n".join(
                    [
                        "- `代碼白名單`：優先以 ETF 代碼套用穩定分類（市值/股利/科技/金融/ESG/主題/海外/平衡/收益/主動）。",
                        "- `名稱關鍵字`：若未命中白名單，再以名稱關鍵字補判（高股息/科技/ESG/金融/主動等）。",
                        "- `其他`：未命中上述關鍵字分類。",
                    ]
                )
            )


def _render_top10_etf_2025_view():
    _render_top10_etf_2026_ytd_view(
        page_title="2026 年截至今日前十大股利型、配息型 ETF（台股）",
        page_key_prefix="top10_dividend_etf_ytd",
        etf_type_filter="股利型",
        card_subject_label="前十大股利型、配息型 ETF",
        strategy_label="前十大股利型ETF等權",
        empty_warning_text="目前沒有可顯示的股利型、配息型 ETF 排行資料。",
    )


def _render_tw_etf_all_types_view():
    title_col, refresh_col, aum_track_col = st.columns([5, 1, 1])
    with title_col:
        st.subheader("台股 ETF 全類型總表（2025 / 2026 YTD）")
    with refresh_col:
        refresh_market = st.button(
            "更新最新市況",
            key="tw_etf_all_types_update_market",
            width="stretch",
            type="primary",
        )
    with aum_track_col:
        refresh_aum_track = st.button(
            "更新規模追蹤",
            key="tw_etf_all_types_update_aum_track",
            width="stretch",
        )
        reset_aum_track = st.button(
            "從今日重置",
            key="tw_etf_all_types_reset_aum_track",
            width="stretch",
        )
    if refresh_market:
        _fetch_twse_snapshot_with_fallback.clear()
        _build_tw_etf_top10_between.clear()
        _load_twii_twse_month_close_map.clear()
        _load_twii_twse_return_between.clear()
        _load_tw_market_return_between.clear()
        _load_tw_etf_daily_change_map.clear()
        _load_tw_snapshot_open_map.clear()
        _load_tw_market_daily_return.clear()
        _build_tw_etf_all_types_performance_table.clear()
        st.rerun()

    with st.container(border=True):
        _render_card_section_header(
            "總表卡", "列出台股 ETF 全名單，並同時比較 2025 全年與 2026 YTD。"
        )
        try:
            table_df, meta = _build_tw_etf_all_types_performance_table(
                ytd_start_yyyymmdd="20251231",
                ytd_end_yyyymmdd=datetime.now().strftime("%Y%m%d"),
                compare_start_yyyymmdd="20241231",
                compare_end_yyyymmdd="20251231",
            )
        except Exception as exc:
            st.error(f"無法建立 ETF 全類型總表：{exc}")
            return

        if table_df.empty:
            st.warning("目前沒有可顯示的台股 ETF 全類型資料。")
            return

        store = _history_store()

        universe_count = int(meta.get("universe_count", len(table_df)))
        type_series = table_df.get("類型", pd.Series(dtype=str)).astype(str)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("ETF檔數", str(len(table_df)))
        m2.metric("母體檔數（可比較）", str(universe_count))
        m3.metric("類型數", str(int(type_series.nunique(dropna=True))))
        m4.metric("市值型", str(int((type_series == "市值型").sum())))
        m5.metric("股利型", str(int((type_series == "股利型").sum())))
        m6.metric("主動式", str(int((type_series == "主動式").sum())))

        ytd_start_used = str(meta.get("ytd_start_used", "")).strip()
        ytd_end_used = str(meta.get("ytd_end_used", "")).strip()
        compare_start_used = str(meta.get("compare_start_used", "")).strip()
        compare_end_used = str(meta.get("compare_end_used", "")).strip()
        if ytd_start_used and ytd_end_used:
            st.caption(f"2026 YTD 區間（實際交易日）：{ytd_start_used} -> {ytd_end_used}")
        if compare_start_used and compare_end_used:
            st.caption(f"2025 區間（實際交易日）：{compare_start_used} -> {compare_end_used}")
        snapshot_health = _build_snapshot_health(
            start_used=ytd_start_used,
            end_used=ytd_end_used,
            target_yyyymmdd=datetime.now().strftime("%Y%m%d"),
        )
        _render_data_health_caption("快照資料健康度", snapshot_health)

        market_2025_return = _safe_float(meta.get("market_2025_return"))
        market_2025_symbol = str(meta.get("market_2025_symbol", "")).strip()
        if market_2025_return is not None and market_2025_symbol:
            st.caption(f"2025 台股大盤：{market_2025_symbol} 區間報酬 {market_2025_return:.2f}%")
        else:
            st.caption("2025 台股大盤：目前無法取得，`輸贏大盤2025(%)` 先顯示空白。")

        market_ytd_return = _safe_float(meta.get("market_ytd_return"))
        market_ytd_symbol = str(meta.get("market_ytd_symbol", "")).strip()
        if market_ytd_return is not None and market_ytd_symbol:
            st.caption(f"2026 YTD 台股大盤：{market_ytd_symbol} 區間報酬 {market_ytd_return:.2f}%")
        else:
            st.caption("2026 YTD 台股大盤：目前無法取得，`輸贏大盤2026YTD(%)` 先顯示空白。")
        market_daily_return = _safe_float(meta.get("market_daily_return"))
        market_daily_symbol = str(meta.get("market_daily_symbol", "")).strip()
        daily_prev_used = str(meta.get("daily_prev_used", "")).strip()
        daily_end_used = str(meta.get("daily_end_used", "")).strip()
        if (
            market_daily_return is not None
            and market_daily_symbol
            and daily_prev_used
            and daily_end_used
        ):
            st.caption(
                f"今日大盤漲幅：{market_daily_symbol} {market_daily_return:.2f}%（{daily_prev_used} -> {daily_end_used}）"
            )
        else:
            st.caption("今日大盤漲幅：目前無法取得，`今日贏大盤%` 先顯示空白。")

        issues = meta.get("issues", [])
        if isinstance(issues, list) and issues:
            _render_sync_issues(
                "大盤資料同步有部分錯誤，已盡量使用本地可用資料", issues, preview_limit=2
            )
        st.caption("資料來源：TWSE MI_INDEX（上市全市場快照）；已納入所有 ETF 類型。")
        st.caption("排序規則：先依 ETF 類型，再依 2026 YTD 績效由高到低。")

        today_trade_token = _resolve_latest_tw_trade_day_token()
        today_trade_date = _resolve_latest_tw_trade_date_iso(today_trade_token)
        aum_track_anchor_date = _load_tw_etf_aum_track_anchor_date(store)
        if not aum_track_anchor_date:
            try:
                removed = store.clear_tw_etf_aum_history()
                _set_tw_etf_aum_track_anchor_date(store, trade_date=today_trade_date)
                aum_track_anchor_date = today_trade_date
                if removed > 0:
                    st.info(
                        f"基金規模追蹤已重新起算（刪除 {removed} 筆舊資料），第一日為 {today_trade_date}。"
                    )
            except Exception as exc:
                aum_track_anchor_date = today_trade_date
                st.warning(f"初始化基金規模追蹤起算日失敗：{exc}")
        aum_track_anchor_date = (
            str(aum_track_anchor_date or today_trade_date).strip() or today_trade_date
        )

        if reset_aum_track:
            try:
                removed = store.clear_tw_etf_aum_history()
                _set_tw_etf_aum_track_anchor_date(store, trade_date=today_trade_date)
                aum_track_anchor_date = today_trade_date
                st.success(f"已重置基金規模追蹤資料（刪除 {removed} 筆），從今日開始累積。")
            except Exception as exc:
                st.warning(f"重置基金規模追蹤失敗：{exc}")

        if refresh_aum_track:
            try:
                with st.spinner("更新今日 ETF 基金規模中..."):
                    aum_rows = _build_tw_etf_aum_snapshot_rows(
                        table_df,
                        aum_map=_load_tw_etf_aum_billion_map(today_trade_token),
                    )
                    if not aum_rows:
                        st.warning("今日沒有可更新的 ETF 規模資料。")
                    else:
                        updated = store.save_tw_etf_aum_snapshot(
                            rows=aum_rows,
                            trade_date=today_trade_date,
                            keep_days=0,  # <=0: DB 永久保留
                        )
                        st.success(
                            f"已累積 ETF 規模追蹤：{updated} 檔（日期 {today_trade_date}）。"
                        )
            except Exception as exc:
                st.warning(f"更新 ETF 規模追蹤失敗：{exc}")

        history_df = store.load_tw_etf_aum_history(
            etf_codes=table_df.get("代碼", pd.Series(dtype=str)).astype(str).tolist(),
            keep_days=0,
        )
        history_wide = _build_tw_etf_aum_history_wide(
            history_df,
            start_date=aum_track_anchor_date,
            max_date_cols=10,
        )
        history_with_links, history_link_config = _decorate_tw_etf_aum_history_links(history_wide)
        st.markdown("#### 基金規模追蹤（最近 10 交易日）")
        st.caption("欄位單位：億（整數顯示）；色塊規則：日增幅 > 10% 以粉紅標示。")
        st.caption(
            f"起算日：{aum_track_anchor_date}；資料庫採累積保存（不覆蓋），畫面僅顯示最近 10 個交易日。"
        )
        if history_wide.empty:
            st.info("尚無規模追蹤資料，請按「更新規模追蹤」。")
        else:
            st.dataframe(
                _style_tw_etf_aum_history_table(history_with_links),
                width="stretch",
                hide_index=True,
                height=min(_full_table_height(history_wide), 720),
                column_config=history_link_config if history_link_config else None,
            )
            st.caption("可點擊 `台股代號` 開啟回測；可點擊 `ETF名稱` 開啟該檔 ETF 成分股熱力圖。")

        table_with_links, table_link_config = _decorate_tw_etf_name_heatmap_links(table_df)
        table_with_links, code_link_config = _decorate_dataframe_backtest_links(table_with_links)
        merged_link_config: dict[str, object] = {}
        if isinstance(code_link_config, dict):
            merged_link_config.update(code_link_config)
        if isinstance(table_link_config, dict):
            merged_link_config.update(table_link_config)
        if merged_link_config:
            st.caption("可直接點擊 `ETF` 中文名稱，在新分頁開啟對應熱力圖（內容同 00935 熱力圖）。")
        st.dataframe(
            _style_tw_today_move_table(table_with_links),
            width="stretch",
            hide_index=True,
            height=min(_full_table_height(table_with_links), 1200),
            column_config=merged_link_config if merged_link_config else None,
        )
        hub_col1, hub_col2 = st.columns([2, 1])
        with hub_col1:
            st.caption("已開啟/已快取的 ETF 熱力圖可在 `熱力圖總表` 分頁集中管理。")
        with hub_col2:
            if st.button("前往 熱力圖總表", key="go_heatmap_hub_page", width="stretch"):
                st.session_state["active_page"] = "熱力圖總表"
                st.rerun()


def _render_heatmap_hub_view():
    st.subheader("熱力圖總表")
    with st.container(border=True):
        _render_card_section_header(
            "已快取 ETF 熱力圖", "集中管理你曾開啟過的 ETF 熱力圖，並可釘選成獨立卡片。"
        )
        entries = [
            row
            for row in _load_heatmap_hub_entries(pinned_only=False)
            if _normalize_heatmap_etf_code(getattr(row, "etf_code", ""))
            not in HEATMAP_CARD_BLOCKLIST
        ]
        if not entries:
            st.caption(
                "目前尚無已開啟的 ETF 熱力圖紀錄。先到「台股 ETF 全類型總表」點擊 ETF 名稱即可新增。"
            )
            return

        total_count = len(entries)
        pinned_count = int(sum(1 for row in entries if bool(getattr(row, "pin_as_card", False))))
        opened_total = int(sum(max(0, int(getattr(row, "open_count", 0) or 0)) for row in entries))
        m1, m2, m3 = st.columns(3)
        m1.metric("已快取 ETF 熱力圖", str(total_count))
        m2.metric("釘選成獨立卡片", str(pinned_count))
        m3.metric("累計開啟次數", str(opened_total))

        h1, h2, h3, h4, h5, h6 = st.columns([1.1, 2.6, 1.6, 1.0, 1.0, 1.0])
        h1.caption("ETF代碼")
        h2.caption("ETF名稱（新分頁）")
        h3.caption("最近開啟")
        h4.caption("開啟次數")
        h5.caption("釘選卡片")
        h6.caption("本頁開啟")

        for entry in entries:
            code = _normalize_heatmap_etf_code(getattr(entry, "etf_code", ""))
            if not code:
                continue
            name = _clean_heatmap_name_for_query(getattr(entry, "etf_name", "")) or code
            last_opened = getattr(entry, "last_opened_at", None)
            if isinstance(last_opened, datetime):
                if last_opened.tzinfo is None:
                    last_opened = last_opened.replace(tzinfo=timezone.utc)
                last_opened_text = last_opened.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_opened_text = str(last_opened or "—").strip() or "—"
            open_count = int(getattr(entry, "open_count", 0) or 0)
            pinned = bool(getattr(entry, "pin_as_card", False))
            open_url = build_heatmap_drill_url(code, name, src="heatmap_hub")
            pin_key = f"heatmap_hub_pin:{code}"
            if pin_key not in st.session_state:
                st.session_state[pin_key] = pinned

            c1, c2, c3, c4, c5, c6 = st.columns([1.1, 2.6, 1.6, 1.0, 1.0, 1.0])
            c1.markdown(f"`{code}`")
            c2.link_button(label=name, url=open_url, width="stretch")
            c3.caption(last_opened_text)
            c4.caption(str(open_count))
            pin_now = c5.checkbox("釘選", key=pin_key, label_visibility="collapsed")
            if bool(pin_now) != pinned:
                _set_heatmap_hub_pin(etf_code=code, pin_as_card=bool(pin_now))
                st.rerun()
            if c6.button("開啟", key=f"heatmap_hub_open:{code}", width="stretch"):
                _upsert_heatmap_hub_entry(etf_code=code, etf_name=name, opened=True)
                st.session_state[HEATMAP_HUB_SESSION_ACTIVE_KEY] = {"code": code, "name": name}
                st.session_state["active_page"] = _heatmap_page_key_for_code(code)
                st.rerun()


def _render_bottom20_etf_2025_view():
    _render_top10_etf_2026_ytd_view(
        page_title="2025 年後20大最差勁 ETF（台股）",
        page_key_prefix="bottom20_etf_2025",
        start_target="20241231",
        end_target="20251231",
        top_n=20,
        sort_ascending=True,
        exclude_split_event=True,
        rank_by_underperform=True,
        compare_start_yyyymmdd="20240101",
        compare_end_yyyymmdd="20241231",
        compare_col_label="2024績效(%)",
        performance_col_label="2025報酬(%)",
        underperform_col_label="輸給台股大盤(%)",
        count_label="後20檔數",
        ratio_label="後20占比",
        benchmark_period_note="同 2025 全年區間",
        compare_period_caption_label="2024 對照區間",
        card_subject_label="2025 後20大最差勁 ETF",
        strategy_label="2025後20ETF等權",
        empty_warning_text="目前沒有可顯示的 2025 後20 ETF 資料。",
    )


def _render_top10_etf_2026_ytd_view(
    *,
    page_title: str = "2026 年截至今日前十大 ETF（台股）",
    page_key_prefix: str = "top10_etf_ytd",
    start_target: str = "20251231",
    end_target: str | None = None,
    top_n: int = 10,
    sort_ascending: bool = False,
    etf_type_filter: str | None = None,
    exclude_split_event: bool = False,
    rank_by_underperform: bool = False,
    compare_start_yyyymmdd: str = "20250101",
    compare_end_yyyymmdd: str = "20251231",
    compare_col_label: str = "2025績效(%)",
    performance_col_label: str = "YTD報酬(%)",
    underperform_col_label: str | None = None,
    count_label: str = "前10檔數",
    ratio_label: str = "前10占比",
    benchmark_period_note: str = "同本頁比較區間",
    compare_period_caption_label: str = "2025 對照區間",
    card_subject_label: str = "前十大 ETF",
    strategy_label: str = "前十大ETF等權",
    empty_warning_text: str = "目前沒有可顯示的 ETF 排行資料。",
):
    store = _history_store()
    etf_type_filter_text = str(etf_type_filter or "").strip()
    payload_key = f"{page_key_prefix}_compare_payload"
    strategy_hover_code = f"{page_key_prefix.upper()}_EW"
    display_n = max(1, int(top_n))

    title_col, refresh_col = st.columns([6, 1])
    with title_col:
        st.subheader(page_title)
    with refresh_col:
        refresh_market = st.button(
            "更新最新市況",
            key=f"{page_key_prefix}_update_market",
            width="stretch",
            type="primary",
        )
    if refresh_market:
        _fetch_twse_snapshot_with_fallback.clear()
        _build_tw_etf_top10_between.clear()
        _build_tw_active_etf_ytd_between.clear()
        _load_twii_twse_month_close_map.clear()
        _load_twii_twse_return_between.clear()
        _load_tw_market_return_between.clear()
        _load_tw_etf_daily_change_map.clear()
        _load_tw_snapshot_open_map.clear()
        _load_tw_market_daily_return.clear()
        st.session_state.pop(payload_key, None)
        st.rerun()

    target_end_token = str(end_target or datetime.now().strftime("%Y%m%d"))
    with st.container(border=True):
        _render_card_section_header("排行卡", "依 TWSE 全市場日收盤快照計算區間報酬率。")
        try:
            fetch_n = 99999 if rank_by_underperform else display_n
            top10, start_used, end_used, universe_count = _build_tw_etf_top10_between(
                start_yyyymmdd=start_target,
                end_yyyymmdd=target_end_token,
                type_filter=etf_type_filter_text or None,
                top_n=fetch_n,
                sort_ascending=bool(sort_ascending),
                exclude_split_event=bool(exclude_split_event),
            )
            if top10.empty:
                st.warning(empty_warning_text)
                return
            top10_etf_df = top10.rename(columns={"區間報酬(%)": performance_col_label}).copy()
        except Exception as exc:
            st.error(f"無法建立 ETF 排行：{exc}")
            return

        market_return_pct: float | None = None
        market_symbol_used = ""
        market_issues: list[str] = []
        try:
            market_return_pct, market_symbol_used, market_issues = _load_tw_market_return_between(
                start_yyyymmdd=start_used,
                end_yyyymmdd=end_used,
                force_sync=False,
            )
        except Exception as exc:
            market_issues = [f"market: {exc}"]

        if rank_by_underperform:
            if market_return_pct is not None and math.isfinite(float(market_return_pct)):
                underperform_name = str(underperform_col_label or "輸給台股大盤(%)")
                top10_etf_df[underperform_name] = (
                    float(market_return_pct)
                    - pd.to_numeric(top10_etf_df[performance_col_label], errors="coerce")
                ).round(2)
                top10_etf_df = (
                    top10_etf_df.sort_values(underperform_name, ascending=False, na_position="last")
                    .head(display_n)
                    .copy()
                )
            else:
                top10_etf_df = (
                    top10_etf_df.sort_values(
                        performance_col_label, ascending=True, na_position="last"
                    )
                    .head(display_n)
                    .copy()
                )
        else:
            top10_etf_df = top10_etf_df.head(display_n).copy()

        top10_symbols = tuple(
            str(x).strip().upper()
            for x in top10_etf_df["代碼"].astype(str).tolist()
            if str(x).strip()
        )
        compare_return_map: dict[str, float] = {}
        hist_compare_start_used = compare_start_yyyymmdd
        hist_compare_end_used = compare_end_yyyymmdd
        try:
            hist_compare_df, hist_compare_start_used, hist_compare_end_used = (
                _build_tw_active_etf_ytd_between(
                    start_yyyymmdd=compare_start_yyyymmdd,
                    end_yyyymmdd=compare_end_yyyymmdd,
                    symbols=top10_symbols,
                )
            )
            compare_return_map = {
                str(row["代碼"]).strip().upper(): float(row["YTD報酬(%)"])
                for _, row in hist_compare_df.iterrows()
                if str(row.get("代碼", "")).strip() and pd.notna(row.get("YTD報酬(%)"))
            }
        except Exception as exc:
            market_issues.append(f"compare_period: {exc}")

        market_compare_return_pct: float | None = None
        market_compare_symbol_used = ""
        try:
            market_compare_return_pct, market_compare_symbol_used, market_compare_issues = (
                _load_tw_market_return_between(
                    start_yyyymmdd=compare_start_yyyymmdd,
                    end_yyyymmdd=hist_compare_end_used,
                    force_sync=False,
                )
            )
            market_issues.extend(market_compare_issues)
        except Exception as exc:
            market_issues.append(f"market_compare: {exc}")

        daily_change_map, daily_end_used, daily_prev_used = _load_tw_etf_daily_change_map(end_used)
        _, daily_open_map = _load_tw_snapshot_open_map(daily_end_used or end_used)
        market_daily_return, market_daily_symbol, _, _, market_daily_issues = (
            _load_tw_market_daily_return(end_used, force_sync=False)
        )
        market_issues.extend(market_daily_issues)
        previous_rank_map: dict[str, int] = {}
        if daily_prev_used and daily_prev_used != end_used:
            try:
                prev_fetch_n = 99999 if rank_by_underperform else display_n
                prev_top10, _, _, _ = _build_tw_etf_top10_between(
                    start_yyyymmdd=start_target,
                    end_yyyymmdd=daily_prev_used,
                    type_filter=etf_type_filter_text or None,
                    top_n=prev_fetch_n,
                    sort_ascending=bool(sort_ascending),
                    exclude_split_event=bool(exclude_split_event),
                )
                if not prev_top10.empty:
                    prev_rank_df = prev_top10.rename(
                        columns={"區間報酬(%)": performance_col_label}
                    ).copy()
                    if rank_by_underperform:
                        if market_return_pct is not None and math.isfinite(
                            float(market_return_pct)
                        ):
                            underperform_name = str(underperform_col_label or "輸給台股大盤(%)")
                            prev_rank_df[underperform_name] = (
                                float(market_return_pct)
                                - pd.to_numeric(
                                    prev_rank_df[performance_col_label], errors="coerce"
                                )
                            ).round(2)
                            prev_rank_df = (
                                prev_rank_df.sort_values(
                                    underperform_name, ascending=False, na_position="last"
                                )
                                .head(display_n)
                                .copy()
                            )
                        else:
                            prev_rank_df = (
                                prev_rank_df.sort_values(
                                    performance_col_label, ascending=True, na_position="last"
                                )
                                .head(display_n)
                                .copy()
                            )
                    else:
                        prev_rank_df = prev_rank_df.head(display_n).copy()
                    previous_rank_map = {
                        str(code).strip().upper(): idx
                        for idx, code in enumerate(
                            prev_rank_df.get("代碼", pd.Series(dtype=str)).astype(str).tolist(),
                            start=1,
                        )
                        if str(code).strip()
                    }
            except Exception as exc:
                market_issues.append(f"rank_movement: {exc}")
        top10_etf_df = _attach_rank_movement_columns(
            top10_etf_df,
            previous_rank_map=previous_rank_map,
        )

        benchmark_code = market_symbol_used or market_compare_symbol_used or "^TWII"
        table_df = _decorate_tw_etf_top10_ytd_table(
            top10_etf_df,
            compare_return_map=compare_return_map,
            market_return_pct=market_return_pct,
            market_compare_return_pct=market_compare_return_pct,
            benchmark_code=benchmark_code,
            end_used=end_used,
            compare_col_label=compare_col_label,
            performance_col_label=performance_col_label,
            underperform_col_label=underperform_col_label if rank_by_underperform else None,
        )
        table_df = _with_tw_today_fields(
            table_df,
            daily_change_map=daily_change_map,
            daily_open_map=daily_open_map,
            market_daily_return_pct=market_daily_return,
        )
        top10_today_df = _with_tw_today_fields(
            top10_etf_df,
            daily_change_map=daily_change_map,
            daily_open_map=daily_open_map,
            market_daily_return_pct=market_daily_return,
        )

        top10_ratio_text = (
            "—" if universe_count <= 0 else f"{(len(top10_etf_df) / universe_count) * 100.0:.1f}%"
        )
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(count_label, str(len(top10_etf_df)))
        m2.metric("母體檔數（可比較）", str(universe_count))
        m3.metric(ratio_label, top10_ratio_text)
        type_series = top10_etf_df.get("類型", pd.Series(dtype=str))
        if etf_type_filter_text == "股利型":
            m4.metric("股利/配息型", str(int((type_series == "股利型").sum())))
            m5.metric("非股利型", str(int((type_series != "股利型").sum())))
        else:
            m4.metric("市值型", str(int((type_series == "市值型").sum())))
            m5.metric("股利型", str(int((type_series == "股利型").sum())))
        m6.metric("今日上漲檔數", str(int((top10_today_df["今日漲幅"] > 0).sum())))
        snapshot_health = _build_snapshot_health(
            start_used=start_used,
            end_used=end_used,
            target_yyyymmdd=target_end_token,
        )
        st.caption(f"計算區間（實際交易日）：{start_used} -> {end_used}")
        st.caption(
            f"{compare_period_caption_label}（實際交易日）：{hist_compare_start_used} -> {hist_compare_end_used}"
        )
        _render_data_health_caption("快照資料健康度", snapshot_health)
        if market_return_pct is not None and market_symbol_used:
            st.caption(
                f"大盤對照：{market_symbol_used} 區間報酬 {market_return_pct:.2f}%（{benchmark_period_note}）"
            )
        else:
            st.caption("大盤對照：目前無法取得，`贏輸台股大盤(%)` 先顯示為空白。")
        if (
            market_daily_return is not None
            and market_daily_symbol
            and daily_prev_used
            and daily_end_used
        ):
            st.caption(
                f"今日大盤漲幅：{market_daily_symbol} {market_daily_return:.2f}%（{daily_prev_used} -> {daily_end_used}）"
            )
        else:
            st.caption("今日大盤漲幅：目前無法取得，`今日贏大盤%` 先顯示空白。")
        if daily_prev_used:
            if previous_rank_map:
                st.caption(
                    f"排名異動：已併入 `排名` 欄位（例：`1  (持平)`、`10  (新進榜)`），比較基準為 {daily_prev_used}。"
                )
            else:
                st.caption("排名異動：暫無可比較的前一交易日榜單，`排名` 會顯示為 `N  (—)`。")
        else:
            st.caption("排名異動：目前無前一交易日可比較，`排名` 會顯示為 `N  (—)`。")
        if market_issues:
            _render_sync_issues("更新大盤資料時有部分同步錯誤", market_issues, preview_limit=2)
        st.caption("資料來源：TWSE MI_INDEX（上市全市場快照）；已排除槓反/期貨/海外與債券商品。")
        if etf_type_filter_text:
            st.caption(
                "篩選條件：僅納入 `股利型`（名稱含高股息/股利/股息/收益/配息/月配/季配/年配）ETF。"
            )
        if exclude_split_event:
            st.caption("區間處理：已排除區間內含分割事件的標的（避免價格比較失真）。")
        if rank_by_underperform:
            st.caption(
                f"排行規則：以 `贏輸台股大盤(%)` 最低（輸給大盤最多）排序，取倒數 {display_n} 名。"
            )
        st.caption("母體檔數採起訖快照交集（經股票型 ETF 過濾）。")
        st.caption(
            f"`{compare_col_label}` 採對照區間內首個可交易日計算；若空白代表該檔對照區間無可用日K。"
        )
        st.caption(f"報酬計算：`贏輸台股大盤(%) = {performance_col_label} - 大盤報酬`。")
        table_with_links, table_link_config = _decorate_tw_etf_name_heatmap_links(
            table_df,
            src=f"{page_key_prefix}_rank_table",
        )
        table_with_links, code_link_config = _decorate_dataframe_backtest_links(table_with_links)
        merged_link_config: dict[str, object] = {}
        if isinstance(code_link_config, dict):
            merged_link_config.update(code_link_config)
        if isinstance(table_link_config, dict):
            merged_link_config.update(table_link_config)
        styled_table_df = _style_tw_today_move_table(table_with_links)
        if page_key_prefix in {"top10_etf_ytd", "top10_dividend_etf_ytd"}:
            # 固定顯示完整 11 列（台股大盤 + 前10），方便一次截圖。
            table_height = min(640, 42 + max(1, int(len(table_df))) * 36)
            st.dataframe(
                styled_table_df,
                width="stretch",
                hide_index=True,
                height=table_height,
                column_config=merged_link_config if merged_link_config else None,
            )
        else:
            st.dataframe(
                styled_table_df,
                width="stretch",
                hide_index=True,
                column_config=merged_link_config if merged_link_config else None,
            )

        with st.expander("分類說明", expanded=False):
            st.markdown(
                "\n".join(
                    [
                        "- `代碼白名單`：優先以 ETF 代碼套用穩定分類（市值/股利/科技/金融/ESG/主題/海外/平衡/收益/主動）。",
                        "- `名稱關鍵字`：若未命中白名單，再以名稱關鍵字補判（高股息/科技/ESG/金融/主動等）。",
                        "- `其他`：未命中上述關鍵字分類。",
                    ]
                )
            )

    symbols = [
        str(x).strip().upper() for x in top10_etf_df["代碼"].astype(str).tolist() if str(x).strip()
    ]
    if not symbols:
        return
    name_map = {
        str(row["代碼"]).strip().upper(): str(row["ETF"]).strip()
        for _, row in top10_etf_df.iterrows()
        if str(row.get("代碼", "")).strip()
    }

    with st.container(border=True):
        _render_card_section_header(
            "Benchmark 對照卡", "策略曲線、基準曲線與每檔 Buy & Hold 同圖比較。"
        )
        st.caption(
            f"差異說明：上方表格的 `{performance_col_label}` 採快照區間報酬；"
            f"本對照卡曲線與下方 `Total Return %` 會以共同比較區間 `{start_used} -> {end_used}` 對齊，"
            "因此同一檔數值可能略有差異。"
        )
        st.markdown(
            (
                "說明："
                f"<span title='把{card_subject_label}平均分配資金後，從區間起點買進並持有到期末，不做換股或調倉。'>"
                f"<code>Strategy Equity（{strategy_label}）</code>（滑鼠移入查看）"
                "</span>"
            ),
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([2, 2, 1])
        benchmark_choice = c1.selectbox(
            "Benchmark",
            options=["twii", "0050", "006208"],
            index=0,
            format_func=lambda x: {"twii": "^TWII", "0050": "0050", "006208": "006208"}.get(x, x),
            key=f"{page_key_prefix}_benchmark",
        )
        sync_before_run = c2.checkbox(
            "執行前同步最新日K（較慢）",
            value=False,
            key=f"{page_key_prefix}_sync_before_run",
        )
        force_refresh = c3.button("重新計算", width="stretch", key=f"{page_key_prefix}_refresh")

        start_dt = datetime.combine(
            datetime.strptime(start_used, "%Y%m%d").date(), datetime.min.time()
        ).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(
            datetime.strptime(end_used, "%Y%m%d").date(), datetime.min.time()
        ).replace(tzinfo=timezone.utc)
        run_key = f"{page_key_prefix}:{start_used}:{end_used}:{benchmark_choice}:{sync_before_run}:{','.join(symbols)}"
        payload = st.session_state.get(payload_key)
        if not isinstance(payload, dict):
            payload = {}

        should_recompute = force_refresh or payload.get("run_key") != run_key
        if should_recompute:
            with st.spinner(f"計算 {card_subject_label} Benchmark 對照中..."):
                payload = _compute_tw_equal_weight_compare_payload(
                    symbols=symbols,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    benchmark_choice=benchmark_choice,
                    sync_before_run=sync_before_run,
                    insufficient_msg=f"可用{card_subject_label}歷史資料不足，無法建立對照圖。",
                )
            payload["run_key"] = run_key
            st.session_state[payload_key] = payload

        payload = st.session_state.get(payload_key, {})
        if not isinstance(payload, dict):
            payload = {}
        error_text = str(payload.get("error", "")).strip()
        if error_text:
            _emit_issue_message(error_text)
            return

        symbol_sync_issues = payload.get("symbol_sync_issues", [])
        if isinstance(symbol_sync_issues, list) and symbol_sync_issues:
            _render_sync_issues(
                "部分 ETF 同步失敗，已盡量使用本地可用資料", symbol_sync_issues, preview_limit=3
            )

        benchmark_sync_issues = payload.get("benchmark_sync_issues", [])
        if isinstance(benchmark_sync_issues, list) and benchmark_sync_issues:
            _render_sync_issues(
                "Benchmark 同步有部分錯誤，已盡量使用本地可用資料",
                benchmark_sync_issues,
                preview_limit=2,
            )

        strategy_equity = payload.get("strategy_equity")
        benchmark_equity = payload.get("benchmark_equity")
        per_symbol_equity = payload.get("per_symbol_equity")
        if not isinstance(strategy_equity, pd.Series) or strategy_equity.empty:
            st.warning("目前無法建立策略曲線。")
            return
        if not isinstance(benchmark_equity, pd.Series):
            benchmark_equity = pd.Series(dtype=float)
        if not isinstance(per_symbol_equity, dict):
            per_symbol_equity = {}
        compare_health = _build_data_health(
            as_of=strategy_equity.index.max() if len(strategy_equity.index) else "",
            data_sources=[_store_data_source(store, "daily_bars")],
            source_chain=[str(payload.get("benchmark_symbol", "") or "benchmark")],
            degraded=bool(symbol_sync_issues or benchmark_sync_issues),
            fallback_depth=len(symbol_sync_issues) + len(benchmark_sync_issues),
            notes="部分標的同步失敗時會盡量使用本地資料",
        )
        _render_data_health_caption("Benchmark 對照資料健康度", compare_health)

        palette = _ui_palette()
        symbol_styles = _build_symbol_line_styles(list(per_symbol_equity.keys()))
        benchmark_label = str(payload.get("benchmark_symbol", "") or "Benchmark")
        chart_lines: list[dict[str, Any]] = [
            {
                "name": f"Strategy Equity（{strategy_label}）",
                "series": strategy_equity,
                "color": str(palette["equity"]),
                "width": 2.4,
                "dash": "solid",
                "hover_code": strategy_hover_code,
                "value_label": "Equity",
                "y_format": ",.0f",
            }
        ]
        if not benchmark_equity.empty:
            style = _benchmark_line_style(palette, width=2.0)
            chart_lines.append(
                {
                    "name": f"Benchmark Equity（{benchmark_label}）",
                    "series": benchmark_equity,
                    "color": str(style["color"]),
                    "width": float(style["width"]),
                    "dash": str(style["dash"]),
                    "hover_code": benchmark_label,
                    "value_label": "Equity",
                    "y_format": ",.0f",
                }
            )
        for symbol in sorted(per_symbol_equity.keys()):
            series = per_symbol_equity[symbol]
            if not isinstance(series, pd.Series) or len(series) < 2:
                continue
            style = symbol_styles.get(symbol, {"color": "#1f77b4", "dash": "solid"})
            label_name = str(name_map.get(symbol, symbol)).strip()
            trace_name = (
                f"Buy-and-Hold（{symbol} {label_name}）"
                if label_name and label_name != symbol
                else f"Buy-and-Hold（{symbol}）"
            )
            chart_lines.append(
                {
                    "name": trace_name,
                    "series": series,
                    "color": str(style["color"]),
                    "width": 1.8,
                    "dash": str(style["dash"]),
                    "hover_code": symbol,
                    "value_label": "Equity",
                    "y_format": ",.0f",
                }
            )
        _render_benchmark_lines_chart(
            lines=chart_lines,
            height=460,
            chart_key=f"{page_key_prefix}_benchmark_chart",
        )

        summary_rows: list[dict[str, object]] = []
        strategy_perf = _series_metrics_basic(strategy_equity)
        summary_rows.append(
            {
                "Series": f"Strategy Equity（{strategy_label}）",
                "Total Return %": round(strategy_perf["total_return"] * 100.0, 2),
                "CAGR %": round(strategy_perf["cagr"] * 100.0, 2),
                "MDD %": round(strategy_perf["max_drawdown"] * 100.0, 2),
                "Sharpe": round(strategy_perf["sharpe"], 2),
            }
        )
        if not benchmark_equity.empty:
            benchmark_perf = _series_metrics_basic(benchmark_equity)
            benchmark_label = str(payload.get("benchmark_symbol", "") or "Benchmark")
            summary_rows.append(
                {
                    "Series": f"Benchmark Equity（{benchmark_label}）",
                    "Total Return %": round(benchmark_perf["total_return"] * 100.0, 2),
                    "CAGR %": round(benchmark_perf["cagr"] * 100.0, 2),
                    "MDD %": round(benchmark_perf["max_drawdown"] * 100.0, 2),
                    "Sharpe": round(benchmark_perf["sharpe"], 2),
                }
            )

        for symbol in sorted(per_symbol_equity.keys()):
            series = per_symbol_equity[symbol]
            if not isinstance(series, pd.Series) or len(series) < 2:
                continue
            perf = _series_metrics_basic(series)
            label_name = str(name_map.get(symbol, symbol)).strip()
            series_name = (
                f"Buy-and-Hold（{symbol} {label_name}）"
                if label_name and label_name != symbol
                else f"Buy-and-Hold（{symbol}）"
            )
            summary_rows.append(
                {
                    "Series": series_name,
                    "Total Return %": round(perf["total_return"] * 100.0, 2),
                    "CAGR %": round(perf["cagr"] * 100.0, 2),
                    "MDD %": round(perf["max_drawdown"] * 100.0, 2),
                    "Sharpe": round(perf["sharpe"], 2),
                }
            )

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
        used_symbols = payload.get("used_symbols", [])
        skipped_symbols = payload.get("skipped_symbols", [])
        if isinstance(used_symbols, list) and isinstance(skipped_symbols, list):
            st.caption(
                f"可用ETF：{len(used_symbols)} 檔 | 資料不足未納入：{len(skipped_symbols)} 檔"
            )


def _render_consensus_representative_etf_view():
    title_col, refresh_col = st.columns([6, 1])
    with title_col:
        st.subheader("共識代表 ETF（前10交集）")
    with refresh_col:
        force_recompute = st.button(
            "重新計算",
            key="consensus_etf_refresh",
            width="stretch",
            type="primary",
        )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.caption("更新節奏建議：每月更新一次；若市場結構快速變化可手動重算。")
    with c2:
        force_refresh_constituents = st.checkbox(
            "重抓成分股（較慢）",
            value=False,
            key="consensus_etf_force_refresh_constituents",
        )

    if force_recompute:
        _build_consensus_representative_between.clear()
        _build_tw_etf_top10_between.clear()
        st.rerun()

    start_target = "20251231"
    end_target = datetime.now().strftime("%Y%m%d")

    with st.container(border=True):
        _render_card_section_header(
            "共識代表卡", "由當期前10 ETF 成分股交集推導最具代表性的單一 ETF。"
        )
        try:
            payload = _build_consensus_representative_between(
                start_yyyymmdd=start_target,
                end_yyyymmdd=end_target,
                force_refresh_constituents=bool(force_refresh_constituents),
            )
        except Exception as exc:
            st.error(f"無法建立共識代表 ETF：{exc}")
            return

        error_text = str(payload.get("error", "")).strip()
        if error_text:
            _emit_issue_message(error_text)
            issues = payload.get("issues", [])
            if isinstance(issues, list) and issues:
                _render_sync_issues("成分股取得有部分錯誤", issues, preview_limit=3)
            return

        start_used = str(payload.get("start_used", "")).strip()
        end_used = str(payload.get("end_used", "")).strip()
        top_pick = payload.get("top_pick", {})
        if not isinstance(top_pick, dict):
            top_pick = {}
        consensus_count = int(payload.get("consensus_count", 0) or 0)
        top10_count = int(payload.get("top10_count", 0) or 0)
        usable_count = int(payload.get("usable_count", 0) or 0)
        threshold_used = int(payload.get("threshold_used", 0) or 0)
        threshold_label = str(payload.get("threshold_label", "")).strip()
        universe_count = int(payload.get("universe_count", 0) or 0)
        fallback_applied = bool(payload.get("fallback_applied", False))

        top_code = str(top_pick.get("ETF代碼", "—")).strip() or "—"
        top_name = str(top_pick.get("ETF名稱", top_code)).strip() or top_code
        top_score = _safe_float(top_pick.get("代表性分數(覆蓋率%)"))
        top_score_text = "—" if top_score is None else f"{top_score:.2f}%"

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("建議 ETF", top_code)
        m2.metric("代表性分數", top_score_text)
        m3.metric("交集股票數", str(consensus_count))
        m4.metric("交集門檻", threshold_label or "—")
        m5.metric("可用 ETF 數", f"{usable_count}/{top10_count}")
        m6.metric("前10母體（可比較）", str(universe_count))

        st.caption(f"建議標的：{top_code} {top_name}")
        st.caption(f"計算區間（實際交易日）：{start_used} -> {end_used}")
        if fallback_applied:
            st.caption(
                f"交集門檻說明：嚴格交集不足，已降級為「至少 {threshold_used}/{usable_count} 檔共持」。"
            )
        else:
            st.caption(f"交集門檻說明：使用嚴格交集（{usable_count}/{usable_count} 檔共持）。")

        snapshot_health = _build_snapshot_health(
            start_used=start_used,
            end_used=end_used,
            target_yyyymmdd=end_target,
        )
        _render_data_health_caption("快照資料健康度", snapshot_health)

        issues = payload.get("issues", [])
        if isinstance(issues, list) and issues:
            _render_sync_issues(
                "部分 ETF 成分股抓取失敗，已用可用資料計算", issues, preview_limit=3
            )

    alternatives_df = payload.get("alternatives_df")
    if isinstance(alternatives_df, pd.DataFrame) and not alternatives_df.empty:
        alternatives_df = _attach_tw_etf_management_fee_column(
            alternatives_df, code_col_candidates=("ETF代碼",)
        )
        with st.container(border=True):
            _render_card_section_header(
                "代表性排名（建議 + 備選）", "以交集權重覆蓋率排序，前3檔可作為核心與備選。"
            )
            show_cols = [
                col
                for col in [
                    "排名",
                    "ETF代碼",
                    "ETF名稱",
                    "管理費(%)",
                    "ETF規模(億)",
                    "交集股數",
                    "交集權重總和(%)",
                    "可用權重總和(%)",
                    "代表性分數(覆蓋率%)",
                ]
                if col in alternatives_df.columns
            ]
            st.dataframe(alternatives_df[show_cols], width="stretch", hide_index=True)

    consensus_df = payload.get("consensus_df")
    if isinstance(consensus_df, pd.DataFrame) and not consensus_df.empty:
        with st.container(border=True):
            _render_card_section_header(
                "共識核心成分股", "顯示交集核心股票與其在前10 ETF 中的覆蓋情況。"
            )
            show_cols = [
                col
                for col in ["代號", "名稱", "被持有檔數", "平均權重(%)", "權重離散度(%)", "持有ETF"]
                if col in consensus_df.columns
            ]
            st.dataframe(
                consensus_df[show_cols],
                width="stretch",
                hide_index=True,
                height=_full_table_height(consensus_df),
            )

    representative_df = payload.get("representative_df")
    if isinstance(representative_df, pd.DataFrame) and not representative_df.empty:
        representative_df = _attach_tw_etf_management_fee_column(
            representative_df, code_col_candidates=("ETF代碼",)
        )
        with st.container(border=True):
            _render_card_section_header("完整代表性排名", "若前3檔分數接近，可在此比較完整名單。")
            show_cols = [
                col
                for col in [
                    "排名",
                    "ETF代碼",
                    "ETF名稱",
                    "管理費(%)",
                    "ETF規模(億)",
                    "交集股數",
                    "交集權重總和(%)",
                    "可用權重總和(%)",
                    "代表性分數(覆蓋率%)",
                    "資料來源",
                ]
                if col in representative_df.columns
            ]
            st.dataframe(representative_df[show_cols], width="stretch", hide_index=True)
            st.caption(
                "註：代表性分數高，代表該 ETF 對共識交集股的權重覆蓋率更高；不等於未來報酬保證。"
            )


def _render_two_etf_pick_view():
    title_col, refresh_col = st.columns([6, 1])
    with title_col:
        st.subheader("兩檔 ETF 推薦（進攻型）")
    with refresh_col:
        force_recompute = st.button(
            "重新計算",
            key="two_etf_pick_refresh",
            width="stretch",
            type="primary",
        )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption("策略定位：核心（共識代表）+ 衛星（低重疊動能），預設每月檢查一次。")
    with c2:
        allow_overseas = st.checkbox(
            "允許海外成分",
            value=False,
            key="two_etf_pick_allow_overseas",
        )
    with c3:
        force_refresh_constituents = st.checkbox(
            "重抓成分股（較慢）",
            value=False,
            key="two_etf_pick_force_refresh_constituents",
        )

    overlap_cap_pct = st.slider(
        "重疊門檻（Jaccard %）",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        key="two_etf_pick_overlap_cap",
        help="先用此門檻挑第2檔；若無候選會自動放寬到 20%，再不行才回退到最高YTD。",
    )

    if force_recompute:
        _build_two_etf_aggressive_picks.clear()
        _build_consensus_representative_between.clear()
        _build_tw_etf_top10_between.clear()
        st.rerun()

    start_target = "20251231"
    end_target = datetime.now().strftime("%Y%m%d")

    with st.container(border=True):
        _render_card_section_header(
            "兩檔推薦卡", "以前10ETF + 共識代表 + 成分股重疊度，輸出核心/衛星兩檔建議。"
        )
        try:
            payload = _build_two_etf_aggressive_picks(
                start_yyyymmdd=start_target,
                end_yyyymmdd=end_target,
                allow_overseas=bool(allow_overseas),
                overlap_cap_pct=float(overlap_cap_pct),
                force_refresh_constituents=bool(force_refresh_constituents),
            )
        except Exception as exc:
            st.error(f"無法建立兩檔 ETF 推薦：{exc}")
            return

        error_text = str(payload.get("error", "")).strip()
        if error_text:
            _emit_issue_message(error_text)
            issues = payload.get("issues", [])
            if isinstance(issues, list) and issues:
                _render_sync_issues("成分股取得有部分錯誤", issues, preview_limit=3)
            return

        pick_1 = payload.get("pick_1", {})
        pick_2 = payload.get("pick_2", {})
        pick_1 = pick_1 if isinstance(pick_1, dict) else {}
        pick_2 = pick_2 if isinstance(pick_2, dict) else {}
        start_used = str(payload.get("start_used", "")).strip()
        end_used = str(payload.get("end_used", "")).strip()
        top10_count = int(payload.get("top10_count", 0) or 0)
        universe_count = int(payload.get("universe_count", 0) or 0)
        overlap_cap_used = _safe_float(payload.get("overlap_cap_used"))
        overlap_pick_2 = _safe_float(pick_2.get("與核心重疊度(%)"))
        fallback_mode = str(payload.get("fallback_mode", "")).strip()

        pick_1_code = str(pick_1.get("ETF代碼", "—")).strip() or "—"
        pick_2_code = str(pick_2.get("ETF代碼", "—")).strip() or "—"
        overlap_text = "—" if overlap_pick_2 is None else f"{overlap_pick_2:.2f}%"
        cap_text = "—" if overlap_cap_used is None else f"{overlap_cap_used:.1f}%"

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("核心 ETF", pick_1_code)
        m2.metric("衛星 ETF", pick_2_code)
        m3.metric("兩檔重疊度", overlap_text)
        m4.metric("重疊門檻", cap_text)
        m5.metric("前10母體（可比較）", str(universe_count))

        st.caption(f"計算區間（實際交易日）：{start_used} -> {end_used}")
        st.caption(
            f"策略參數：風格=進攻型｜調整頻率=每月｜允許海外={'是' if bool(payload.get('allow_overseas', False)) else '否'}"
            f"｜候選池=前10 ETF（共 {top10_count} 檔）"
        )
        if fallback_mode == "strict_overlap":
            st.caption("第2檔選擇規則：在原始重疊門檻內，選 YTD 報酬最高的候選。")
        elif fallback_mode == "relaxed_overlap_20":
            st.caption("第2檔選擇規則：原始門檻無候選，已放寬到 20% 重疊門檻後選取。")
        else:
            st.caption("第2檔選擇規則：放寬門檻仍無候選，已回退為可投資候選中 YTD 最高。")

        snapshot_health = _build_snapshot_health(
            start_used=start_used,
            end_used=end_used,
            target_yyyymmdd=end_target,
        )
        _render_data_health_caption("快照資料健康度", snapshot_health)

        issues = payload.get("issues", [])
        if isinstance(issues, list) and issues:
            _render_sync_issues("部分資料抓取失敗，已用可用資料計算", issues, preview_limit=3)
        excluded_overseas = payload.get("excluded_overseas_codes", [])
        if isinstance(excluded_overseas, list) and excluded_overseas:
            st.caption(
                f"海外限制：已排除 {len(excluded_overseas)} 檔候選（{', '.join(excluded_overseas[:5])}）。"
            )

    recommendation_df = payload.get("recommendation_df")
    if isinstance(recommendation_df, pd.DataFrame) and not recommendation_df.empty:
        recommendation_df = _attach_tw_etf_management_fee_column(
            recommendation_df, code_col_candidates=("ETF代碼",)
        )
        with st.container(border=True):
            _render_card_section_header("推薦結果", "核心檔重代表性，衛星檔重低重疊動能。")
            show_cols = [
                col
                for col in [
                    "角色",
                    "ETF代碼",
                    "ETF名稱",
                    "管理費(%)",
                    "ETF規模(億)",
                    "ETF類型",
                    "YTD報酬(%)",
                    "與核心重疊度(%)",
                    "說明",
                ]
                if col in recommendation_df.columns
            ]
            st.dataframe(recommendation_df[show_cols], width="stretch", hide_index=True)

    candidate_df = payload.get("candidate_df")
    if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
        candidate_df = _attach_tw_etf_management_fee_column(
            candidate_df, code_col_candidates=("ETF代碼",)
        )
        with st.container(border=True):
            _render_card_section_header(
                "候選比較", "顯示前10候選的報酬、重疊度與是否納入本次推薦。"
            )
            show_cols = [
                col
                for col in [
                    "前10排名",
                    "ETF代碼",
                    "ETF名稱",
                    "管理費(%)",
                    "ETF規模(億)",
                    "ETF類型",
                    "YTD報酬(%)",
                    "與核心重疊度(%)",
                    "成分股數",
                    "是否海外成分",
                    "可納入候選",
                    "排除原因",
                ]
                if col in candidate_df.columns
            ]
            st.dataframe(candidate_df[show_cols], width="stretch", hide_index=True)
            st.caption("註：此卡為選股規則透明化，非未來報酬保證；請搭配資金配置與風險控管。")


def _render_active_etf_2026_ytd_view():
    store = _history_store()
    title_col, refresh_col = st.columns([6, 1])
    with title_col:
        st.subheader("2026 年主動式 ETF YTD（Buy & Hold）")
    with refresh_col:
        refresh_market = st.button(
            "更新最新市況", key="active_etf_ytd_update_market", width="stretch", type="primary"
        )
    if refresh_market:
        _fetch_twse_snapshot_with_fallback.clear()
        _build_tw_active_etf_ytd_between.clear()
        _load_twii_twse_month_close_map.clear()
        _load_twii_twse_return_between.clear()
        _load_tw_market_return_between.clear()
        _load_tw_etf_daily_change_map.clear()
        _load_tw_snapshot_open_map.clear()
        _load_tw_market_daily_return.clear()
        st.session_state.pop("active_etf_ytd_compare_payload", None)
        st.session_state.pop("active_etf_2025_compare_payload", None)
        st.rerun()

    start_target = "20260101"
    end_target = datetime.now().strftime("%Y%m%d")

    with st.container(border=True):
        _render_card_section_header(
            "主動式 ETF 績效卡", "台股主動式 ETF，2026 YTD Buy & Hold（復權版）。"
        )
        try:
            etf_df, start_used, end_used = _build_tw_active_etf_ytd_between(
                start_yyyymmdd=start_target, end_yyyymmdd=end_target
            )
            if etf_df.empty:
                st.warning("目前沒有可顯示的台股主動式 ETF YTD 資料。")
                return
            etf_symbols = tuple(
                str(x).strip().upper()
                for x in etf_df["代碼"].astype(str).tolist()
                if str(x).strip() and not str(x).strip().upper().startswith("^")
            )
            hist_2025_df, hist_2025_start_used, hist_2025_end_used = (
                _build_tw_active_etf_ytd_between(
                    start_yyyymmdd="20250101",
                    end_yyyymmdd="20251231",
                    symbols=etf_symbols,
                )
            )
            y2025_map = {
                str(row["代碼"]).strip().upper(): float(row["YTD報酬(%)"])
                for _, row in hist_2025_df.iterrows()
                if str(row.get("代碼", "")).strip() and pd.notna(row.get("YTD報酬(%)"))
            }
        except Exception as exc:
            st.error(f"無法建立主動式 ETF YTD 清單：{exc}")
            return
        market_return_pct: float | None = None
        market_symbol_used = ""
        market_issues: list[str] = []
        try:
            market_return_pct, market_symbol_used, market_issues = _load_tw_market_return_between(
                start_yyyymmdd=start_used,
                end_yyyymmdd=end_used,
                force_sync=False,
            )
        except Exception as exc:
            market_issues = [f"market: {exc}"]

        market_2025_return_pct: float | None = None
        market_2025_symbol_used = ""
        try:
            market_2025_return_pct, market_2025_symbol_used, market_2025_issues = (
                _load_tw_market_return_between(
                    start_yyyymmdd="20250101",
                    end_yyyymmdd=hist_2025_end_used,
                    force_sync=False,
                )
            )
            market_issues.extend(market_2025_issues)
        except Exception as exc:
            market_issues.append(f"market_2025: {exc}")

        etf_df["2025績效(%)"] = etf_df["代碼"].astype(str).str.strip().str.upper().map(y2025_map)
        etf_df["2025績效(%)"] = _truncate_series(etf_df["2025績效(%)"], digits=2)
        etf_df["贏輸台股大盤(%)"] = np.nan
        if market_return_pct is not None and math.isfinite(float(market_return_pct)):
            etf_df["贏輸台股大盤(%)"] = (
                pd.to_numeric(etf_df["YTD報酬(%)"], errors="coerce") - float(market_return_pct)
            ).round(2)

        benchmark_code = market_symbol_used or market_2025_symbol_used or "^TWII"
        benchmark_row = {
            "排名": "—",
            "代碼": benchmark_code,
            "ETF": "台股大盤",
            "開盤": np.nan,
            "收盤": np.nan,
            "績效起算日": "—",
            "績效終點日": end_used,
            "2025績效(%)": (
                _truncate_value(market_2025_return_pct, digits=2)
                if market_2025_return_pct is not None
                else np.nan
            ),
            "YTD報酬(%)": round(float(market_return_pct), 2)
            if market_return_pct is not None
            else np.nan,
            "贏輸台股大盤(%)": 0.0 if market_return_pct is not None else np.nan,
        }
        table_df = pd.concat([pd.DataFrame([benchmark_row]), etf_df], ignore_index=True)
        daily_change_map, daily_end_used, daily_prev_used = _load_tw_etf_daily_change_map(end_used)
        _, daily_open_map = _load_tw_snapshot_open_map(daily_end_used or end_used)
        market_daily_return, market_daily_symbol, _, _, market_daily_issues = (
            _load_tw_market_daily_return(end_used, force_sync=False)
        )
        market_issues.extend(market_daily_issues)
        table_df = _with_tw_today_fields(
            table_df,
            daily_change_map=daily_change_map,
            daily_open_map=daily_open_map,
            market_daily_return_pct=market_daily_return,
        )
        etf_today_df = _with_tw_today_fields(
            etf_df,
            daily_change_map=daily_change_map,
            daily_open_map=daily_open_map,
            market_daily_return_pct=market_daily_return,
        )
        table_df = _attach_tw_etf_management_fee_column(table_df, code_col_candidates=("代碼",))
        if "排名" in table_df.columns:
            table_df["排名"] = table_df["排名"].map(lambda v: str(v) if pd.notna(v) else "")
        columns_order = [
            "排名",
            "代碼",
            "ETF",
            "管理費(%)",
            "ETF規模(億)",
            "開盤",
            "收盤",
            "績效起算日",
            "績效終點日",
            "今日漲幅",
            "今日贏大盤%",
            "2025績效(%)",
            "YTD報酬(%)",
            "贏輸台股大盤(%)",
        ]
        table_df = table_df[[col for col in columns_order if col in table_df.columns]]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("樣本數", str(len(etf_df)))
        m2.metric("正報酬檔數", str(int((etf_df["YTD報酬(%)"] > 0).sum())))
        m3.metric("負報酬檔數", str(int((etf_df["YTD報酬(%)"] < 0).sum())))
        m4.metric("今日上漲檔數", str(int((etf_today_df["今日漲幅"] > 0).sum())))
        snapshot_health = _build_snapshot_health(
            start_used=start_used,
            end_used=end_used,
            target_yyyymmdd=end_target,
        )
        st.caption(f"計算區間（實際交易日）：{start_used} -> {end_used}")
        st.caption(f"2025 對照區間（實際交易日）：{hist_2025_start_used} -> {hist_2025_end_used}")
        _render_data_health_caption("快照資料健康度", snapshot_health)
        if market_return_pct is not None and market_symbol_used:
            st.caption(
                f"大盤對照：{market_symbol_used} 區間報酬 {market_return_pct:.2f}%（同 2026 YTD 區間）"
            )
        else:
            st.caption("大盤對照：目前無法取得，`贏輸台股大盤(%)` 先顯示為空白。")
        if (
            market_daily_return is not None
            and market_daily_symbol
            and daily_prev_used
            and daily_end_used
        ):
            st.caption(
                f"今日大盤漲幅：{market_daily_symbol} {market_daily_return:.2f}%（{daily_prev_used} -> {daily_end_used}）"
            )
        else:
            st.caption("今日大盤漲幅：目前無法取得，`今日贏大盤%` 先顯示空白。")
        if market_issues:
            _render_sync_issues("更新大盤資料時有部分同步錯誤", market_issues, preview_limit=2)
        st.caption(
            "資料來源：TWSE MI_INDEX（上市全市場快照）。主動式規則：代碼結尾 A 且名稱含「主動」。"
        )
        st.caption(
            "`2025績效(%)` 採各檔在 2025 區間內首個可交易日計算；若空白代表該檔 2025 區間無可用日K。"
        )
        st.caption(
            "報酬計算：Buy & Hold（復權版，套用已知 split 事件）；`贏輸台股大盤(%) = YTD報酬 - 大盤報酬`。"
        )
        st.dataframe(table_df, width="stretch", hide_index=True)

    symbols = [
        str(x).strip().upper() for x in etf_df["代碼"].astype(str).tolist() if str(x).strip()
    ]
    if not symbols:
        return
    name_map = {
        str(row["代碼"]).strip().upper(): str(row["ETF"]).strip()
        for _, row in etf_df.iterrows()
        if str(row.get("代碼", "")).strip()
    }

    def _render_active_etf_benchmark_card(
        *,
        card_title: str,
        period_caption: str,
        date_start: str,
        date_end: str,
        key_prefix: str,
        spinner_text: str,
    ):
        with st.container(border=True):
            _render_card_section_header(
                card_title, "策略曲線、基準曲線與每檔 Buy & Hold 同圖比較。"
            )
            st.caption(period_caption)
            st.markdown(
                (
                    "說明："
                    "<span title='把可用主動式ETF平均分配資金後，從區間起點買進並持有到期末，不做換股或調倉。'>"
                    "<code>Strategy Equity（主動式ETF等權）</code>（滑鼠移入查看）"
                    "</span>"
                ),
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns([2, 2, 1])
            benchmark_choice = c1.selectbox(
                "Benchmark",
                options=["twii", "0050", "006208"],
                index=0,
                format_func=lambda x: {"twii": "^TWII", "0050": "0050", "006208": "006208"}.get(
                    x, x
                ),
                key=f"{key_prefix}_benchmark",
            )
            sync_before_run = c2.checkbox(
                "執行前同步最新日K（較慢）",
                value=False,
                key=f"{key_prefix}_sync_before_run",
            )
            force_refresh = c3.button("重新計算", width="stretch", key=f"{key_prefix}_refresh")

            try:
                start_dt = datetime.combine(
                    datetime.strptime(date_start, "%Y%m%d").date(), datetime.min.time()
                ).replace(tzinfo=timezone.utc)
                end_dt = datetime.combine(
                    datetime.strptime(date_end, "%Y%m%d").date(), datetime.min.time()
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                st.warning(f"{card_title} 日期格式不正確，無法建立對照圖。")
                return
            if end_dt <= start_dt:
                st.warning(f"{card_title} 日期區間不足，無法建立對照圖。")
                return

            run_key = f"{key_prefix}:{date_start}:{date_end}:{benchmark_choice}:{sync_before_run}:{','.join(symbols)}"
            payload_key = f"{key_prefix}_compare_payload"
            payload = st.session_state.get(payload_key)
            if not isinstance(payload, dict):
                payload = {}

            should_recompute = force_refresh or payload.get("run_key") != run_key
            if should_recompute:
                with st.spinner(spinner_text):
                    payload = _compute_tw_equal_weight_compare_payload(
                        symbols=symbols,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        benchmark_choice=benchmark_choice,
                        sync_before_run=sync_before_run,
                        insufficient_msg="可用主動式 ETF 歷史資料不足，無法建立對照圖。",
                    )
                payload["run_key"] = run_key
                st.session_state[payload_key] = payload

            payload = st.session_state.get(payload_key, {})
            if not isinstance(payload, dict):
                payload = {}
            error_text = str(payload.get("error", "")).strip()
            if error_text:
                _emit_issue_message(error_text)
                return

            symbol_sync_issues = payload.get("symbol_sync_issues", [])
            if isinstance(symbol_sync_issues, list) and symbol_sync_issues:
                _render_sync_issues(
                    "部分 ETF 同步失敗，已盡量使用本地可用資料", symbol_sync_issues, preview_limit=3
                )

            benchmark_sync_issues = payload.get("benchmark_sync_issues", [])
            if isinstance(benchmark_sync_issues, list) and benchmark_sync_issues:
                _render_sync_issues(
                    "Benchmark 同步有部分錯誤，已盡量使用本地可用資料",
                    benchmark_sync_issues,
                    preview_limit=2,
                )

            strategy_equity = payload.get("strategy_equity")
            benchmark_equity = payload.get("benchmark_equity")
            per_symbol_equity = payload.get("per_symbol_equity")
            if not isinstance(strategy_equity, pd.Series) or strategy_equity.empty:
                st.warning("目前無法建立策略曲線。")
                return
            if not isinstance(benchmark_equity, pd.Series):
                benchmark_equity = pd.Series(dtype=float)
            if not isinstance(per_symbol_equity, dict):
                per_symbol_equity = {}
            compare_health = _build_data_health(
                as_of=strategy_equity.index.max() if len(strategy_equity.index) else "",
                data_sources=[_store_data_source(store, "daily_bars")],
                source_chain=[str(payload.get("benchmark_symbol", "") or "benchmark")],
                degraded=bool(symbol_sync_issues or benchmark_sync_issues),
                fallback_depth=len(symbol_sync_issues) + len(benchmark_sync_issues),
                notes="部分標的同步失敗時會盡量使用本地資料",
            )
            _render_data_health_caption("Benchmark 對照資料健康度", compare_health)

            palette = _ui_palette()
            symbol_styles = _build_symbol_line_styles(list(per_symbol_equity.keys()))
            benchmark_label = str(payload.get("benchmark_symbol", "") or "Benchmark")
            chart_lines: list[dict[str, Any]] = [
                {
                    "name": "Strategy Equity（主動式ETF等權）",
                    "series": strategy_equity,
                    "color": str(palette["equity"]),
                    "width": 2.4,
                    "dash": "solid",
                    "hover_code": f"{key_prefix.upper()}_EW",
                    "value_label": "Equity",
                    "y_format": ",.0f",
                }
            ]
            if not benchmark_equity.empty:
                style = _benchmark_line_style(palette, width=2.0)
                chart_lines.append(
                    {
                        "name": f"Benchmark Equity（{benchmark_label}）",
                        "series": benchmark_equity,
                        "color": str(style["color"]),
                        "width": float(style["width"]),
                        "dash": str(style["dash"]),
                        "hover_code": benchmark_label,
                        "value_label": "Equity",
                        "y_format": ",.0f",
                    }
                )
            for symbol in sorted(per_symbol_equity.keys()):
                series = per_symbol_equity[symbol]
                if not isinstance(series, pd.Series) or len(series) < 2:
                    continue
                style = symbol_styles.get(symbol, {"color": "#1f77b4", "dash": "solid"})
                label_name = str(name_map.get(symbol, symbol)).strip()
                trace_name = (
                    f"Buy-and-Hold（{symbol} {label_name}）"
                    if label_name and label_name != symbol
                    else f"Buy-and-Hold（{symbol}）"
                )
                chart_lines.append(
                    {
                        "name": trace_name,
                        "series": series,
                        "color": str(style["color"]),
                        "width": 1.8,
                        "dash": str(style["dash"]),
                        "hover_code": symbol,
                        "value_label": "Equity",
                        "y_format": ",.0f",
                    }
                )
            _render_benchmark_lines_chart(
                lines=chart_lines,
                height=460,
                chart_key=f"{key_prefix}_benchmark",
            )

            summary_rows: list[dict[str, object]] = []
            strategy_perf = _series_metrics_basic(strategy_equity)
            summary_rows.append(
                {
                    "Series": "Strategy Equity（主動式ETF等權）",
                    "Total Return %": round(strategy_perf["total_return"] * 100.0, 2),
                    "CAGR %": round(strategy_perf["cagr"] * 100.0, 2),
                    "MDD %": round(strategy_perf["max_drawdown"] * 100.0, 2),
                    "Sharpe": round(strategy_perf["sharpe"], 2),
                }
            )
            if not benchmark_equity.empty:
                benchmark_perf = _series_metrics_basic(benchmark_equity)
                benchmark_label = str(payload.get("benchmark_symbol", "") or "Benchmark")
                summary_rows.append(
                    {
                        "Series": f"Benchmark Equity（{benchmark_label}）",
                        "Total Return %": round(benchmark_perf["total_return"] * 100.0, 2),
                        "CAGR %": round(benchmark_perf["cagr"] * 100.0, 2),
                        "MDD %": round(benchmark_perf["max_drawdown"] * 100.0, 2),
                        "Sharpe": round(benchmark_perf["sharpe"], 2),
                    }
                )

            for symbol in sorted(per_symbol_equity.keys()):
                series = per_symbol_equity[symbol]
                if not isinstance(series, pd.Series) or len(series) < 2:
                    continue
                perf = _series_metrics_basic(series)
                label_name = str(name_map.get(symbol, symbol)).strip()
                series_name = (
                    f"Buy-and-Hold（{symbol} {label_name}）"
                    if label_name and label_name != symbol
                    else f"Buy-and-Hold（{symbol}）"
                )
                summary_rows.append(
                    {
                        "Series": series_name,
                        "Total Return %": round(perf["total_return"] * 100.0, 2),
                        "CAGR %": round(perf["cagr"] * 100.0, 2),
                        "MDD %": round(perf["max_drawdown"] * 100.0, 2),
                        "Sharpe": round(perf["sharpe"], 2),
                    }
                )

            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
            used_symbols = payload.get("used_symbols", [])
            skipped_symbols = payload.get("skipped_symbols", [])
            if isinstance(used_symbols, list) and isinstance(skipped_symbols, list):
                st.caption(
                    f"可用ETF：{len(used_symbols)} 檔 | 資料不足未納入：{len(skipped_symbols)} 檔"
                )

    _render_active_etf_benchmark_card(
        card_title="Benchmark 對照卡",
        period_caption=(
            f"差異說明：上方表格的 `YTD報酬(%)` 採各檔在區間內首個可交易日起算；"
            f"本對照卡曲線與下方 `Total Return %` 會以共同比較區間 `{start_used} -> {end_used}` 對齊，"
            "因此同一檔數值可能略有差異。"
        ),
        date_start=start_used,
        date_end=end_used,
        key_prefix="active_etf_ytd",
        spinner_text="計算主動式 ETF Benchmark 對照中...",
    )
    _render_active_etf_benchmark_card(
        card_title="2025 全年 Benchmark 對照卡",
        period_caption=(
            f"差異說明：上方表格的 `2025績效(%)` 採各檔在 2025 區間首個可交易日起算；"
            f"本對照卡曲線與下方 `Total Return %` 會以共同比較區間 `{hist_2025_start_used} -> {hist_2025_end_used}` 對齊，"
            "因此同一檔數值可能略有差異。"
        ),
        date_start=hist_2025_start_used,
        date_end=hist_2025_end_used,
        key_prefix="active_etf_2025",
        spinner_text="計算主動式 ETF 2025 Benchmark 對照中...",
    )


def _render_quality_bar(ctx, refresh_sec: int):
    provider_labels = {
        "fugle_ws": "Fugle WebSocket",
        "tw_fugle_rest": "Fugle Historical",
        "tw_mis": "TW MIS",
        "tw_openapi": "TW OpenAPI",
        "tw_tpex": "TPEx OpenAPI",
        "twelvedata": "Twelve Data",
        "yahoo": "Yahoo",
        "stooq": "Stooq",
    }

    def _label(name: str) -> str:
        key = str(name or "").strip().lower()
        if not key:
            return "unknown"
        return provider_labels.get(key, key)

    def _label_data_source(name: str) -> str:
        key = str(name or "").strip().lower()
        if not key:
            return "unknown"
        if key.startswith("cache:last_good:"):
            base = key[len("cache:last_good:") :]
            return f"快取({_label(base)})"
        if key.startswith("cache:"):
            return "快取"
        if key.endswith("_ticks"):
            base = key[: -len("_ticks")]
            return f"{_label(base)}(tick聚合)"
        if key.endswith("_tail"):
            base = key[: -len("_tail")]
            return f"{_label(base)}(日K尾段)"
        if key.endswith("_resampled"):
            base = key[: -len("_resampled")]
            return f"{_label(base)}(重採樣)"
        if key.endswith("_quote_derived"):
            base = key[: -len("_quote_derived")]
            return f"{_label(base)}(報價推導)"
        return _label(key)

    quote = ctx.quote
    quality = ctx.quality
    source_chain = [
        str(s or "").strip() for s in list(getattr(ctx, "source_chain", [])) if str(s or "").strip()
    ]
    chain_text = " -> ".join([_label(s) for s in source_chain]) if source_chain else "—"
    current_source = _label(str(getattr(quote, "source", "") or "unknown"))
    st.caption(f"即時來源：{current_source} | 鏈路：{chain_text}")
    intraday_source = _label_data_source(str(getattr(ctx, "intraday_source", "") or "unknown"))
    intraday_bars = 0
    try:
        intraday_bars = int(len(getattr(ctx, "intraday", pd.DataFrame())))
    except Exception:
        intraday_bars = 0
    st.caption(f"即時走勢來源：{intraday_source} | K數={intraday_bars}")

    health = _build_data_health(
        as_of=getattr(quote, "ts", ""),
        data_sources=[str(getattr(quote, "source", "") or "unknown")],
        source_chain=source_chain,
        degraded=bool(getattr(quality, "degraded", False)),
        fallback_depth=int(getattr(quality, "fallback_depth", 0) or 0),
        freshness_sec=int(getattr(quality, "freshness_sec", 0) or 0)
        if getattr(quality, "freshness_sec", None) is not None
        else None,
        notes=f"refresh={refresh_sec}s | delayed={'yes' if quote.is_delayed else 'no'}",
    )
    _render_data_health_caption("資料品質", health)
    st.caption(f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（fragment 局部刷新）")


_UI_RUNTIME_CONFIGURED = False


def _collect_runtime_values(required_names: tuple[str, ...]) -> dict[str, object]:
    module_globals = globals()
    missing = [name for name in required_names if name not in module_globals]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise RuntimeError(f"missing app runtime symbols: {preview}{suffix}")
    return {name: module_globals[name] for name in required_names}


def _ensure_ui_runtime_configured() -> None:
    global _UI_RUNTIME_CONFIGURED
    if _UI_RUNTIME_CONFIGURED:
        return
    from ui.core import charts as charts_module
    from ui.pages import backtest as backtest_page
    from ui.pages import live as live_page

    charts_module.configure_runtime(_collect_runtime_values(charts_module.REQUIRED_RUNTIME_NAMES))
    live_page.configure_runtime(_collect_runtime_values(live_page.REQUIRED_RUNTIME_NAMES))
    backtest_page.configure_runtime(_collect_runtime_values(backtest_page.REQUIRED_RUNTIME_NAMES))
    _UI_RUNTIME_CONFIGURED = True


def _render_benchmark_lines_chart(*args, **kwargs):
    _ensure_ui_runtime_configured()
    from ui.core.charts import _render_benchmark_lines_chart as impl

    return impl(*args, **kwargs)


def _render_live_chart(*args, **kwargs):
    _ensure_ui_runtime_configured()
    from ui.core.charts import _render_live_chart as impl

    return impl(*args, **kwargs)


def _render_indicator_panels(*args, **kwargs):
    _ensure_ui_runtime_configured()
    from ui.core.charts import _render_indicator_panels as impl

    return impl(*args, **kwargs)


def _render_live_view():
    _ensure_ui_runtime_configured()
    from ui.pages.live import _render_live_view as impl

    return impl()


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


def _cache_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if math.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _cache_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _build_sync_rows(symbols: list[str], reports: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        report = reports.get(symbol)
        rows.append(
            {
                "symbol": symbol,
                "rows": int(getattr(report, "rows_upserted", 0) or 0),
                "source": str(getattr(report, "source", "unknown") or "unknown"),
                "fallback": int(getattr(report, "fallback_depth", 0) or 0),
                "error": str(getattr(report, "error", "") or ""),
            }
        )
    return rows


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _frame_to_split_payload(df: pd.DataFrame) -> dict[str, object]:
    if not isinstance(df, pd.DataFrame):
        return {}
    try:
        payload = json.loads(df.to_json(orient="split", date_format="iso"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _series_to_split_payload(series: pd.Series) -> dict[str, object]:
    if not isinstance(series, pd.Series):
        return {}
    try:
        payload = json.loads(series.to_json(orient="split", date_format="iso"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _frame_from_split_payload(
    payload: object, *, default_columns: list[str] | None = None
) -> pd.DataFrame:
    defaults = default_columns or []
    if not isinstance(payload, dict) or not payload:
        return pd.DataFrame(columns=defaults)
    try:
        frame = pd.read_json(StringIO(json.dumps(payload, ensure_ascii=False)), orient="split")
        idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
        if len(idx) and not idx.isna().all():
            frame.index = idx
        return frame
    except Exception:
        return pd.DataFrame(columns=defaults)


def _series_from_split_payload(payload: object) -> pd.Series:
    if not isinstance(payload, dict) or not payload:
        return pd.Series(dtype=float)
    try:
        series = pd.read_json(
            StringIO(json.dumps(payload, ensure_ascii=False)),
            orient="split",
            typ="series",
        )
        idx = pd.to_datetime(series.index, utc=True, errors="coerce")
        if len(idx) and not idx.isna().all():
            series.index = idx
        return series
    except Exception:
        return pd.Series(dtype=float)


def _serialize_backtest_metrics(metrics: BacktestMetrics) -> dict[str, object]:
    return {
        "total_return": _cache_float(getattr(metrics, "total_return", 0.0)),
        "cagr": _cache_float(getattr(metrics, "cagr", 0.0)),
        "max_drawdown": _cache_float(getattr(metrics, "max_drawdown", 0.0)),
        "sharpe": _cache_float(getattr(metrics, "sharpe", 0.0)),
        "win_rate": _cache_float(getattr(metrics, "win_rate", 0.0)),
        "avg_win": _cache_float(getattr(metrics, "avg_win", 0.0)),
        "avg_loss": _cache_float(getattr(metrics, "avg_loss", 0.0)),
        "trades": _cache_int(getattr(metrics, "trades", 0)),
    }


def _deserialize_backtest_metrics(payload: object) -> BacktestMetrics:
    data = payload if isinstance(payload, dict) else {}
    return BacktestMetrics(
        total_return=_cache_float(data.get("total_return")),
        cagr=_cache_float(data.get("cagr")),
        max_drawdown=_cache_float(data.get("max_drawdown")),
        sharpe=_cache_float(data.get("sharpe")),
        win_rate=_cache_float(data.get("win_rate")),
        avg_win=_cache_float(data.get("avg_win")),
        avg_loss=_cache_float(data.get("avg_loss")),
        trades=_cache_int(data.get("trades")),
    )


def _serialize_trade(trade: Trade) -> dict[str, object]:
    return {
        "entry_date": pd.Timestamp(trade.entry_date).isoformat(),
        "entry_price": _cache_float(trade.entry_price),
        "exit_date": pd.Timestamp(trade.exit_date).isoformat(),
        "exit_price": _cache_float(trade.exit_price),
        "qty": _cache_float(trade.qty),
        "fee": _cache_float(trade.fee),
        "tax": _cache_float(trade.tax),
        "slippage": _cache_float(trade.slippage),
        "pnl": _cache_float(trade.pnl),
        "pnl_pct": _cache_float(trade.pnl_pct),
    }


def _deserialize_trade(payload: object) -> Trade | None:
    if not isinstance(payload, dict):
        return None
    try:
        return Trade(
            entry_date=_to_utc_timestamp(payload.get("entry_date")),
            entry_price=_cache_float(payload.get("entry_price")),
            exit_date=_to_utc_timestamp(payload.get("exit_date")),
            exit_price=_cache_float(payload.get("exit_price")),
            qty=_cache_float(payload.get("qty")),
            fee=_cache_float(payload.get("fee")),
            tax=_cache_float(payload.get("tax")),
            slippage=_cache_float(payload.get("slippage")),
            pnl=_cache_float(payload.get("pnl")),
            pnl_pct=_cache_float(payload.get("pnl_pct")),
        )
    except Exception:
        return None


def _serialize_single_backtest_result(result: BacktestResult) -> dict[str, object]:
    return {
        "kind": "single",
        "equity_curve": _frame_to_split_payload(result.equity_curve),
        "trades": [_serialize_trade(t) for t in (result.trades or [])],
        "metrics": _serialize_backtest_metrics(result.metrics),
        "drawdown_series": _series_to_split_payload(result.drawdown_series),
        "yearly_returns": {
            str(k): _cache_float(v) for k, v in (result.yearly_returns or {}).items()
        },
        "signals": _series_to_split_payload(result.signals),
    }


def _deserialize_single_backtest_result(payload: object) -> BacktestResult | None:
    data = payload if isinstance(payload, dict) else {}
    if not data:
        return None
    trades: list[Trade] = []
    for item in data.get("trades", []) if isinstance(data.get("trades"), list) else []:
        trade = _deserialize_trade(item)
        if trade is not None:
            trades.append(trade)
    yearly_raw = data.get("yearly_returns", {})
    yearly_returns = {}
    if isinstance(yearly_raw, dict):
        yearly_returns = {str(k): _cache_float(v) for k, v in yearly_raw.items()}
    return BacktestResult(
        equity_curve=_frame_from_split_payload(
            data.get("equity_curve"), default_columns=["equity"]
        ),
        trades=trades,
        metrics=_deserialize_backtest_metrics(data.get("metrics")),
        drawdown_series=_series_from_split_payload(data.get("drawdown_series")),
        yearly_returns=yearly_returns,
        signals=_series_from_split_payload(data.get("signals")).fillna(0).astype(int),
    )


def _serialize_portfolio_backtest_result(result: PortfolioBacktestResult) -> dict[str, object]:
    return {
        "kind": "portfolio",
        "equity_curve": _frame_to_split_payload(result.equity_curve),
        "metrics": _serialize_backtest_metrics(result.metrics),
        "drawdown_series": _series_to_split_payload(result.drawdown_series),
        "yearly_returns": {
            str(k): _cache_float(v) for k, v in (result.yearly_returns or {}).items()
        },
        "trades": _frame_to_split_payload(result.trades),
        "signals": _frame_to_split_payload(result.signals),
        "component_results": {
            str(sym): _serialize_single_backtest_result(comp)
            for sym, comp in result.component_results.items()
        },
    }


def _deserialize_portfolio_backtest_result(payload: object) -> PortfolioBacktestResult | None:
    data = payload if isinstance(payload, dict) else {}
    if not data:
        return None
    comp_raw = data.get("component_results", {})
    component_results: dict[str, BacktestResult] = {}
    if isinstance(comp_raw, dict):
        for sym, comp_payload in comp_raw.items():
            comp_result = _deserialize_single_backtest_result(comp_payload)
            if comp_result is not None:
                component_results[str(sym)] = comp_result
    yearly_raw = data.get("yearly_returns", {})
    yearly_returns = {}
    if isinstance(yearly_raw, dict):
        yearly_returns = {str(k): _cache_float(v) for k, v in yearly_raw.items()}
    return PortfolioBacktestResult(
        equity_curve=_frame_from_split_payload(
            data.get("equity_curve"), default_columns=["equity"]
        ),
        metrics=_deserialize_backtest_metrics(data.get("metrics")),
        drawdown_series=_series_from_split_payload(data.get("drawdown_series")),
        yearly_returns=yearly_returns,
        trades=_frame_from_split_payload(data.get("trades")),
        signals=_frame_from_split_payload(data.get("signals")),
        component_results=component_results,
    )


def _serialize_backtest_run_payload_v2(run_payload: dict[str, object]) -> dict[str, object]:
    mode = str(run_payload.get("mode", "")).strip()
    serialized: dict[str, object] = {
        "mode": mode,
        "walk_forward": bool(run_payload.get("walk_forward")),
        "initial_capital": _cache_float(run_payload.get("initial_capital"), default=1_000_000.0),
    }
    if mode == "single":
        serialized["symbol"] = str(run_payload.get("symbol", "")).strip().upper()
    else:
        symbols_raw = run_payload.get("symbols", [])
        if isinstance(symbols_raw, list):
            serialized["symbols"] = [str(s).strip().upper() for s in symbols_raw if str(s).strip()]

    bars_map = run_payload.get("bars_by_symbol", {})
    serialized_bars: dict[str, object] = {}
    if isinstance(bars_map, dict):
        for symbol, bars in bars_map.items():
            if isinstance(bars, pd.DataFrame):
                serialized_bars[str(symbol).strip().upper()] = _frame_to_split_payload(bars)
    serialized["bars_by_symbol"] = serialized_bars

    result = run_payload.get("result")
    if mode == "portfolio" and isinstance(result, PortfolioBacktestResult):
        serialized["result"] = _serialize_portfolio_backtest_result(result)
    elif mode == "single" and isinstance(result, BacktestResult):
        serialized["result"] = _serialize_single_backtest_result(result)
    else:
        serialized["result"] = {}

    if bool(run_payload.get("walk_forward")):
        train = run_payload.get("train_result")
        if mode == "portfolio" and isinstance(train, PortfolioBacktestResult):
            serialized["train_result"] = _serialize_portfolio_backtest_result(train)
        elif mode == "single" and isinstance(train, BacktestResult):
            serialized["train_result"] = _serialize_single_backtest_result(train)
        serialized["split_date"] = (
            pd.Timestamp(run_payload.get("split_date")).isoformat()
            if run_payload.get("split_date") is not None
            else ""
        )
        best_params = run_payload.get("best_params", {})
        serialized["best_params"] = best_params if isinstance(best_params, dict) else {}
        candidates = run_payload.get("candidates", {})
        if isinstance(candidates, (dict, int, float)):
            serialized["candidates"] = candidates
    return serialized


def _deserialize_backtest_run_payload_v2(payload: object) -> dict[str, object] | None:
    data = payload if isinstance(payload, dict) else {}
    mode = str(data.get("mode", "")).strip()
    if mode not in {"single", "portfolio"}:
        return None

    bars_map_raw = data.get("bars_by_symbol", {})
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    if isinstance(bars_map_raw, dict):
        for symbol, bars_payload in bars_map_raw.items():
            bars_df = _frame_from_split_payload(bars_payload)
            if not bars_df.empty:
                bars_by_symbol[str(symbol).strip().upper()] = bars_df
    if not bars_by_symbol:
        return None

    result_payload = data.get("result", {})
    if mode == "portfolio":
        result = _deserialize_portfolio_backtest_result(result_payload)
    else:
        result = _deserialize_single_backtest_result(result_payload)
    if result is None:
        return None
    if mode == "portfolio" and isinstance(result, PortfolioBacktestResult):
        if any(sym not in result.component_results for sym in bars_by_symbol):
            return None

    run_payload: dict[str, object] = {
        "mode": mode,
        "walk_forward": bool(data.get("walk_forward")),
        "initial_capital": _cache_float(data.get("initial_capital"), default=1_000_000.0),
        "bars_by_symbol": bars_by_symbol,
        "result": result,
    }
    if mode == "single":
        run_payload["symbol"] = str(data.get("symbol", "")).strip().upper() or next(
            iter(bars_by_symbol.keys())
        )
    else:
        raw_symbols = data.get("symbols", [])
        if isinstance(raw_symbols, list):
            symbols = [str(s).strip().upper() for s in raw_symbols if str(s).strip()]
            run_payload["symbols"] = symbols or list(bars_by_symbol.keys())
        else:
            run_payload["symbols"] = list(bars_by_symbol.keys())

    if run_payload["walk_forward"]:
        train_payload = data.get("train_result", {})
        if mode == "portfolio":
            train_result = _deserialize_portfolio_backtest_result(train_payload)
        else:
            train_result = _deserialize_single_backtest_result(train_payload)
        if train_result is not None:
            run_payload["train_result"] = train_result
        split_date_raw = str(data.get("split_date", "")).strip()
        if split_date_raw:
            try:
                run_payload["split_date"] = _to_utc_timestamp(split_date_raw)
            except Exception:
                pass
        best_params = data.get("best_params", {})
        if isinstance(best_params, dict):
            run_payload["best_params"] = best_params
        candidates = data.get("candidates")
        if isinstance(candidates, (dict, int, float)):
            run_payload["candidates"] = candidates
    return run_payload


def _serialize_backtest_run_payload(run_payload: dict[str, object]) -> dict[str, object]:
    legacy = _serialize_backtest_run_payload_v2(run_payload)
    mode = str(legacy.get("mode", "")).strip()
    walk_forward = bool(legacy.get("walk_forward"))
    meta: dict[str, object] = {
        "mode": mode,
        "walk_forward": walk_forward,
        "initial_capital": _cache_float(legacy.get("initial_capital"), default=1_000_000.0),
    }
    if mode == "single":
        meta["symbol"] = str(legacy.get("symbol", "")).strip().upper()
    else:
        symbols_raw = legacy.get("symbols", [])
        if isinstance(symbols_raw, list):
            meta["symbols"] = [str(s).strip().upper() for s in symbols_raw if str(s).strip()]
    if walk_forward:
        meta["split_date"] = str(legacy.get("split_date", "") or "")
        best_params = legacy.get("best_params", {})
        if isinstance(best_params, dict):
            meta["best_params"] = best_params
        candidates = legacy.get("candidates")
        if isinstance(candidates, (dict, int, float)):
            meta["candidates"] = candidates

    train_payload = legacy.get("train_result", {})
    if not isinstance(train_payload, dict):
        train_payload = {}
    return {
        "format_version": 3,
        "meta": meta,
        "bars": {"by_symbol": legacy.get("bars_by_symbol", {})},
        "results": {
            "test": legacy.get("result", {}),
            "train": train_payload,
        },
    }


def _deserialize_backtest_run_payload(payload: object) -> dict[str, object] | None:
    data = payload if isinstance(payload, dict) else {}
    if int(data.get("format_version", 0) or 0) == 3:
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            return None
        bars_layer = data.get("bars", {})
        bars_by_symbol = bars_layer.get("by_symbol", {}) if isinstance(bars_layer, dict) else {}
        results_layer = data.get("results", {})
        if not isinstance(results_layer, dict):
            return None

        mode = str(meta.get("mode", "")).strip()
        if mode not in {"single", "portfolio"}:
            return None
        legacy_payload: dict[str, object] = {
            "mode": mode,
            "walk_forward": bool(meta.get("walk_forward")),
            "initial_capital": _cache_float(meta.get("initial_capital"), default=1_000_000.0),
            "bars_by_symbol": bars_by_symbol if isinstance(bars_by_symbol, dict) else {},
            "result": results_layer.get("test", {}),
        }
        if mode == "single":
            legacy_payload["symbol"] = str(meta.get("symbol", "")).strip().upper()
        else:
            symbols_raw = meta.get("symbols", [])
            if isinstance(symbols_raw, list):
                legacy_payload["symbols"] = [
                    str(s).strip().upper() for s in symbols_raw if str(s).strip()
                ]
        if bool(meta.get("walk_forward")):
            legacy_payload["train_result"] = results_layer.get("train", {})
            legacy_payload["split_date"] = str(meta.get("split_date", "") or "")
            best_params = meta.get("best_params", {})
            if isinstance(best_params, dict):
                legacy_payload["best_params"] = best_params
            candidates = meta.get("candidates")
            if isinstance(candidates, (dict, int, float)):
                legacy_payload["candidates"] = candidates
        return _deserialize_backtest_run_payload_v2(legacy_payload)
    return _deserialize_backtest_run_payload_v2(payload)


def _load_cached_backtest_payload(
    *,
    store: HistoryStore,
    run_key: str,
    expected_schema: int,
    expected_hash: str,
) -> tuple[dict[str, object] | None, str]:
    cached_replay = store.load_latest_backtest_replay_run(run_key)
    if cached_replay is None or not isinstance(cached_replay.payload, dict):
        return None, ""
    cached_params = cached_replay.params if isinstance(cached_replay.params, dict) else {}
    cached_schema = int(cached_params.get("schema_version", 0) or 0)
    cached_hash = str(cached_params.get("source_hash", "")).strip()
    compatible_schemas = {int(expected_schema)}
    if int(expected_schema) >= 3:
        compatible_schemas.add(2)
    if cached_schema not in compatible_schemas or cached_hash != expected_hash:
        return None, "偵測到舊版快取或參數簽章不一致，本次會重新計算回測。"
    restored = _deserialize_backtest_run_payload(cached_replay.payload)
    if restored is None:
        return None, "偵測到快取內容不完整，本次會重新計算回測。"
    return restored, "已載入同條件的本地快取回測結果。"


def _render_backtest_view():
    _ensure_ui_runtime_configured()
    from ui.pages.backtest import _render_backtest_view as impl

    return impl()


def _render_tutorial_view():
    st.subheader("新手教學（完整功能版）")
    st.caption("目標：先知道每頁在做什麼，再跑回測，最後才做比較與微調。")

    st.markdown("### 0) 全站功能地圖（先看這張）")
    page_df = pd.DataFrame(
        [
            {
                "分頁": "即時看盤",
                "你會看到什麼": "即時行情、即時趨勢、技術快照、建議卡",
                "什麼時候用": "盤中快速看狀態",
            },
            {
                "分頁": "回測工作台",
                "你會看到什麼": "單檔/投組回測、Walk-Forward、Benchmark 比較、回放",
                "什麼時候用": "驗證策略與參數",
            },
            {
                "分頁": "2026 YTD 前十大股利型、配息型 ETF",
                "你會看到什麼": "2026 年迄今股利/配息型 Top10、2025 對照、大盤勝負、Benchmark 對照卡",
                "什麼時候用": "看今年高股息族群領先 ETF",
            },
            {
                "分頁": "2026 YTD 前十大 ETF",
                "你會看到什麼": "2026 年迄今 Top10、2025 對照、大盤勝負、Benchmark 對照卡",
                "什麼時候用": "看今年領先 ETF",
            },
            {
                "分頁": "台股 ETF 全類型總表",
                "你會看到什麼": "台股 ETF 全名單、類型分類、2025/2026YTD 與大盤勝負",
                "什麼時候用": "一次盤點全市場 ETF 的跨年度相對強弱",
            },
            {
                "分頁": "2025 後20大最差勁 ETF",
                "你會看到什麼": "2025 全年報酬後20名排行",
                "什麼時候用": "快速盤點全年落後族群",
            },
            {
                "分頁": "共識代表 ETF",
                "你會看到什麼": "以前10 ETF 成分股交集推導建議ETF與備選",
                "什麼時候用": "想要從多檔收斂到單一核心ETF",
            },
            {
                "分頁": "兩檔 ETF 推薦",
                "你會看到什麼": "核心+衛星兩檔建議（低重疊動能）",
                "什麼時候用": "想快速落地成可執行兩檔組合",
            },
            {
                "分頁": "2026 YTD 主動式 ETF",
                "你會看到什麼": "主動式 ETF 排行、2025 對照、大盤勝負、Benchmark 對照卡",
                "什麼時候用": "追蹤主動式 ETF 表現",
            },
            {
                "分頁": "ETF 輪動策略",
                "你會看到什麼": "固定 ETF 池月調倉回測、調倉明細、持有最久排名",
                "什麼時候用": "看規則化輪動是否優於基準",
            },
            {
                "分頁": "00910 熱力圖",
                "你會看到什麼": "全球成分股 YTD 分組熱力圖 + 台股子集合進階回測",
                "什麼時候用": "同時看國內/海外成分股相對表現",
            },
            {
                "分頁": "00935 熱力圖",
                "你會看到什麼": "成分股相對大盤熱力圖 + 公司簡介",
                "什麼時候用": "看 00935 內部強弱分布",
            },
            {
                "分頁": "00735 熱力圖",
                "你會看到什麼": "成分股相對大盤熱力圖 + 公司簡介",
                "什麼時候用": "看 00735 內部強弱分布",
            },
            {
                "分頁": "0050 熱力圖",
                "你會看到什麼": "成分股相對大盤熱力圖 + 公司簡介（依權重排序）",
                "什麼時候用": "看台灣 50 內部強弱",
            },
            {
                "分頁": "0052 熱力圖",
                "你會看到什麼": "成分股相對大盤熱力圖 + 公司簡介",
                "什麼時候用": "看 0052 內部強弱分布",
            },
            {
                "分頁": "資料庫檢視",
                "你會看到什麼": "DuckDB/SQLite 表格總覽、欄位、分頁資料",
                "什麼時候用": "確認資料是否有進本地資料庫",
            },
            {
                "分頁": "新手教學",
                "你會看到什麼": "名詞解釋、快取邏輯、操作順序、常見誤解",
                "什麼時候用": "剛上手或看不懂數字時",
            },
        ]
    )
    st.dataframe(page_df, width="stretch", hide_index=True)

    st.markdown("### 1) 第一次使用，建議照這個順序")
    st.markdown(
        "\n".join(
            [
                "1. 到 `回測工作台`，先跑一個 `buy_hold`（單檔、近 1~3 年）。",
                "2. 確認你看得懂 `總報酬/CAGR/MDD/Sharpe` 與成交明細。",
                "3. 再到 `2026 YTD 前十大 ETF`、`2026 YTD 前十大股利型、配息型 ETF`、`台股 ETF 全類型總表` 與 `2025 後20大最差勁 ETF` 看橫向比較，接著看 `共識代表 ETF` 收斂核心，再用 `兩檔 ETF 推薦` 產出可執行組合。",
                "4. 想看 ETF 內部成分股強弱，再進 `00910 / 00935 / 00735 / 0050 / 0052 熱力圖`。",
                "5. 最後才用 `ETF 輪動策略` 或 `2026 YTD 主動式 ETF` 做進階比較。",
            ]
        )
    )
    st.info("若你只想先確認流程有跑通，第一步固定用 `buy_hold` 最穩。")

    st.markdown("### 2) 更新按鈕與快取邏輯（最常問）")
    cache_df = pd.DataFrame(
        [
            {
                "項目": "更新最新市況（Top10/主動式）",
                "作用": "清除該頁快取並重抓最新快照",
                "不按會怎樣": "會先顯示上次可用結果",
            },
            {
                "項目": "重新計算（共識代表 ETF）",
                "作用": "重算前10交集與代表性分數",
                "不按會怎樣": "沿用最近一次共識計算結果",
            },
            {
                "項目": "重新計算（兩檔 ETF 推薦）",
                "作用": "重算核心/衛星推薦與重疊門檻",
                "不按會怎樣": "沿用最近一次兩檔推薦結果",
            },
            {
                "項目": "更新 成分股（熱力圖）",
                "作用": "重抓 ETF 成分股清單並更新快取",
                "不按會怎樣": "沿用 `universe_snapshots` 既有快取",
            },
            {
                "項目": "執行前同步最新日K（較慢）",
                "作用": "先補齊資料再跑",
                "不按會怎樣": "優先用本地資料庫，不足才補同步",
            },
            {
                "項目": "重新計算（Benchmark 對照卡）",
                "作用": "在目前參數下重算曲線與績效表",
                "不按會怎樣": "沿用本頁已算過結果",
            },
        ]
    )
    st.dataframe(cache_df, width="stretch", hide_index=True)

    st.markdown("### 3) 資料會存在哪裡？")
    storage_df = pd.DataFrame(
        [
            {
                "項目": "歷史日K（含 Benchmark）",
                "位置": "Parquet（預設 iCloud，可由環境變數覆蓋）",
                "用途": "回測與比較主資料來源",
            },
            {"項目": "回測摘要", "位置": "DuckDB `backtest_runs`", "用途": "保存回測重點結果"},
            {
                "項目": "回測回放快取",
                "位置": "DuckDB `backtest_replay_runs`",
                "用途": "同條件可直接載入完整回放結果",
            },
            {
                "項目": "熱力圖結果",
                "位置": "DuckDB `heatmap_runs`",
                "用途": "熱力圖頁可先顯示上次結果",
            },
            {
                "項目": "ETF 輪動結果",
                "位置": "DuckDB `rotation_runs`",
                "用途": "輪動頁可先顯示上次結果",
            },
            {
                "項目": "成分股清單",
                "位置": "DuckDB `universe_snapshots`",
                "用途": "避免每次重抓成分股",
            },
            {
                "項目": "即時資料",
                "位置": "Session 記憶體 + Parquet `intraday_ticks`",
                "用途": "即時看盤與補圖",
            },
        ]
    )
    st.dataframe(storage_df, width="stretch", hide_index=True)

    st.markdown("### 4) 回測流程（工作台核心）")
    flow_df = pd.DataFrame(
        [
            {"步驟": "A. 選市場與代碼", "說明": "可單檔或投組（逗號分隔）。"},
            {"步驟": "B. 選日期區間", "說明": "決定樣本資料。"},
            {"步驟": "C. 選策略", "說明": "新手先用 `buy_hold`。"},
            {"步驟": "D. 設成本", "說明": "手續費/稅/滑價都會改變結果。"},
            {"步驟": "E. 設實際投入起點", "說明": "這是會改績效的關鍵參數。"},
            {"步驟": "F. 自動回測", "說明": "條件輸入完成就會自動計算，並優先載入同條件快取。"},
            {"步驟": "G. 需要時再開 Walk-Forward", "說明": "用 Train/Test 檢查穩健性。"},
        ]
    )
    st.dataframe(flow_df, width="stretch", hide_index=True)

    st.markdown("### 5) 回測安全門檻（資料筆數）")
    threshold_df = pd.DataFrame(
        [
            {"模式": "buy_hold", "最少K數": "2", "原因": "至少要有起點與終點。"},
            {"模式": "一般策略（SMA/EMA/RSI/MACD）", "最少K數": "40", "原因": "指標需要基本樣本。"},
            {
                "模式": "日K趨勢策略（sma_trend_filter / donchian_breakout）",
                "最少K數": "120",
                "原因": "需要長視窗濾網。",
            },
            {
                "模式": "Walk-Forward",
                "最少K數": "至少 80（長視窗策略更高）",
                "原因": "Train/Test 都要有足夠樣本。",
            },
        ]
    )
    st.dataframe(threshold_df, width="stretch", hide_index=True)

    st.markdown("### 6) 指標怎麼讀（Top10/主動式/Benchmark）")
    metric_df = pd.DataFrame(
        [
            {"欄位": "YTD報酬(%)", "意思": "今年截至目前區間報酬"},
            {"欄位": "2025績效(%)", "意思": "2025 年區間報酬（各檔以區間內第一個可交易日起算）"},
            {"欄位": "贏輸台股大盤(%)", "意思": "ETF 報酬 - 大盤報酬；正值=贏大盤"},
            {"欄位": "Strategy Equity（等權）", "意思": "把多檔 ETF 等權持有後的資產曲線"},
            {"欄位": "Benchmark Equity", "意思": "同初始資產下，基準標的的資產曲線"},
        ]
    )
    st.dataframe(metric_df, width="stretch", hide_index=True)
    st.caption(
        "你若看到同一檔 ETF 在上方表格與下方 Benchmark 對照卡數值略有差異，通常是因為比較區間對齊方式不同（快照區間 vs 共同比較區間）。"
    )

    st.markdown("### 7) 熱力圖怎麼讀")
    heatmap_df = pd.DataFrame(
        [
            {"顏色": "綠色", "代表": "贏過基準（越綠代表超額越高）"},
            {"顏色": "紅色", "代表": "輸給基準（越紅代表落後越大）"},
            {"欄位": "strategy_return_pct", "代表": "該成分股策略報酬"},
            {"欄位": "benchmark_return_pct", "代表": "同區間基準報酬"},
            {"欄位": "excess_pct", "代表": "超額報酬 = 策略報酬 - 基準報酬"},
        ]
    )
    st.dataframe(heatmap_df, width="stretch", hide_index=True)

    st.markdown("### 8) 常見混淆（會不會改回測結果）")
    confusion_df = pd.DataFrame(
        [
            {"控制項": "起始日期 / 結束日期", "會不會改回測結果": "會", "重點": "改變樣本區間"},
            {
                "控制項": "實際投入起點（日期或第幾根K）",
                "會不會改回測結果": "會",
                "重點": "改變本金開始參與時間",
            },
            {
                "控制項": "回放位置（K棒/日期）",
                "會不會改回測結果": "不會",
                "重點": "只改圖表播放位置",
            },
            {"控制項": "回放視窗 / 生長位置", "會不會改回測結果": "不會", "重點": "只改視覺呈現"},
        ]
    )
    st.dataframe(confusion_df, width="stretch", hide_index=True)

    st.markdown("### 9) 訊號點 vs 實際成交點")
    point_df = pd.DataFrame(
        [
            {
                "類型": "訊號點（價格圖）",
                "代表意思": "策略判斷該進場/出場",
                "適合用途": "看策略邏輯翻多翻空",
            },
            {
                "類型": "實際成交點（資產圖）",
                "代表意思": "回測規則真正成交位置",
                "適合用途": "檢查績效計算合理性",
            },
        ]
    )
    st.dataframe(point_df, width="stretch", hide_index=True)
    st.caption("本系統成交規則：`T 日收盤出訊號，T+1 開盤成交`。")

    st.markdown("### 10) 參數白話解釋（常用）")
    params_df = pd.DataFrame(
        [
            {
                "參數": "Fast",
                "白話意思": "短天期均線",
                "單位/格式": "天數（整數）",
                "新手建議": "10~20",
            },
            {
                "參數": "Slow",
                "白話意思": "長天期均線",
                "單位/格式": "天數（整數）",
                "新手建議": "30~120 且大於 Fast",
            },
            {
                "參數": "Trend Filter",
                "白話意思": "長期趨勢濾網均線",
                "單位/格式": "天數（整數）",
                "新手建議": "120",
            },
            {
                "參數": "Breakout Lookback",
                "白話意思": "突破回朔天數",
                "單位/格式": "天數（整數）",
                "新手建議": "55",
            },
            {
                "參數": "Exit Lookback",
                "白話意思": "出場回朔天數",
                "單位/格式": "天數（整數）",
                "新手建議": "20 且小於 Breakout",
            },
            {
                "參數": "RSI Buy Below",
                "白話意思": "低於此值才考慮買入",
                "單位/格式": "0~100",
                "新手建議": "30",
            },
            {
                "參數": "RSI Sell Above",
                "白話意思": "高於此值才考慮賣出",
                "單位/格式": "0~100",
                "新手建議": "55~65",
            },
            {
                "參數": "Fee Rate",
                "白話意思": "手續費比例",
                "單位/格式": "小數比例",
                "新手建議": "台股常見 0.001425",
            },
            {
                "參數": "Sell Tax",
                "白話意思": "賣出交易稅比例",
                "單位/格式": "小數比例",
                "新手建議": "股票 0.003 / ETF 0.001",
            },
            {
                "參數": "Slippage",
                "白話意思": "理論價與成交價偏差",
                "單位/格式": "小數比例",
                "新手建議": "0.0005（0.05%）",
            },
            {
                "參數": "Train 比例",
                "白話意思": "Walk-Forward 訓練占比",
                "單位/格式": "0~1",
                "新手建議": "0.70",
            },
            {
                "參數": "參數挑選目標",
                "白話意思": "Train 區選最佳參數指標",
                "單位/格式": "sharpe/cagr/total_return/mdd",
                "新手建議": "sharpe",
            },
        ]
    )
    st.dataframe(params_df, width="stretch", hide_index=True)

    st.markdown("### 11) 技術指標速讀（RSI / MACD / 布林通道 / KD）")
    indicators_df = pd.DataFrame(
        [
            {
                "指標": "RSI",
                "白話解釋": "看短期買賣力道是否過熱或過冷",
                "常見看法": "低檔區偏弱、高檔區偏強",
                "新手提醒": "不要單看超買超賣，先看大方向趨勢",
            },
            {
                "指標": "MACD",
                "白話解釋": "看趨勢加速度與轉折",
                "常見看法": "快慢線黃金交叉偏多、死亡交叉偏空",
                "新手提醒": "盤整盤容易來回假訊號",
            },
            {
                "指標": "布林通道",
                "白話解釋": "看波動區間與價格相對位置",
                "常見看法": "貼上軌代表強勢、貼下軌代表弱勢",
                "新手提醒": "貼軌不等於立刻反轉，可能是趨勢延續",
            },
            {
                "指標": "KD",
                "白話解釋": "看短線動能轉折速度",
                "常見看法": "K 上穿 D 偏多、K 下穿 D 偏空",
                "新手提醒": "在強趨勢中容易過早反向判斷",
            },
        ]
    )
    st.dataframe(indicators_df, width="stretch", hide_index=True)
    st.caption(
        "建議做法：先用趨勢類指標（如 MACD）判方向，再用 RSI/KD 找節奏，最後用布林通道觀察波動風險。"
    )

    st.markdown("### 12) 新手建議操作（先簡單再進階）")
    st.markdown(
        "\n".join(
            [
                "1. 先在 `回測工作台` 用 `buy_hold` 跑單檔，確認資料與報表都正常。",
                "2. 再改成 `sma_trend_filter` 或 `donchian_breakout`，比較是否優於 `buy_hold`。",
                "3. 接著用 `2026 YTD 前十大 ETF` + `2026 YTD 前十大股利型、配息型 ETF` + `台股 ETF 全類型總表` + `2025 後20大最差勁 ETF` + `共識代表 ETF` + `兩檔 ETF 推薦` 做橫向排名、收斂與落地組合。",
                "4. 最後才進 `ETF 輪動策略`、`2026 YTD 主動式 ETF` 與各熱力圖做進階判讀。",
            ]
        )
    )


def _render_excess_heatmap_panel(
    rows_df: pd.DataFrame, *, title: str, colorbar_title: str = "超額報酬 %"
):
    if rows_df is None or rows_df.empty:
        st.info(f"{title}：目前無可用資料。")
        return

    frame = rows_df.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["name"] = frame["name"].astype(str)
    frame["excess_pct"] = pd.to_numeric(frame["excess_pct"], errors="coerce")
    frame["asset_return_pct"] = pd.to_numeric(frame["asset_return_pct"], errors="coerce")
    frame["benchmark_return_pct"] = pd.to_numeric(frame["benchmark_return_pct"], errors="coerce")
    frame["bars"] = pd.to_numeric(frame["bars"], errors="coerce").fillna(0).astype(int)
    frame = frame.dropna(subset=["excess_pct", "asset_return_pct", "benchmark_return_pct"])
    if frame.empty:
        st.info(f"{title}：目前無可用資料。")
        return
    frame = frame.sort_values("excess_pct", ascending=False).reset_index(drop=True)

    st.markdown(f"#### {title}")
    winners = int((frame["excess_pct"] > 0).sum())
    losers = int((frame["excess_pct"] < 0).sum())
    ties = int((frame["excess_pct"] == 0).sum())
    avg_excess = float(frame["excess_pct"].mean())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("勝過基準檔數", str(winners))
    m2.metric("輸給基準檔數", str(losers))
    m3.metric("持平檔數", str(ties))
    m4.metric("平均超額報酬", f"{avg_excess:+.2f}%")

    palette = _ui_palette()
    tiles_per_row = 6
    tile_rows = int(math.ceil(len(frame) / tiles_per_row))
    z = np.full((tile_rows, tiles_per_row), np.nan)
    txt = np.full((tile_rows, tiles_per_row), "", dtype=object)
    custom = np.empty((tile_rows, tiles_per_row, 7), dtype=object)
    custom[:, :, :] = None

    for i, row in frame.iterrows():
        r = i // tiles_per_row
        c = i % tiles_per_row
        z[r, c] = float(row["excess_pct"])
        label = str(row["symbol"]).strip()
        name_text = str(row.get("name", "")).strip()
        weight_text = _format_weight_pct_label(row.get("weight_pct"))
        if name_text and not _is_unresolved_symbol_name(label, name_text):
            txt[r, c] = (
                f"<b>{label}</b><br>{name_text}<br>權重 {weight_text}<br>{row['excess_pct']:+.2f}%"
            )
        else:
            txt[r, c] = f"<b>{label}</b><br>權重 {weight_text}<br>{row['excess_pct']:+.2f}%"
        custom[r, c, 0] = float(row["asset_return_pct"])
        custom[r, c, 1] = float(row["benchmark_return_pct"])
        custom[r, c, 2] = str(row.get("benchmark_symbol", ""))
        custom[r, c, 3] = int(row.get("bars", 0))
        custom[r, c, 4] = str(row.get("market_tag", ""))
        custom[r, c, 5] = weight_text
        custom[r, c, 6] = name_text

    max_abs = _heatmap_max_abs(z)
    fig_heat = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                text=txt,
                texttemplate="%{text}",
                textfont=dict(
                    size=12, color=HEATMAP_TEXT_COLOR, family="Noto Sans TC, Segoe UI, sans-serif"
                ),
                customdata=custom,
                zmin=-max_abs,
                zmax=max_abs,
                zmid=0.0,
                colorscale=HEATMAP_EXCESS_COLORSCALE,
                xgap=6,
                ygap=6,
                hoverongaps=False,
                colorbar=dict(title=colorbar_title, thickness=14, len=0.78),
                hovertemplate=(
                    "名稱：%{customdata[6]}<br>"
                    "標的報酬：%{customdata[0]:+.2f}%<br>"
                    "基準報酬：%{customdata[1]:+.2f}%<br>"
                    "超額：%{z:+.2f}%<br>"
                    "基準：%{customdata[2]}<br>"
                    "市場：%{customdata[4]}<br>"
                    "對齊Bars：%{customdata[3]}<br>"
                    "權重：%{customdata[5]}"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig_heat.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig_heat.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"
    )
    fig_heat.update_layout(
        height=max(250, 90 * tile_rows),
        margin=dict(l=10, r=10, t=10, b=10),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"]), family="Noto Sans TC, Segoe UI, sans-serif"),
        hoverlabel=_plot_hoverlabel_style(palette),
    )
    _enable_plotly_draw_tools(fig_heat)
    _apply_plotly_watermark(fig_heat, text=str(title), palette=palette)
    _render_plotly_chart(
        fig_heat,
        chart_key=f"plotly:heatmap:{title}",
        filename=f"heatmap_{title}",
        scale=2,
        width="stretch",
        watermark_text=str(title),
        palette=palette,
    )

    out_df = frame.copy()
    for col in ["weight_pct", "asset_return_pct", "benchmark_return_pct", "excess_pct"]:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce").round(2)
    rename_map = {
        "symbol": "symbol",
        "name": "name",
        "market_tag": "market",
        "benchmark_symbol": "benchmark",
        "weight_pct": "weight_pct",
        "asset_return_pct": "asset_return_pct",
        "benchmark_return_pct": "benchmark_return_pct",
        "excess_pct": "excess_pct",
        "bars": "bars",
        "start_actual": "start_actual",
        "end_actual": "end_actual",
    }
    keep_cols = [c for c in rename_map if c in out_df.columns]
    out_df = out_df[keep_cols].rename(
        columns={k: v for k, v in rename_map.items() if k in keep_cols}
    )
    st.dataframe(out_df, width="stretch", hide_index=True)


def _render_00910_global_ytd_block(
    *,
    store: HistoryStore,
    service: MarketDataService,
    page_key: str,
    etf_code: str,
    full_rows: list[dict[str, object]],
):
    etf_text = str(etf_code or "").strip().upper()
    st.markdown(f"### {etf_text} 全球成分股 YTD 熱力圖（Buy & Hold）")
    today = date.today()
    ytd_start_date = date(today.year, 1, 1)
    ytd_end_date = today
    start_dt = datetime.combine(ytd_start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(ytd_end_date, datetime.min.time()).replace(tzinfo=timezone.utc)

    c1, c2, c3, c4 = st.columns(4)
    tw_benchmark = c1.selectbox(
        "台股基準", options=["^TWII", "0050"], index=0, key=f"{page_key}_global_bench_tw"
    )
    us_benchmark = c2.selectbox(
        "美股基準", options=["QQQ", "^IXIC"], index=0, key=f"{page_key}_global_bench_us"
    )
    jp_benchmark = c3.selectbox(
        "日股基準", options=["^N225"], index=0, key=f"{page_key}_global_bench_jp"
    )
    ks_benchmark = c4.selectbox(
        "韓股基準", options=["^KS11"], index=0, key=f"{page_key}_global_bench_ks"
    )
    s1, s2 = st.columns(2)
    sync_before_run = s1.checkbox(
        "執行前同步最新日K（推薦）",
        value=True,
        key=f"{page_key}_global_sync_ytd",
    )
    parallel_sync = s2.checkbox(
        "平行同步",
        value=True,
        key=f"{page_key}_global_parallel_ytd",
    )
    st.caption(
        f"YTD 區間：{ytd_start_date.isoformat()} ~ {ytd_end_date.isoformat()}。"
        "台股成分對台股基準；海外成分依市場對應基準（US/JP/KS）。"
    )

    universe_id = f"TW:{etf_text}:GLOBAL_YTD"
    payload_key = f"{page_key}_global_ytd_payload"
    run_key = (
        f"{etf_text}_global_ytd:{ytd_start_date}:{ytd_end_date}:{tw_benchmark}:"
        f"{us_benchmark}:{jp_benchmark}:{ks_benchmark}"
    )
    cached_run = store.load_latest_heatmap_run(universe_id)
    if payload_key not in st.session_state and cached_run and isinstance(cached_run.payload, dict):
        initial_payload = dict(cached_run.payload)
        initial_payload.setdefault("generated_at", cached_run.created_at.isoformat())
        st.session_state[payload_key] = initial_payload

    def _dedupe_keep_order(values: list[str]) -> list[str]:
        out: list[str] = []
        for value in values:
            text = str(value or "").strip().upper()
            if text and text not in out:
                out.append(text)
        return out

    run_now = st.button(
        f"執行 {etf_text} YTD 全球熱力圖",
        type="primary",
        width="stretch",
        key=f"{page_key}_global_run_ytd",
    )
    if run_now:
        rows_full = list(full_rows)
        if not rows_full:
            rows_full, _ = service.get_etf_constituents_full(
                etf_text, limit=None, force_refresh=True
            )
        if not rows_full:
            st.error(f"目前抓不到 {etf_text} 完整成分股（含海外），請稍後重試。")
            return
        rows_full = _enrich_rows_with_tw_names(rows=rows_full, service=service)

        market_benchmark_map = {
            "US": us_benchmark,
            "JP": jp_benchmark,
            "KS": ks_benchmark,
        }
        universe_items: list[dict[str, object]] = []
        seen_keys: set[tuple[str, str]] = set()
        for row in rows_full:
            raw_symbol = str(row.get("symbol", "")).strip().upper()
            name = str(row.get("name", raw_symbol)).strip() or raw_symbol
            market_tag = str(row.get("market", "")).strip().upper()
            tw_code = str(row.get("tw_code", "")).strip().upper()
            weight_pct = _safe_float(row.get("weight_pct"))

            if tw_code and re.fullmatch(r"\d{4}", tw_code):
                asset_symbol = tw_code
                asset_market = "TW"
                group = "TW"
                benchmark_symbol = tw_benchmark
                benchmark_market = "TW"
                market_tag = "TW"
            else:
                if not raw_symbol:
                    continue
                if not market_tag:
                    market_tag = raw_symbol.split(".")[-1] if "." in raw_symbol else "US"
                asset_symbol = raw_symbol
                asset_market = "US"
                group = "OVERSEAS"
                benchmark_symbol = market_benchmark_map.get(market_tag, us_benchmark)
                benchmark_market = "US"

            dedupe_key = (asset_market, asset_symbol)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            universe_items.append(
                {
                    "symbol": asset_symbol,
                    "name": name,
                    "group": group,
                    "market_tag": market_tag,
                    "market": asset_market,
                    "benchmark_symbol": benchmark_symbol,
                    "benchmark_market": benchmark_market,
                    "weight_pct": weight_pct,
                }
            )

        if not universe_items:
            st.error(f"{etf_text} 成分股清單為空，無法執行。")
            return

        sync_issues: list[str] = []
        if sync_before_run:
            tw_symbols = _dedupe_keep_order(
                [
                    *[str(it["symbol"]) for it in universe_items if str(it["market"]) == "TW"],
                    *[
                        str(it["benchmark_symbol"])
                        for it in universe_items
                        if str(it["benchmark_market"]) == "TW"
                    ],
                ]
            )
            us_symbols = _dedupe_keep_order(
                [
                    *[str(it["symbol"]) for it in universe_items if str(it["market"]) == "US"],
                    *[
                        str(it["benchmark_symbol"])
                        for it in universe_items
                        if str(it["benchmark_market"]) == "US"
                    ],
                ]
            )
            with st.spinner(f"同步 {etf_text} YTD 所需資料中..."):
                if tw_symbols:
                    _, issues_tw = _sync_symbols_history(
                        store,
                        market="TW",
                        symbols=tw_symbols,
                        start=start_dt,
                        end=end_dt,
                        parallel=parallel_sync,
                    )
                    sync_issues.extend(issues_tw)
                if us_symbols:
                    _, issues_us = _sync_symbols_history(
                        store,
                        market="US",
                        symbols=us_symbols,
                        start=start_dt,
                        end=end_dt,
                        parallel=parallel_sync,
                    )
                    sync_issues.extend(issues_us)

        close_cache: dict[tuple[str, str], pd.Series] = {}

        def _load_close_series(symbol: str, market: str) -> pd.Series:
            key = (str(market).upper(), str(symbol).upper())
            if key in close_cache:
                return close_cache[key]
            bars = store.load_daily_bars(symbol=symbol, market=market, start=start_dt, end=end_dt)
            bars = normalize_ohlcv_frame(bars)
            if bars.empty and not sync_before_run:
                report = store.sync_symbol_history(
                    symbol=symbol, market=market, start=start_dt, end=end_dt
                )
                if report.error:
                    sync_issues.append(f"{symbol}: {report.error}")
                bars = store.load_daily_bars(
                    symbol=symbol, market=market, start=start_dt, end=end_dt
                )
                bars = normalize_ohlcv_frame(bars)
            if bars.empty:
                close_cache[key] = pd.Series(dtype=float)
                return close_cache[key]

            bars, _ = apply_split_adjustment(
                bars=bars,
                symbol=symbol,
                market=market,
                use_known=True,
                use_auto_detect=True,
            )
            close = pd.to_numeric(bars["close"], errors="coerce").dropna()
            close.index = pd.to_datetime(close.index, utc=True, errors="coerce")
            close = close.dropna()
            close = close[(close.index >= start_dt) & (close.index <= end_dt)]
            close_cache[key] = close
            return close

        result_rows: list[dict[str, object]] = []
        for item in universe_items:
            symbol = str(item["symbol"])
            market = str(item["market"])
            benchmark_symbol = str(item["benchmark_symbol"])
            benchmark_market = str(item["benchmark_market"])
            asset_close = _load_close_series(symbol=symbol, market=market)
            benchmark_close = _load_close_series(symbol=benchmark_symbol, market=benchmark_market)
            if asset_close.empty or benchmark_close.empty:
                continue
            comp = pd.concat(
                [asset_close.rename("asset"), benchmark_close.rename("benchmark")],
                axis=1,
                join="inner",
            ).dropna()
            if len(comp) < 2:
                continue

            asset_return_pct = float(comp["asset"].iloc[-1] / comp["asset"].iloc[0] - 1.0) * 100.0
            benchmark_return_pct = (
                float(comp["benchmark"].iloc[-1] / comp["benchmark"].iloc[0] - 1.0) * 100.0
            )
            excess_pct = asset_return_pct - benchmark_return_pct
            result_rows.append(
                {
                    "symbol": symbol,
                    "name": str(item["name"]),
                    "group": str(item["group"]),
                    "market_tag": str(item["market_tag"]),
                    "benchmark_symbol": benchmark_symbol,
                    "weight_pct": item.get("weight_pct"),
                    "asset_return_pct": asset_return_pct,
                    "benchmark_return_pct": benchmark_return_pct,
                    "excess_pct": excess_pct,
                    "bars": int(len(comp)),
                    "start_actual": pd.Timestamp(comp.index[0]).date().isoformat(),
                    "end_actual": pd.Timestamp(comp.index[-1]).date().isoformat(),
                }
            )

        payload = {
            "run_key": run_key,
            "rows": result_rows,
            "start_date": ytd_start_date.isoformat(),
            "end_date": ytd_end_date.isoformat(),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "sync_issues": sync_issues,
            "benchmarks": {
                "TW": tw_benchmark,
                "US": us_benchmark,
                "JP": jp_benchmark,
                "KS": ks_benchmark,
            },
        }
        st.session_state[payload_key] = payload
        store.save_heatmap_run(universe_id=universe_id, payload=payload)

    payload = st.session_state.get(payload_key)
    if not payload:
        st.info(f"按下「執行 {etf_text} YTD 全球熱力圖」後，會顯示國內/海外分組結果。")
        return
    if payload.get("run_key") != run_key:
        st.caption("目前顯示的是上一次執行結果；若要套用目前基準設定，請重新執行。")

    sync_issues = payload.get("sync_issues")
    if isinstance(sync_issues, list) and sync_issues:
        preview = [" ".join(str(item).split()) for item in sync_issues[:3]]
        preview_text = " | ".join(
            [item if len(item) <= 120 else f"{item[:117]}..." for item in preview]
        )
        remain = len(sync_issues) - len(preview)
        remain_text = f" | 其餘 {remain} 筆請查看終端 log。" if remain > 0 else ""
        st.warning(f"部分標的同步失敗，已盡量使用可用資料：{preview_text}{remain_text}")

    generated_at_text = str(payload.get("generated_at", "")).strip()
    if generated_at_text:
        try:
            generated_at = datetime.fromisoformat(generated_at_text)
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            generated_at_text = generated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    st.caption(
        f"上次執行：{generated_at_text or 'N/A'} | "
        f"YTD：{payload.get('start_date')} ~ {payload.get('end_date')} | "
        "策略：buy_hold"
    )

    rows_df = pd.DataFrame(payload.get("rows", []))
    if rows_df.empty:
        st.warning("沒有可用結果（可能部分海外標的資料尚未覆蓋 YTD）。")
        return
    rows_df = _fill_unresolved_tw_names(
        frame=rows_df,
        service=service,
        full_rows=full_rows,
        symbol_col="symbol",
        name_col="name",
    )

    tw_df = rows_df[rows_df["group"] == "TW"].copy()
    overseas_df = rows_df[rows_df["group"] == "OVERSEAS"].copy()
    _render_excess_heatmap_panel(tw_df, title="國內成分股（TW vs 台股基準）")
    _render_excess_heatmap_panel(overseas_df, title="海外成分股（Overseas vs 對應海外基準）")


def _render_tw_etf_heatmap_view(
    etf_code: str, page_desc: str, *, auto_run_if_missing: bool = False
):
    store = _history_store()
    service = _market_service()
    perf_timer = PerfTimer(enabled=perf_debug_enabled())
    etf_text = str(etf_code).strip().upper()
    page_key = f"tw{etf_text.lower()}"

    st.subheader(f"{etf_text} 成分股熱力圖回測（相對大盤）")
    st.caption(
        f"以 {etf_text}{page_desc}成分股逐檔回測，與大盤同區間比較；綠色代表贏過、紅色代表輸給。"
    )
    _render_etf_index_method_summary(etf_text)

    c1, c2, c3, c4 = st.columns(4)
    start_date = c1.date_input(
        "起始日期", value=date(date.today().year - 5, 1, 1), key=f"{page_key}_start"
    )
    end_date = c2.date_input("結束日期", value=date.today(), key=f"{page_key}_end")
    benchmark_choice = c3.selectbox(
        "Benchmark",
        options=["twii", "0050", "006208"],
        format_func=lambda x: {"twii": "^TWII", "0050": "0050", "006208": "006208"}.get(x, x),
        index=0,
        key=f"{page_key}_benchmark",
    )
    heatmap_strategy_options = ["buy_hold", "sma_trend_filter", "donchian_breakout", "sma_cross"]
    strategy = c4.selectbox(
        "回測策略",
        options=heatmap_strategy_options,
        index=0,
        key=f"{page_key}_strategy",
        format_func=_strategy_label,
    )

    strategy_params: dict[str, float] = {}
    if strategy == "sma_cross":
        p1, p2 = st.columns(2)
        fast = p1.slider("Fast", min_value=3, max_value=60, value=10, key=f"{page_key}_fast")
        slow = p2.slider("Slow", min_value=10, max_value=180, value=30, key=f"{page_key}_slow")
        strategy_params = {"fast": float(fast), "slow": float(slow)}
    elif strategy == "sma_trend_filter":
        p1, p2, p3 = st.columns(3)
        fast = p1.slider("Fast", min_value=5, max_value=80, value=20, key=f"{page_key}_fast")
        slow = p2.slider("Slow", min_value=20, max_value=220, value=60, key=f"{page_key}_slow")
        trend = p3.slider(
            "Trend Filter", min_value=60, max_value=300, value=120, key=f"{page_key}_trend"
        )
        strategy_params = {"fast": float(fast), "slow": float(slow), "trend": float(trend)}
    elif strategy == "donchian_breakout":
        p1, p2, p3 = st.columns(3)
        entry_n = p1.slider(
            "Breakout Lookback", min_value=20, max_value=120, value=55, key=f"{page_key}_entry"
        )
        exit_max = max(10, int(entry_n) - 1)
        exit_n = p2.slider(
            "Exit Lookback",
            min_value=5,
            max_value=exit_max,
            value=min(20, exit_max),
            key=f"{page_key}_exit",
        )
        trend = p3.slider(
            "Trend Filter", min_value=60, max_value=300, value=120, key=f"{page_key}_trend"
        )
        strategy_params = {
            "entry_n": float(entry_n),
            "exit_n": float(exit_n),
            "trend": float(trend),
        }

    with st.expander("成本參數（台股預設）", expanded=False):
        k1, k2, k3 = st.columns(3)
        fee_rate = k1.number_input(
            "Fee Rate",
            min_value=0.0,
            max_value=0.02,
            value=0.001425,
            step=0.0001,
            format="%.6f",
            key=f"{page_key}_fee",
        )
        sell_tax = k2.number_input(
            "Sell Tax",
            min_value=0.0,
            max_value=0.02,
            value=0.0030,
            step=0.0001,
            format="%.6f",
            key=f"{page_key}_tax",
        )
        slippage = k3.number_input(
            "Slippage",
            min_value=0.0,
            max_value=0.02,
            value=0.0005,
            step=0.0001,
            format="%.6f",
            key=f"{page_key}_slip",
        )

    universe_id = f"TW:{etf_text}"
    payload_key = f"{page_key}_heatmap_payload"
    cached_run = store.load_latest_heatmap_run(universe_id)
    if payload_key not in st.session_state and cached_run and isinstance(cached_run.payload, dict):
        initial_payload = dict(cached_run.payload)
        initial_payload.setdefault("generated_at", cached_run.created_at.isoformat())
        st.session_state[payload_key] = initial_payload

    full_rows_00910: list[dict[str, object]] = []
    full_rows_for_name_lookup: list[dict[str, object]] = []
    has_overseas_constituents = False
    snapshot = store.load_universe_snapshot(universe_id)
    u1, u2 = st.columns([1, 4])
    refresh_constituents = u1.button(f"更新 {etf_text} 成分股", width="stretch")
    if refresh_constituents:
        with st.spinner(f"抓取 {etf_text} 成分股中..."):
            symbols_new, source_new = service.get_tw_etf_constituents(etf_text, limit=None)
            if symbols_new:
                store.save_universe_snapshot(
                    universe_id=universe_id, symbols=symbols_new, source=source_new
                )
                snapshot = store.load_universe_snapshot(universe_id)
    if snapshot and snapshot.symbols:
        expected_count = service.get_tw_etf_expected_count(etf_text)
        if expected_count is None:
            count_text = f"{len(snapshot.symbols)}"
            completeness = "無預設檔數"
        else:
            count_text = f"{len(snapshot.symbols)} / {expected_count}"
            completeness = "完整" if len(snapshot.symbols) >= expected_count else "可能不完整"
        u2.caption(
            f"已載入成分股：{count_text}（{completeness}） | 來源：{snapshot.source} | "
            f"快取時間：{snapshot.fetched_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        try:
            rows_full, full_source = service.get_etf_constituents_full(
                etf_text,
                limit=None,
                force_refresh=bool(refresh_constituents),
            )
        except Exception:
            rows_full = []
            full_source = "unavailable"
        full_rows_for_name_lookup = _enrich_rows_with_tw_names(
            rows=list(rows_full), service=service
        )
        if etf_text == "00910":
            full_rows_00910 = list(full_rows_for_name_lookup)
            if not full_rows_00910:
                st.caption(
                    "00910 完整成分股（含海外）目前抓取失敗，請稍後按「更新 00910 成分股」重試。"
                )
        has_overseas_constituents = _render_full_constituents_if_has_overseas(
            etf_code=etf_text,
            full_rows=full_rows_for_name_lookup,
            source=full_source,
        )
        has_overseas_constituents = bool(
            has_overseas_constituents or etf_text in ETF_FORCE_GLOBAL_HEATMAP
        )

        snapshot_expander_title = (
            f"查看台股可回測成分股（{len(snapshot.symbols)}）"
            if has_overseas_constituents
            else f"查看全部成分股（{len(snapshot.symbols)}）"
        )
        with st.expander(snapshot_expander_title, expanded=False):
            name_map_all = _resolve_tw_symbol_names(
                service=service,
                symbols=snapshot.symbols,
                full_rows=full_rows_for_name_lookup,
            )
            const_rows = [
                {"symbol": sym, "name": name_map_all.get(sym, sym)} for sym in snapshot.symbols
            ]
            st.dataframe(pd.DataFrame(const_rows), width="stretch", hide_index=True)
    else:
        u2.caption(
            f"尚未載入 {etf_text} 成分股快取。你仍可先查看上次回測結果，或按「更新 {etf_text} 成分股」後再重新回測。"
        )

    if has_overseas_constituents:
        _render_00910_global_ytd_block(
            store=store,
            service=service,
            page_key=page_key,
            etf_code=etf_text,
            full_rows=full_rows_for_name_lookup,
        )
        st.markdown("---")
        st.markdown("#### 台股子集合進階熱力圖（自訂區間/策略）")
        st.caption(f"下方為 {etf_text} 台股子集合回測，僅比較可回測的台股成分。")

    symbol_options = list(snapshot.symbols) if snapshot and snapshot.symbols else []
    symbol_key = f"{page_key}_symbol_pick"
    current_pick = st.session_state.get(symbol_key, symbol_options)
    if not isinstance(current_pick, list):
        current_pick = symbol_options
    current_pick = [s for s in current_pick if s in symbol_options]
    if not current_pick:
        current_pick = symbol_options
    st.session_state[symbol_key] = current_pick
    symbol_name_map_for_pick = _resolve_tw_symbol_names(
        service=service,
        symbols=symbol_options,
        full_rows=full_rows_for_name_lookup,
    )
    symbol_weight_map_for_pick: dict[str, float] = {}
    for row in full_rows_for_name_lookup:
        if not isinstance(row, dict):
            continue
        symbol_token = _normalize_constituent_symbol(row.get("symbol"), row.get("tw_code"))
        if not symbol_token:
            continue
        weight_pct = _safe_weight_pct(row.get("weight_pct"))
        if weight_pct is None:
            continue
        old_value = symbol_weight_map_for_pick.get(symbol_token)
        if old_value is None or float(weight_pct) > float(old_value):
            symbol_weight_map_for_pick[symbol_token] = float(weight_pct)

    def _format_symbol_option(sym: str) -> str:
        code = str(sym)
        name = str(symbol_name_map_for_pick.get(code, code)).strip()
        if name and name != code:
            return f"{code} {name}"
        return code

    selected_symbols = st.multiselect(
        "納入比較標的（預設全選）",
        options=symbol_options,
        key=symbol_key,
        format_func=_format_symbol_option,
    )

    if not selected_symbols:
        st.info("目前未選取標的。你仍可先看上次結果，或選取後按下執行。")

    date_is_valid = end_date >= start_date
    if not date_is_valid:
        st.warning("結束日期不可早於起始日期。")

    start_dt = (
        datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        if date_is_valid
        else None
    )
    end_dt = (
        datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        if date_is_valid
        else None
    )
    s1, s2 = st.columns(2)
    sync_before_run = s1.checkbox(
        "執行前同步最新日K（較慢）",
        value=False,
        key=f"{page_key}_sync_before_run",
        help="預設關閉：優先使用本地資料庫；資料不足時才補同步。",
    )
    parallel_sync = s2.checkbox(
        "平行同步多標的",
        value=True,
        key=f"{page_key}_parallel_sync",
        help="標的較多時通常更快；若網路不穩可關閉改逐檔同步。",
    )

    strategy_token = _stable_json_dumps(
        strategy_params if isinstance(strategy_params, dict) else {}
    )
    run_key = (
        f"{page_key}_heatmap:{start_date}:{end_date}:{benchmark_choice}:{strategy}:{strategy_token}:"
        f"{fee_rate}:{sell_tax}:{slippage}:{','.join(selected_symbols)}"
    )
    run_clicked = st.button(f"執行 {etf_text} 熱力圖回測", type="primary", width="stretch")
    payload_before_run = st.session_state.get(payload_key)

    def _should_auto_refresh_cached_heatmap(payload_obj: object) -> bool:
        if not isinstance(payload_obj, dict) or not payload_obj:
            return False
        if start_dt is None or end_dt is None:
            return False

        generated_ts = pd.to_datetime(
            str(payload_obj.get("generated_at", "")).strip(), utc=True, errors="coerce"
        )
        if snapshot is not None:
            snapshot_ts = pd.Timestamp(snapshot.fetched_at)
            if snapshot_ts.tzinfo is None:
                snapshot_ts = snapshot_ts.tz_localize("UTC")
            else:
                snapshot_ts = snapshot_ts.tz_convert("UTC")
            if not pd.isna(generated_ts) and snapshot_ts > generated_ts:
                return True

        payload_end_ts = pd.to_datetime(
            str(payload_obj.get("end_date", "")).strip(), utc=True, errors="coerce"
        )
        if pd.isna(payload_end_ts):
            return False

        selected_end_ts = pd.Timestamp(end_dt)
        if selected_end_ts.tzinfo is None:
            selected_end_ts = selected_end_ts.tz_localize("UTC")
        else:
            selected_end_ts = selected_end_ts.tz_convert("UTC")

        benchmark_symbol_for_check = (
            str(
                payload_obj.get("benchmark_symbol")
                or {"twii": "^TWII", "0050": "0050", "006208": "006208"}.get(
                    benchmark_choice, "^TWII"
                )
            )
            .strip()
            .upper()
        )
        if not benchmark_symbol_for_check:
            return False

        benchmark_bars_for_check = normalize_ohlcv_frame(
            store.load_daily_bars(
                symbol=benchmark_symbol_for_check,
                market="TW",
                start=start_dt,
                end=end_dt,
            )
        )
        if benchmark_bars_for_check.empty:
            return False

        latest_local_ts = pd.Timestamp(benchmark_bars_for_check.index.max())
        if latest_local_ts.tzinfo is None:
            latest_local_ts = latest_local_ts.tz_localize("UTC")
        else:
            latest_local_ts = latest_local_ts.tz_convert("UTC")

        return (
            latest_local_ts.normalize() <= selected_end_ts.normalize()
            and latest_local_ts.normalize() > payload_end_ts.normalize()
        )

    auto_trigger_reason = ""
    if auto_run_if_missing and not run_clicked and date_is_valid:
        if not isinstance(payload_before_run, dict) or not payload_before_run:
            auto_trigger_reason = "missing_payload"
        elif _should_auto_refresh_cached_heatmap(payload_before_run):
            auto_trigger_reason = "market_updated"

    if run_clicked or auto_trigger_reason:
        if not date_is_valid:
            st.error("日期區間無效，請先修正起訖日期。")
            return

        run_symbols = list(selected_symbols)
        symbol_options_before_refresh = list(symbol_options)
        selected_all_before = bool(symbol_options_before_refresh) and set(selected_symbols) == set(
            symbol_options_before_refresh
        )
        should_refresh_snapshot = bool(run_clicked or auto_trigger_reason == "missing_payload")

        if should_refresh_snapshot:
            with st.spinner("同步最新成分股中..."):
                symbols_latest, source_latest = service.get_tw_etf_constituents(
                    etf_text, limit=None
                )
                if symbols_latest:
                    store.save_universe_snapshot(
                        universe_id=universe_id, symbols=symbols_latest, source=source_latest
                    )
                    snapshot = store.load_universe_snapshot(universe_id)
                    symbol_options = list(snapshot.symbols) if snapshot and snapshot.symbols else []
                    if selected_all_before or not selected_symbols:
                        run_symbols = list(symbols_latest)
                    else:
                        run_symbols = [s for s in selected_symbols if s in symbols_latest]
                        if not run_symbols:
                            run_symbols = list(symbols_latest)

        if not symbol_options:
            if auto_trigger_reason:
                st.warning(f"尚未載入 {etf_text} 成分股，請先按「更新 {etf_text} 成分股」。")
            else:
                st.error(f"尚未載入 {etf_text} 成分股，請先按「更新 {etf_text} 成分股」。")
            return
        if not run_symbols:
            if auto_trigger_reason:
                run_symbols = list(symbol_options)
            else:
                st.error("請至少選擇 1 檔成分股。")
                return

        if auto_trigger_reason == "missing_payload":
            st.caption(f"首次開啟 {etf_text} 熱力圖：已自動更新成分股並建立快取。")
            st.caption(
                "自動建圖僅使用本地資料，不主動打外部同步；若要補齊缺資料，請手動按「執行熱力圖回測」。"
            )
        elif auto_trigger_reason == "market_updated":
            st.caption(f"{etf_text} 市場資料較快取更新，已自動重算熱力圖。")
            st.caption(
                "自動重算僅使用本地資料，不主動打外部同步；若要補齊缺資料，請手動按「執行熱力圖回測」。"
            )

        run_key = (
            f"{page_key}_heatmap:{start_date}:{end_date}:{benchmark_choice}:{strategy}:{strategy_token}:"
            f"{fee_rate}:{sell_tax}:{slippage}:{','.join(run_symbols)}"
        )
        min_required = get_strategy_min_bars(strategy)
        with st.spinner("整理成分股資料中..."):
            prepared = prepare_heatmap_bars(
                store=store,
                symbols=run_symbols,
                start_dt=start_dt,
                end_dt=end_dt,
                min_required=min_required,
                sync_before_run=bool(sync_before_run),
                parallel_sync=bool(parallel_sync),
                lazy_sync_on_insufficient=not bool(auto_trigger_reason),
                normalize_ohlcv_frame=normalize_ohlcv_frame,
            )
        bars_cache = dict(prepared.bars_cache)
        symbol_sync_issues = list(prepared.sync_issues)
        perf_timer.mark("heatmap_bars_prepared")

        benchmark_close, benchmark_symbol, benchmark_sync_issues = _load_tw_benchmark_close(
            store=store,
            choice=benchmark_choice,
            start_dt=start_dt,
            end_dt=end_dt,
            sync_first=sync_before_run,
            allow_twii_fallback=False,
        )
        _render_sync_issues(
            "Benchmark 同步有部分錯誤，已盡量使用本地可用資料", benchmark_sync_issues
        )
        if benchmark_close.empty:
            st.error("Benchmark 取得失敗，請改選其他基準（0050 或 006208）後重試。")
            return
        perf_timer.mark("heatmap_benchmark_loaded")

        progress = st.progress(0.0)
        cost_model = CostModel(
            fee_rate=float(fee_rate), sell_tax_rate=float(sell_tax), slippage_rate=float(slippage)
        )
        name_map = _resolve_tw_symbol_names(
            service=service,
            symbols=run_symbols,
            full_rows=full_rows_for_name_lookup,
        )
        rows = compute_heatmap_rows(
            run_symbols=run_symbols,
            bars_cache=bars_cache,
            benchmark_close=benchmark_close,
            strategy=strategy,
            strategy_params=strategy_params if isinstance(strategy_params, dict) else {},
            cost_model=cost_model,
            name_map=name_map,
            min_required=min_required,
            progress_callback=lambda ratio: progress.progress(float(ratio)),
            max_workers=6 if bool(parallel_sync) else 1,
        )
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol_token = str(item.get("symbol", "")).strip().upper()
            item["weight_pct"] = symbol_weight_map_for_pick.get(symbol_token)

        progress.empty()
        perf_timer.mark("heatmap_rows_computed")
        _render_sync_issues("部分成分股同步失敗，已盡量使用本地可用資料", symbol_sync_issues)
        payload = {
            "run_key": run_key,
            "rows": rows,
            "benchmark_symbol": benchmark_symbol,
            "selected_count": len(run_symbols),
            "universe_count": len(snapshot.symbols) if snapshot else len(run_symbols),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "strategy": strategy,
            "strategy_label": _strategy_label(strategy),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        st.session_state[payload_key] = payload
        store.save_heatmap_run(universe_id=universe_id, payload=payload)

    payload = st.session_state.get(payload_key)
    if not payload:
        st.info(f"設定好條件後，按下「執行 {etf_text} 熱力圖回測」。")
        return
    if payload.get("run_key") != run_key:
        st.caption("目前顯示的是上一次執行結果；若要套用目前設定，請重新按下執行。")
    heatmap_health = _build_data_health(
        as_of=payload.get("generated_at", ""),
        data_sources=[_store_data_source(store, "heatmap_runs")],
        source_chain=[str(payload.get("benchmark_symbol", "") or "benchmark")],
        degraded=payload.get("run_key") != run_key,
        fallback_depth=1 if payload.get("run_key") != run_key else 0,
        notes=f"selected={payload.get('selected_count', 0)}/{payload.get('universe_count', 0)}",
    )
    _render_data_health_caption("熱力圖資料健康度", heatmap_health)

    rows_df = pd.DataFrame(payload.get("rows", []))
    if rows_df.empty:
        st.warning("沒有可用回測結果（可能資料不足或期間太短）。")
        return
    rows_df = _fill_unresolved_tw_names(
        frame=rows_df,
        service=service,
        full_rows=full_rows_for_name_lookup,
        symbol_col="symbol",
        name_col="name",
    )
    rows_df["name"] = rows_df["name"].astype(str).str.strip()
    if "weight_pct" not in rows_df.columns:
        rows_df["weight_pct"] = np.nan
    rows_df["weight_pct"] = rows_df["weight_pct"].where(
        rows_df["weight_pct"].notna(), rows_df["symbol"].map(symbol_weight_map_for_pick)
    )

    rows_df = rows_df.sort_values("excess_pct", ascending=False).reset_index(drop=True)
    winners = int((rows_df["excess_pct"] > 0).sum())
    losers = int((rows_df["excess_pct"] < 0).sum())
    ties = int((rows_df["excess_pct"] == 0).sum())
    avg_excess = float(rows_df["excess_pct"].mean())
    max_win = float(rows_df["excess_pct"].max())
    max_loss = float(rows_df["excess_pct"].min())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("勝過大盤檔數", f"{winners}")
    m2.metric("輸給大盤檔數", f"{losers}")
    m3.metric("持平檔數", f"{ties}")
    m4.metric("平均超額報酬", f"{avg_excess:+.2f}%")
    m5.metric("最佳/最差", f"{max_win:+.2f}% / {max_loss:+.2f}%")
    with st.expander("結果怎麼看（越大/越小）", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- `勝過大盤檔數`：越大越好（代表更多成分股跑贏基準）。",
                    "- `輸給大盤檔數`：越小越好（越大代表弱勢標的較多）。",
                    "- `持平檔數`：中性，通常代表差異接近 0%。",
                    "- `平均超額報酬`：越大越好；`> 0` 代表整體贏過基準。",
                    "- `最佳/最差`：最佳越大越好；最差希望不要太負值（絕對值越小越穩）。",
                    "- `Bars`：策略與 Benchmark 對齊後可比較的 K 棒數，不是交易次數；通常越大結果越有參考性。",
                ]
            )
        )
    generated_at_text = str(payload.get("generated_at", "")).strip()
    if generated_at_text:
        try:
            generated_at = datetime.fromisoformat(generated_at_text)
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            generated_at_text = generated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    st.caption(
        f"上次執行：{generated_at_text or 'N/A'} | "
        f"Benchmark: {payload.get('benchmark_symbol', 'N/A')} | "
        f"區間：{payload.get('start_date')} ~ {payload.get('end_date')} | "
        f"策略：{payload.get('strategy_label', payload.get('strategy'))} | "
        f"回測標的數：{payload.get('selected_count')} | 全成分股數：{payload.get('universe_count', 'N/A')}"
    )

    palette = _ui_palette()
    tiles_per_row = 8
    tile_rows = int(math.ceil(len(rows_df) / tiles_per_row))
    z = np.full((tile_rows, tiles_per_row), np.nan)
    txt = np.full((tile_rows, tiles_per_row), "", dtype=object)
    custom = np.empty((tile_rows, tiles_per_row, 5), dtype=object)
    custom[:, :, :] = None

    for i, row in rows_df.iterrows():
        r = i // tiles_per_row
        c = i % tiles_per_row
        z[r, c] = float(row["excess_pct"])
        label = str(row["symbol"]).strip()
        name_text = str(row.get("name", "")).strip()
        weight_text = _format_weight_pct_label(row.get("weight_pct"))
        if name_text and not _is_unresolved_symbol_name(label, name_text):
            txt[r, c] = (
                f"<b>{label}</b><br>{name_text}<br>權重 {weight_text}<br>{row['excess_pct']:+.2f}%"
            )
        else:
            txt[r, c] = f"<b>{label}</b><br>權重 {weight_text}<br>{row['excess_pct']:+.2f}%"
        custom[r, c, 0] = float(row["strategy_return_pct"])
        custom[r, c, 1] = float(row["benchmark_return_pct"])
        custom[r, c, 2] = str(row["status"])
        custom[r, c, 3] = str(row["name"])
        custom[r, c, 4] = weight_text

    max_abs = _heatmap_max_abs(z)
    fig_heat = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                text=txt,
                texttemplate="%{text}",
                textfont=dict(
                    size=12, color=HEATMAP_TEXT_COLOR, family="Noto Sans TC, Segoe UI, sans-serif"
                ),
                customdata=custom,
                zmin=-max_abs,
                zmax=max_abs,
                zmid=0.0,
                colorscale=HEATMAP_EXCESS_COLORSCALE,
                xgap=6,
                ygap=6,
                hoverongaps=False,
                colorbar=dict(title="相對大盤 %", thickness=14, len=0.8),
                hovertemplate=(
                    "公司：%{customdata[3]}<br>"
                    "策略報酬：%{customdata[0]:+.2f}%<br>"
                    "大盤報酬：%{customdata[1]:+.2f}%<br>"
                    "超額：%{z:+.2f}%<br>"
                    "狀態：%{customdata[2]}<br>"
                    "權重：%{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig_heat.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig_heat.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"
    )
    fig_heat.update_layout(
        height=max(280, 90 * tile_rows),
        margin=dict(l=10, r=10, t=20, b=10),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"]), family="Noto Sans TC, Segoe UI, sans-serif"),
        hoverlabel=_plot_hoverlabel_style(palette),
    )
    _enable_plotly_draw_tools(fig_heat)
    _apply_plotly_watermark(fig_heat, text=f"{etf_text} 熱力圖", palette=palette)
    _render_plotly_chart(
        fig_heat,
        chart_key=f"plotly:heatmap:{etf_text}",
        filename=f"heatmap_{etf_text}",
        scale=2,
        width="stretch",
        watermark_text=f"{etf_text} 熱力圖",
        palette=palette,
    )

    out_df = rows_df.copy()
    if "weight_pct" in out_df.columns:
        out_df["weight_pct"] = out_df["weight_pct"].map(_format_weight_pct_label)
    out_df["strategy_return_pct"] = out_df["strategy_return_pct"].map(lambda v: round(float(v), 2))
    out_df["benchmark_return_pct"] = out_df["benchmark_return_pct"].map(
        lambda v: round(float(v), 2)
    )
    out_df["excess_pct"] = out_df["excess_pct"].map(lambda v: round(float(v), 2))
    output_cols = [
        col
        for col in [
            "symbol",
            "name",
            "weight_pct",
            "strategy_return_pct",
            "benchmark_return_pct",
            "excess_pct",
            "status",
            "bars",
        ]
        if col in out_df.columns
    ]
    st.dataframe(
        out_df[output_cols],
        width="stretch",
        hide_index=True,
    )

    if snapshot and snapshot.symbols:
        if etf_text == "00735" and has_overseas_constituents:
            _render_00735_heatmap_intro_tabs(
                snapshot_symbols=list(snapshot.symbols),
                service=service,
                full_rows=full_rows_for_name_lookup,
            )
        else:
            _render_heatmap_constituent_intro_sections(
                etf_code=etf_text,
                snapshot_symbols=list(snapshot.symbols),
                service=service,
                full_rows_00910=full_rows_for_name_lookup,
            )

    perf_timer.mark("heatmap_render_complete")
    if perf_timer.enabled:
        st.caption(perf_timer.summary_text(prefix=f"perf/heatmap/{etf_text}"))


def _render_tw_etf_rotation_view():
    store = _history_store()
    service = _market_service()
    palette = _ui_palette()
    perf_timer = PerfTimer(enabled=perf_debug_enabled())

    universe_id = "TW:ROTATION:CORE6"
    payload_key = "tw_rotation_payload"
    cached_run = store.load_latest_rotation_run(universe_id)
    if payload_key not in st.session_state and cached_run and isinstance(cached_run.payload, dict):
        initial_payload = dict(cached_run.payload)
        initial_payload.setdefault("generated_at", cached_run.created_at.isoformat())
        initial_payload.setdefault("run_key", cached_run.run_key)
        st.session_state[payload_key] = initial_payload

    st.subheader("台股多 ETF 輪動策略（日K）")
    st.caption(
        "固定標的池：0050、0052、00935、0056、00878、00919。每月第一個交易日評分，"
        "並於下一交易日開盤調整持股；大盤不在 60MA 上方時全數空手。"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    start_date = c1.date_input(
        "起始日期", value=date(date.today().year - 5, 1, 1), key="rotation_start"
    )
    end_date = c2.date_input("結束日期", value=date.today(), key="rotation_end")
    top_n = c3.slider(
        "每月持有檔數 Top N",
        min_value=1,
        max_value=len(ROTATION_DEFAULT_UNIVERSE),
        value=3,
        key="rotation_topn",
    )
    benchmark_choice = c4.selectbox(
        "Benchmark",
        options=["twii", "0050", "006208"],
        index=0,
        format_func=lambda x: {
            "twii": "^TWII（Auto fallback）",
            "0050": "0050",
            "006208": "006208",
        }.get(x, x),
        key="rotation_benchmark",
    )
    initial_capital = c5.number_input(
        "初始資產",
        min_value=10_000.0,
        max_value=1_000_000_000.0,
        value=1_000_000.0,
        step=50_000.0,
        key="rotation_initial_capital",
    )

    with st.expander("成本參數（台股ETF預設）", expanded=False):
        k1, k2, k3 = st.columns(3)
        fee_rate = k1.number_input(
            "Fee Rate",
            min_value=0.0,
            max_value=0.02,
            value=0.001425,
            step=0.0001,
            format="%.6f",
            key="rotation_fee",
            help="手續費比例，0.001425 = 0.1425%。",
        )
        sell_tax = k2.number_input(
            "Sell Tax",
            min_value=0.0,
            max_value=0.02,
            value=0.0010,
            step=0.0001,
            format="%.6f",
            key="rotation_tax",
            help="ETF 常見賣出交易稅可用 0.001（0.1%）。",
        )
        slippage = k3.number_input(
            "Slippage",
            min_value=0.0,
            max_value=0.02,
            value=0.0005,
            step=0.0001,
            format="%.6f",
            key="rotation_slippage",
            help="滑價比例，0.0005 = 0.05%。",
        )

    st.markdown(
        "\n".join(
            [
                "- 市場濾網：`Benchmark close > SMA60` 才允許持有。",
                "- 動能分數：`0.2*20日 + 0.5*60日 + 0.3*120日`。",
                "- 個股濾網：`score > 0` 且 `close > SMA60` 才納入排名。",
                "- 調倉：每月一次，訊號後的下一交易日開盤成交。",
            ]
        )
    )

    date_is_valid = end_date >= start_date
    if not date_is_valid:
        st.warning("結束日期不可早於起始日期。")

    start_dt = (
        datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        if date_is_valid
        else None
    )
    end_dt = (
        datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        if date_is_valid
        else None
    )
    s1, s2 = st.columns(2)
    sync_before_run = s1.checkbox(
        "執行前同步最新日K（較慢）",
        value=False,
        key="rotation_sync_before_run",
        help="預設關閉：優先使用本地資料庫；資料不足時才補同步。",
    )
    parallel_sync = s2.checkbox(
        "平行同步多標的",
        value=True,
        key="rotation_parallel_sync",
        help="多檔 ETF 時通常更快；若網路不穩可關閉改逐檔同步。",
    )

    run_key = (
        f"tw_rotation:{start_date}:{end_date}:{benchmark_choice}:"
        f"{top_n}:{fee_rate}:{sell_tax}:{slippage}:{initial_capital}"
    )

    if st.button("執行 ETF 輪動策略回測", type="primary", width="stretch"):
        if not date_is_valid:
            st.error("日期區間無效，請先修正起訖日期。")
            return

        progress = st.progress(0.0)
        with st.spinner("整理 ETF 池資料中..."):
            prepared = prepare_rotation_bars(
                store=store,
                symbols=list(ROTATION_DEFAULT_UNIVERSE),
                start_dt=start_dt,
                end_dt=end_dt,
                sync_before_run=bool(sync_before_run),
                parallel_sync=bool(parallel_sync),
                normalize_ohlcv_frame=normalize_ohlcv_frame,
                min_required=ROTATION_MIN_BARS,
                progress_callback=lambda ratio: progress.progress(float(ratio)),
                max_workers=6 if bool(parallel_sync) else 1,
            )
        progress.empty()
        bars_by_symbol = dict(prepared.bars_by_symbol)
        skipped_symbols = list(prepared.skipped_symbols)
        symbol_sync_issues = list(prepared.sync_issues)
        perf_timer.mark("rotation_bars_prepared")
        _render_sync_issues("部分 ETF 同步失敗，已盡量使用本地可用資料", symbol_sync_issues)

        if not bars_by_symbol:
            st.error(f"可用資料不足（每檔至少需 {ROTATION_MIN_BARS} 根K），無法執行。")
            return

        benchmark_bars, benchmark_symbol, benchmark_sync_issues = _load_tw_benchmark_bars(
            store=store,
            choice=benchmark_choice,
            start_dt=start_dt,
            end_dt=end_dt,
            sync_first=sync_before_run,
            allow_twii_fallback=True,
            min_rows=60,
        )
        _render_sync_issues(
            "Benchmark 同步有部分錯誤，已盡量使用本地可用資料", benchmark_sync_issues
        )
        if benchmark_bars.empty:
            st.error("Benchmark 取得失敗，請改選 0050 或 006208 後重試。")
            return
        perf_timer.mark("rotation_benchmark_loaded")

        top_n_effective = min(int(top_n), len(bars_by_symbol))
        cost_model = CostModel(
            fee_rate=float(fee_rate),
            sell_tax_rate=float(sell_tax),
            slippage_rate=float(slippage),
        )

        try:
            result = run_tw_etf_rotation_backtest(
                bars_by_symbol=bars_by_symbol,
                benchmark_bars=benchmark_bars,
                top_n=top_n_effective,
                initial_capital=float(initial_capital),
                cost_model=cost_model,
            )
        except Exception as exc:
            st.error(f"輪動回測失敗：{exc}")
            return
        perf_timer.mark("rotation_backtest_executed")

        eq_idx = result.equity_curve.index
        buy_hold_equity = build_buy_hold_equity(
            bars_by_symbol=bars_by_symbol,
            target_index=eq_idx,
            initial_capital=float(initial_capital),
        )
        benchmark_close = (
            pd.to_numeric(benchmark_bars["close"], errors="coerce").reindex(eq_idx).ffill()
        )
        benchmark_non_na = benchmark_close.dropna()
        if benchmark_non_na.empty or float(benchmark_non_na.iloc[0]) <= 0:
            benchmark_equity = pd.Series(dtype=float)
        else:
            benchmark_equity = float(initial_capital) * (
                benchmark_close / float(benchmark_non_na.iloc[0])
            )

        rebalance_rows: list[dict[str, object]] = []
        selected_symbol_lists: list[list[str]] = []
        for rec in result.rebalance_records:
            selected_symbol_lists.append(list(rec.selected_symbols))
            rebalance_rows.append(
                {
                    "signal_date": rec.signal_date.isoformat(),
                    "effective_date": rec.effective_date.isoformat(),
                    "market_filter_on": bool(rec.market_filter_on),
                    "selected_symbols": rec.selected_symbols,
                    "weights": rec.weights,
                    "scores": rec.scores,
                }
            )

        name_map = service.get_tw_symbol_names(list(ROTATION_DEFAULT_UNIVERSE))
        holding_rank = runner_build_rotation_holding_rank(
            weights_df=result.weights,
            selected_symbol_lists=selected_symbol_lists,
            universe_symbols=list(ROTATION_DEFAULT_UNIVERSE),
            name_map=name_map,
        )

        trades_df = result.trades.copy()
        if not trades_df.empty and "date" in trades_df.columns:
            trades_df["date"] = pd.to_datetime(
                trades_df["date"], utc=True, errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        payload = runner_build_rotation_payload(
            run_key=run_key,
            benchmark_symbol=benchmark_symbol,
            universe_symbols=list(ROTATION_DEFAULT_UNIVERSE),
            bars_by_symbol=bars_by_symbol,
            skipped_symbols=skipped_symbols,
            start_date=start_date,
            end_date=end_date,
            top_n=int(top_n_effective),
            initial_capital=float(initial_capital),
            metrics=result.metrics.__dict__,
            equity_series=result.equity_curve["equity"],
            benchmark_equity=benchmark_equity,
            buy_hold_equity=buy_hold_equity,
            rebalance_records=rebalance_rows,
            trades_df=trades_df,
            holding_rank=holding_rank,
        )
        st.session_state[payload_key] = payload
        store.save_rotation_run(
            universe_id=universe_id,
            run_key=run_key,
            params={
                "start_date": str(start_date),
                "end_date": str(end_date),
                "benchmark_choice": benchmark_choice,
                "top_n": int(top_n_effective),
                "fee_rate": float(fee_rate),
                "sell_tax": float(sell_tax),
                "slippage": float(slippage),
                "initial_capital": float(initial_capital),
            },
            payload=payload,
        )
        perf_timer.mark("rotation_run_cached")

    payload = st.session_state.get(payload_key)
    if not payload:
        st.info("設定條件後按下「執行 ETF 輪動策略回測」，之後會自動顯示最近一次快取結果。")
        return
    if payload.get("run_key") != run_key:
        st.caption("目前顯示的是上一次執行結果；若要套用目前設定，請重新按下執行。")
    rotation_health = _build_data_health(
        as_of=payload.get("generated_at", ""),
        data_sources=[_store_data_source(store, "rotation_runs")],
        source_chain=[str(payload.get("benchmark_symbol", "") or "benchmark")],
        degraded=payload.get("run_key") != run_key,
        fallback_depth=1 if payload.get("run_key") != run_key else 0,
        notes=f"used={len(payload.get('used_symbols', []))}/{len(payload.get('universe_symbols', []))}",
    )
    _render_data_health_caption("輪動資料健康度", rotation_health)

    strategy_df = pd.DataFrame(payload.get("equity_curve", []))
    if (
        strategy_df.empty
        or "date" not in strategy_df.columns
        or "equity" not in strategy_df.columns
    ):
        st.warning("快取內容不完整，請重新執行一次輪動回測。")
        return
    strategy_df["date"] = pd.to_datetime(strategy_df["date"], utc=True, errors="coerce")
    strategy_df["equity"] = pd.to_numeric(strategy_df["equity"], errors="coerce")
    strategy_df = strategy_df.dropna(subset=["date", "equity"]).set_index("date").sort_index()
    strategy_series = strategy_df["equity"]

    benchmark_df = pd.DataFrame(payload.get("benchmark_curve", []))
    benchmark_series = pd.Series(dtype=float)
    if not benchmark_df.empty and {"date", "equity"}.issubset(benchmark_df.columns):
        benchmark_df["date"] = pd.to_datetime(benchmark_df["date"], utc=True, errors="coerce")
        benchmark_df["equity"] = pd.to_numeric(benchmark_df["equity"], errors="coerce")
        benchmark_df = benchmark_df.dropna(subset=["date", "equity"]).set_index("date").sort_index()
        benchmark_series = benchmark_df["equity"]

    buy_hold_df = pd.DataFrame(payload.get("buy_hold_curve", []))
    buy_hold_series = pd.Series(dtype=float)
    if not buy_hold_df.empty and {"date", "equity"}.issubset(buy_hold_df.columns):
        buy_hold_df["date"] = pd.to_datetime(buy_hold_df["date"], utc=True, errors="coerce")
        buy_hold_df["equity"] = pd.to_numeric(buy_hold_df["equity"], errors="coerce")
        buy_hold_df = buy_hold_df.dropna(subset=["date", "equity"]).set_index("date").sort_index()
        buy_hold_series = buy_hold_df["equity"]

    generated_at_text = str(payload.get("generated_at", "")).strip()
    if generated_at_text:
        try:
            generated_dt = datetime.fromisoformat(generated_at_text)
            if generated_dt.tzinfo is None:
                generated_dt = generated_dt.replace(tzinfo=timezone.utc)
            generated_at_text = generated_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    st.caption(
        f"上次執行：{generated_at_text or 'N/A'} | "
        f"Benchmark: {payload.get('benchmark_symbol', 'N/A')} | "
        f"區間：{payload.get('start_date')} ~ {payload.get('end_date')} | "
        f"持有檔數：Top {payload.get('top_n')} | "
        f"可用標的：{len(payload.get('used_symbols', []))} / {len(payload.get('universe_symbols', []))}"
    )
    if payload.get("skipped_symbols"):
        st.caption(f"資料不足未納入：{', '.join(payload.get('skipped_symbols', []))}")

    chart_lines: list[dict[str, Any]] = [
        {
            "name": "Strategy Equity (ETF Rotation)",
            "series": strategy_series,
            "color": str(palette["equity"]),
            "width": 2.3,
            "dash": "solid",
            "hover_code": "ROTATION",
            "value_label": "Equity",
            "y_format": ",.0f",
        }
    ]
    if not benchmark_series.empty:
        benchmark_label = str(payload.get("benchmark_symbol", "Benchmark")).strip() or "Benchmark"
        style = _benchmark_line_style(palette, width=2.0)
        chart_lines.append(
            {
                "name": f"Benchmark Equity ({benchmark_label})",
                "series": benchmark_series,
                "color": str(style["color"]),
                "width": float(style["width"]),
                "dash": str(style["dash"]),
                "hover_code": benchmark_label,
                "value_label": "Equity",
                "y_format": ",.0f",
            }
        )
    if not buy_hold_series.empty:
        chart_lines.append(
            {
                "name": "Buy-and-Hold Equity (Frictionless, Equal-Weight ETF Pool)",
                "series": buy_hold_series,
                "color": str(palette["buy_hold"]),
                "width": 1.9,
                "dash": "solid",
                "hover_code": "EW_POOL",
                "value_label": "Equity",
                "y_format": ",.0f",
            }
        )
    _render_benchmark_lines_chart(
        lines=chart_lines,
        height=500,
        chart_key="rotation_benchmark",
    )

    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Return", f"{float(metrics.get('total_return', 0.0)) * 100.0:+.2f}%")
        m2.metric("CAGR", f"{float(metrics.get('cagr', 0.0)) * 100.0:+.2f}%")
        m3.metric("Sharpe", f"{float(metrics.get('sharpe', 0.0)):.2f}")
        m4.metric("MDD", f"{float(metrics.get('max_drawdown', 0.0)) * 100.0:+.2f}%")
        m5.metric("Trades", f"{int(metrics.get('trades', 0))}")

    if not benchmark_series.empty:
        comp = pd.concat(
            [strategy_series.rename("strategy"), benchmark_series.rename("benchmark")], axis=1
        ).dropna()
        if len(comp) >= 2:
            strat_ret = float(comp["strategy"].iloc[-1] / comp["strategy"].iloc[0] - 1.0)
            bench_ret = float(comp["benchmark"].iloc[-1] / comp["benchmark"].iloc[0] - 1.0)
            excess = (strat_ret - bench_ret) * 100.0
            verdict = "贏過大盤" if excess > 0 else ("輸給大盤" if excess < 0 else "與大盤持平")
            st.info(
                f"相對Benchmark：{verdict} {excess:+.2f}%（策略 {strat_ret * 100:+.2f}% / Benchmark {bench_ret * 100:+.2f}%）"
            )

    holding_rank = payload.get("holding_rank")
    if not isinstance(holding_rank, list):
        holding_rank = []
    if not holding_rank:
        selected_symbol_lists: list[list[str]] = []
        for item in payload.get("rebalance_records", []):
            if isinstance(item, dict):
                symbols = item.get("selected_symbols")
                if isinstance(symbols, list):
                    selected_symbol_lists.append([str(s) for s in symbols])
        name_map = service.get_tw_symbol_names(list(ROTATION_DEFAULT_UNIVERSE))
        holding_rank = runner_build_rotation_holding_rank(
            weights_df=None,
            selected_symbol_lists=selected_symbol_lists,
            universe_symbols=list(ROTATION_DEFAULT_UNIVERSE),
            name_map=name_map,
        )

    top3_rank = [row for row in holding_rank if isinstance(row, dict)][:3]
    if top3_rank:
        st.markdown("#### 持有最久 ETF（策略推薦 Top3）")
        cols = st.columns(min(3, len(top3_rank)))
        for idx, row in enumerate(top3_rank):
            symbol = str(row.get("symbol", "N/A"))
            name = str(row.get("name", symbol))
            hold_days = int(float(row.get("hold_days", 0) or 0))
            hold_ratio = float(row.get("hold_ratio_pct", 0.0) or 0.0)
            selected_months = int(float(row.get("selected_months", 0) or 0))
            cols[idx].metric(
                f"Top {idx + 1}",
                f"{symbol} {name}".strip(),
                f"持有 {hold_days} 天（{hold_ratio:.1f}%）｜入選 {selected_months} 次",
            )
        st.caption(
            "說明：此排名反映本次回測參數下的策略偏好（持有天數/入選次數），不代表未來保證報酬。"
        )

    rebalance_df = pd.DataFrame(payload.get("rebalance_records", []))
    if not rebalance_df.empty:
        name_map = service.get_tw_symbol_names(list(ROTATION_DEFAULT_UNIVERSE))

        def _format_selected(v: object) -> str:
            if not isinstance(v, list):
                return "—"
            if not v:
                return "空手"
            return "、".join([f"{sym} {name_map.get(sym, '')}".strip() for sym in v])

        rebalance_df["signal_date"] = pd.to_datetime(
            rebalance_df["signal_date"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        rebalance_df["effective_date"] = pd.to_datetime(
            rebalance_df["effective_date"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        rebalance_df["selected"] = rebalance_df["selected_symbols"].map(_format_selected)
        rebalance_df["market_filter"] = rebalance_df["market_filter_on"].map(
            lambda v: "ON" if bool(v) else "OFF"
        )
        rebalance_df["scores"] = rebalance_df["scores"].map(
            lambda obj: (
                "、".join([f"{k}:{float(v):+.3f}" for k, v in obj.items()])
                if isinstance(obj, dict) and obj
                else "—"
            )
        )
        st.markdown("#### 每月調倉明細")
        st.dataframe(
            rebalance_df[["signal_date", "effective_date", "market_filter", "selected", "scores"]],
            width="stretch",
            hide_index=True,
        )

    trades_df = pd.DataFrame(payload.get("trades", []))
    if not trades_df.empty:
        st.markdown("#### 成交紀錄（前200筆）")
        show_cols = [
            c
            for c in [
                "date",
                "symbol",
                "side",
                "qty",
                "price",
                "notional",
                "fee",
                "tax",
                "slippage",
                "pnl",
                "target_weight",
            ]
            if c in trades_df.columns
        ]
        st.dataframe(trades_df[show_cols].head(200), width="stretch", hide_index=True)

    perf_timer.mark("rotation_render_complete")
    if perf_timer.enabled:
        st.caption(perf_timer.summary_text(prefix="perf/rotation"))


def _render_00935_heatmap_view():
    _render_tw_etf_heatmap_view("00935", page_desc="科技類")


def _render_00735_heatmap_view():
    _render_tw_etf_heatmap_view("00735", page_desc="科技類")


def _render_00910_heatmap_view():
    _render_tw_etf_heatmap_view("00910", page_desc="第一金太空衛星")


def _render_0050_heatmap_view():
    _render_tw_etf_heatmap_view("0050", page_desc="台灣50")


def _render_0052_heatmap_view():
    _render_tw_etf_heatmap_view("0052", page_desc="科技ETF")


def _render_db_browser_view():
    store = _history_store()
    backend_name = str(getattr(store, "backend_name", "duckdb") or "duckdb").strip().lower()
    backend_label = "DuckDB" if backend_name == "duckdb" else "SQLite"

    st.subheader(f"{backend_label} 資料庫檢視")
    db_path = store.db_path

    st.caption(f"資料庫路徑：`{db_path}` | backend: `{backend_label}`")
    if not db_path.exists():
        st.error(f"找不到資料庫檔案：`{db_path}`。")
        return

    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    st.caption(f"檔案大小：約 {db_size_mb:.2f} MB")

    st.markdown("#### 市場基礎資料預載")
    latest_bootstrap = store.load_latest_bootstrap_run()
    if latest_bootstrap is None:
        st.info("尚未有預載紀錄。可先執行一次台股/美股核心資料預載。")
    else:
        finished_text = "進行中"
        if latest_bootstrap.finished_at is not None:
            finished_text = latest_bootstrap.finished_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        st.caption(
            "最近一次任務："
            f"{latest_bootstrap.scope}｜狀態 {latest_bootstrap.status}｜"
            f"啟動 {latest_bootstrap.started_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}｜"
            f"完成 {finished_text}｜"
            f"總數 {latest_bootstrap.total_symbols} / 成功 {latest_bootstrap.synced_symbols} / 失敗 {latest_bootstrap.failed_symbols}"
        )
        if latest_bootstrap.error:
            st.warning(f"最近一次任務錯誤：{latest_bootstrap.error}")

    b1, b2, b3, b4 = st.columns([1.5, 1, 1, 1])
    scope_label = b1.selectbox(
        "預載範圍",
        options=["台股+美股核心", "僅台股", "僅美股核心"],
        index=0,
        key="db_bootstrap_scope",
    )
    years = int(b2.selectbox("歷史年數", options=[3, 5, 8], index=1, key="db_bootstrap_years"))
    workers = int(
        b3.selectbox("平行工作數", options=[2, 4, 6, 8], index=2, key="db_bootstrap_workers")
    )
    tw_limit_input = int(
        b4.number_input(
            "台股筆數上限",
            min_value=0,
            value=0,
            step=50,
            key="db_bootstrap_tw_limit",
            help="0 代表全量台股；可先輸入 200 做試跑。",
        )
    )
    scope_token = {
        "台股+美股核心": "both",
        "僅台股": "tw",
        "僅美股核心": "us",
    }.get(scope_label, "both")
    tw_limit = None if tw_limit_input <= 0 else tw_limit_input

    a1, a2 = st.columns([1, 1])
    if a1.button("啟動基礎資料預載", type="primary", width="stretch", key="db_run_bootstrap"):
        with st.spinner("預載中（會同步 metadata 與日線歷史）..."):
            summary = run_market_data_bootstrap(
                store=store,
                scope=scope_token,
                years=years,
                parallel=True,
                max_workers=workers,
                tw_limit=tw_limit,
                sync_mode="min_rows",
            )
        st.session_state["db_manual_bootstrap_summary"] = summary
        if str(summary.get("status", "")) == "completed":
            st.success("基礎資料預載完成。")
        else:
            st.warning(f"預載完成，但有部分失敗（狀態：{summary.get('status', 'unknown')}）。")

    if a2.button("執行一次增量更新", width="stretch", key="db_run_incremental_refresh"):
        with st.spinner("增量更新中（只補缺漏或最新區間）..."):
            summary = run_incremental_refresh(
                store=store,
                years=5,
                parallel=True,
                max_workers=min(workers, 4),
                tw_limit=180,
                us_limit=80,
            )
        st.session_state["db_incremental_refresh_summary"] = summary
        if str(summary.get("status", "")) == "completed":
            st.success("增量更新完成。")
        else:
            st.warning(f"增量更新完成，但有部分失敗（狀態：{summary.get('status', 'unknown')}）。")

    manual_summary = st.session_state.get("db_manual_bootstrap_summary")
    if isinstance(manual_summary, dict):
        preview = {
            "任務ID": manual_summary.get("run_id", ""),
            "範圍": manual_summary.get("scope", ""),
            "總數": manual_summary.get("total_symbols", 0),
            "成功": manual_summary.get("synced_success", 0),
            "跳過": manual_summary.get("skipped_symbols", 0),
            "失敗": manual_summary.get("failed_symbols", 0),
            "狀態": manual_summary.get("status", ""),
        }
        st.caption("最近一次手動預載摘要")
        st.dataframe(pd.DataFrame([preview]), width="stretch", hide_index=True)
        issues = manual_summary.get("issues")
        if isinstance(issues, list) and issues:
            _render_sync_issues(
                "手動預載有部分同步錯誤", [str(item) for item in issues], preview_limit=3
            )

    incremental_summary = st.session_state.get("db_incremental_refresh_summary")
    if isinstance(incremental_summary, dict):
        preview = {
            "任務ID": incremental_summary.get("run_id", ""),
            "總數": incremental_summary.get("total_symbols", 0),
            "成功": incremental_summary.get("synced_success", 0),
            "跳過": incremental_summary.get("skipped_symbols", 0),
            "失敗": incremental_summary.get("failed_symbols", 0),
            "狀態": incremental_summary.get("status", ""),
        }
        st.caption("最近一次增量更新摘要")
        st.dataframe(pd.DataFrame([preview]), width="stretch", hide_index=True)
        issues = incremental_summary.get("issues")
        if isinstance(issues, list) and issues:
            _render_sync_issues(
                "增量更新有部分同步錯誤", [str(item) for item in issues], preview_limit=3
            )

    conn_sqlite: sqlite3.Connection | None = None
    conn_duck = None
    try:
        if backend_name == "duckdb":
            if duckdb is None:
                st.error("目前環境缺少 duckdb 套件，無法開啟 DuckDB 檢視。")
                return
            conn_duck = duckdb.connect(str(db_path), read_only=True)
            tables_df = conn_duck.execute(
                """
                SELECT table_name AS name
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                ORDER BY table_name ASC
                """
            ).df()
        else:
            conn_sqlite = sqlite3.connect(str(db_path))
            tables_df = pd.read_sql_query(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name ASC
                """,
                conn_sqlite,
            )
        if tables_df.empty:
            st.info("目前資料庫中沒有可檢視的資料表。")
            return

        table_names = tables_df["name"].astype(str).tolist()
        summary_rows: list[dict[str, object]] = []
        count_map: dict[str, int] = {}
        for table in table_names:
            escaped = table.replace('"', '""')
            if backend_name == "duckdb":
                assert conn_duck is not None
                count = int(conn_duck.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0])
            else:
                assert conn_sqlite is not None
                count = int(conn_sqlite.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0])
            count_map[table] = count
            summary_rows.append({"資料表": table, "筆數": count})

        st.markdown("#### 資料表總覽")
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        c1, c2, c3 = st.columns([2, 1, 1])
        selected_table = c1.selectbox("選擇資料表", options=table_names, key="db_view_table")
        page_size = int(
            c2.selectbox(
                "每頁筆數", options=[20, 50, 100, 200, 500], index=2, key="db_view_page_size"
            )
        )
        order_mode = c3.selectbox(
            "排序", options=["最新在前", "舊到新"], index=0, key="db_view_order_mode"
        )

        total_rows = int(count_map.get(selected_table, 0))
        total_pages = max(1, int(math.ceil(total_rows / max(page_size, 1))))
        page = int(
            st.number_input(
                "頁碼",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key=f"db_view_page_{selected_table}",
            )
        )
        offset = (page - 1) * page_size
        st.caption(f"目前資料表 `{selected_table}`：共 {total_rows} 筆，頁 {page} / {total_pages}")

        escaped_table = selected_table.replace('"', '""')
        escaped_table_literal = selected_table.replace("'", "''")
        if backend_name == "duckdb":
            assert conn_duck is not None
            schema_df = conn_duck.execute(f"PRAGMA table_info('{escaped_table_literal}')").df()
        else:
            assert conn_sqlite is not None
            schema_df = pd.read_sql_query(f'PRAGMA table_info("{escaped_table}")', conn_sqlite)
        col_names = schema_df["name"].astype(str).tolist() if not schema_df.empty else []
        order_candidates = ["updated_at", "created_at", "date", "fetched_at", "id"]
        order_col = next((col for col in order_candidates if col in col_names), None)
        direction = "DESC" if order_mode == "最新在前" else "ASC"

        query = f'SELECT * FROM "{escaped_table}"'
        if order_col:
            query += f' ORDER BY "{order_col}" {direction}'
        query += " LIMIT ? OFFSET ?"

        if backend_name == "duckdb":
            assert conn_duck is not None
            data_df = conn_duck.execute(query, [page_size, offset]).df()
        else:
            assert conn_sqlite is not None
            data_df = pd.read_sql_query(query, conn_sqlite, params=[page_size, offset])
        st.markdown("#### 資料表內容")
        if data_df.empty:
            st.info("此頁沒有資料。")
        else:
            st.dataframe(data_df, width="stretch", hide_index=True)

        with st.expander("欄位結構（Schema）", expanded=False):
            if schema_df.empty:
                st.caption("無法讀取欄位資訊。")
            else:
                schema_out = schema_df.rename(
                    columns={
                        "name": "欄位",
                        "type": "型別",
                        "notnull": "不可為空",
                        "dflt_value": "預設值",
                        "pk": "主鍵",
                    }
                )
                show_cols = [
                    c
                    for c in ["cid", "欄位", "型別", "不可為空", "預設值", "主鍵"]
                    if c in schema_out.columns
                ]
                st.dataframe(schema_out[show_cols], width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"讀取資料庫失敗：{exc}")
    finally:
        if conn_sqlite is not None:
            conn_sqlite.close()
        if conn_duck is not None:
            conn_duck.close()


def main():
    st.set_page_config(page_title="即時看盤 + 回測平台", layout="wide")
    _inject_ui_styles()
    _consume_heatmap_drilldown_query()
    _consume_backtest_drilldown_query()
    _auto_run_daily_incremental_refresh(_history_store())
    st.title("即時走勢 / 多來源資料 / 回測平台")
    st.caption(_runtime_stack_caption())
    _render_design_toolbox()
    active_page = _render_page_cards_nav()
    page_renderers = {
        "即時看盤": _render_live_view,
        "回測工作台": _render_backtest_view,
        "2026 YTD 前十大股利型、配息型 ETF": _render_top10_etf_2025_view,
        "2026 YTD 前十大 ETF": _render_top10_etf_2026_ytd_view,
        "台股 ETF 全類型總表": _render_tw_etf_all_types_view,
        "2025 後20大最差勁 ETF": _render_bottom20_etf_2025_view,
        "共識代表 ETF": _render_consensus_representative_etf_view,
        "兩檔 ETF 推薦": _render_two_etf_pick_view,
        "2026 YTD 主動式 ETF": _render_active_etf_2026_ytd_view,
        "ETF 輪動策略": _render_tw_etf_rotation_view,
        "熱力圖總表": _render_heatmap_hub_view,
        "00910 熱力圖": _render_00910_heatmap_view,
        "00935 熱力圖": _render_00935_heatmap_view,
        "00735 熱力圖": _render_00735_heatmap_view,
        "0050 熱力圖": _render_0050_heatmap_view,
        "0052 熱力圖": _render_0052_heatmap_view,
        "資料庫檢視": _render_db_browser_view,
        "新手教學": _render_tutorial_view,
    }
    page_renderers.update(_dynamic_heatmap_page_renderers())
    render_fn = page_renderers.get(active_page)
    if render_fn is None:
        st.error("頁面載入失敗，請重新整理後再試。")
        return
    render_fn()


if __name__ == "__main__":
    main()
