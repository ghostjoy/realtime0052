from __future__ import annotations

import json
import math
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from io import StringIO
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest import (
    BacktestMetrics,
    BacktestResult,
    CostModel,
    PortfolioBacktestResult,
    ROTATION_DEFAULT_UNIVERSE,
    ROTATION_MIN_BARS,
    Trade,
    apply_start_to_bars_map,
    apply_split_adjustment,
    build_buy_hold_equity,
    interval_return,
    run_tw_etf_rotation_backtest,
    run_backtest,
    run_portfolio_backtest,
    get_strategy_min_bars,
    required_walkforward_bars,
    walk_forward_portfolio,
    walk_forward_single,
)
from backtest.adjustments import known_split_events
from indicators import add_indicators
from providers import TwMisProvider
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
    ordered_symbols: list[str] = []
    for symbol in symbols:
        text = str(symbol or "").strip().upper()
        if text and text not in ordered_symbols:
            ordered_symbols.append(text)
    reports: dict[str, object] = {}
    issues: list[str] = []
    if not ordered_symbols:
        return reports, issues

    def _run_one(symbol: str) -> object:
        return store.sync_symbol_history(symbol=symbol, market=market, start=start, end=end)

    use_parallel = parallel and len(ordered_symbols) > 1
    if not use_parallel:
        for symbol in ordered_symbols:
            try:
                report = _run_one(symbol)
                reports[symbol] = report
                err = str(getattr(report, "error", "") or "").strip()
                if err:
                    issues.append(f"{symbol}: {err}")
            except Exception as exc:
                reports[symbol] = None
                issues.append(f"{symbol}: {exc}")
        return reports, issues

    workers = max(1, min(int(max_workers), len(ordered_symbols)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, symbol): symbol for symbol in ordered_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                report = future.result()
                reports[symbol] = report
                err = str(getattr(report, "error", "") or "").strip()
                if err:
                    issues.append(f"{symbol}: {err}")
            except Exception as exc:
                reports[symbol] = None
                issues.append(f"{symbol}: {exc}")
    return reports, issues


def _bars_need_backfill(bars: pd.DataFrame, *, start: datetime, end: datetime) -> bool:
    if bars is None or bars.empty:
        return True
    idx = pd.to_datetime(bars.index, utc=True, errors="coerce")
    idx = idx.dropna()
    if idx.empty:
        return True
    first_ts = pd.Timestamp(idx.min()).to_pydatetime().replace(tzinfo=timezone.utc)
    last_ts = pd.Timestamp(idx.max()).to_pydatetime().replace(tzinfo=timezone.utc)
    return first_ts > start or last_ts < end


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
    should_flush = bool(buffer) and (len(buffer) >= max(1, int(batch_size)) or now_ts - last_flush >= float(flush_interval_sec))
    if should_flush:
        try:
            store.save_intraday_ticks(symbol=symbol, market=market, ticks=buffer)
            buffer = []
            st.session_state[flush_key] = now_ts
        except Exception as exc:
            st.warning(f"即時資料寫入 SQLite 失敗（已略過本輪）：{exc}")
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
            ("元大投信 0050 基本資訊（Index Profile / Index Methodology）", "https://www.yuantaetfs.com/product/detail/0050/Basic_information"),
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
            ("TWSE ETF e添富：00935 新上市資訊", "https://www.twse.com.tw/zh/ETFortune/newsDetail/ff8080818b7e232e018b83ab7de9002b"),
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
            ("第一金投信 00910 專頁（標的指數編製方法）", "https://www.fsitc.com.tw/act/202206_asetf/"),
        ],
    },
}


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
        source_lines = [f"- [{str(label)}]({str(url)})" for label, url in sources if str(label).strip() and str(url).strip()]
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
        use_container_width=True,
        hide_index=True,
        height=_full_table_height(out_df),
    )


def _render_00910_constituent_intro_table(
    *,
    service: MarketDataService,
    full_rows: list[dict[str, object]],
):
    rows_full_for_intro = list(full_rows)
    if not rows_full_for_intro:
        rows_full_for_intro, _ = service.get_etf_constituents_full("00910", limit=None, force_refresh=False)
    if not rows_full_for_intro:
        return

    intro_df = pd.DataFrame(rows_full_for_intro)
    if "rank" in intro_df.columns:
        intro_df["rank"] = pd.to_numeric(intro_df["rank"], errors="coerce")
        intro_df = intro_df.sort_values("rank", ascending=True, na_position="last")
    intro_df["symbol"] = intro_df.get("symbol", pd.Series(dtype=str)).astype(str)
    intro_df["name"] = intro_df.get("name", pd.Series(dtype=str)).astype(str)
    intro_df["market"] = intro_df.get("market", pd.Series(dtype=str)).astype(str).str.upper()
    intro_df["weight_pct"] = pd.to_numeric(intro_df.get("weight_pct"), errors="coerce").round(2)
    briefs: list[str] = []
    for _, row in intro_df.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        name = str(row.get("name", "")).strip()
        market = str(row.get("market", "")).strip().upper()
        brief = _company_brief_for_00910(symbol=symbol, name=name, market_tag=market)
        briefs.append(brief)
    intro_df["brief"] = briefs
    show_cols = [c for c in ["rank", "symbol", "name", "market", "weight_pct", "brief"] if c in intro_df.columns]
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
        use_container_width=True,
        hide_index=True,
        height=_full_table_height(out_intro),
    )


PAGE_CARDS = [
    {"key": "即時看盤", "desc": "台股/美股即時報價、即時走勢與技術快照。"},
    {"key": "回測工作台", "desc": "日K同步、策略回測、回放與績效比較。"},
    {"key": "2025 前十大 ETF", "desc": "2025 年全年台股 ETF 報酬率前十名。"},
    {"key": "2026 YTD 前十大 ETF", "desc": "2026 年截至今日的台股 ETF 報酬率前十名。"},
    {"key": "2026 YTD 主動式 ETF", "desc": "台股主動式 ETF 在 2026 年截至今日的 Buy & Hold 績效。"},
    {"key": "ETF 輪動策略", "desc": "6檔台股ETF月頻輪動與基準對照。"},
    {"key": "00910 熱力圖", "desc": "00910 成分股回測的相對大盤熱力圖。"},
    {"key": "00935 熱力圖", "desc": "00935 成分股回測的相對大盤熱力圖。"},
    {"key": "00993A 熱力圖", "desc": "00993A 成分股回測的相對大盤熱力圖。"},
    {"key": "0050 熱力圖", "desc": "0050 成分股回測的相對大盤熱力圖。"},
    {"key": "資料庫檢視", "desc": "直接查看 SQLite 各表筆數、欄位與內容。"},
    {"key": "新手教學", "desc": "參數白話解釋與常見回測誤區。"},
]


def _strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(str(name), str(name))


def _format_price(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.2f}"


def _format_int(v: Optional[int]) -> str:
    if v is None:
        return "—"
    return f"{v:,}"


def _safe_float(value: object) -> Optional[float]:
    try:
        fv = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def _resolve_live_change_metrics(
    quote,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
) -> tuple[Optional[float], Optional[float], str]:
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

    daily_norm = _normalize_ohlcv_frame(daily)
    if (prev_close is None or abs(prev_close) < 1e-12) and not daily_norm.empty and "close" in daily_norm.columns:
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

    intraday_norm = _normalize_ohlcv_frame(intraday)
    if (prev_close is None or abs(prev_close) < 1e-12) and not intraday_norm.empty and "close" in intraday_norm.columns:
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
    if change_pct is None and change is not None and prev_close is not None and abs(prev_close) >= 1e-12:
        change_pct = change / prev_close * 100.0

    source_name = str(getattr(quote, "source", "unknown") or "unknown")
    basis_text = f"source={source_name}, prev_close={prev_close_basis}"
    return change, change_pct, basis_text


def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["open", "high", "low", "close", "volume"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=base_cols)

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        renamed: list[str] = []
        for col in out.columns:
            parts = [str(part).strip().lower() for part in col if str(part).strip()]
            candidate = ""
            for item in reversed(parts):
                if item in {"open", "high", "low", "close", "adj close", "adj_close", "volume", "price"}:
                    candidate = item
                    break
            renamed.append(candidate or (parts[-1] if parts else ""))
        out.columns = renamed
    else:
        out.columns = [str(col).strip().lower() for col in out.columns]

    if "adj close" in out.columns and "adj_close" not in out.columns:
        out = out.rename(columns={"adj close": "adj_close"})

    if "price" in out.columns and "close" not in out.columns:
        out["close"] = out["price"]

    if "close" not in out.columns:
        return pd.DataFrame(columns=base_cols)

    close = pd.to_numeric(out["close"], errors="coerce")
    for col in ["open", "high", "low"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(close)
        else:
            out[col] = close
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0
    out["close"] = close

    out = out[base_cols].dropna(subset=["open", "high", "low", "close"], how="any")
    out = out.sort_index()
    return out


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
    return (
        "Date=%{x|%Y-%m-%d}<br>"
        f"代碼={token}<br>"
        f"{value_label}=%{{y:{y_format}}}"
        "<extra></extra>"
    )


def _benchmark_line_style(
    palette: dict[str, object],
    *,
    width: float = 2.0,
    dash: Optional[str] = None,
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
    return list(_THEME_PALETTES.keys())


def _current_theme_name() -> str:
    default_theme = "灰白專業（Soft Gray）"
    theme = str(st.session_state.get("ui_theme", default_theme))
    if theme not in _THEME_PALETTES:
        theme = default_theme
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
            margin-bottom: 0.12rem;
        }}
        .card-section-sub {{
            font-size: 0.82rem;
            color: {text_muted};
            margin-bottom: 0.5rem;
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
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.link_button("Figma", "https://www.figma.com", use_container_width=True)
        c2.link_button("Pencil", "https://pencil.evolus.vn", use_container_width=True)


def _render_page_cards_nav() -> str:
    page_options = [item["key"] for item in PAGE_CARDS]
    active_page = str(st.session_state.get("active_page", page_options[0]))
    if active_page not in page_options:
        active_page = page_options[0]
        st.session_state["active_page"] = active_page

    st.markdown("#### 功能卡片")
    cols = st.columns(5, gap="small")
    for idx, item in enumerate(PAGE_CARDS):
        key = item["key"]
        desc = item["desc"]
        is_active = key == active_page
        with cols[idx % 5]:
            st.markdown(
                (
                    f"<div class='page-nav-card{' active' if is_active else ''}'>"
                    f"<div class='page-card-title'>{key}</div>"
                    f"<div class='page-card-desc'>{desc}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            clicked = st.button(
                "已開啟" if is_active else "開啟",
                key=f"page-card:{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            )
            if clicked and not is_active:
                st.session_state["active_page"] = key
                st.rerun()

    return str(st.session_state.get("active_page", active_page))


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_twse_snapshot_with_fallback(target_yyyymmdd: str, lookback_days: int = 14) -> tuple[str, pd.DataFrame]:
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
        selected: Optional[dict] = None
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
            out_rows.append({"code": code, "name": name, "close": close})
        if out_rows:
            return date_token, pd.DataFrame(out_rows)
        errors.append(f"{date_token}:no parsable rows")

    raise RuntimeError("TWSE snapshot fetch failed: " + " | ".join(errors[-5:]))


def _classify_tw_etf(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return "其他"
    dividend_keywords = ("高股息", "股利", "股息", "收益", "月配", "季配", "年配")
    market_cap_keywords = ("台灣50", "臺灣50", "台灣中型100", "臺灣中型100", "市值", "台灣領袖")
    if any(key in text for key in dividend_keywords):
        return "股利型"
    if any(key in text for key in market_cap_keywords):
        return "市值型"
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


def _split_factor_and_events_between(symbol: str, start_used: str, end_used: str) -> tuple[float, str]:
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
def _build_tw_etf_top10_between(start_yyyymmdd: str, end_yyyymmdd: str) -> tuple[pd.DataFrame, str, str]:
    start_used, start_df = _fetch_twse_snapshot_with_fallback(start_yyyymmdd)
    end_used, end_df = _fetch_twse_snapshot_with_fallback(end_yyyymmdd)

    # 台股股票型 ETF（排除槓反/期貨/海外與債券商品）。
    start_df = start_df[start_df.apply(lambda r: _is_tw_equity_etf(str(r.get("code", "")), str(r.get("name", ""))), axis=1)].copy()
    end_df = end_df[end_df.apply(lambda r: _is_tw_equity_etf(str(r.get("code", "")), str(r.get("name", ""))), axis=1)].copy()
    if start_df.empty or end_df.empty:
        return pd.DataFrame(), start_used, end_used

    merged = end_df.merge(
        start_df[["code", "close"]].rename(columns={"close": "start_close"}),
        on="code",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), start_used, end_used
    merged = merged.rename(columns={"name": "name", "close": "end_close"})
    factor_info = merged["code"].map(lambda c: _split_factor_and_events_between(symbol=str(c), start_used=start_used, end_used=end_used))
    merged["split_factor"] = factor_info.map(lambda x: float(x[0]))
    merged["split_events"] = factor_info.map(lambda x: str(x[1]))
    merged["adj_start_close"] = pd.to_numeric(merged["start_close"], errors="coerce") * pd.to_numeric(merged["split_factor"], errors="coerce")
    merged["return_pct"] = (pd.to_numeric(merged["end_close"], errors="coerce") / pd.to_numeric(merged["adj_start_close"], errors="coerce") - 1.0) * 100.0
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_pct", "start_close", "end_close", "adj_start_close"])
    if merged.empty:
        return pd.DataFrame(), start_used, end_used

    merged["type"] = merged["name"].map(_classify_tw_etf)
    merged = merged.sort_values("return_pct", ascending=False).head(10).copy()
    merged["rank"] = range(1, len(merged) + 1)
    out = merged[["rank", "code", "name", "type", "start_close", "adj_start_close", "end_close", "split_events", "return_pct"]].copy()
    out = out.rename(
        columns={
            "rank": "排名",
            "code": "代碼",
            "name": "ETF",
            "type": "類型",
            "start_close": "期初收盤",
            "adj_start_close": "復權期初",
            "end_close": "期末收盤",
            "split_events": "復權事件",
            "return_pct": "區間報酬(%)",
        }
    )
    out["區間報酬(%)"] = pd.to_numeric(out["區間報酬(%)"], errors="coerce").round(2)
    out["期初收盤"] = pd.to_numeric(out["期初收盤"], errors="coerce").round(2)
    out["復權期初"] = pd.to_numeric(out["復權期初"], errors="coerce").round(2)
    out["期末收盤"] = pd.to_numeric(out["期末收盤"], errors="coerce").round(2)
    out["復權事件"] = out["復權事件"].replace("", "—")
    return out, start_used, end_used


@st.cache_data(ttl=3600, show_spinner=False)
def _build_tw_active_etf_ytd_between(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    symbols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, str, str]:
    start_used, start_df = _fetch_twse_snapshot_with_fallback(start_yyyymmdd)
    end_used, end_df = _fetch_twse_snapshot_with_fallback(end_yyyymmdd)

    end_active_df = end_df[end_df.apply(lambda r: _is_tw_active_etf(str(r.get("code", "")), str(r.get("name", ""))), axis=1)].copy()
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
        universe_symbols = [str(v).strip().upper() for v in end_active_df["code"].astype(str).tolist() if str(v).strip()]

    if not universe_symbols:
        return pd.DataFrame(), start_used, end_used

    end_name_map = {
        str(row["code"]).strip().upper(): str(row["name"]).strip()
        for _, row in end_active_df.iterrows()
        if str(row.get("code", "")).strip()
    }

    start_dt = datetime.combine(datetime.strptime(start_yyyymmdd, "%Y%m%d").date(), datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(datetime.strptime(end_used, "%Y%m%d").date(), datetime.min.time()).replace(tzinfo=timezone.utc)
    store = _history_store()

    rows: list[dict[str, object]] = []
    for symbol in universe_symbols:
        bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt))
        need_sync = True
        if isinstance(bars, pd.DataFrame) and not bars.empty:
            idx = pd.to_datetime(bars.index, utc=True, errors="coerce").dropna()
            if not idx.empty:
                last_ts = pd.Timestamp(idx.max()).to_pydatetime().replace(tzinfo=timezone.utc)
                need_sync = last_ts < end_dt
        if need_sync:
            store.sync_symbol_history(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt))
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
        split_factor, split_events = _split_factor_and_events_between(symbol=symbol, start_used=first_token, end_used=last_token)
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
            "start_close",
            "adj_start_close",
            "end_close",
            "start_trade_date",
            "end_trade_date",
            "split_events",
            "return_pct",
        ]
    ].copy()
    out = out.rename(
        columns={
            "rank": "排名",
            "code": "代碼",
            "name": "ETF",
            "start_close": "期初收盤",
            "adj_start_close": "復權期初",
            "end_close": "期末收盤",
            "start_trade_date": "績效起算日",
            "end_trade_date": "績效終點日",
            "split_events": "復權事件",
            "return_pct": "YTD報酬(%)",
        }
    )
    out["YTD報酬(%)"] = pd.to_numeric(out["YTD報酬(%)"], errors="coerce").round(2)
    out["期初收盤"] = pd.to_numeric(out["期初收盤"], errors="coerce").round(2)
    out["復權期初"] = pd.to_numeric(out["復權期初"], errors="coerce").round(2)
    out["期末收盤"] = pd.to_numeric(out["期末收盤"], errors="coerce").round(2)
    out["復權事件"] = out["復權事件"].replace("", "—")
    return out, start_used, end_used


@st.cache_data(ttl=600, show_spinner=False)
def _load_tw_market_return_between(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    *,
    force_sync: bool = False,
) -> tuple[Optional[float], str, list[str]]:
    start_dt = datetime.combine(datetime.strptime(start_yyyymmdd, "%Y%m%d").date(), datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(datetime.strptime(end_yyyymmdd, "%Y%m%d").date(), datetime.min.time()).replace(tzinfo=timezone.utc)
    if end_dt <= start_dt:
        return None, "", []

    store = _history_store()
    issues: list[str] = []
    candidates = ["^TWII", "0050", "006208"]
    for symbol in candidates:
        if force_sync:
            report = store.sync_symbol_history(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            err = str(getattr(report, "error", "") or "").strip()
            if err:
                issues.append(f"{symbol}: {err}")

        bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt))
        if _bars_need_backfill(bars, start=start_dt, end=end_dt):
            report = store.sync_symbol_history(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            err = str(getattr(report, "error", "") or "").strip()
            if err:
                issues.append(f"{symbol}: {err}")
            bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt))
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


def _decorate_tw_etf_top10_ytd_table(
    top10_df: pd.DataFrame,
    *,
    y2025_map: dict[str, float],
    market_return_pct: Optional[float],
    market_2025_return_pct: Optional[float],
    benchmark_code: str,
    end_used: str,
) -> pd.DataFrame:
    etf_df = top10_df.copy()
    if "區間報酬(%)" in etf_df.columns and "YTD報酬(%)" not in etf_df.columns:
        etf_df = etf_df.rename(columns={"區間報酬(%)": "YTD報酬(%)"})
    if "YTD報酬(%)" not in etf_df.columns:
        etf_df["YTD報酬(%)"] = np.nan

    code_series = etf_df.get("代碼", pd.Series(dtype=str)).astype(str).str.strip().str.upper()
    etf_df["2025績效(%)"] = code_series.map(y2025_map)
    etf_df["2025績效(%)"] = pd.to_numeric(etf_df["2025績效(%)"], errors="coerce").round(2)
    etf_df["贏輸台股大盤(%)"] = np.nan
    if market_return_pct is not None and math.isfinite(float(market_return_pct)):
        etf_df["贏輸台股大盤(%)"] = (
            pd.to_numeric(etf_df["YTD報酬(%)"], errors="coerce") - float(market_return_pct)
        ).round(2)

    benchmark_row = {
        "排名": "—",
        "代碼": str(benchmark_code or "^TWII"),
        "ETF": "台股大盤",
        "類型": "大盤",
        "期初收盤": np.nan,
        "復權期初": np.nan,
        "期末收盤": np.nan,
        "復權事件": "—",
        "2025績效(%)": round(float(market_2025_return_pct), 2) if market_2025_return_pct is not None else np.nan,
        "YTD報酬(%)": round(float(market_return_pct), 2) if market_return_pct is not None else np.nan,
        "贏輸台股大盤(%)": 0.0 if market_return_pct is not None else np.nan,
        "績效終點日": str(end_used),
    }
    table_df = pd.concat([pd.DataFrame([benchmark_row]), etf_df], ignore_index=True)
    if "排名" in table_df.columns:
        table_df["排名"] = table_df["排名"].map(lambda v: str(v) if pd.notna(v) else "")
    columns_order = [
        "排名",
        "代碼",
        "ETF",
        "類型",
        "期初收盤",
        "復權期初",
        "期末收盤",
        "復權事件",
        "2025績效(%)",
        "YTD報酬(%)",
        "贏輸台股大盤(%)",
    ]
    return table_df[[col for col in columns_order if col in table_df.columns]]


def _render_tw_etf_top10_page(title: str, start_yyyymmdd: str, end_yyyymmdd: str):
    st.subheader(title)
    with st.container(border=True):
        _render_card_section_header("排行卡", "依 TWSE 全市場日收盤快照計算區間報酬率。")
        try:
            top10, start_used, end_used = _build_tw_etf_top10_between(start_yyyymmdd=start_yyyymmdd, end_yyyymmdd=end_yyyymmdd)
        except Exception as exc:
            st.error(f"無法建立 ETF 排行：{exc}")
            return
        if top10.empty:
            st.warning("目前沒有可顯示的 ETF 排行資料。")
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("樣本數", str(len(top10)))
        m2.metric("市值型", str(int((top10["類型"] == "市值型").sum())))
        m3.metric("股利型", str(int((top10["類型"] == "股利型").sum())))
        m4.metric("有復權事件", str(int((top10["復權事件"] != "—").sum())))
        st.caption(f"計算區間（實際交易日）：{start_used} -> {end_used}")
        st.caption("資料來源：TWSE MI_INDEX（上市全市場快照）；已排除槓反/期貨/海外與債券商品。")
        st.caption("報酬計算：以復權期初（套用已知 split 事件）對比期末收盤。")
        st.dataframe(top10, use_container_width=True, hide_index=True)

        with st.expander("分類說明", expanded=False):
            st.markdown(
                "\n".join(
                    [
                        "- `市值型`：名稱含台灣50/中型100/市值等關鍵字。",
                        "- `股利型`：名稱含高股息/股利/收益/月配/季配等關鍵字。",
                        "- `其他`：其餘主題型、產業型、槓反型、主動式等。",
                    ]
                )
            )


def _render_top10_etf_2025_view():
    _render_tw_etf_top10_page(
        title="2025 年前十大 ETF（台股）",
        start_yyyymmdd="20241231",
        end_yyyymmdd="20251231",
    )


def _render_top10_etf_2026_ytd_view():
    title_col, refresh_col = st.columns([6, 1])
    with title_col:
        st.subheader("2026 年截至今日前十大 ETF（台股）")
    with refresh_col:
        refresh_market = st.button("更新最新市況", key="top10_etf_ytd_update_market", use_container_width=True, type="primary")
    if refresh_market:
        _fetch_twse_snapshot_with_fallback.clear()
        _build_tw_etf_top10_between.clear()
        _build_tw_active_etf_ytd_between.clear()
        _load_tw_market_return_between.clear()
        st.session_state.pop("top10_etf_ytd_compare_payload", None)
        st.rerun()

    start_target = "20251231"
    end_target = datetime.now().strftime("%Y%m%d")
    with st.container(border=True):
        _render_card_section_header("排行卡", "依 TWSE 全市場日收盤快照計算區間報酬率。")
        try:
            top10, start_used, end_used = _build_tw_etf_top10_between(start_yyyymmdd=start_target, end_yyyymmdd=end_target)
            if top10.empty:
                st.warning("目前沒有可顯示的 ETF 排行資料。")
                return
            top10_etf_df = top10.rename(columns={"區間報酬(%)": "YTD報酬(%)"}).copy()
            top10_symbols = tuple(str(x).strip().upper() for x in top10_etf_df["代碼"].astype(str).tolist() if str(x).strip())
            hist_2025_df, hist_2025_start_used, hist_2025_end_used = _build_tw_active_etf_ytd_between(
                start_yyyymmdd="20250101",
                end_yyyymmdd="20251231",
                symbols=top10_symbols,
            )
            y2025_map = {
                str(row["代碼"]).strip().upper(): float(row["YTD報酬(%)"])
                for _, row in hist_2025_df.iterrows()
                if str(row.get("代碼", "")).strip() and pd.notna(row.get("YTD報酬(%)"))
            }
        except Exception as exc:
            st.error(f"無法建立 ETF 排行：{exc}")
            return

        market_return_pct: Optional[float] = None
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

        market_2025_return_pct: Optional[float] = None
        market_2025_symbol_used = ""
        try:
            market_2025_return_pct, market_2025_symbol_used, market_2025_issues = _load_tw_market_return_between(
                start_yyyymmdd="20250101",
                end_yyyymmdd=hist_2025_end_used,
                force_sync=False,
            )
            market_issues.extend(market_2025_issues)
        except Exception as exc:
            market_issues.append(f"market_2025: {exc}")

        benchmark_code = market_symbol_used or market_2025_symbol_used or "^TWII"
        table_df = _decorate_tw_etf_top10_ytd_table(
            top10_etf_df,
            y2025_map=y2025_map,
            market_return_pct=market_return_pct,
            market_2025_return_pct=market_2025_return_pct,
            benchmark_code=benchmark_code,
            end_used=end_used,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("樣本數", str(len(top10_etf_df)))
        m2.metric("市值型", str(int((top10_etf_df.get("類型", pd.Series(dtype=str)) == "市值型").sum())))
        m3.metric("股利型", str(int((top10_etf_df.get("類型", pd.Series(dtype=str)) == "股利型").sum())))
        m4.metric("有復權事件", str(int((top10_etf_df["復權事件"] != "—").sum())))
        st.caption(f"計算區間（實際交易日）：{start_used} -> {end_used}")
        st.caption(f"2025 對照區間（實際交易日）：{hist_2025_start_used} -> {hist_2025_end_used}")
        if market_return_pct is not None and market_symbol_used:
            st.caption(f"大盤對照：{market_symbol_used} 區間報酬 {market_return_pct:.2f}%（同 2026 YTD 區間）")
        else:
            st.caption("大盤對照：目前無法取得，`贏輸台股大盤(%)` 先顯示為空白。")
        if market_issues:
            preview = [" ".join(str(item).split()) for item in market_issues[:2]]
            preview_text = " | ".join([item if len(item) <= 120 else f"{item[:117]}..." for item in preview])
            st.warning(f"更新大盤資料時有部分同步錯誤：{preview_text}")
        st.caption("資料來源：TWSE MI_INDEX（上市全市場快照）；已排除槓反/期貨/海外與債券商品。")
        st.caption("`2025績效(%)` 採各檔在 2025 區間內首個可交易日計算；若空白代表該檔 2025 區間無可用日K。")
        st.caption("報酬計算：`贏輸台股大盤(%) = YTD報酬 - 大盤報酬`。")
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        with st.expander("分類說明", expanded=False):
            st.markdown(
                "\n".join(
                    [
                        "- `市值型`：名稱含台灣50/中型100/市值等關鍵字。",
                        "- `股利型`：名稱含高股息/股利/收益/月配/季配等關鍵字。",
                        "- `其他`：其餘主題型、產業型、槓反型、主動式等。",
                    ]
                )
            )

    symbols = [str(x).strip().upper() for x in top10_etf_df["代碼"].astype(str).tolist() if str(x).strip()]
    if not symbols:
        return
    name_map = {
        str(row["代碼"]).strip().upper(): str(row["ETF"]).strip()
        for _, row in top10_etf_df.iterrows()
        if str(row.get("代碼", "")).strip()
    }

    with st.container(border=True):
        _render_card_section_header("Benchmark 對照卡", "策略曲線、基準曲線與每檔 Buy & Hold 同圖比較。")
        st.caption(
            (
                f"差異說明：上方表格的 `YTD報酬(%)` 採快照區間報酬；"
                f"本對照卡曲線與下方 `Total Return %` 會以共同比較區間 `{start_used} -> {end_used}` 對齊，"
                "因此同一檔數值可能略有差異。"
            )
        )
        st.markdown(
            (
                "說明："
                "<span title='把前十大ETF平均分配資金後，從區間起點買進並持有到期末，不做換股或調倉。'>"
                "<code>Strategy Equity（前十大ETF等權）</code>（滑鼠移入查看）"
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
            key="top10_etf_ytd_benchmark",
        )
        sync_before_run = c2.checkbox(
            "執行前同步最新日K（較慢）",
            value=False,
            key="top10_etf_ytd_sync_before_run",
        )
        force_refresh = c3.button("重新計算", use_container_width=True, key="top10_etf_ytd_refresh")

        start_dt = datetime.combine(datetime.strptime(start_used, "%Y%m%d").date(), datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(datetime.strptime(end_used, "%Y%m%d").date(), datetime.min.time()).replace(tzinfo=timezone.utc)
        run_key = f"top10_etf_ytd:{start_used}:{end_used}:{benchmark_choice}:{sync_before_run}:{','.join(symbols)}"
        payload_key = "top10_etf_ytd_compare_payload"
        payload = st.session_state.get(payload_key)
        if not isinstance(payload, dict):
            payload = {}

        should_recompute = force_refresh or payload.get("run_key") != run_key
        if should_recompute:
            store = _history_store()
            symbol_sync_issues: list[str] = []
            benchmark_sync_issues: list[str] = []

            def _benchmark_candidates_tw(choice: str) -> list[str]:
                mapping = {
                    "twii": ["^TWII"],
                    "0050": ["0050"],
                    "006208": ["006208"],
                }
                return mapping.get(str(choice or "").strip().lower(), ["^TWII"])

            with st.spinner("計算前十大 ETF Benchmark 對照中..."):
                if sync_before_run:
                    _, symbol_sync_issues = _sync_symbols_history(
                        store,
                        market="TW",
                        symbols=symbols,
                        start=start_dt,
                        end=end_dt,
                        parallel=True,
                    )

                bars_by_symbol: dict[str, pd.DataFrame] = {}
                skipped_symbols: list[str] = []
                for symbol in symbols:
                    bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt))
                    if len(bars) < 2 and not sync_before_run:
                        report = store.sync_symbol_history(symbol=symbol, market="TW", start=start_dt, end=end_dt)
                        if report.error:
                            symbol_sync_issues.append(f"{symbol}: {report.error}")
                        bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt))
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
                    payload = {
                        "run_key": run_key,
                        "error": "可用前十大 ETF 歷史資料不足，無法建立對照圖。",
                        "symbol_sync_issues": symbol_sync_issues,
                        "benchmark_sync_issues": benchmark_sync_issues,
                    }
                    st.session_state[payload_key] = payload
                else:
                    target_index = pd.DatetimeIndex([])
                    for bars in bars_by_symbol.values():
                        target_index = target_index.union(pd.DatetimeIndex(bars.index))
                    target_index = target_index.sort_values()

                    initial_capital = 1_000_000.0
                    strategy_equity = build_buy_hold_equity(
                        bars_by_symbol=bars_by_symbol,
                        target_index=target_index,
                        initial_capital=initial_capital,
                    ).dropna()

                    per_symbol_equity: dict[str, pd.Series] = {}
                    for symbol, bars in bars_by_symbol.items():
                        eq_sym = build_buy_hold_equity(
                            bars_by_symbol={symbol: bars},
                            target_index=target_index,
                            initial_capital=initial_capital,
                        ).dropna()
                        if not eq_sym.empty:
                            per_symbol_equity[symbol] = eq_sym

                    benchmark_symbol_used = ""
                    benchmark_equity = pd.Series(dtype=float)
                    for candidate in _benchmark_candidates_tw(benchmark_choice):
                        if sync_before_run:
                            report = store.sync_symbol_history(symbol=candidate, market="TW", start=start_dt, end=end_dt)
                            if report.error:
                                benchmark_sync_issues.append(f"{candidate}: {report.error}")
                        bench_bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=candidate, market="TW", start=start_dt, end=end_dt))
                        if bench_bars.empty and not sync_before_run:
                            report = store.sync_symbol_history(symbol=candidate, market="TW", start=start_dt, end=end_dt)
                            if report.error:
                                benchmark_sync_issues.append(f"{candidate}: {report.error}")
                            bench_bars = _normalize_ohlcv_frame(
                                store.load_daily_bars(symbol=candidate, market="TW", start=start_dt, end=end_dt)
                            )
                        if bench_bars.empty:
                            continue
                        bench_bars, _ = apply_split_adjustment(
                            bars=bench_bars,
                            symbol=candidate,
                            market="TW",
                            use_known=True,
                            use_auto_detect=True,
                        )
                        if bench_bars.empty or "close" not in bench_bars.columns:
                            continue
                        bench_close = pd.to_numeric(bench_bars["close"], errors="coerce").dropna().sort_index()
                        if len(bench_close) < 2:
                            continue
                        aligned = bench_close.reindex(target_index).ffill()
                        valid = aligned.dropna()
                        if len(valid) < 2:
                            continue
                        base_val = float(valid.iloc[0])
                        if not math.isfinite(base_val) or base_val <= 0:
                            continue
                        benchmark_equity = (aligned.loc[valid.index[0] :] / base_val) * initial_capital
                        benchmark_equity = benchmark_equity.dropna()
                        benchmark_symbol_used = candidate
                        break

                    common_index = pd.DatetimeIndex(strategy_equity.index)
                    if not benchmark_equity.empty:
                        common_index = common_index.intersection(pd.DatetimeIndex(benchmark_equity.index))
                    common_index = common_index.sort_values()
                    if len(common_index) < 2:
                        payload = {
                            "run_key": run_key,
                            "error": "Strategy 與 Benchmark 缺少足夠重疊交易日，無法建立對照圖。",
                            "symbol_sync_issues": symbol_sync_issues,
                            "benchmark_sync_issues": benchmark_sync_issues,
                        }
                        st.session_state[payload_key] = payload
                    else:
                        strategy_plot = strategy_equity.reindex(common_index).ffill().dropna()
                        benchmark_plot = benchmark_equity.reindex(common_index).ffill().dropna() if not benchmark_equity.empty else pd.Series(dtype=float)
                        per_symbol_plot: dict[str, pd.Series] = {}
                        for symbol, series in per_symbol_equity.items():
                            aligned = series.reindex(common_index).ffill().dropna()
                            if len(aligned) >= 2:
                                per_symbol_plot[symbol] = aligned

                        payload = {
                            "run_key": run_key,
                            "error": "",
                            "benchmark_symbol": benchmark_symbol_used,
                            "strategy_equity": strategy_plot,
                            "benchmark_equity": benchmark_plot,
                            "per_symbol_equity": per_symbol_plot,
                            "used_symbols": sorted(list(bars_by_symbol.keys())),
                            "skipped_symbols": sorted(skipped_symbols),
                            "symbol_sync_issues": symbol_sync_issues,
                            "benchmark_sync_issues": benchmark_sync_issues,
                        }
                        st.session_state[payload_key] = payload

        payload = st.session_state.get(payload_key, {})
        if not isinstance(payload, dict):
            payload = {}
        error_text = str(payload.get("error", "")).strip()
        if error_text:
            st.warning(error_text)
            return

        symbol_sync_issues = payload.get("symbol_sync_issues", [])
        if isinstance(symbol_sync_issues, list) and symbol_sync_issues:
            preview = [" ".join(str(item).split()) for item in symbol_sync_issues[:3]]
            preview_text = " | ".join([item if len(item) <= 120 else f"{item[:117]}..." for item in preview])
            remain = len(symbol_sync_issues) - len(preview)
            remain_text = f" | 其餘 {remain} 筆請查看終端 log。" if remain > 0 else ""
            st.warning(f"部分 ETF 同步失敗，已盡量使用本地可用資料：{preview_text}{remain_text}")

        benchmark_sync_issues = payload.get("benchmark_sync_issues", [])
        if isinstance(benchmark_sync_issues, list) and benchmark_sync_issues:
            preview = [" ".join(str(item).split()) for item in benchmark_sync_issues[:2]]
            preview_text = " | ".join([item if len(item) <= 120 else f"{item[:117]}..." for item in preview])
            st.warning(f"Benchmark 同步有部分錯誤，已盡量使用本地可用資料：{preview_text}")

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

        palette = _ui_palette()
        symbol_styles = _build_symbol_line_styles(list(per_symbol_equity.keys()))
        fig_cmp = go.Figure()
        fig_cmp.add_trace(
            go.Scatter(
                x=strategy_equity.index,
                y=strategy_equity.values,
                mode="lines",
                name="Strategy Equity（前十大ETF等權）",
                line=dict(color=str(palette["equity"]), width=2.4),
                hovertemplate=_hovertemplate_with_code("TOP10_EW", value_label="Equity", y_format=",.0f"),
            )
        )
        if not benchmark_equity.empty:
            benchmark_label = str(payload.get("benchmark_symbol", "") or "Benchmark")
            fig_cmp.add_trace(
                go.Scatter(
                    x=benchmark_equity.index,
                    y=benchmark_equity.values,
                    mode="lines",
                    name=f"Benchmark Equity（{benchmark_label}）",
                    line=_benchmark_line_style(palette, width=2.0),
                    hovertemplate=_hovertemplate_with_code(benchmark_label, value_label="Equity", y_format=",.0f"),
                )
            )

        for symbol in sorted(per_symbol_equity.keys()):
            series = per_symbol_equity[symbol]
            if not isinstance(series, pd.Series) or len(series) < 2:
                continue
            style = symbol_styles.get(symbol, {"color": "#1f77b4", "dash": "solid"})
            label_name = str(name_map.get(symbol, symbol)).strip()
            trace_name = f"Buy-and-Hold（{symbol} {label_name}）" if label_name and label_name != symbol else f"Buy-and-Hold（{symbol}）"
            fig_cmp.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=trace_name,
                    line=dict(color=str(style["color"]), width=1.8, dash=str(style["dash"])),
                    hovertemplate=_hovertemplate_with_code(symbol, value_label="Equity", y_format=",.0f"),
                )
            )

        fig_cmp.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=30, b=10),
            template=str(palette["plot_template"]),
            paper_bgcolor=str(palette["paper_bg"]),
            plot_bgcolor=str(palette["plot_bg"]),
            font=dict(color=str(palette["text_color"])),
        )
        fig_cmp.update_xaxes(gridcolor=str(palette["grid"]))
        fig_cmp.update_yaxes(gridcolor=str(palette["grid"]))
        st.plotly_chart(fig_cmp, use_container_width=True)

        summary_rows: list[dict[str, object]] = []
        strategy_perf = _series_metrics_basic(strategy_equity)
        summary_rows.append(
            {
                "Series": "Strategy Equity（前十大ETF等權）",
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
            series_name = f"Buy-and-Hold（{symbol} {label_name}）" if label_name and label_name != symbol else f"Buy-and-Hold（{symbol}）"
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
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        used_symbols = payload.get("used_symbols", [])
        skipped_symbols = payload.get("skipped_symbols", [])
        if isinstance(used_symbols, list) and isinstance(skipped_symbols, list):
            st.caption(f"可用ETF：{len(used_symbols)} 檔 | 資料不足未納入：{len(skipped_symbols)} 檔")

def _render_active_etf_2026_ytd_view():
    title_col, refresh_col = st.columns([6, 1])
    with title_col:
        st.subheader("2026 年主動式 ETF YTD（Buy & Hold）")
    with refresh_col:
        refresh_market = st.button("更新最新市況", key="active_etf_ytd_update_market", use_container_width=True, type="primary")
    if refresh_market:
        _fetch_twse_snapshot_with_fallback.clear()
        _build_tw_active_etf_ytd_between.clear()
        _load_tw_market_return_between.clear()
        st.session_state.pop("active_etf_ytd_compare_payload", None)
        st.session_state.pop("active_etf_2025_compare_payload", None)
        st.rerun()

    start_target = "20260101"
    end_target = datetime.now().strftime("%Y%m%d")

    with st.container(border=True):
        _render_card_section_header("主動式 ETF 績效卡", "台股主動式 ETF，2026 YTD Buy & Hold（復權版）。")
        try:
            etf_df, start_used, end_used = _build_tw_active_etf_ytd_between(start_yyyymmdd=start_target, end_yyyymmdd=end_target)
            if etf_df.empty:
                st.warning("目前沒有可顯示的台股主動式 ETF YTD 資料。")
                return
            etf_symbols = tuple(
                str(x).strip().upper()
                for x in etf_df["代碼"].astype(str).tolist()
                if str(x).strip() and not str(x).strip().upper().startswith("^")
            )
            hist_2025_df, hist_2025_start_used, hist_2025_end_used = _build_tw_active_etf_ytd_between(
                start_yyyymmdd="20250101",
                end_yyyymmdd="20251231",
                symbols=etf_symbols,
            )
            y2025_map = {
                str(row["代碼"]).strip().upper(): float(row["YTD報酬(%)"])
                for _, row in hist_2025_df.iterrows()
                if str(row.get("代碼", "")).strip() and pd.notna(row.get("YTD報酬(%)"))
            }
        except Exception as exc:
            st.error(f"無法建立主動式 ETF YTD 清單：{exc}")
            return
        market_return_pct: Optional[float] = None
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

        market_2025_return_pct: Optional[float] = None
        market_2025_symbol_used = ""
        try:
            market_2025_return_pct, market_2025_symbol_used, market_2025_issues = _load_tw_market_return_between(
                start_yyyymmdd="20250101",
                end_yyyymmdd=hist_2025_end_used,
                force_sync=False,
            )
            market_issues.extend(market_2025_issues)
        except Exception as exc:
            market_issues.append(f"market_2025: {exc}")

        etf_df["2025績效(%)"] = etf_df["代碼"].astype(str).str.strip().str.upper().map(y2025_map)
        etf_df["2025績效(%)"] = pd.to_numeric(etf_df["2025績效(%)"], errors="coerce").round(2)
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
            "期初收盤": np.nan,
            "復權期初": np.nan,
            "期末收盤": np.nan,
            "績效起算日": "—",
            "績效終點日": end_used,
            "復權事件": "—",
            "2025績效(%)": round(float(market_2025_return_pct), 2) if market_2025_return_pct is not None else np.nan,
            "YTD報酬(%)": round(float(market_return_pct), 2) if market_return_pct is not None else np.nan,
            "贏輸台股大盤(%)": 0.0 if market_return_pct is not None else np.nan,
        }
        table_df = pd.concat([pd.DataFrame([benchmark_row]), etf_df], ignore_index=True)
        if "排名" in table_df.columns:
            table_df["排名"] = table_df["排名"].map(lambda v: str(v) if pd.notna(v) else "")
        columns_order = [
            "排名",
            "代碼",
            "ETF",
            "期初收盤",
            "復權期初",
            "期末收盤",
            "績效起算日",
            "績效終點日",
            "復權事件",
            "2025績效(%)",
            "YTD報酬(%)",
            "贏輸台股大盤(%)",
        ]
        table_df = table_df[[col for col in columns_order if col in table_df.columns]]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("樣本數", str(len(etf_df)))
        m2.metric("正報酬檔數", str(int((etf_df["YTD報酬(%)"] > 0).sum())))
        m3.metric("負報酬檔數", str(int((etf_df["YTD報酬(%)"] < 0).sum())))
        m4.metric("有復權事件", str(int((etf_df["復權事件"] != "—").sum())))
        st.caption(f"計算區間（實際交易日）：{start_used} -> {end_used}")
        st.caption(f"2025 對照區間（實際交易日）：{hist_2025_start_used} -> {hist_2025_end_used}")
        if market_return_pct is not None and market_symbol_used:
            st.caption(f"大盤對照：{market_symbol_used} 區間報酬 {market_return_pct:.2f}%（同 2026 YTD 區間）")
        else:
            st.caption("大盤對照：目前無法取得，`贏輸台股大盤(%)` 先顯示為空白。")
        if market_issues:
            preview = [" ".join(str(item).split()) for item in market_issues[:2]]
            preview_text = " | ".join([item if len(item) <= 120 else f"{item[:117]}..." for item in preview])
            st.warning(f"更新大盤資料時有部分同步錯誤：{preview_text}")
        st.caption("資料來源：TWSE MI_INDEX（上市全市場快照）。主動式規則：代碼結尾 A 且名稱含「主動」。")
        st.caption("`2025績效(%)` 採各檔在 2025 區間內首個可交易日計算；若空白代表該檔 2025 區間無可用日K。")
        st.caption("報酬計算：Buy & Hold（復權版，套用已知 split 事件）；`贏輸台股大盤(%) = YTD報酬 - 大盤報酬`。")
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    symbols = [str(x).strip().upper() for x in etf_df["代碼"].astype(str).tolist() if str(x).strip()]
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
            _render_card_section_header(card_title, "策略曲線、基準曲線與每檔 Buy & Hold 同圖比較。")
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
                format_func=lambda x: {"twii": "^TWII", "0050": "0050", "006208": "006208"}.get(x, x),
                key=f"{key_prefix}_benchmark",
            )
            sync_before_run = c2.checkbox(
                "執行前同步最新日K（較慢）",
                value=False,
                key=f"{key_prefix}_sync_before_run",
            )
            force_refresh = c3.button("重新計算", use_container_width=True, key=f"{key_prefix}_refresh")

            try:
                start_dt = datetime.combine(datetime.strptime(date_start, "%Y%m%d").date(), datetime.min.time()).replace(tzinfo=timezone.utc)
                end_dt = datetime.combine(datetime.strptime(date_end, "%Y%m%d").date(), datetime.min.time()).replace(tzinfo=timezone.utc)
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
                store = _history_store()
                symbol_sync_issues: list[str] = []
                benchmark_sync_issues: list[str] = []

                def _benchmark_candidates_tw(choice: str) -> list[str]:
                    mapping = {
                        "twii": ["^TWII"],
                        "0050": ["0050"],
                        "006208": ["006208"],
                    }
                    return mapping.get(str(choice or "").strip().lower(), ["^TWII"])

                with st.spinner(spinner_text):
                    if sync_before_run:
                        _, symbol_sync_issues = _sync_symbols_history(
                            store,
                            market="TW",
                            symbols=symbols,
                            start=start_dt,
                            end=end_dt,
                            parallel=True,
                        )

                    bars_by_symbol: dict[str, pd.DataFrame] = {}
                    skipped_symbols: list[str] = []
                    for symbol in symbols:
                        bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt))
                        if len(bars) < 2 and not sync_before_run:
                            report = store.sync_symbol_history(symbol=symbol, market="TW", start=start_dt, end=end_dt)
                            if report.error:
                                symbol_sync_issues.append(f"{symbol}: {report.error}")
                            bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt))
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
                        payload = {
                            "run_key": run_key,
                            "error": "可用主動式 ETF 歷史資料不足，無法建立對照圖。",
                            "symbol_sync_issues": symbol_sync_issues,
                            "benchmark_sync_issues": benchmark_sync_issues,
                        }
                        st.session_state[payload_key] = payload
                    else:
                        target_index = pd.DatetimeIndex([])
                        for bars in bars_by_symbol.values():
                            target_index = target_index.union(pd.DatetimeIndex(bars.index))
                        target_index = target_index.sort_values()

                        initial_capital = 1_000_000.0
                        strategy_equity = build_buy_hold_equity(
                            bars_by_symbol=bars_by_symbol,
                            target_index=target_index,
                            initial_capital=initial_capital,
                        ).dropna()

                        per_symbol_equity: dict[str, pd.Series] = {}
                        for symbol, bars in bars_by_symbol.items():
                            eq_sym = build_buy_hold_equity(
                                bars_by_symbol={symbol: bars},
                                target_index=target_index,
                                initial_capital=initial_capital,
                            ).dropna()
                            if not eq_sym.empty:
                                per_symbol_equity[symbol] = eq_sym

                        benchmark_symbol_used = ""
                        benchmark_equity = pd.Series(dtype=float)
                        for candidate in _benchmark_candidates_tw(benchmark_choice):
                            if sync_before_run:
                                report = store.sync_symbol_history(symbol=candidate, market="TW", start=start_dt, end=end_dt)
                                if report.error:
                                    benchmark_sync_issues.append(f"{candidate}: {report.error}")
                            bench_bars = _normalize_ohlcv_frame(store.load_daily_bars(symbol=candidate, market="TW", start=start_dt, end=end_dt))
                            if bench_bars.empty and not sync_before_run:
                                report = store.sync_symbol_history(symbol=candidate, market="TW", start=start_dt, end=end_dt)
                                if report.error:
                                    benchmark_sync_issues.append(f"{candidate}: {report.error}")
                                bench_bars = _normalize_ohlcv_frame(
                                    store.load_daily_bars(symbol=candidate, market="TW", start=start_dt, end=end_dt)
                                )
                            if bench_bars.empty:
                                continue
                            bench_bars, _ = apply_split_adjustment(
                                bars=bench_bars,
                                symbol=candidate,
                                market="TW",
                                use_known=True,
                                use_auto_detect=True,
                            )
                            if bench_bars.empty or "close" not in bench_bars.columns:
                                continue
                            bench_close = pd.to_numeric(bench_bars["close"], errors="coerce").dropna().sort_index()
                            if len(bench_close) < 2:
                                continue
                            aligned = bench_close.reindex(target_index).ffill()
                            valid = aligned.dropna()
                            if len(valid) < 2:
                                continue
                            base_val = float(valid.iloc[0])
                            if not math.isfinite(base_val) or base_val <= 0:
                                continue
                            benchmark_equity = (aligned.loc[valid.index[0] :] / base_val) * initial_capital
                            benchmark_equity = benchmark_equity.dropna()
                            benchmark_symbol_used = candidate
                            break

                        common_index = pd.DatetimeIndex(strategy_equity.index)
                        if not benchmark_equity.empty:
                            common_index = common_index.intersection(pd.DatetimeIndex(benchmark_equity.index))
                        common_index = common_index.sort_values()
                        if len(common_index) < 2:
                            payload = {
                                "run_key": run_key,
                                "error": "Strategy 與 Benchmark 缺少足夠重疊交易日，無法建立對照圖。",
                                "symbol_sync_issues": symbol_sync_issues,
                                "benchmark_sync_issues": benchmark_sync_issues,
                            }
                            st.session_state[payload_key] = payload
                        else:
                            strategy_plot = strategy_equity.reindex(common_index).ffill().dropna()
                            benchmark_plot = (
                                benchmark_equity.reindex(common_index).ffill().dropna() if not benchmark_equity.empty else pd.Series(dtype=float)
                            )
                            per_symbol_plot: dict[str, pd.Series] = {}
                            for symbol, series in per_symbol_equity.items():
                                aligned = series.reindex(common_index).ffill().dropna()
                                if len(aligned) >= 2:
                                    per_symbol_plot[symbol] = aligned

                            payload = {
                                "run_key": run_key,
                                "error": "",
                                "benchmark_symbol": benchmark_symbol_used,
                                "strategy_equity": strategy_plot,
                                "benchmark_equity": benchmark_plot,
                                "per_symbol_equity": per_symbol_plot,
                                "used_symbols": sorted(list(bars_by_symbol.keys())),
                                "skipped_symbols": sorted(skipped_symbols),
                                "symbol_sync_issues": symbol_sync_issues,
                                "benchmark_sync_issues": benchmark_sync_issues,
                            }
                            st.session_state[payload_key] = payload

            payload = st.session_state.get(payload_key, {})
            if not isinstance(payload, dict):
                payload = {}
            error_text = str(payload.get("error", "")).strip()
            if error_text:
                st.warning(error_text)
                return

            symbol_sync_issues = payload.get("symbol_sync_issues", [])
            if isinstance(symbol_sync_issues, list) and symbol_sync_issues:
                preview = [" ".join(str(item).split()) for item in symbol_sync_issues[:3]]
                preview_text = " | ".join([item if len(item) <= 120 else f"{item[:117]}..." for item in preview])
                remain = len(symbol_sync_issues) - len(preview)
                remain_text = f" | 其餘 {remain} 筆請查看終端 log。" if remain > 0 else ""
                st.warning(f"部分 ETF 同步失敗，已盡量使用本地可用資料：{preview_text}{remain_text}")

            benchmark_sync_issues = payload.get("benchmark_sync_issues", [])
            if isinstance(benchmark_sync_issues, list) and benchmark_sync_issues:
                preview = [" ".join(str(item).split()) for item in benchmark_sync_issues[:2]]
                preview_text = " | ".join([item if len(item) <= 120 else f"{item[:117]}..." for item in preview])
                st.warning(f"Benchmark 同步有部分錯誤，已盡量使用本地可用資料：{preview_text}")

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

            palette = _ui_palette()
            symbol_styles = _build_symbol_line_styles(list(per_symbol_equity.keys()))
            fig_cmp = go.Figure()
            fig_cmp.add_trace(
                go.Scatter(
                    x=strategy_equity.index,
                    y=strategy_equity.values,
                    mode="lines",
                    name="Strategy Equity（主動式ETF等權）",
                    line=dict(color=str(palette["equity"]), width=2.4),
                    hovertemplate=_hovertemplate_with_code(f"{key_prefix.upper()}_EW", value_label="Equity", y_format=",.0f"),
                )
            )
            if not benchmark_equity.empty:
                benchmark_label = str(payload.get("benchmark_symbol", "") or "Benchmark")
                fig_cmp.add_trace(
                    go.Scatter(
                        x=benchmark_equity.index,
                        y=benchmark_equity.values,
                        mode="lines",
                        name=f"Benchmark Equity（{benchmark_label}）",
                        line=_benchmark_line_style(palette, width=2.0),
                        hovertemplate=_hovertemplate_with_code(benchmark_label, value_label="Equity", y_format=",.0f"),
                    )
                )

            for symbol in sorted(per_symbol_equity.keys()):
                series = per_symbol_equity[symbol]
                if not isinstance(series, pd.Series) or len(series) < 2:
                    continue
                style = symbol_styles.get(symbol, {"color": "#1f77b4", "dash": "solid"})
                label_name = str(name_map.get(symbol, symbol)).strip()
                trace_name = (
                    f"Buy-and-Hold（{symbol} {label_name}）" if label_name and label_name != symbol else f"Buy-and-Hold（{symbol}）"
                )
                fig_cmp.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode="lines",
                        name=trace_name,
                        line=dict(color=str(style["color"]), width=1.8, dash=str(style["dash"])),
                        hovertemplate=_hovertemplate_with_code(symbol, value_label="Equity", y_format=",.0f"),
                    )
                )

            fig_cmp.update_layout(
                height=460,
                margin=dict(l=10, r=10, t=30, b=10),
                template=str(palette["plot_template"]),
                paper_bgcolor=str(palette["paper_bg"]),
                plot_bgcolor=str(palette["plot_bg"]),
                font=dict(color=str(palette["text_color"])),
            )
            fig_cmp.update_xaxes(gridcolor=str(palette["grid"]))
            fig_cmp.update_yaxes(gridcolor=str(palette["grid"]))
            st.plotly_chart(fig_cmp, use_container_width=True)

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
                    f"Buy-and-Hold（{symbol} {label_name}）" if label_name and label_name != symbol else f"Buy-and-Hold（{symbol}）"
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
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            used_symbols = payload.get("used_symbols", [])
            skipped_symbols = payload.get("skipped_symbols", [])
            if isinstance(used_symbols, list) and isinstance(skipped_symbols, list):
                st.caption(f"可用ETF：{len(used_symbols)} 檔 | 資料不足未納入：{len(skipped_symbols)} 檔")

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
    source_chain = [str(s or "").strip() for s in list(getattr(ctx, "source_chain", [])) if str(s or "").strip()]
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

    freshness = "—" if quality.freshness_sec is None else f"{quality.freshness_sec}s"
    st.caption(
        f"資料品質：source={quote.source} | delayed={'yes' if quote.is_delayed else 'no'} | "
        f"fallback_depth={quality.fallback_depth} | freshness={freshness} | refresh={refresh_sec}s"
    )
    st.caption(f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（fragment 局部刷新）")


def _render_live_chart(ind: pd.DataFrame):
    palette = _ui_palette()
    price_up_line = str(palette.get("price_up_line", palette["price_up"]))
    price_down_line = str(palette.get("price_down_line", palette["price_down"]))
    price_up_fill = str(palette.get("price_up_fill", _to_rgba(price_up_line, 0.42)))
    price_down_fill = str(palette.get("price_down_fill", _to_rgba(price_down_line, 0.42)))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.72, 0.28])
    fig.add_trace(
        go.Candlestick(
            x=ind.index,
            open=ind["open"],
            high=ind["high"],
            low=ind["low"],
            close=ind["close"],
            name="K線",
            increasing_line_color=price_up_line,
            increasing_fillcolor=price_up_fill,
            decreasing_line_color=price_down_line,
            decreasing_fillcolor=price_down_fill,
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


def _render_indicator_panels(
    ind: pd.DataFrame,
    *,
    chart_key: str,
    height: int = 420,
    x_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
):
    if not isinstance(ind, pd.DataFrame) or ind.empty or "close" not in ind.columns:
        st.caption("指標副圖：資料不足。")
        return

    frame = ind.copy().sort_index()
    for col in [
        "close",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "sma_20",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "macd",
        "macd_signal",
        "macd_hist",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    palette = _ui_palette()
    rsi_color = str(palette.get("rsi_line", palette["sma20"]))
    kd_k_color = str(palette.get("kd_k", palette["signal_buy"]))
    kd_d_color = str(palette.get("kd_d", palette["signal_sell"]))
    macd_line_color = str(palette.get("macd_line", palette["equity"]))
    macd_signal_color = str(palette.get("macd_signal", palette["buy_hold"]))
    panel_height = max(280, min(int(height), 520))
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.36, 0.18, 0.18, 0.28],
    )

    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            mode="lines",
            name="Close",
            line=dict(color=str(palette["text_muted"]), width=1.35),
        ),
        row=1,
        col=1,
    )
    bb_mid_col = "bb_mid" if "bb_mid" in frame.columns else ("sma_20" if "sma_20" in frame.columns else "")
    if bb_mid_col:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[bb_mid_col],
                mode="lines",
                name="BB中軌",
                line=dict(color=str(palette["sma20"]), width=1.0),
            ),
            row=1,
            col=1,
        )
    if "bb_upper" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["bb_upper"],
                mode="lines",
                name="BB上軌",
                line=dict(color=str(palette["bb_upper"]), width=1.0),
            ),
            row=1,
            col=1,
        )
    if "bb_lower" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["bb_lower"],
                mode="lines",
                name="BB下軌",
                line=dict(color=str(palette["bb_lower"]), width=1.0),
            ),
            row=1,
            col=1,
        )

    if "rsi_14" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["rsi_14"],
                mode="lines",
                name="RSI14",
                line=dict(color=rsi_color, width=1.2),
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=70, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    if "stoch_k" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["stoch_k"],
                mode="lines",
                name="KD-K",
                line=dict(color=kd_k_color, width=1.2),
            ),
            row=3,
            col=1,
        )
    if "stoch_d" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["stoch_d"],
                mode="lines",
                name="KD-D",
                line=dict(color=kd_d_color, width=1.2),
            ),
            row=3,
            col=1,
        )
    fig.add_hline(y=80, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=20, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    if "macd_hist" in frame.columns:
        hist_vals = pd.to_numeric(frame["macd_hist"], errors="coerce").fillna(0.0)
        hist_colors = np.where(hist_vals >= 0, str(palette["volume_up"]), str(palette["volume_down"]))
        fig.add_trace(
            go.Bar(x=frame.index, y=hist_vals.values, name="MACD Hist", marker_color=hist_colors),
            row=4,
            col=1,
        )
    if "macd" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["macd"],
                mode="lines",
                name="MACD",
                line=dict(color=macd_line_color, width=1.2),
            ),
            row=4,
            col=1,
        )
    if "macd_signal" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["macd_signal"],
                mode="lines",
                name="MACD Signal",
                line=dict(color=macd_signal_color, width=1.1),
            ),
            row=4,
            col=1,
        )
    fig.add_hline(y=0, line=dict(color=str(palette["text_muted"]), width=1, dash="dot"), row=4, col=1)

    if x_range is not None:
        x0, x1 = x_range
        for row in (1, 2, 3, 4):
            fig.update_xaxes(range=[x0, x1], row=row, col=1)

    fig.update_xaxes(gridcolor=str(palette["grid"]))
    fig.update_yaxes(gridcolor=str(palette["grid"]))
    fig.update_layout(
        height=panel_height,
        margin=dict(l=10, r=10, t=18, b=8),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"])),
        legend_orientation="h",
        legend_y=1.02,
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _render_live_view():
    service = _market_service()
    store = _history_store()

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
                        st.caption("即時走勢：本輪改用本地 SQLite 即時快取。")
                    else:
                        st.caption(f"即時走勢：K數偏少，已用本地 SQLite 補齊（{before} -> {len(bars_intraday)}）。")
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
                _render_live_chart(ind)
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
                    st.dataframe(ind.iloc[-1][show_cols].to_frame("value").T, use_container_width=True)

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


def _frame_from_split_payload(payload: object, *, default_columns: Optional[list[str]] = None) -> pd.DataFrame:
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


def _deserialize_trade(payload: object) -> Optional[Trade]:
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
        "yearly_returns": {str(k): _cache_float(v) for k, v in (result.yearly_returns or {}).items()},
        "signals": _series_to_split_payload(result.signals),
    }


def _deserialize_single_backtest_result(payload: object) -> Optional[BacktestResult]:
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
        equity_curve=_frame_from_split_payload(data.get("equity_curve"), default_columns=["equity"]),
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
        "yearly_returns": {str(k): _cache_float(v) for k, v in (result.yearly_returns or {}).items()},
        "trades": _frame_to_split_payload(result.trades),
        "signals": _frame_to_split_payload(result.signals),
        "component_results": {str(sym): _serialize_single_backtest_result(comp) for sym, comp in result.component_results.items()},
    }


def _deserialize_portfolio_backtest_result(payload: object) -> Optional[PortfolioBacktestResult]:
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
        equity_curve=_frame_from_split_payload(data.get("equity_curve"), default_columns=["equity"]),
        metrics=_deserialize_backtest_metrics(data.get("metrics")),
        drawdown_series=_series_from_split_payload(data.get("drawdown_series")),
        yearly_returns=yearly_returns,
        trades=_frame_from_split_payload(data.get("trades")),
        signals=_frame_from_split_payload(data.get("signals")),
        component_results=component_results,
    )


def _serialize_backtest_run_payload(run_payload: dict[str, object]) -> dict[str, object]:
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
        serialized["split_date"] = pd.Timestamp(run_payload.get("split_date")).isoformat() if run_payload.get("split_date") is not None else ""
        best_params = run_payload.get("best_params", {})
        serialized["best_params"] = best_params if isinstance(best_params, dict) else {}
        candidates = run_payload.get("candidates", {})
        if isinstance(candidates, (dict, int, float)):
            serialized["candidates"] = candidates
    return serialized


def _deserialize_backtest_run_payload(payload: object) -> Optional[dict[str, object]]:
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
        if any(sym not in result.component_results for sym in bars_by_symbol.keys()):
            return None

    run_payload: dict[str, object] = {
        "mode": mode,
        "walk_forward": bool(data.get("walk_forward")),
        "initial_capital": _cache_float(data.get("initial_capital"), default=1_000_000.0),
        "bars_by_symbol": bars_by_symbol,
        "result": result,
    }
    if mode == "single":
        run_payload["symbol"] = str(data.get("symbol", "")).strip().upper() or next(iter(bars_by_symbol.keys()))
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
            out.attrs["source"] = f"sqlite:{source_text}" if source_text else "sqlite"
            return out

        return pd.DataFrame(columns=["close"])

    store = _history_store()
    service = _market_service()

    _render_card_section_header("回測設定卡", "先設定基本條件，再調整策略/成本與進階回放選項。")
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
    strategy = c4.selectbox(
        "策略",
        options=DAILY_STRATEGY_OPTIONS,
        index=1,
        format_func=_strategy_label,
    )
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
        strategy_params = {"entry_n": float(entry_n), "exit_n": float(exit_n), "trend": float(trend)}
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
    adj1, adj2, adj3 = st.columns(3)
    use_split_adjustment = adj1.checkbox("分割調整（復權）", value=True)
    auto_detect_split = adj2.checkbox("自動偵測分割事件", value=True)
    use_total_return_adjustment = adj3.checkbox(
        "還原權息計算（Adj Close）",
        value=False,
        help="有 `adj_close` 時，會將 OHLC 轉成還原權息價格。若已還原，會略過額外分割調整避免重複計算。",
    )
    if use_total_return_adjustment and use_split_adjustment:
        st.caption("已啟用還原權息：若標的 `adj_close` 覆蓋率足夠，將優先使用還原權息並略過分割調整。")
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

    sync_parallel = st.checkbox(
        "多標的同步平行處理",
        value=True,
        help="多檔標的時通常更快；若網路不穩可關閉改為逐檔同步。",
    )
    auto_sync = st.checkbox(
        "App 啟動時自動增量同步（較慢）",
        value=False,
        help="關閉可先直接使用本地 SQLite；需要時再手動同步。",
    )
    auto_fill_gaps = st.checkbox(
        "回測前自動補資料缺口（推薦）",
        value=True,
        help="若發現本地資料起訖缺口，會只同步缺口標的；可減少「回測天數不夠」問題。",
    )
    sync_key = f"synced:{market}:{','.join(symbols)}:{start_date}:{end_date}"
    gapfill_key = f"gapfill:{market}:{','.join(symbols)}:{start_date}:{end_date}"
    sync_start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    sync_end = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    if auto_sync and not st.session_state.get(sync_key):
        reports, sync_issues = _sync_symbols_history(
            store,
            market=market,
            symbols=symbols,
            start=sync_start,
            end=sync_end,
            parallel=sync_parallel,
        )
        sync_rows = []
        for symbol in symbols:
            report = reports.get(symbol)
            sync_rows.append(
                {
                    "symbol": symbol,
                    "rows": int(getattr(report, "rows_upserted", 0) or 0),
                    "source": str(getattr(report, "source", "unknown") or "unknown"),
                    "fallback": int(getattr(report, "fallback_depth", 0) or 0),
                    "error": str(getattr(report, "error", "") or ""),
                }
            )
        st.session_state[sync_key] = True
        sync_df = pd.DataFrame(sync_rows)
        if sync_issues:
            st.warning("部分同步失敗，請檢查下方同步結果。")
        st.dataframe(sync_df, use_container_width=True, hide_index=True)

    if auto_fill_gaps and not st.session_state.get(gapfill_key):
        gap_symbols: list[str] = []
        for symbol in symbols:
            bars = store.load_daily_bars(symbol=symbol, market=market, start=sync_start, end=sync_end)
            if _bars_need_backfill(bars, start=sync_start, end=sync_end):
                gap_symbols.append(symbol)
        if gap_symbols:
            reports, sync_issues = _sync_symbols_history(
                store,
                market=market,
                symbols=gap_symbols,
                start=sync_start,
                end=sync_end,
                parallel=sync_parallel,
            )
            sync_rows = []
            for symbol in gap_symbols:
                report = reports.get(symbol)
                sync_rows.append(
                    {
                        "symbol": symbol,
                        "rows": int(getattr(report, "rows_upserted", 0) or 0),
                        "source": str(getattr(report, "source", "unknown") or "unknown"),
                        "fallback": int(getattr(report, "fallback_depth", 0) or 0),
                        "error": str(getattr(report, "error", "") or ""),
                    }
                )
            if sync_issues:
                st.warning("缺口補齊時有部分同步失敗，請檢查下方結果。")
            else:
                st.caption("已自動補齊回測區間資料缺口。")
            st.dataframe(pd.DataFrame(sync_rows), use_container_width=True, hide_index=True)
        st.session_state[gapfill_key] = True

    if st.button("同步歷史資料", use_container_width=True):
        reports, sync_issues = _sync_symbols_history(
            store,
            market=market,
            symbols=symbols,
            start=sync_start,
            end=sync_end,
            parallel=sync_parallel,
        )
        sync_rows = []
        for symbol in symbols:
            report = reports.get(symbol)
            sync_rows.append(
                {
                    "symbol": symbol,
                    "rows": int(getattr(report, "rows_upserted", 0) or 0),
                    "source": str(getattr(report, "source", "unknown") or "unknown"),
                    "fallback": int(getattr(report, "fallback_depth", 0) or 0),
                    "error": str(getattr(report, "error", "") or ""),
                }
            )
        sync_df = pd.DataFrame(sync_rows)
        if sync_issues:
            st.error("部分同步失敗，請檢查同步結果。")
        else:
            st.success("同步成功。")
        st.dataframe(sync_df, use_container_width=True, hide_index=True)

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    availability_rows = []
    for symbol in symbols:
        bars = store.load_daily_bars(symbol=symbol, market=market, start=sync_start, end=sync_end)
        if bars.empty:
            availability_rows.append({"symbol": symbol, "rows": 0, "sources": "", "status": "EMPTY", "adj_mode": ""})
            continue
        bars = bars.sort_index()
        adj_info: dict[str, object] = {"applied": False}
        if use_total_return_adjustment:
            bars, adj_info = _apply_total_return_adjustment(bars)
        if use_split_adjustment and not bool(adj_info.get("applied")):
            bars, split_events = apply_split_adjustment(
                bars=bars,
                symbol=symbol,
                market=market,
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
                "sources": ",".join(sorted(set(bars["source"].dropna().astype(str)))) if "source" in bars.columns else "",
                "status": "OK",
                "adj_mode": adj_mode,
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
    base_min_bars = get_strategy_min_bars(strategy)
    min_required_bars = (
        required_walkforward_bars(strategy_name=strategy, train_ratio=float(train_ratio))
        if enable_wf
        else base_min_bars
    )
    bars_by_symbol = {sym: bars for sym, bars in bars_by_symbol.items() if len(bars) >= min_required_bars}
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

    run_key = (
        f"bt_result:{market}:{','.join(symbols)}:{strategy}:{start_date}:{end_date}:"
        f"{int(enable_wf)}:{train_ratio}:{objective}:{int(initial_capital)}:"
        f"{int(use_split_adjustment)}:{int(auto_detect_split)}:{int(use_total_return_adjustment)}:"
        f"{invest_start_mode}:{invest_start_date}:{invest_start_k}"
    )
    with st.container(border=True):
        _render_card_section_header("回測執行", "輸入完成後會自動回測；同條件會優先讀取 SQLite 快取。")

    run_params = {
        "market": market,
        "mode": mode,
        "symbols": list(symbols),
        "strategy": strategy,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "enable_walk_forward": bool(enable_wf),
        "train_ratio": float(train_ratio),
        "objective": objective if enable_wf else "",
        "initial_capital": float(initial_capital),
        "fee_rate": float(fee_rate),
        "sell_tax_rate": float(sell_tax),
        "slippage_rate": float(slippage),
        "use_split_adjustment": bool(use_split_adjustment),
        "auto_detect_split": bool(auto_detect_split),
        "use_total_return_adjustment": bool(use_total_return_adjustment),
        "invest_start_mode": str(invest_start_mode),
        "invest_start_date": str(invest_start_date) if invest_start_date is not None else "",
        "invest_start_k": int(invest_start_k) if invest_start_k is not None else -1,
    }

    payload = st.session_state.get(run_key)
    if payload is None:
        cached_replay = store.load_latest_backtest_replay_run(run_key)
        if cached_replay and isinstance(cached_replay.payload, dict):
            restored = _deserialize_backtest_run_payload(cached_replay.payload)
            if restored is not None:
                payload = restored
                st.session_state[run_key] = restored
                st.caption("已載入同條件的 SQLite 快取回測結果。")

    if payload is None:
        cost_model = CostModel(fee_rate=fee_rate, sell_tax_rate=sell_tax, slippage_rate=slippage)
        run_payload = {}
        try:
            with st.spinner("回測計算中..."):
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
        payload = run_payload
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
        replay_payload = _serialize_backtest_run_payload(run_payload)
        store.save_backtest_replay_run(
            run_key=run_key,
            params=run_params,
            payload=replay_payload,
        )

    if not payload:
        st.info("尚未有可用回測結果。")
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
    benchmark_adj_info: dict[str, object] = {"applied": False}
    if use_total_return_adjustment and not benchmark_raw.empty:
        benchmark_raw, benchmark_adj_info = _apply_total_return_adjustment(benchmark_raw)
    benchmark_split_events = []
    if use_split_adjustment and not benchmark_raw.empty:
        if not bool(benchmark_adj_info.get("applied")):
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

    with st.container(border=True):
        _render_card_section_header("績效卡", "總報酬、CAGR、MDD 與相對大盤結果。")
        metric_rows = _metrics_to_rows(result.metrics)
        metric_cols = st.columns(4)
        for idx, (label, val) in enumerate(metric_rows):
            metric_cols[idx % 4].metric(label, val)
        if benchmark_choice != "off" and not benchmark_equity.empty:
            rel = pd.concat(
                [
                    strategy_equity.rename("strategy"),
                    benchmark_equity.rename("benchmark"),
                ],
                axis=1,
            ).dropna()
            if len(rel) >= 2:
                strategy_ret = float(rel["strategy"].iloc[-1] / rel["strategy"].iloc[0] - 1.0)
                benchmark_ret = float(rel["benchmark"].iloc[-1] / rel["benchmark"].iloc[0] - 1.0)
                diff_pct = (strategy_ret - benchmark_ret) * 100.0
                verdict = "贏過大盤" if diff_pct > 0 else ("輸給大盤" if diff_pct < 0 else "與大盤持平")
                r1, r2 = st.columns(2)
                r1.metric("相對大盤結果", verdict)
                r2.metric("贏/輸大盤（百分比）", f"{diff_pct:+.2f}%")
                st.caption(
                    "計算基準：同一重疊區間的 Total Return；"
                    f"Strategy={strategy_ret * 100:.2f}% vs Benchmark={benchmark_ret * 100:.2f}%"
                )
            else:
                st.caption("Benchmark 可用資料不足，暫時無法判斷是否贏過大盤。")
        elif benchmark_choice != "off":
            st.caption("目前沒有可用的 Benchmark 資料，無法計算是否贏過大盤。")

    if payload.get("walk_forward"):
        with st.container(border=True):
            _render_card_section_header("Walk-Forward 卡", "Train/Test 分段結果與最佳參數。")
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
    indicator_panel_key = f"play_indicator_panel:{run_key}"
    indicator_compact_key = f"play_indicator_compact:{run_key}"
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
    if indicator_panel_key not in st.session_state:
        st.session_state[indicator_panel_key] = True
    if indicator_compact_key not in st.session_state:
        st.session_state[indicator_compact_key] = True
    if viewport_mode_key not in st.session_state:
        st.session_state[viewport_mode_key] = "完整區間"
    if viewport_anchor_key not in st.session_state:
        st.session_state[viewport_anchor_key] = 70
    if focus_key not in st.session_state:
        st.session_state[focus_key] = selected_symbols[0]
    focus_symbol = st.selectbox("回放焦點標的", options=selected_symbols, key=focus_key)

    focus_bars = bars_by_symbol[focus_symbol].sort_index().dropna(subset=["open", "high", "low", "close"], how="any")
    if focus_bars.empty:
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
    if viewport_window_key not in st.session_state:
        st.session_state[viewport_window_key] = min(180, window_max)
    else:
        st.session_state[viewport_window_key] = int(
            min(max(int(st.session_state[viewport_window_key]), window_min), window_max)
        )
    date_options = [pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in focus_bars.index]
    date_to_idx = {d: i for i, d in enumerate(date_options)}

    with st.container(border=True):
        _render_card_section_header("回放控制卡", "播放速度、標記模式與視窗控制。")
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 2, 2, 3])
        if c1.button("Play", use_container_width=True):
            st.session_state[play_key] = True
        if c2.button("Pause", use_container_width=True):
            st.session_state[play_key] = False
        if c3.button("Reset", use_container_width=True):
            st.session_state[play_key] = False
            st.session_state[idx_key] = replay_reset_idx
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
        st.checkbox(
            "顯示技術指標副圖（RSI / MACD / 布林 / KD）",
            key=indicator_panel_key,
        )
        st.checkbox(
            "緊湊指標副圖（筆電建議）",
            key=indicator_compact_key,
        )
        st.caption("買賣點模式說明：訊號點=策略切換點；實際成交點=依回測規則 T+1 開盤成交；同時顯示=兩者一起顯示。")
        st.caption(
            f"回放預設位置：第 {default_play_idx} 根K（完整區間末端）。"
            f"若要重播動畫可按 Reset 回到第 {replay_reset_idx} 根。"
        )

    speed_steps = {"0.5x": 1, "1x": 2, "2x": 4, "5x": 8, "10x": 16}

    @st.fragment(run_every="0.5s")
    def playback():
        palette = _ui_palette()
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
        benchmark_now = benchmark_equity.reindex(equity_now.index).ffill() if not benchmark_equity.empty else pd.Series(dtype=float)
        panel_x_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.018, row_heights=[0.68, 0.32])
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
            panel_x_range = (pd.Timestamp(x_start), pd.Timestamp(x_end))
        elif len(bars_now.index) >= 2:
            panel_x_range = (pd.Timestamp(bars_now.index[0]), pd.Timestamp(bars_now.index[-1]))

        fig.update_xaxes(gridcolor=str(palette["grid"]))
        fig.update_yaxes(gridcolor=str(palette["grid"]))
        fig.update_layout(
            height=560,
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
        if st.session_state.get(indicator_panel_key):
            _render_indicator_panels(
                ind_now,
                chart_key=f"play_indicator_chart:{run_key}:{focus_symbol}",
                height=300 if st.session_state.get(indicator_compact_key) else 380,
                x_range=panel_x_range,
            )
        st.caption(f"目前回放到：第 {idx + 1} 根K（{bars_now.index[-1].strftime('%Y-%m-%d')}）")

    with st.container(border=True):
        _render_card_section_header("回放圖卡", "K線 + Equity + Benchmark 動態回放。")
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

    _render_card_section_header("Benchmark 對照卡", "策略曲線、基準曲線與買進持有同圖比較。")
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
    if bool(benchmark_adj_info.get("applied")):
        st.caption(f"Benchmark 已套用還原權息（Adj Close 覆蓋率 {benchmark_adj_info.get('coverage_pct', 0)}%）。")
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
                multi_symbol_compare = len([c for c in norm.columns if c.startswith("asset:")]) >= 2
                fig_cmp = go.Figure()
                fig_cmp.add_trace(
                    go.Scatter(
                        x=norm.index,
                        y=norm["strategy"],
                        mode="lines",
                        name="Strategy Equity",
                        line=dict(color=str(palette["equity"]), width=2.2),
                        hovertemplate=(
                            _hovertemplate_with_code("MULTI_STRATEGY", value_label="Normalized", y_format=".4f")
                            if multi_symbol_compare
                            else None
                        ),
                    )
                )
                fig_cmp.add_trace(
                    go.Scatter(
                        x=norm.index,
                        y=norm["benchmark"],
                        mode="lines",
                        name="Benchmark Equity",
                        line=_benchmark_line_style(palette, width=2.0),
                        hovertemplate=(
                            _hovertemplate_with_code(benchmark_symbol or "BENCHMARK", value_label="Normalized", y_format=".4f")
                            if multi_symbol_compare
                            else None
                        ),
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
                            hovertemplate=(
                                _hovertemplate_with_code("EW_POOL", value_label="Normalized", y_format=".4f")
                                if multi_symbol_compare
                                else None
                            ),
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
                            hovertemplate=(
                                _hovertemplate_with_code(sym, value_label="Normalized", y_format=".4f")
                                if multi_symbol_compare
                                else None
                            ),
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

    _render_card_section_header("策略 vs 買進持有卡", "看整段與指定區間的報酬率差異。")
    hold_label = "買進持有（等權投組）" if is_portfolio else f"買進持有（{selected_symbols[0]}）"
    st.caption("買進持有以回測標的收盤價計算；若為投組則採等權配置。")

    comparison_rows = [
        {
            "比較區間": "整段",
            "比較項目": "策略",
            "起始交易日(實際)": strategy_equity.index[0].strftime("%Y-%m-%d"),
            "結束交易日(實際)": strategy_equity.index[-1].strftime("%Y-%m-%d"),
            "報酬率%": round((strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1.0) * 100.0, 2),
        }
    ]
    if not buy_hold_equity.empty:
        comparison_rows.append(
            {
                "比較區間": "整段",
                "比較項目": hold_label,
                "起始交易日(實際)": buy_hold_equity.index[0].strftime("%Y-%m-%d"),
                "結束交易日(實際)": buy_hold_equity.index[-1].strftime("%Y-%m-%d"),
                "報酬率%": round((buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0] - 1.0) * 100.0, 2),
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
        strategy_interval = interval_return(strategy_equity, start_date=interval_start, end_date=interval_end)
        hold_interval = interval_return(buy_hold_equity, start_date=interval_start, end_date=interval_end)
        if strategy_interval.get("ok"):
            comparison_rows.append(
                {
                    "比較區間": "指定區間",
                    "比較項目": "策略",
                    "起始交易日(實際)": strategy_interval["start_used"].strftime("%Y-%m-%d"),
                    "結束交易日(實際)": strategy_interval["end_used"].strftime("%Y-%m-%d"),
                    "報酬率%": round(float(strategy_interval["return"]) * 100.0, 2),
                }
            )
        if hold_interval.get("ok"):
            comparison_rows.append(
                {
                    "比較區間": "指定區間",
                    "比較項目": hold_label,
                    "起始交易日(實際)": hold_interval["start_used"].strftime("%Y-%m-%d"),
                    "結束交易日(實際)": hold_interval["end_used"].strftime("%Y-%m-%d"),
                    "報酬率%": round(float(hold_interval["return"]) * 100.0, 2),
                }
            )
    if comparison_rows:
        out_df = pd.DataFrame(comparison_rows)
        st.dataframe(out_df, use_container_width=True, hide_index=True)
    else:
        st.caption("目前沒有可計算的比較資料。")

    _render_card_section_header("逐年報酬卡")
    if result.yearly_returns:
        yr = pd.DataFrame([{"年度": y, "報酬率%": round(v * 100.0, 2)} for y, v in result.yearly_returns.items()])
        st.dataframe(yr.sort_values("年度"), use_container_width=True, hide_index=True)
    else:
        st.caption("樣本不足，無逐年報酬。")

    if is_portfolio:
        _render_card_section_header("投組分項績效卡")
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

    _render_card_section_header("交易明細卡")
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
    st.subheader("新手教學（完整功能版）")
    st.caption("目標：先知道每頁在做什麼，再跑回測，最後才做比較與微調。")

    st.markdown("### 0) 全站功能地圖（先看這張）")
    page_df = pd.DataFrame(
        [
            {"分頁": "即時看盤", "你會看到什麼": "即時行情、即時趨勢、技術快照、建議卡", "什麼時候用": "盤中快速看狀態"},
            {"分頁": "回測工作台", "你會看到什麼": "單檔/投組回測、Walk-Forward、Benchmark 比較、回放", "什麼時候用": "驗證策略與參數"},
            {"分頁": "2025 前十大 ETF", "你會看到什麼": "2025 全年 Top10 報酬排行", "什麼時候用": "回顧去年全年度強弱"},
            {"分頁": "2026 YTD 前十大 ETF", "你會看到什麼": "2026 年迄今 Top10、2025 對照、大盤勝負、Benchmark 對照卡", "什麼時候用": "看今年領先 ETF"},
            {"分頁": "2026 YTD 主動式 ETF", "你會看到什麼": "主動式 ETF 排行、2025 對照、大盤勝負、Benchmark 對照卡", "什麼時候用": "追蹤主動式 ETF 表現"},
            {"分頁": "ETF 輪動策略", "你會看到什麼": "固定 ETF 池月調倉回測、調倉明細、持有最久排名", "什麼時候用": "看規則化輪動是否優於基準"},
            {"分頁": "00910 熱力圖", "你會看到什麼": "全球成分股 YTD 分組熱力圖 + 台股子集合進階回測", "什麼時候用": "同時看國內/海外成分股相對表現"},
            {"分頁": "00935 熱力圖", "你會看到什麼": "成分股相對大盤熱力圖 + 公司簡介", "什麼時候用": "看 00935 內部強弱分布"},
            {"分頁": "00993A 熱力圖", "你會看到什麼": "成分股相對大盤熱力圖", "什麼時候用": "看 00993A 內部強弱分布"},
            {"分頁": "0050 熱力圖", "你會看到什麼": "成分股相對大盤熱力圖 + 公司簡介（依權重排序）", "什麼時候用": "看台灣 50 內部強弱"},
            {"分頁": "資料庫檢視", "你會看到什麼": "SQLite 表格總覽、欄位、分頁資料", "什麼時候用": "確認資料是否有進 SQLite"},
            {"分頁": "新手教學", "你會看到什麼": "名詞解釋、快取邏輯、操作順序、常見誤解", "什麼時候用": "剛上手或看不懂數字時"},
        ]
    )
    st.dataframe(page_df, use_container_width=True, hide_index=True)

    st.markdown("### 1) 第一次使用，建議照這個順序")
    st.markdown(
        "\n".join(
            [
                "1. 到 `回測工作台`，先跑一個 `buy_hold`（單檔、近 1~3 年）。",
                "2. 確認你看得懂 `總報酬/CAGR/MDD/Sharpe` 與成交明細。",
                "3. 再到 `2026 YTD 前十大 ETF` 或 `2026 YTD 主動式 ETF` 看橫向比較。",
                "4. 想看 ETF 內部成分股強弱，再進 `00910 / 00935 / 00993A / 0050 熱力圖`。",
                "5. 最後才用 `ETF 輪動策略` 做規則化比較。",
            ]
        )
    )
    st.info("若你只想先確認流程有跑通，第一步固定用 `buy_hold` 最穩。")

    st.markdown("### 2) 更新按鈕與快取邏輯（最常問）")
    cache_df = pd.DataFrame(
        [
            {"項目": "更新最新市況（Top10/主動式）", "作用": "清除該頁快取並重抓最新快照", "不按會怎樣": "會先顯示上次可用結果"},
            {"項目": "更新 成分股（熱力圖）", "作用": "重抓 ETF 成分股清單並更新快取", "不按會怎樣": "沿用 `universe_snapshots` 既有快取"},
            {"項目": "執行前同步最新日K（較慢）", "作用": "先補齊資料再跑", "不按會怎樣": "優先用本地 SQLite，不足才補同步"},
            {"項目": "重新計算（Benchmark 對照卡）", "作用": "在目前參數下重算曲線與績效表", "不按會怎樣": "沿用本頁已算過結果"},
        ]
    )
    st.dataframe(cache_df, use_container_width=True, hide_index=True)

    st.markdown("### 3) 資料會存在哪裡？")
    storage_df = pd.DataFrame(
        [
            {"項目": "歷史日K（含 Benchmark）", "位置": "SQLite（預設 iCloud，可由環境變數覆蓋）", "用途": "回測與比較主資料來源"},
            {"項目": "回測摘要", "位置": "SQLite `backtest_runs`", "用途": "保存回測重點結果"},
            {"項目": "回測回放快取", "位置": "SQLite `backtest_replay_runs`", "用途": "同條件可直接載入完整回放結果"},
            {"項目": "熱力圖結果", "位置": "SQLite `heatmap_runs`", "用途": "熱力圖頁可先顯示上次結果"},
            {"項目": "ETF 輪動結果", "位置": "SQLite `rotation_runs`", "用途": "輪動頁可先顯示上次結果"},
            {"項目": "成分股清單", "位置": "SQLite `universe_snapshots`", "用途": "避免每次重抓成分股"},
            {"項目": "即時資料", "位置": "Session 記憶體 + SQLite `intraday_ticks`", "用途": "即時看盤與補圖"},
        ]
    )
    st.dataframe(storage_df, use_container_width=True, hide_index=True)

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
    st.dataframe(flow_df, use_container_width=True, hide_index=True)

    st.markdown("### 5) 回測安全門檻（資料筆數）")
    threshold_df = pd.DataFrame(
        [
            {"模式": "buy_hold", "最少K數": "2", "原因": "至少要有起點與終點。"},
            {"模式": "一般策略（SMA/EMA/RSI/MACD）", "最少K數": "40", "原因": "指標需要基本樣本。"},
            {"模式": "日K趨勢策略（sma_trend_filter / donchian_breakout）", "最少K數": "120", "原因": "需要長視窗濾網。"},
            {"模式": "Walk-Forward", "最少K數": "至少 80（長視窗策略更高）", "原因": "Train/Test 都要有足夠樣本。"},
        ]
    )
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)

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
    st.dataframe(metric_df, use_container_width=True, hide_index=True)
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
    st.dataframe(heatmap_df, use_container_width=True, hide_index=True)

    st.markdown("### 8) 常見混淆（會不會改回測結果）")
    confusion_df = pd.DataFrame(
        [
            {"控制項": "起始日期 / 結束日期", "會不會改回測結果": "會", "重點": "改變樣本區間"},
            {"控制項": "實際投入起點（日期或第幾根K）", "會不會改回測結果": "會", "重點": "改變本金開始參與時間"},
            {"控制項": "回放位置（K棒/日期）", "會不會改回測結果": "不會", "重點": "只改圖表播放位置"},
            {"控制項": "回放視窗 / 生長位置", "會不會改回測結果": "不會", "重點": "只改視覺呈現"},
            {"控制項": "顯示成交連線", "會不會改回測結果": "不會", "重點": "僅做圖上對照"},
        ]
    )
    st.dataframe(confusion_df, use_container_width=True, hide_index=True)

    st.markdown("### 9) 訊號點 vs 實際成交點")
    point_df = pd.DataFrame(
        [
            {"類型": "訊號點（價格圖）", "代表意思": "策略判斷該進場/出場", "適合用途": "看策略邏輯翻多翻空"},
            {"類型": "實際成交點（資產圖）", "代表意思": "回測規則真正成交位置", "適合用途": "檢查績效計算合理性"},
        ]
    )
    st.dataframe(point_df, use_container_width=True, hide_index=True)
    st.caption("本系統成交規則：`T 日收盤出訊號，T+1 開盤成交`。")

    st.markdown("### 10) 參數白話解釋（常用）")
    params_df = pd.DataFrame(
        [
            {"參數": "Fast", "白話意思": "短天期均線", "單位/格式": "天數（整數）", "新手建議": "10~20"},
            {"參數": "Slow", "白話意思": "長天期均線", "單位/格式": "天數（整數）", "新手建議": "30~120 且大於 Fast"},
            {"參數": "Trend Filter", "白話意思": "長期趨勢濾網均線", "單位/格式": "天數（整數）", "新手建議": "120"},
            {"參數": "Breakout Lookback", "白話意思": "突破回朔天數", "單位/格式": "天數（整數）", "新手建議": "55"},
            {"參數": "Exit Lookback", "白話意思": "出場回朔天數", "單位/格式": "天數（整數）", "新手建議": "20 且小於 Breakout"},
            {"參數": "RSI Buy Below", "白話意思": "低於此值才考慮買入", "單位/格式": "0~100", "新手建議": "30"},
            {"參數": "RSI Sell Above", "白話意思": "高於此值才考慮賣出", "單位/格式": "0~100", "新手建議": "55~65"},
            {"參數": "Fee Rate", "白話意思": "手續費比例", "單位/格式": "小數比例", "新手建議": "台股常見 0.001425"},
            {"參數": "Sell Tax", "白話意思": "賣出交易稅比例", "單位/格式": "小數比例", "新手建議": "股票 0.003 / ETF 0.001"},
            {"參數": "Slippage", "白話意思": "理論價與成交價偏差", "單位/格式": "小數比例", "新手建議": "0.0005（0.05%）"},
            {"參數": "Train 比例", "白話意思": "Walk-Forward 訓練占比", "單位/格式": "0~1", "新手建議": "0.70"},
            {"參數": "參數挑選目標", "白話意思": "Train 區選最佳參數指標", "單位/格式": "sharpe/cagr/total_return/mdd", "新手建議": "sharpe"},
        ]
    )
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("### 11) 技術指標速讀（RSI / MACD / 布林通道 / KD）")
    indicators_df = pd.DataFrame(
        [
            {"指標": "RSI", "白話解釋": "看短期買賣力道是否過熱或過冷", "常見看法": "低檔區偏弱、高檔區偏強", "新手提醒": "不要單看超買超賣，先看大方向趨勢"},
            {"指標": "MACD", "白話解釋": "看趨勢加速度與轉折", "常見看法": "快慢線黃金交叉偏多、死亡交叉偏空", "新手提醒": "盤整盤容易來回假訊號"},
            {"指標": "布林通道", "白話解釋": "看波動區間與價格相對位置", "常見看法": "貼上軌代表強勢、貼下軌代表弱勢", "新手提醒": "貼軌不等於立刻反轉，可能是趨勢延續"},
            {"指標": "KD", "白話解釋": "看短線動能轉折速度", "常見看法": "K 上穿 D 偏多、K 下穿 D 偏空", "新手提醒": "在強趨勢中容易過早反向判斷"},
        ]
    )
    st.dataframe(indicators_df, use_container_width=True, hide_index=True)
    st.caption("建議做法：先用趨勢類指標（如 MACD）判方向，再用 RSI/KD 找節奏，最後用布林通道觀察波動風險。")

    st.markdown("### 12) 新手建議操作（先簡單再進階）")
    st.markdown(
        "\n".join(
            [
                "1. 先在 `回測工作台` 用 `buy_hold` 跑單檔，確認資料與報表都正常。",
                "2. 再改成 `sma_trend_filter` 或 `donchian_breakout`，比較是否優於 `buy_hold`。",
                "3. 接著用 `2026 YTD 前十大 ETF` / `2026 YTD 主動式 ETF` 做橫向排名比較。",
                "4. 最後才進 `ETF 輪動策略` 與各熱力圖做進階判讀。",
            ]
        )
    )


def _render_excess_heatmap_panel(rows_df: pd.DataFrame, *, title: str, colorbar_title: str = "超額報酬 %"):
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
    custom = np.empty((tile_rows, tiles_per_row, 6), dtype=object)
    custom[:, :, :] = None

    for i, row in frame.iterrows():
        r = i // tiles_per_row
        c = i % tiles_per_row
        z[r, c] = float(row["excess_pct"])
        label = str(row["symbol"]).strip()
        txt[r, c] = f"<b>{label}</b><br>{row['excess_pct']:+.2f}%"
        custom[r, c, 0] = float(row["asset_return_pct"])
        custom[r, c, 1] = float(row["benchmark_return_pct"])
        custom[r, c, 2] = str(row.get("benchmark_symbol", ""))
        custom[r, c, 3] = int(row.get("bars", 0))
        custom[r, c, 4] = str(row.get("market_tag", ""))
        custom[r, c, 5] = str(row.get("weight_pct", ""))

    max_abs = _heatmap_max_abs(z)
    fig_heat = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                text=txt,
                texttemplate="%{text}",
                textfont=dict(size=12, color=HEATMAP_TEXT_COLOR, family="Noto Sans TC, Segoe UI, sans-serif"),
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
    fig_heat.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
    fig_heat.update_layout(
        height=max(250, 90 * tile_rows),
        margin=dict(l=10, r=10, t=10, b=10),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"]), family="Noto Sans TC, Segoe UI, sans-serif"),
        hoverlabel=_plot_hoverlabel_style(palette),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

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
    keep_cols = [c for c in rename_map.keys() if c in out_df.columns]
    out_df = out_df[keep_cols].rename(columns={k: v for k, v in rename_map.items() if k in keep_cols})
    st.dataframe(out_df, use_container_width=True, hide_index=True)


def _render_00910_global_ytd_block(
    *,
    store: HistoryStore,
    service: MarketDataService,
    page_key: str,
    full_rows: list[dict[str, object]],
):
    st.markdown("### 00910 全球成分股 YTD 熱力圖（Buy & Hold）")
    today = date.today()
    ytd_start_date = date(today.year, 1, 1)
    ytd_end_date = today
    start_dt = datetime.combine(ytd_start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(ytd_end_date, datetime.min.time()).replace(tzinfo=timezone.utc)

    c1, c2, c3, c4 = st.columns(4)
    tw_benchmark = c1.selectbox("台股基準", options=["^TWII", "0050"], index=0, key=f"{page_key}_00910_bench_tw")
    us_benchmark = c2.selectbox("美股基準", options=["QQQ", "^IXIC"], index=0, key=f"{page_key}_00910_bench_us")
    jp_benchmark = c3.selectbox("日股基準", options=["^N225"], index=0, key=f"{page_key}_00910_bench_jp")
    ks_benchmark = c4.selectbox("韓股基準", options=["^KS11"], index=0, key=f"{page_key}_00910_bench_ks")
    s1, s2 = st.columns(2)
    sync_before_run = s1.checkbox(
        "執行前同步最新日K（推薦）",
        value=True,
        key=f"{page_key}_00910_sync_ytd",
    )
    parallel_sync = s2.checkbox(
        "平行同步",
        value=True,
        key=f"{page_key}_00910_parallel_ytd",
    )
    st.caption(
        f"YTD 區間：{ytd_start_date.isoformat()} ~ {ytd_end_date.isoformat()}。"
        "台股成分對台股基準；海外成分依市場對應基準（US/JP/KS）。"
    )

    universe_id = "TW:00910:GLOBAL_YTD"
    payload_key = f"{page_key}_global_ytd_payload"
    run_key = (
        f"00910_global_ytd:{ytd_start_date}:{ytd_end_date}:{tw_benchmark}:"
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

    run_now = st.button("執行 00910 YTD 全球熱力圖", type="primary", use_container_width=True, key=f"{page_key}_00910_run_ytd")
    if run_now:
        rows_full = list(full_rows)
        if not rows_full:
            rows_full, _ = service.get_etf_constituents_full("00910", limit=None, force_refresh=True)
        if not rows_full:
            st.error("目前抓不到 00910 完整成分股（含海外），請稍後重試。")
            return

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
            st.error("00910 成分股清單為空，無法執行。")
            return

        sync_issues: list[str] = []
        if sync_before_run:
            tw_symbols = _dedupe_keep_order(
                [
                    *[str(it["symbol"]) for it in universe_items if str(it["market"]) == "TW"],
                    *[str(it["benchmark_symbol"]) for it in universe_items if str(it["benchmark_market"]) == "TW"],
                ]
            )
            us_symbols = _dedupe_keep_order(
                [
                    *[str(it["symbol"]) for it in universe_items if str(it["market"]) == "US"],
                    *[str(it["benchmark_symbol"]) for it in universe_items if str(it["benchmark_market"]) == "US"],
                ]
            )
            with st.spinner("同步 00910 YTD 所需資料中..."):
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
            bars = _normalize_ohlcv_frame(bars)
            if bars.empty and not sync_before_run:
                report = store.sync_symbol_history(symbol=symbol, market=market, start=start_dt, end=end_dt)
                if report.error:
                    sync_issues.append(f"{symbol}: {report.error}")
                bars = store.load_daily_bars(symbol=symbol, market=market, start=start_dt, end=end_dt)
                bars = _normalize_ohlcv_frame(bars)
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
            comp = pd.concat([asset_close.rename("asset"), benchmark_close.rename("benchmark")], axis=1, join="inner").dropna()
            if len(comp) < 2:
                continue

            asset_return_pct = float(comp["asset"].iloc[-1] / comp["asset"].iloc[0] - 1.0) * 100.0
            benchmark_return_pct = float(comp["benchmark"].iloc[-1] / comp["benchmark"].iloc[0] - 1.0) * 100.0
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
        st.info("按下「執行 00910 YTD 全球熱力圖」後，會顯示國內/海外分組結果。")
        return
    if payload.get("run_key") != run_key:
        st.caption("目前顯示的是上一次執行結果；若要套用目前基準設定，請重新執行。")

    sync_issues = payload.get("sync_issues")
    if isinstance(sync_issues, list) and sync_issues:
        preview = [" ".join(str(item).split()) for item in sync_issues[:3]]
        preview_text = " | ".join([item if len(item) <= 120 else f"{item[:117]}..." for item in preview])
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

    tw_df = rows_df[rows_df["group"] == "TW"].copy()
    overseas_df = rows_df[rows_df["group"] == "OVERSEAS"].copy()
    _render_excess_heatmap_panel(tw_df, title="國內成分股（TW vs 台股基準）")
    _render_excess_heatmap_panel(overseas_df, title="海外成分股（Overseas vs 對應海外基準）")

def _render_tw_etf_heatmap_view(etf_code: str, page_desc: str):
    store = _history_store()
    service = _market_service()
    etf_text = str(etf_code).strip().upper()
    page_key = f"tw{etf_text.lower()}"

    st.subheader(f"{etf_text} 成分股熱力圖回測（相對大盤）")
    st.caption(f"以 {etf_text}{page_desc}成分股逐檔回測，與大盤同區間比較；綠色代表贏過、紅色代表輸給。")
    _render_etf_index_method_summary(etf_text)

    c1, c2, c3, c4 = st.columns(4)
    start_date = c1.date_input("起始日期", value=date(date.today().year - 5, 1, 1), key=f"{page_key}_start")
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
        trend = p3.slider("Trend Filter", min_value=60, max_value=300, value=120, key=f"{page_key}_trend")
        strategy_params = {"fast": float(fast), "slow": float(slow), "trend": float(trend)}
    elif strategy == "donchian_breakout":
        p1, p2, p3 = st.columns(3)
        entry_n = p1.slider("Breakout Lookback", min_value=20, max_value=120, value=55, key=f"{page_key}_entry")
        exit_max = max(10, int(entry_n) - 1)
        exit_n = p2.slider(
            "Exit Lookback",
            min_value=5,
            max_value=exit_max,
            value=min(20, exit_max),
            key=f"{page_key}_exit",
        )
        trend = p3.slider("Trend Filter", min_value=60, max_value=300, value=120, key=f"{page_key}_trend")
        strategy_params = {"entry_n": float(entry_n), "exit_n": float(exit_n), "trend": float(trend)}

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
    snapshot = store.load_universe_snapshot(universe_id)
    u1, u2 = st.columns([1, 4])
    refresh_constituents = u1.button(f"更新 {etf_text} 成分股", use_container_width=True)
    if refresh_constituents:
        with st.spinner(f"抓取 {etf_text} 成分股中..."):
            symbols_new, source_new = service.get_tw_etf_constituents(etf_text, limit=None)
            if symbols_new:
                store.save_universe_snapshot(universe_id=universe_id, symbols=symbols_new, source=source_new)
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
        with st.expander(f"查看全部成分股（{len(snapshot.symbols)}）", expanded=False):
            name_map_all = service.get_tw_symbol_names(snapshot.symbols)
            const_rows = [{"symbol": sym, "name": name_map_all.get(sym, sym)} for sym in snapshot.symbols]
            st.dataframe(pd.DataFrame(const_rows), use_container_width=True, hide_index=True)

        if etf_text == "00910":
            full_rows, full_source = service.get_etf_constituents_full(
                etf_text, limit=None, force_refresh=bool(refresh_constituents)
            )
            full_rows_00910 = list(full_rows)
            if full_rows:
                full_df = pd.DataFrame(full_rows)
                tw_subset_count = int((full_df["tw_code"].astype(str) != "").sum()) if "tw_code" in full_df.columns else 0
                st.caption(
                    f"00910 完整成分股（含海外）共 {len(full_rows)} 檔 | 來源：{full_source} | "
                    f"其中台股可回測 {tw_subset_count} 檔。"
                )
                with st.expander(f"查看完整成分股（含海外，共 {len(full_rows)} 檔）", expanded=False):
                    out_df = full_df.rename(
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
                    st.dataframe(out_df, use_container_width=True, hide_index=True)
            else:
                st.caption("00910 完整成分股（含海外）目前抓取失敗，請稍後按「更新 00910 成分股」重試。")
    else:
        u2.caption(f"尚未載入 {etf_text} 成分股快取。你仍可先查看上次回測結果，或按「更新 {etf_text} 成分股」後再重新回測。")

    if etf_text == "00910":
        _render_00910_global_ytd_block(
            store=store,
            service=service,
            page_key=page_key,
            full_rows=full_rows_00910,
        )
        st.markdown("---")
        st.markdown("#### 台股子集合進階熱力圖（自訂區間/策略）")
        st.caption("下方為 00910 台股子集合回測，僅比較可回測的台股成分。")

    symbol_options = list(snapshot.symbols) if snapshot and snapshot.symbols else []
    symbol_key = f"{page_key}_symbol_pick"
    current_pick = st.session_state.get(symbol_key, symbol_options)
    if not isinstance(current_pick, list):
        current_pick = symbol_options
    current_pick = [s for s in current_pick if s in symbol_options]
    if not current_pick:
        current_pick = symbol_options
    st.session_state[symbol_key] = current_pick
    symbol_name_map_for_pick: dict[str, str] = {}
    if etf_text == "00935":
        symbol_name_map_for_pick = service.get_tw_symbol_names(symbol_options)

    def _format_symbol_option(sym: str) -> str:
        code = str(sym)
        if etf_text != "00935":
            return code
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

    start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc) if date_is_valid else None
    end_dt = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc) if date_is_valid else None
    s1, s2 = st.columns(2)
    sync_before_run = s1.checkbox(
        "執行前同步最新日K（較慢）",
        value=False,
        key=f"{page_key}_sync_before_run",
        help="預設關閉：優先使用本地 SQLite；資料不足時才補同步。",
    )
    parallel_sync = s2.checkbox(
        "平行同步多標的",
        value=True,
        key=f"{page_key}_parallel_sync",
        help="標的較多時通常更快；若網路不穩可關閉改逐檔同步。",
    )

    def _show_sync_issues(prefix: str, issues: list[str]):
        if not issues:
            return
        preview = [" ".join(str(item).split()) for item in issues[:3]]
        preview_text = " | ".join([item if len(item) <= 120 else f"{item[:117]}..." for item in preview])
        remain = len(issues) - len(preview)
        remain_text = f" | 其餘 {remain} 筆請查看終端 log。" if remain > 0 else ""
        st.warning(f"{prefix}：{preview_text}{remain_text}")

    def _benchmark_candidates_tw(choice: str) -> list[str]:
        mapping = {
            "twii": ["^TWII"],
            "0050": ["0050"],
            "006208": ["006208"],
        }
        return mapping.get(choice, ["^TWII"])

    def _load_benchmark_close(choice: str, *, sync_first: bool) -> tuple[pd.Series, str, list[str]]:
        sync_issues: list[str] = []
        for benchmark_symbol in _benchmark_candidates_tw(choice):
            if sync_first:
                report = store.sync_symbol_history(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
                if report.error:
                    sync_issues.append(f"{benchmark_symbol}: {report.error}")
            bench_bars = store.load_daily_bars(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
            if bench_bars.empty and not sync_first:
                report = store.sync_symbol_history(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
                if report.error:
                    sync_issues.append(f"{benchmark_symbol}: {report.error}")
                bench_bars = store.load_daily_bars(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
            bench_bars = _normalize_ohlcv_frame(bench_bars)
            if bench_bars.empty:
                continue
            bench_bars, _ = apply_split_adjustment(
                bars=bench_bars,
                symbol=benchmark_symbol,
                market="TW",
                use_known=True,
                use_auto_detect=True,
            )
            close = pd.to_numeric(bench_bars["close"], errors="coerce").dropna()
            if len(close) >= 2:
                return close, benchmark_symbol, sync_issues
        return pd.Series(dtype=float), "", sync_issues

    strategy_token = json.dumps(strategy_params, sort_keys=True, ensure_ascii=False)
    run_key = (
        f"{page_key}_heatmap:{start_date}:{end_date}:{benchmark_choice}:{strategy}:{strategy_token}:"
        f"{fee_rate}:{sell_tax}:{slippage}:{','.join(selected_symbols)}"
    )
    if st.button(f"執行 {etf_text} 熱力圖回測", type="primary", use_container_width=True):
        if not date_is_valid:
            st.error("日期區間無效，請先修正起訖日期。")
            return
        if not symbol_options:
            st.error(f"尚未載入 {etf_text} 成分股，請先按「更新 {etf_text} 成分股」。")
            return
        if not selected_symbols:
            st.error("請至少選擇 1 檔成分股。")
            return

        run_symbols = list(selected_symbols)
        with st.spinner("同步最新成分股中..."):
            symbols_latest, source_latest = service.get_tw_etf_constituents(etf_text, limit=None)
            if symbols_latest:
                store.save_universe_snapshot(universe_id=universe_id, symbols=symbols_latest, source=source_latest)
                snapshot = store.load_universe_snapshot(universe_id)
                if set(selected_symbols) == set(symbol_options):
                    run_symbols = symbols_latest
                else:
                    run_symbols = [s for s in selected_symbols if s in symbols_latest]
                    if not run_symbols:
                        run_symbols = symbols_latest

        run_key = (
            f"{page_key}_heatmap:{start_date}:{end_date}:{benchmark_choice}:{strategy}:{strategy_token}:"
            f"{fee_rate}:{sell_tax}:{slippage}:{','.join(run_symbols)}"
        )
        symbol_sync_issues: list[str] = []
        if sync_before_run:
            with st.spinner("同步成分股日K中..."):
                _, symbol_sync_issues = _sync_symbols_history(
                    store,
                    market="TW",
                    symbols=run_symbols,
                    start=start_dt,
                    end=end_dt,
                    parallel=parallel_sync,
                )

        bars_cache: dict[str, pd.DataFrame] = {}
        min_required = get_strategy_min_bars(strategy)
        symbols_need_sync: list[str] = []
        for symbol in run_symbols:
            bars_local = store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            bars_local = _normalize_ohlcv_frame(bars_local)
            bars_cache[symbol] = bars_local
            if len(bars_local) < min_required:
                symbols_need_sync.append(symbol)

        if symbols_need_sync and not sync_before_run:
            with st.spinner("本地資料不足，補同步中..."):
                _, lazy_sync_issues = _sync_symbols_history(
                    store,
                    market="TW",
                    symbols=symbols_need_sync,
                    start=start_dt,
                    end=end_dt,
                    parallel=parallel_sync,
                )
                symbol_sync_issues.extend(lazy_sync_issues)
            for symbol in symbols_need_sync:
                bars_local = store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
                bars_cache[symbol] = _normalize_ohlcv_frame(bars_local)

        benchmark_close, benchmark_symbol, benchmark_sync_issues = _load_benchmark_close(
            benchmark_choice,
            sync_first=sync_before_run,
        )
        _show_sync_issues("Benchmark 同步有部分錯誤，已盡量使用本地可用資料", benchmark_sync_issues)
        if benchmark_close.empty:
            st.error("Benchmark 取得失敗，請改選其他基準（0050 或 006208）後重試。")
            return

        progress = st.progress(0.0)
        rows: list[dict[str, object]] = []
        cost_model = CostModel(fee_rate=float(fee_rate), sell_tax_rate=float(sell_tax), slippage_rate=float(slippage))
        name_map = service.get_tw_symbol_names(run_symbols)

        for idx, symbol in enumerate(run_symbols):
            bars = bars_cache.get(symbol, pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
            if len(bars) < min_required:
                progress.progress((idx + 1) / max(len(run_symbols), 1))
                continue

            bars, _ = apply_split_adjustment(
                bars=bars,
                symbol=symbol,
                market="TW",
                use_known=True,
                use_auto_detect=True,
            )
            if len(bars) < min_required:
                progress.progress((idx + 1) / max(len(run_symbols), 1))
                continue

            try:
                bt = run_backtest(
                    bars=bars,
                    strategy_name=strategy,
                    strategy_params=strategy_params,
                    cost_model=cost_model,
                    initial_capital=1_000_000.0,
                )
            except Exception:
                progress.progress((idx + 1) / max(len(run_symbols), 1))
                continue

            strategy_curve = pd.to_numeric(bt.equity_curve["equity"], errors="coerce").dropna()
            if len(strategy_curve) < 2:
                progress.progress((idx + 1) / max(len(run_symbols), 1))
                continue

            comp = pd.concat(
                [
                    strategy_curve.rename("strategy"),
                    benchmark_close.reindex(strategy_curve.index).ffill().rename("benchmark"),
                ],
                axis=1,
            ).dropna()
            if len(comp) < 2:
                progress.progress((idx + 1) / max(len(run_symbols), 1))
                continue

            strategy_ret = float(comp["strategy"].iloc[-1] / comp["strategy"].iloc[0] - 1.0)
            benchmark_ret = float(comp["benchmark"].iloc[-1] / comp["benchmark"].iloc[0] - 1.0)
            excess_pct = (strategy_ret - benchmark_ret) * 100.0
            rows.append(
                {
                    "symbol": symbol,
                    "name": name_map.get(symbol, symbol),
                    "strategy_return_pct": strategy_ret * 100.0,
                    "benchmark_return_pct": benchmark_ret * 100.0,
                    "excess_pct": excess_pct,
                    "status": "WIN" if excess_pct > 0 else ("LOSE" if excess_pct < 0 else "TIE"),
                    "bars": int(len(comp)),
                }
            )
            progress.progress((idx + 1) / max(len(run_symbols), 1))

        progress.empty()
        _show_sync_issues("部分成分股同步失敗，已盡量使用本地可用資料", symbol_sync_issues)
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

    rows_df = pd.DataFrame(payload.get("rows", []))
    if rows_df.empty:
        st.warning("沒有可用回測結果（可能資料不足或期間太短）。")
        return
    if "name" not in rows_df.columns:
        name_map = service.get_tw_symbol_names(rows_df["symbol"].astype(str).tolist())
        rows_df["name"] = rows_df["symbol"].map(name_map).fillna(rows_df["symbol"])
    rows_df["name"] = rows_df["name"].astype(str)

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
    custom = np.empty((tile_rows, tiles_per_row, 4), dtype=object)
    custom[:, :, :] = None

    for i, row in rows_df.iterrows():
        r = i // tiles_per_row
        c = i % tiles_per_row
        z[r, c] = float(row["excess_pct"])
        label = str(row["symbol"]).strip()
        name_text = str(row.get("name", "")).strip()
        if etf_text == "00935" and name_text and name_text != label:
            txt[r, c] = f"<b>{label}</b><br>{name_text}<br>{row['excess_pct']:+.2f}%"
        else:
            txt[r, c] = f"<b>{label}</b><br>{row['excess_pct']:+.2f}%"
        custom[r, c, 0] = float(row["strategy_return_pct"])
        custom[r, c, 1] = float(row["benchmark_return_pct"])
        custom[r, c, 2] = str(row["status"])
        custom[r, c, 3] = str(row["name"])

    max_abs = _heatmap_max_abs(z)
    fig_heat = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                text=txt,
                texttemplate="%{text}",
                textfont=dict(size=12, color=HEATMAP_TEXT_COLOR, family="Noto Sans TC, Segoe UI, sans-serif"),
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
                    "狀態：%{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig_heat.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig_heat.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
    fig_heat.update_layout(
        height=max(280, 90 * tile_rows),
        margin=dict(l=10, r=10, t=20, b=10),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"]), family="Noto Sans TC, Segoe UI, sans-serif"),
        hoverlabel=_plot_hoverlabel_style(palette),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    out_df = rows_df.copy()
    out_df["strategy_return_pct"] = out_df["strategy_return_pct"].map(lambda v: round(float(v), 2))
    out_df["benchmark_return_pct"] = out_df["benchmark_return_pct"].map(lambda v: round(float(v), 2))
    out_df["excess_pct"] = out_df["excess_pct"].map(lambda v: round(float(v), 2))
    st.dataframe(
        out_df[["symbol", "name", "strategy_return_pct", "benchmark_return_pct", "excess_pct", "status", "bars"]],
        use_container_width=True,
        hide_index=True,
    )

    if etf_text in {"00935", "00993A", "0050"} and snapshot and snapshot.symbols:
        st.markdown("---")
        _render_tw_constituent_intro_table(
            etf_code=etf_text,
            symbols=list(snapshot.symbols),
            service=service,
        )
    elif etf_text == "00910":
        st.markdown("---")
        _render_00910_constituent_intro_table(
            service=service,
            full_rows=full_rows_00910,
        )


def _render_tw_etf_rotation_view():
    store = _history_store()
    service = _market_service()
    palette = _ui_palette()

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
    start_date = c1.date_input("起始日期", value=date(date.today().year - 5, 1, 1), key="rotation_start")
    end_date = c2.date_input("結束日期", value=date.today(), key="rotation_end")
    top_n = c3.slider("每月持有檔數 Top N", min_value=1, max_value=len(ROTATION_DEFAULT_UNIVERSE), value=3, key="rotation_topn")
    benchmark_choice = c4.selectbox(
        "Benchmark",
        options=["twii", "0050", "006208"],
        index=0,
        format_func=lambda x: {"twii": "^TWII（Auto fallback）", "0050": "0050", "006208": "006208"}.get(x, x),
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

    start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc) if date_is_valid else None
    end_dt = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc) if date_is_valid else None
    s1, s2 = st.columns(2)
    sync_before_run = s1.checkbox(
        "執行前同步最新日K（較慢）",
        value=False,
        key="rotation_sync_before_run",
        help="預設關閉：優先使用本地 SQLite；資料不足時才補同步。",
    )
    parallel_sync = s2.checkbox(
        "平行同步多標的",
        value=True,
        key="rotation_parallel_sync",
        help="多檔 ETF 時通常更快；若網路不穩可關閉改逐檔同步。",
    )

    def _show_sync_issues(prefix: str, issues: list[str]):
        if not issues:
            return
        preview = [" ".join(str(item).split()) for item in issues[:3]]
        preview_text = " | ".join([item if len(item) <= 120 else f"{item[:117]}..." for item in preview])
        remain = len(issues) - len(preview)
        remain_text = f" | 其餘 {remain} 筆請查看終端 log。" if remain > 0 else ""
        st.warning(f"{prefix}：{preview_text}{remain_text}")

    def _benchmark_candidates_tw(choice: str) -> list[str]:
        mapping = {
            "twii": ["^TWII", "0050", "006208"],
            "0050": ["0050"],
            "006208": ["006208"],
        }
        return mapping.get(choice, ["^TWII", "0050", "006208"])

    def _load_benchmark_bars(choice: str, *, sync_first: bool) -> tuple[pd.DataFrame, str, list[str]]:
        sync_issues: list[str] = []
        for benchmark_symbol in _benchmark_candidates_tw(choice):
            if sync_first:
                report = store.sync_symbol_history(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
                if report.error:
                    sync_issues.append(f"{benchmark_symbol}: {report.error}")
            bench = store.load_daily_bars(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
            if bench.empty and not sync_first:
                report = store.sync_symbol_history(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
                if report.error:
                    sync_issues.append(f"{benchmark_symbol}: {report.error}")
                bench = store.load_daily_bars(symbol=benchmark_symbol, market="TW", start=start_dt, end=end_dt)
            bench = _normalize_ohlcv_frame(bench)
            if bench.empty:
                continue
            bench, _ = apply_split_adjustment(
                bars=bench,
                symbol=benchmark_symbol,
                market="TW",
                use_known=True,
                use_auto_detect=True,
            )
            if len(bench) >= 60:
                return bench, benchmark_symbol, sync_issues
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), "", sync_issues

    def _build_rotation_holding_rank(
        *,
        weights_df: Optional[pd.DataFrame],
        selected_symbol_lists: list[list[str]],
    ) -> list[dict[str, object]]:
        name_map = service.get_tw_symbol_names(list(ROTATION_DEFAULT_UNIVERSE))
        selected_counts = {sym: 0 for sym in ROTATION_DEFAULT_UNIVERSE}
        for symbols in selected_symbol_lists:
            for sym in symbols:
                if sym in selected_counts:
                    selected_counts[sym] += 1

        total_days = int(len(weights_df)) if isinstance(weights_df, pd.DataFrame) and not weights_df.empty else 0
        total_signals = max(1, len(selected_symbol_lists))
        rows: list[dict[str, object]] = []
        for sym in ROTATION_DEFAULT_UNIVERSE:
            hold_days = 0
            if total_days > 0 and weights_df is not None and sym in weights_df.columns:
                hold_days = int((pd.to_numeric(weights_df[sym], errors="coerce").fillna(0.0) > 1e-10).sum())
            selected_months = int(selected_counts.get(sym, 0))
            if hold_days <= 0 and selected_months <= 0:
                continue
            hold_ratio_pct = (hold_days / total_days * 100.0) if total_days > 0 else 0.0
            selected_ratio_pct = selected_months / total_signals * 100.0
            rows.append(
                {
                    "symbol": sym,
                    "name": name_map.get(sym, sym),
                    "hold_days": hold_days,
                    "hold_ratio_pct": hold_ratio_pct,
                    "selected_months": selected_months,
                    "selected_ratio_pct": selected_ratio_pct,
                }
            )
        rows.sort(
            key=lambda r: (
                int(r.get("hold_days", 0)),
                int(r.get("selected_months", 0)),
                str(r.get("symbol", "")),
            ),
            reverse=True,
        )
        return rows

    run_key = (
        f"tw_rotation:{start_date}:{end_date}:{benchmark_choice}:"
        f"{top_n}:{fee_rate}:{sell_tax}:{slippage}:{initial_capital}"
    )

    if st.button("執行 ETF 輪動策略回測", type="primary", use_container_width=True):
        if not date_is_valid:
            st.error("日期區間無效，請先修正起訖日期。")
            return

        symbol_sync_issues: list[str] = []
        if sync_before_run:
            with st.spinner("同步 ETF 池日K中..."):
                _, symbol_sync_issues = _sync_symbols_history(
                    store,
                    market="TW",
                    symbols=list(ROTATION_DEFAULT_UNIVERSE),
                    start=start_dt,
                    end=end_dt,
                    parallel=parallel_sync,
                )

        bars_cache: dict[str, pd.DataFrame] = {}
        symbols_need_sync: list[str] = []
        for symbol in ROTATION_DEFAULT_UNIVERSE:
            bars_local = store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
            bars_local = _normalize_ohlcv_frame(bars_local)
            bars_cache[symbol] = bars_local
            if len(bars_local) < ROTATION_MIN_BARS:
                symbols_need_sync.append(symbol)

        if symbols_need_sync and not sync_before_run:
            with st.spinner("本地資料不足，補同步 ETF 日K中..."):
                _, lazy_sync_issues = _sync_symbols_history(
                    store,
                    market="TW",
                    symbols=symbols_need_sync,
                    start=start_dt,
                    end=end_dt,
                    parallel=parallel_sync,
                )
                symbol_sync_issues.extend(lazy_sync_issues)
            for symbol in symbols_need_sync:
                bars_local = store.load_daily_bars(symbol=symbol, market="TW", start=start_dt, end=end_dt)
                bars_cache[symbol] = _normalize_ohlcv_frame(bars_local)

        bars_by_symbol: dict[str, pd.DataFrame] = {}
        skipped_symbols: list[str] = []
        progress = st.progress(0.0)
        for idx, symbol in enumerate(ROTATION_DEFAULT_UNIVERSE):
            bars = bars_cache.get(symbol, pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
            if bars.empty:
                skipped_symbols.append(symbol)
                progress.progress((idx + 1) / len(ROTATION_DEFAULT_UNIVERSE))
                continue
            bars, _ = apply_split_adjustment(
                bars=bars,
                symbol=symbol,
                market="TW",
                use_known=True,
                use_auto_detect=True,
            )
            if len(bars) < ROTATION_MIN_BARS:
                skipped_symbols.append(symbol)
            else:
                bars_by_symbol[symbol] = bars
            progress.progress((idx + 1) / len(ROTATION_DEFAULT_UNIVERSE))
        progress.empty()
        _show_sync_issues("部分 ETF 同步失敗，已盡量使用本地可用資料", symbol_sync_issues)

        if not bars_by_symbol:
            st.error(f"可用資料不足（每檔至少需 {ROTATION_MIN_BARS} 根K），無法執行。")
            return

        benchmark_bars, benchmark_symbol, benchmark_sync_issues = _load_benchmark_bars(
            benchmark_choice,
            sync_first=sync_before_run,
        )
        _show_sync_issues("Benchmark 同步有部分錯誤，已盡量使用本地可用資料", benchmark_sync_issues)
        if benchmark_bars.empty:
            st.error("Benchmark 取得失敗，請改選 0050 或 006208 後重試。")
            return

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

        eq_idx = result.equity_curve.index
        buy_hold_equity = build_buy_hold_equity(
            bars_by_symbol=bars_by_symbol,
            target_index=eq_idx,
            initial_capital=float(initial_capital),
        )
        benchmark_close = pd.to_numeric(benchmark_bars["close"], errors="coerce").reindex(eq_idx).ffill()
        benchmark_non_na = benchmark_close.dropna()
        if benchmark_non_na.empty or float(benchmark_non_na.iloc[0]) <= 0:
            benchmark_equity = pd.Series(dtype=float)
        else:
            benchmark_equity = float(initial_capital) * (benchmark_close / float(benchmark_non_na.iloc[0]))

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

        holding_rank = _build_rotation_holding_rank(
            weights_df=result.weights,
            selected_symbol_lists=selected_symbol_lists,
        )

        trades_df = result.trades.copy()
        if not trades_df.empty and "date" in trades_df.columns:
            trades_df["date"] = pd.to_datetime(trades_df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")

        payload = {
            "run_key": run_key,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "strategy": "tw_etf_rotation_v1",
            "benchmark_symbol": benchmark_symbol,
            "universe_symbols": list(ROTATION_DEFAULT_UNIVERSE),
            "used_symbols": sorted(list(bars_by_symbol.keys())),
            "skipped_symbols": sorted(skipped_symbols),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "top_n": int(top_n_effective),
            "initial_capital": float(initial_capital),
            "metrics": result.metrics.__dict__,
            "equity_curve": [
                {"date": pd.Timestamp(idx).isoformat(), "equity": float(val)}
                for idx, val in result.equity_curve["equity"].items()
            ],
            "benchmark_curve": [
                {"date": pd.Timestamp(idx).isoformat(), "equity": float(val)}
                for idx, val in benchmark_equity.dropna().items()
            ],
            "buy_hold_curve": [
                {"date": pd.Timestamp(idx).isoformat(), "equity": float(val)}
                for idx, val in buy_hold_equity.dropna().items()
            ],
            "rebalance_records": rebalance_rows,
            "trades": trades_df.to_dict("records"),
            "holding_rank": holding_rank,
        }
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

    payload = st.session_state.get(payload_key)
    if not payload:
        st.info("設定條件後按下「執行 ETF 輪動策略回測」，之後會自動顯示最近一次快取結果。")
        return
    if payload.get("run_key") != run_key:
        st.caption("目前顯示的是上一次執行結果；若要套用目前設定，請重新按下執行。")

    strategy_df = pd.DataFrame(payload.get("equity_curve", []))
    if strategy_df.empty or "date" not in strategy_df.columns or "equity" not in strategy_df.columns:
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

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strategy_series.index,
            y=strategy_series.values,
            mode="lines",
            name="Strategy Equity (ETF Rotation)",
            line=dict(color=str(palette["equity"]), width=2.3),
        )
    )
    if not benchmark_series.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_series.index,
                y=benchmark_series.values,
                mode="lines",
                name=f"Benchmark Equity ({payload.get('benchmark_symbol', 'Benchmark')})",
                line=_benchmark_line_style(palette, width=2.0),
            )
        )
    if not buy_hold_series.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_hold_series.index,
                y=buy_hold_series.values,
                mode="lines",
                name="Buy-and-Hold Equity (Equal-Weight ETF Pool)",
                line=dict(color=str(palette["buy_hold"]), width=1.9),
            )
        )
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        template=str(palette["plot_template"]),
        paper_bgcolor=str(palette["paper_bg"]),
        plot_bgcolor=str(palette["plot_bg"]),
        font=dict(color=str(palette["text_color"])),
    )
    fig.update_xaxes(gridcolor=str(palette["grid"]))
    fig.update_yaxes(gridcolor=str(palette["grid"]))
    st.plotly_chart(fig, use_container_width=True)

    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Return", f"{float(metrics.get('total_return', 0.0)) * 100.0:+.2f}%")
        m2.metric("CAGR", f"{float(metrics.get('cagr', 0.0)) * 100.0:+.2f}%")
        m3.metric("Sharpe", f"{float(metrics.get('sharpe', 0.0)):.2f}")
        m4.metric("MDD", f"{float(metrics.get('max_drawdown', 0.0)) * 100.0:+.2f}%")
        m5.metric("Trades", f"{int(metrics.get('trades', 0))}")

    if not benchmark_series.empty:
        comp = pd.concat([strategy_series.rename("strategy"), benchmark_series.rename("benchmark")], axis=1).dropna()
        if len(comp) >= 2:
            strat_ret = float(comp["strategy"].iloc[-1] / comp["strategy"].iloc[0] - 1.0)
            bench_ret = float(comp["benchmark"].iloc[-1] / comp["benchmark"].iloc[0] - 1.0)
            excess = (strat_ret - bench_ret) * 100.0
            verdict = "贏過大盤" if excess > 0 else ("輸給大盤" if excess < 0 else "與大盤持平")
            st.info(f"相對Benchmark：{verdict} {excess:+.2f}%（策略 {strat_ret*100:+.2f}% / Benchmark {bench_ret*100:+.2f}%）")

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
        holding_rank = _build_rotation_holding_rank(weights_df=None, selected_symbol_lists=selected_symbol_lists)

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
        st.caption("說明：此排名反映本次回測參數下的策略偏好（持有天數/入選次數），不代表未來保證報酬。")

    rebalance_df = pd.DataFrame(payload.get("rebalance_records", []))
    if not rebalance_df.empty:
        name_map = service.get_tw_symbol_names(list(ROTATION_DEFAULT_UNIVERSE))

        def _format_selected(v: object) -> str:
            if not isinstance(v, list):
                return "—"
            if not v:
                return "空手"
            return "、".join([f"{sym} {name_map.get(sym, '')}".strip() for sym in v])

        rebalance_df["signal_date"] = pd.to_datetime(rebalance_df["signal_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        rebalance_df["effective_date"] = pd.to_datetime(rebalance_df["effective_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        rebalance_df["selected"] = rebalance_df["selected_symbols"].map(_format_selected)
        rebalance_df["market_filter"] = rebalance_df["market_filter_on"].map(lambda v: "ON" if bool(v) else "OFF")
        rebalance_df["scores"] = rebalance_df["scores"].map(
            lambda obj: "、".join([f"{k}:{float(v):+.3f}" for k, v in obj.items()]) if isinstance(obj, dict) and obj else "—"
        )
        st.markdown("#### 每月調倉明細")
        st.dataframe(
            rebalance_df[["signal_date", "effective_date", "market_filter", "selected", "scores"]],
            use_container_width=True,
            hide_index=True,
        )

    trades_df = pd.DataFrame(payload.get("trades", []))
    if not trades_df.empty:
        st.markdown("#### 成交紀錄（前200筆）")
        show_cols = [c for c in ["date", "symbol", "side", "qty", "price", "notional", "fee", "tax", "slippage", "pnl", "target_weight"] if c in trades_df.columns]
        st.dataframe(trades_df[show_cols].head(200), use_container_width=True, hide_index=True)


def _render_00935_heatmap_view():
    _render_tw_etf_heatmap_view("00935", page_desc="科技類")


def _render_00910_heatmap_view():
    _render_tw_etf_heatmap_view("00910", page_desc="第一金太空衛星")


def _render_0050_heatmap_view():
    _render_tw_etf_heatmap_view("0050", page_desc="台灣50")


def _render_00993a_heatmap_view():
    _render_tw_etf_heatmap_view("00993A", page_desc="台股ETF")


def _render_db_browser_view():
    st.subheader("SQLite 資料庫檢視")
    store = _history_store()
    db_path = store.db_path

    st.caption(f"資料庫路徑：`{db_path}`")
    if not db_path.exists():
        st.error(f"找不到資料庫檔案：`{db_path}`。")
        return

    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    st.caption(f"檔案大小：約 {db_size_mb:.2f} MB")

    conn = sqlite3.connect(str(db_path))
    try:
        tables_df = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name ASC
            """,
            conn,
        )
        if tables_df.empty:
            st.info("目前資料庫中沒有可檢視的資料表。")
            return

        table_names = tables_df["name"].astype(str).tolist()
        summary_rows: list[dict[str, object]] = []
        count_map: dict[str, int] = {}
        for table in table_names:
            escaped = table.replace('"', '""')
            count = int(conn.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0])
            count_map[table] = count
            summary_rows.append({"資料表": table, "筆數": count})

        st.markdown("#### 資料表總覽")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns([2, 1, 1])
        selected_table = c1.selectbox("選擇資料表", options=table_names, key="db_view_table")
        page_size = int(c2.selectbox("每頁筆數", options=[20, 50, 100, 200, 500], index=2, key="db_view_page_size"))
        order_mode = c3.selectbox("排序", options=["最新在前", "舊到新"], index=0, key="db_view_order_mode")

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
        schema_df = pd.read_sql_query(f'PRAGMA table_info("{escaped_table}")', conn)
        col_names = schema_df["name"].astype(str).tolist() if not schema_df.empty else []
        order_candidates = ["updated_at", "created_at", "date", "fetched_at", "id"]
        order_col = next((col for col in order_candidates if col in col_names), None)
        direction = "DESC" if order_mode == "最新在前" else "ASC"

        query = f'SELECT * FROM "{escaped_table}"'
        if order_col:
            query += f' ORDER BY "{order_col}" {direction}'
        query += " LIMIT ? OFFSET ?"

        data_df = pd.read_sql_query(query, conn, params=[page_size, offset])
        st.markdown("#### 資料表內容")
        if data_df.empty:
            st.info("此頁沒有資料。")
        else:
            st.dataframe(data_df, use_container_width=True, hide_index=True)

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
                show_cols = [c for c in ["cid", "欄位", "型別", "不可為空", "預設值", "主鍵"] if c in schema_out.columns]
                st.dataframe(schema_out[show_cols], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"讀取資料庫失敗：{exc}")
    finally:
        conn.close()


def main():
    st.set_page_config(page_title="即時看盤 + 回測平台", layout="wide")
    _inject_ui_styles()
    st.title("即時走勢 / 多來源資料 / 回測平台")
    _render_design_toolbox()
    active_page = _render_page_cards_nav()
    page_renderers = {
        "即時看盤": _render_live_view,
        "回測工作台": _render_backtest_view,
        "2025 前十大 ETF": _render_top10_etf_2025_view,
        "2026 YTD 前十大 ETF": _render_top10_etf_2026_ytd_view,
        "2026 YTD 主動式 ETF": _render_active_etf_2026_ytd_view,
        "ETF 輪動策略": _render_tw_etf_rotation_view,
        "00910 熱力圖": _render_00910_heatmap_view,
        "00935 熱力圖": _render_00935_heatmap_view,
        "00993A 熱力圖": _render_00993a_heatmap_view,
        "0050 熱力圖": _render_0050_heatmap_view,
        "資料庫檢視": _render_db_browser_view,
        "新手教學": _render_tutorial_view,
    }
    render_fn = page_renderers.get(active_page)
    if render_fn is None:
        st.error("頁面載入失敗，請重新整理後再試。")
        return
    render_fn()


if __name__ == "__main__":
    main()
