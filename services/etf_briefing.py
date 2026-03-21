from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests

from services.chart_export import export_backtest_chart_artifact
from services.sync_orchestrator import normalize_symbols
from services.tw_etf_super_export import export_tw_etf_super_table_artifact
from storage import HistoryStore
from utils import normalize_ohlcv_frame


DEFAULT_ETF_BRIEFING_ROOT = Path("~/Downloads/etf").expanduser()
DEFAULT_ETF_BRIEFING_THEME = "soft-gray"
DEFAULT_ETF_BRIEFING_SINGLE_SYMBOL = "00935"
DEFAULT_ETF_BRIEFING_START = "2023-01-01"
DEFAULT_ETF_BRIEFING_NEWS_WINDOW_DAYS = 14


def export_etf_briefing_artifact(
    *,
    store: HistoryStore,
    out_root: str | None = None,
    start: str | None = None,
    end: str | None = None,
    single_symbol: str | None = None,
    theme: str = DEFAULT_ETF_BRIEFING_THEME,
    news_window_days: int = DEFAULT_ETF_BRIEFING_NEWS_WINDOW_DAYS,
    include_extra_splits: bool = True,
) -> dict[str, object]:
    now = _resolve_now(end)
    briefing_dir = resolve_etf_briefing_output_dir(
        out_root=out_root,
        run_date=now.date(),
    )
    charts_dir = briefing_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    briefing_dir.mkdir(parents=True, exist_ok=True)

    start_dt = _resolve_start_datetime(start)
    end_dt = now
    single_code = str(single_symbol or DEFAULT_ETF_BRIEFING_SINGLE_SYMBOL).strip().upper()

    super_export_result = export_tw_etf_super_table_artifact(
        store=store,
        out=str(briefing_dir),
        ytd_end=now.strftime("%Y%m%d"),
    )
    super_export_path = Path(str(super_export_result["output_path"]))
    super_export_df = _load_super_export_frame(super_export_path)
    tech_df = _filter_group_frame(super_export_df, type_label="科技型")
    active_df = _filter_group_frame(super_export_df, type_label="主動式")
    tech_csv_path = briefing_dir / f"tech_etf_from_super_export_{super_export_result['trade_date_anchor']}.csv"
    active_csv_path = briefing_dir / f"active_etf_from_super_export_{super_export_result['trade_date_anchor']}.csv"
    _write_csv(tech_csv_path, tech_df)
    _write_csv(active_csv_path, active_df)

    issues: list[str] = []
    tech_symbols = _prepare_chartable_symbols(
        store=store,
        symbols=tech_df.get("代碼", pd.Series(dtype=str)).astype(str).tolist(),
        start=start_dt,
        end=end_dt,
    )
    active_symbols = _prepare_chartable_symbols(
        store=store,
        symbols=active_df.get("代碼", pd.Series(dtype=str)).astype(str).tolist(),
        start=start_dt,
        end=end_dt,
    )
    if len(tech_symbols["excluded"]) > 0:
        issues.append(f"科技型排除無足夠資料標的：{', '.join(tech_symbols['excluded'])}")
    if len(active_symbols["excluded"]) > 0:
        issues.append(f"主動式排除無足夠資料標的：{', '.join(active_symbols['excluded'])}")

    chart_outputs: dict[str, object] = {}
    chart_outputs["tech_combined"] = _export_chart_safe(
        store=store,
        symbols=tech_symbols["included"],
        layout="combined",
        start=start_dt,
        end=end_dt,
        theme=theme,
        out_path=charts_dir / f"tech_etf_combined_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.png",
        issues=issues,
        label="科技型 combined",
    )
    chart_outputs["active_combined"] = _export_chart_safe(
        store=store,
        symbols=active_symbols["included"],
        layout="combined",
        start=start_dt,
        end=end_dt,
        theme=theme,
        out_path=charts_dir / f"active_etf_combined_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.png",
        issues=issues,
        label="主動式 combined",
    )
    chart_outputs["single_00935"] = _export_chart_safe(
        store=store,
        symbols=[single_code],
        layout="single",
        start=start_dt,
        end=end_dt,
        theme=theme,
        out_path=charts_dir / f"{single_code}_single_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.png",
        issues=issues,
        label=f"{single_code} single",
    )

    extra_outputs: dict[str, object] = {}
    if include_extra_splits:
        extra_outputs["tech_split"] = _export_split_chart_group(
            store=store,
            symbols=tech_symbols["included"],
            start=start_dt,
            end=end_dt,
            theme=theme,
            out_dir=charts_dir / "tech_etf_split",
            issues=issues,
            label="科技型 split",
        )
        extra_outputs["active_split"] = _export_split_chart_group(
            store=store,
            symbols=active_symbols["included"],
            start=start_dt,
            end=end_dt,
            theme=theme,
            out_dir=charts_dir / "active_etf_split",
            issues=issues,
            label="主動式 split",
        )

    top_tech_symbol = _select_top_symbol(tech_df)
    top_active_symbol = _select_top_symbol(active_df)
    if top_tech_symbol:
        extra_outputs["top_tech_single"] = _export_chart_safe(
            store=store,
            symbols=[top_tech_symbol],
            layout="single",
            start=start_dt,
            end=end_dt,
            theme=theme,
            out_path=charts_dir / f"{top_tech_symbol}_top_tech_single_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.png",
            issues=issues,
            label="科技型領頭 single",
        )
    if top_active_symbol:
        extra_outputs["top_active_single"] = _export_chart_safe(
            store=store,
            symbols=[top_active_symbol],
            layout="single",
            start=start_dt,
            end=end_dt,
            theme=theme,
            out_path=charts_dir / f"{top_active_symbol}_top_active_single_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.png",
            issues=issues,
            label="主動式領頭 single",
        )

    news_payload = collect_recent_geopolitics_news(now=now, window_days=news_window_days)
    news_sources_path = briefing_dir / "news_sources.json"
    news_sources_path.write_text(
        json.dumps(news_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    context = build_etf_briefing_context(
        super_export_path=super_export_path,
        super_export_df=super_export_df,
        tech_df=tech_df,
        active_df=active_df,
        chart_outputs=chart_outputs,
        extra_outputs=extra_outputs,
        single_symbol=single_code,
        start_dt=start_dt,
        end_dt=end_dt,
        trade_date_anchor=str(super_export_result["trade_date_anchor"]),
        news_payload=news_payload,
        issues=issues,
    )
    report_html = render_sima_yi_html_report(context)
    report_html_path = briefing_dir / "report_sima_yi.html"
    report_html_path.write_text(report_html, encoding="utf-8")

    fb_text = render_sima_yi_fb_post(context)
    fb_text_path = briefing_dir / "fb_post_sima_yi.txt"
    fb_text_path.write_text(fb_text, encoding="utf-8")
    fb_html_path = briefing_dir / "fb_post_sima_yi.html"
    fb_html_path.write_text(render_fb_preview_html(fb_text), encoding="utf-8")

    manifest = {
        "generated_at": now.isoformat(),
        "briefing_dir": str(briefing_dir),
        "super_export_csv": str(super_export_path),
        "tech_csv": str(tech_csv_path),
        "active_csv": str(active_csv_path),
        "tech_symbols": tech_symbols["included"],
        "active_symbols": active_symbols["included"],
        "excluded_symbols": {
            "tech": tech_symbols["excluded"],
            "active": active_symbols["excluded"],
        },
        "chart_outputs": _json_safe({**chart_outputs, **extra_outputs}),
        "report_html": str(report_html_path),
        "fb_text": str(fb_text_path),
        "fb_html": str(fb_html_path),
        "news_sources": str(news_sources_path),
        "issues": issues,
    }
    manifest_path = briefing_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "briefing_dir": str(briefing_dir),
        "super_export_csv": str(super_export_path),
        "tech_csv": str(tech_csv_path),
        "active_csv": str(active_csv_path),
        "report_html": str(report_html_path),
        "fb_text": str(fb_text_path),
        "fb_html": str(fb_html_path),
        "manifest": str(manifest_path),
        "news_sources": str(news_sources_path),
        "issues": issues,
    }


def resolve_etf_briefing_output_dir(*, out_root: str | None, run_date) -> Path:
    root = Path(str(out_root or DEFAULT_ETF_BRIEFING_ROOT)).expanduser()
    return root / f"briefing_{pd.Timestamp(run_date).strftime('%Y%m%d')}"


def collect_recent_geopolitics_news(
    *,
    now: datetime,
    window_days: int,
) -> dict[str, object]:
    queries = [
        (
            "taiwan_us_china",
            "台海與美中",
            "Taiwan Strait OR US China tariffs semiconductors when:{days}d",
            {"reuters", "associated press", "ap news", "ap", "路透"},
            {"taiwan", "china", "us", "semiconductor", "tariff", "chip", "strait"},
        ),
        (
            "ukraine_russia",
            "俄烏戰局",
            "Russia Ukraine ceasefire OR frontline when:{days}d",
            {"reuters", "associated press", "ap news", "ap", "路透"},
            {"russia", "ukraine", "ceasefire", "frontline", "missile", "drone", "moscow", "kyiv"},
        ),
        (
            "middle_east_energy",
            "中東能源與航運",
            "Middle East oil shipping Red Sea Hormuz when:{days}d",
            {"reuters", "associated press", "ap news", "ap", "wsj", "wall street journal", "路透"},
            {"middle east", "oil", "shipping", "red sea", "hormuz", "saudi", "aramco", "tanker"},
        ),
    ]
    groups: list[dict[str, object]] = []
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )
    for key, label, query_tpl, trusted_sources, title_keywords in queries:
        query = query_tpl.format(days=max(1, int(window_days)))
        url = (
            "https://news.google.com/rss/search?q="
            + quote_plus(query)
            + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        )
        items = _fetch_google_news_rss(session=session, url=url, now=now, window_days=window_days)
        filtered_items: list[dict[str, object]] = []
        for item in items:
            source = str(item.get("source") or "").strip().lower()
            title = str(item.get("title") or "").strip().lower()
            if trusted_sources and source and source not in trusted_sources:
                continue
            if title_keywords and not any(keyword in title for keyword in title_keywords):
                continue
            filtered_items.append(item)
        groups.append(
            {
                "key": key,
                "label": label,
                "query": query,
                "items": filtered_items[:5],
            }
        )
    return {
        "generated_at": now.isoformat(),
        "window_days": int(window_days),
        "groups": groups,
    }


def build_etf_briefing_context(
    *,
    super_export_path: Path,
    super_export_df: pd.DataFrame,
    tech_df: pd.DataFrame,
    active_df: pd.DataFrame,
    chart_outputs: dict[str, object],
    extra_outputs: dict[str, object],
    single_symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    trade_date_anchor: str,
    news_payload: dict[str, object],
    issues: list[str],
) -> dict[str, object]:
    market_row = _extract_symbol_row(super_export_df, "^TWII")
    single_row = _extract_symbol_row(super_export_df, single_symbol)
    return {
        "trade_date_anchor": trade_date_anchor,
        "period_label": f"{start_dt.strftime('%Y-%m-%d')} 至 {end_dt.strftime('%Y-%m-%d')}",
        "super_export_name": super_export_path.name,
        "hero_line": _build_hero_line(tech_df=tech_df, active_df=active_df, market_row=market_row),
        "market_summary": _build_market_summary(market_row=market_row, tech_df=tech_df, active_df=active_df),
        "tech_count": int(len(tech_df)),
        "active_count": int(len(active_df)),
        "tech_summary": _build_group_summary("科技型", tech_df),
        "active_summary": _build_group_summary("主動式", active_df),
        "single_symbol": single_symbol,
        "single_summary": _build_single_symbol_summary(single_symbol, single_row),
        "single_chart": chart_outputs.get("single_00935"),
        "news_sections": _build_news_sections(news_payload),
        "tech_rows": _table_rows(tech_df),
        "active_rows": _table_rows(active_df),
        "chart_outputs": chart_outputs,
        "extra_outputs": extra_outputs,
        "issues": list(issues),
    }


def render_sima_yi_html_report(context: dict[str, object]) -> str:
    tech_table = _render_html_table(list(context.get("tech_rows") or []))
    active_table = _render_html_table(list(context.get("active_rows") or []))
    extra_gallery = _render_extra_gallery(context.get("extra_outputs") or {})
    news_blocks = "\n".join(
        f"<section class='news-block'><h3>{escape(str(item['title']))}</h3><p>{escape(str(item['body']))}</p></section>"
        for item in list(context.get("news_sections") or [])
    )
    issues_html = ""
    if list(context.get("issues") or []):
        issues_html = (
            "<section class='issues'><h2>軍情附記</h2><ul>"
            + "".join(f"<li>{escape(str(item))}</li>" for item in list(context.get("issues") or []))
            + "</ul></section>"
        )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>司馬懿密奏 ETF 戰報</title>
  <style>
    :root {{
      --bg:#0b1020; --panel:#121a2f; --panel2:#17213a; --text:#e8ecf4; --muted:#9aa5bd;
      --line:#2b3a5c; --accent:#c9a86a; --green:#22c55e; --red:#ef4444; --blue:#60a5fa;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif; background:radial-gradient(circle at top,#17213a 0%,#0b1020 55%,#070b16 100%); color:var(--text); }}
    .wrap {{ max-width:1400px; margin:0 auto; padding:32px 24px 60px; }}
    .hero {{ padding:28px 30px; background:linear-gradient(135deg,rgba(201,168,106,0.12),rgba(23,33,58,0.96)); border:1px solid rgba(201,168,106,0.28); border-radius:22px; box-shadow:0 20px 80px rgba(0,0,0,0.28); }}
    .hero h1 {{ margin:0 0 10px; font-size:34px; letter-spacing:0.06em; }}
    .hero .meta {{ color:var(--muted); font-size:14px; }}
    .hero .lead {{ margin-top:18px; font-size:22px; line-height:1.7; color:#f4e7c7; }}
    .section {{ margin-top:26px; background:rgba(18,26,47,0.92); border:1px solid rgba(154,165,189,0.14); border-radius:18px; padding:22px; }}
    .section h2 {{ margin:0 0 12px; font-size:24px; color:#f6f1e7; }}
    .section p {{ color:var(--text); line-height:1.85; font-size:16px; }}
    .cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-top:14px; }}
    .card {{ background:var(--panel2); border:1px solid rgba(96,165,250,0.16); border-radius:14px; padding:16px; }}
    .card .label {{ color:var(--muted); font-size:13px; }}
    .card .value {{ margin-top:6px; font-size:24px; font-weight:700; }}
    .chart {{ margin-top:16px; border:1px solid rgba(154,165,189,0.14); border-radius:14px; overflow:hidden; background:#0d1426; }}
    .chart img {{ display:block; width:100%; height:auto; }}
    table {{ width:100%; border-collapse:collapse; margin-top:14px; }}
    th, td {{ padding:10px 12px; border-bottom:1px solid rgba(154,165,189,0.12); text-align:left; font-size:14px; }}
    th {{ color:#f5e9ce; background:rgba(201,168,106,0.08); }}
    td {{ color:var(--text); }}
    .gallery {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:16px; }}
    .gallery .tile {{ background:var(--panel2); border:1px solid rgba(154,165,189,0.14); border-radius:14px; padding:12px; }}
    .gallery .tile img {{ width:100%; border-radius:10px; display:block; }}
    .gallery .tile .caption {{ margin-top:10px; color:var(--muted); font-size:14px; line-height:1.7; }}
    .news-block + .news-block {{ margin-top:14px; padding-top:14px; border-top:1px dashed rgba(154,165,189,0.2); }}
    .news-block h3 {{ margin:0 0 8px; font-size:18px; color:#f3e5bf; }}
    .issues ul {{ margin:0; padding-left:20px; }}
    .footer-note {{ margin-top:28px; color:var(--muted); font-size:13px; }}
    @media (max-width: 980px) {{ .cards,.gallery {{ grid-template-columns:1fr; }} .hero h1 {{ font-size:28px; }} .hero .lead {{ font-size:19px; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="meta">密奏成文日：{escape(str(context.get("trade_date_anchor") or ""))} ｜ 回看區間：{escape(str(context.get("period_label") or ""))}</div>
      <h1>司馬懿為主公進呈 ETF 戰報</h1>
      <div class="lead">{escape(str(context.get("hero_line") or ""))}</div>
    </section>

    <section class="section">
      <h2>全局先判</h2>
      <p>{escape(str(context.get("market_summary") or ""))}</p>
      <div class="cards">
        <div class="card"><div class="label">科技型標的數</div><div class="value">{int(context.get("tech_count") or 0)}</div></div>
        <div class="card"><div class="label">主動式標的數</div><div class="value">{int(context.get("active_count") or 0)}</div></div>
        <div class="card"><div class="label">焦點單檔</div><div class="value">{escape(str(context.get("single_symbol") or ""))}</div></div>
      </div>
    </section>

    <section class="section">
      <h2>科技型 ETF 兵勢</h2>
      <p>{escape(str(context.get("tech_summary") or ""))}</p>
      <div class="chart"><img src="{escape(_rel_chart(context.get('chart_outputs', {}).get('tech_combined')))}" alt="tech combined"></div>
      {tech_table}
    </section>

    <section class="section">
      <h2>主動式 ETF 兵勢</h2>
      <p>{escape(str(context.get("active_summary") or ""))}</p>
      <div class="chart"><img src="{escape(_rel_chart(context.get('chart_outputs', {}).get('active_combined')))}" alt="active combined"></div>
      {active_table}
    </section>

    <section class="section">
      <h2>{escape(str(context.get("single_symbol") or ""))} 單檔觀察</h2>
      <p>{escape(str(context.get("single_summary") or ""))}</p>
      <div class="chart"><img src="{escape(_rel_chart(context.get('single_chart')))}" alt="single"></div>
    </section>

    <section class="section">
      <h2>臣所加畫的附圖</h2>
      <p>主圖只看大勢，未必見枝節；臣故將個別圖另列，使主公觀兵如觀陣，不只看誰聲勢大，也看誰腳步亂。</p>
      {extra_gallery}
    </section>

    <section class="section">
      <h2>地緣局勢與盤面暗線</h2>
      <p>臣不敢只看盤中漲跌，外頭風聲若變，內裡資金便先動。故臣已將近訊與盤面一併勘合，得此數端：</p>
      {news_blocks}
    </section>

    {issues_html}

    <div class="footer-note">臣以盤面、資金、ETF 類型與近訊交叉觀之，務求為主公先定局、後定策。</div>
  </div>
</body>
</html>
"""


def render_sima_yi_fb_post(context: dict[str, object]) -> str:
    news_lines = []
    for item in list(context.get("news_sections") or [])[:3]:
        news_lines.append(f"{item['title']}：{item['short_line']}")
    hashtags = "#ETF #台股 #科技ETF #主動式ETF #00935 #市場觀察"
    return "\n".join(
        [
            "主公。",
            str(context.get("hero_line") or ""),
            "",
            f"今臣先看科技型，再看主動式，最後回到 {context.get('single_symbol')} 這一檔。",
            str(context.get("tech_summary") or ""),
            str(context.get("active_summary") or ""),
            str(context.get("single_summary") or ""),
            "",
            "再說外局。臣以為，盤面真正的壓力不在表面的紅綠，而在外圍風向怎麼逼資金選邊：",
            *news_lines,
            "",
            "主公若要的是運籌帷幄，不是追逐喧嘩，便該先辨誰能打持久戰，誰只是趁亂抬轎。",
            hashtags,
        ]
    ).strip() + "\n"


def render_fb_preview_html(text: str) -> str:
    safe = escape(str(text or "")).replace("\n", "<br>")
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FB Post Preview</title>
<style>body{{margin:0;background:#f5f7fb;font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif;color:#111827;}}
.wrap{{max-width:760px;margin:32px auto;padding:0 18px;}} .card{{background:#fff;border:1px solid #dbe2ee;border-radius:18px;padding:22px;box-shadow:0 12px 40px rgba(15,23,42,.08);line-height:1.9;font-size:18px;white-space:normal;}}
h1{{font-size:22px;}}</style></head><body><div class="wrap"><h1>FB 貼文預覽</h1><div class="card">{safe}</div></div></body></html>"""


def _load_super_export_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _filter_group_frame(frame: pd.DataFrame, *, type_label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    if "代碼" not in work.columns or "類型" not in work.columns:
        return pd.DataFrame(columns=work.columns)
    work["代碼"] = work["代碼"].astype(str).str.replace('="', "", regex=False).str.replace('"', "", regex=False).str.strip().str.upper()
    work = work[~work["代碼"].astype(str).str.startswith("^")].copy()
    work = work[work["類型"].astype(str).str.strip() == str(type_label).strip()].copy()
    if "YTD績效(%)" in work.columns:
        scores = pd.to_numeric(work["YTD績效(%)"], errors="coerce")
        work = work.assign(__score=scores).sort_values("__score", ascending=False, na_position="last").drop(columns="__score")
    return work.reset_index(drop=True)


def _prepare_chartable_symbols(
    *,
    store: HistoryStore,
    symbols: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, list[str]]:
    clean_symbols = normalize_symbols(symbols)
    if not clean_symbols:
        return {"included": [], "excluded": []}
    included: list[str] = []
    excluded: list[str] = []
    for symbol in clean_symbols:
        bars = normalize_ohlcv_frame(
            store.load_daily_bars(symbol=symbol, market="TW", start=start, end=end)
        )
        if len(bars) >= 2:
            included.append(symbol)
        else:
            excluded.append(symbol)
    return {"included": included, "excluded": excluded}


def _export_chart_safe(
    *,
    store: HistoryStore,
    symbols: list[str],
    layout: str,
    start: datetime,
    end: datetime,
    theme: str,
    out_path: Path,
    issues: list[str],
    label: str,
) -> dict[str, object]:
    clean_symbols = normalize_symbols(symbols)
    if not clean_symbols:
        issues.append(f"{label}: no eligible symbols")
        return {"path": "", "symbols": [], "layout": layout}
    try:
        result = export_backtest_chart_artifact(
            store=store,
            symbols=clean_symbols,
            layout=layout,
            market="TW",
            start=start,
            end=end,
            strategy="buy_hold",
            benchmark_choice="auto",
            initial_capital=1_000_000.0,
            fee_rate=None,
            sell_tax=None,
            slippage=None,
            sync_before_run=False,
            use_split_adjustment=True,
            use_total_return_adjustment=True,
            theme=theme,
            width=1800,
            height=960,
            scale=2,
            out=str(out_path),
            out_dir=None,
            include_ew_portfolio=False,
        )
        return {"path": str(out_path), "symbols": clean_symbols, "layout": layout, "result": result}
    except Exception as exc:
        issues.append(f"{label}: {exc}")
        return {"path": "", "symbols": clean_symbols, "layout": layout, "error": str(exc)}


def _export_split_chart_group(
    *,
    store: HistoryStore,
    symbols: list[str],
    start: datetime,
    end: datetime,
    theme: str,
    out_dir: Path,
    issues: list[str],
    label: str,
) -> dict[str, object]:
    clean_symbols = normalize_symbols(symbols)
    if not clean_symbols:
        issues.append(f"{label}: no eligible symbols")
        return {"paths": [], "symbols": []}
    try:
        result = export_backtest_chart_artifact(
            store=store,
            symbols=clean_symbols,
            layout="split",
            market="TW",
            start=start,
            end=end,
            strategy="buy_hold",
            benchmark_choice="auto",
            initial_capital=1_000_000.0,
            fee_rate=None,
            sell_tax=None,
            slippage=None,
            sync_before_run=False,
            use_split_adjustment=True,
            use_total_return_adjustment=True,
            theme=theme,
            width=1600,
            height=900,
            scale=2,
            out=None,
            out_dir=str(out_dir),
        )
        return {"paths": [str(item.get("path")) for item in result.get("items", []) if isinstance(item, dict)], "symbols": clean_symbols}
    except Exception as exc:
        issues.append(f"{label}: {exc}")
        return {"paths": [], "symbols": clean_symbols, "error": str(exc)}


def _select_top_symbol(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "代碼" not in frame.columns:
        return ""
    return str(frame.iloc[0]["代碼"]).strip().upper()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    safe = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    safe.to_csv(path, index=False, encoding="utf-8-sig")


def _resolve_now(end: str | None) -> datetime:
    if not str(end or "").strip():
        return datetime.now().astimezone()
    parsed = pd.Timestamp(str(end).strip())
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


def _resolve_start_datetime(start: str | None) -> datetime:
    token = str(start or DEFAULT_ETF_BRIEFING_START).strip()
    parsed = pd.Timestamp(token)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


def _fetch_google_news_rss(
    *,
    session: requests.Session,
    url: str,
    now: datetime,
    window_days: int,
) -> list[dict[str, object]]:
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
    except Exception:
        return []
    try:
        root = ET.fromstring(response.text)
    except Exception:
        return []
    items: list[dict[str, object]] = []
    cutoff = now - timedelta(days=max(1, int(window_days)))
    for item in root.findall(".//item"):
        title_raw = _xml_text(item.find("title"))
        link = _xml_text(item.find("link"))
        pub_date_raw = _xml_text(item.find("pubDate"))
        pub_dt = _parse_rfc2822_datetime(pub_date_raw)
        if pub_dt is None:
            continue
        if pub_dt < cutoff:
            continue
        title, source = _strip_google_news_source(title_raw)
        if not title:
            continue
        items.append(
            {
                "title": title,
                "source": source,
                "url": link,
                "published_at": pub_dt.astimezone(timezone.utc).isoformat(),
                "published_date": pub_dt.astimezone(timezone.utc).strftime("%Y-%m-%d"),
            }
        )
    return items[:5]


def _xml_text(node) -> str:
    return str(getattr(node, "text", "") or "").strip()


def _parse_rfc2822_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _strip_google_news_source(title: str) -> tuple[str, str]:
    text = re.sub(r"\s+", " ", str(title or "").strip())
    if " - " not in text:
        return text, ""
    main, source = text.rsplit(" - ", 1)
    return main.strip(), source.strip()


def _extract_symbol_row(frame: pd.DataFrame, symbol: str) -> dict[str, object]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "代碼" not in frame.columns:
        return {}
    code_series = frame["代碼"].astype(str).str.replace('="', "", regex=False).str.replace('"', "", regex=False).str.strip().str.upper()
    matched = frame.loc[code_series == str(symbol).strip().upper()]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def _build_hero_line(*, tech_df: pd.DataFrame, active_df: pd.DataFrame, market_row: dict[str, object]) -> str:
    tech_top = _select_top_symbol(tech_df)
    active_top = _select_top_symbol(active_df)
    market_ytd = _safe_number(market_row.get("YTD績效(%)"))
    if market_ytd is not None and market_ytd >= 0:
        market_clause = f"大勢仍未失血，YTD 尚有 {market_ytd:.2f}% 的餘溫"
    elif market_ytd is not None:
        market_clause = f"大勢已有退潮之象，YTD 落至 {market_ytd:.2f}%"
    else:
        market_clause = "大勢雖未盡明，然資金去向已露端倪"
    return (
        f"主公，臣觀今局，{market_clause}；科技型之中以 {tech_top or '少數強者'} 最能領兵，"
        f"主動式之中則以 {active_top or '新兵數檔'} 最見鋒芒。此時可觀強，不可貪多；可借勢，不可亂追。"
    )


def _build_market_summary(*, market_row: dict[str, object], tech_df: pd.DataFrame, active_df: pd.DataFrame) -> str:
    market_ytd = _safe_number(market_row.get("YTD績效(%)"))
    tech_count = len(tech_df)
    active_count = len(active_df)
    tech_avg = _mean_pct(tech_df.get("YTD績效(%)", pd.Series(dtype=float)))
    active_avg = _mean_pct(active_df.get("YTD績效(%)", pd.Series(dtype=float)))
    parts = []
    if market_ytd is not None:
        parts.append(f"大盤 YTD 約為 {market_ytd:.2f}%")
    parts.append(f"科技型樣本 {tech_count} 檔")
    if tech_avg is not None:
        parts.append(f"均值約 {tech_avg:.2f}%")
    parts.append(f"主動式樣本 {active_count} 檔")
    if active_avg is not None:
        parts.append(f"均值約 {active_avg:.2f}%")
    return (
        "臣先以大表總覽全局，再分兵細看板塊。"
        + "，".join(parts)
        + "。若把資金比作軍糧，今日最要緊的，不是追問誰叫得最大聲，而是看誰能把隊伍帶到最後。"
    )


def _build_group_summary(group_label: str, frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return f"{group_label}一線，眼下尚未形成可稱一隊之勢，宜暫列觀察。"
    top = frame.iloc[0].to_dict()
    leader = str(top.get("代碼") or "").strip().upper()
    leader_name = str(top.get("ETF") or "").strip()
    leader_ytd = _safe_number(top.get("YTD績效(%)"))
    avg = _mean_pct(frame.get("YTD績效(%)", pd.Series(dtype=float)))
    dispersion = _dispersion_pct(frame.get("YTD績效(%)", pd.Series(dtype=float)))
    line = f"{group_label}這一路，領頭者暫看 {leader} {leader_name}".strip()
    if leader_ytd is not None:
        line += f"，YTD 約 {leader_ytd:.2f}%"
    if avg is not None:
        line += f"；全隊均值約 {avg:.2f}%"
    if dispersion is not None:
        line += f"，強弱分化約 {dispersion:.2f} 個百分點"
    line += "。臣意，此類若走得整齊，可視為可用之兵；若只剩一兩檔硬撐，便只是孤軍冒進。"
    return line


def _build_single_symbol_summary(symbol: str, row: dict[str, object]) -> str:
    if not row:
        return f"臣對 {symbol} 已另畫單檔圖。其勢可觀，但仍以圖形與資金曲線為準，不以一時聲量為準。"
    ytd = _safe_number(row.get("YTD績效(%)"))
    kind = str(row.get("類型") or "").strip()
    text = f"{symbol} 屬 {kind or 'ETF'}。"
    if ytd is not None:
        text += f" 其 YTD 約 {ytd:.2f}%。"
    text += " 臣特別將其單獨拉出，是因這一檔常介於題材與紀律之間，若看得懂節奏，便知何時守、何時取。"
    return text


def _build_news_sections(news_payload: dict[str, object]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for group in list(news_payload.get("groups") or []):
        label = str(group.get("label") or "").strip()
        items = list(group.get("items") or [])
        if not items:
            sections.append(
                {
                    "title": f"{label}一線",
                    "body": "臣近觀此線，公開消息不足以定大局，故不敢妄下重語；然盤面仍需為此預留一分戒心。",
                    "short_line": "近十四日未見足以改寫盤勢的公開新變，宜守而不躁。",
                }
            )
            continue
        first = items[0]
        title = str(first.get("title") or "").strip()
        date_text = str(first.get("published_date") or "").strip()
        sections.append(
            {
                "title": f"{label}一線：{date_text}",
                "body": (
                    f"臣遍觀近訊，此線最值得記下的，是 {date_text} 前後所見的「{title}」。"
                    f" 這未必立時掀翻盤勢，卻足以改變資金對風險、供應鏈與估值的忍耐度。"
                    f" 故凡與科技權值、主題資產、運價與能源預期相關者，都不可只看表面漲跌。"
                ),
                "short_line": f"{date_text} 前後風聲最重，市場不一定立刻表態，但資金已先調腳步。",
            }
        )
    return sections


def _table_rows(frame: pd.DataFrame) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return out
    keep_cols = [col for col in ["代碼", "ETF", "類型", "YTD績效(%)", "ETF規模(億)"] if col in frame.columns]
    for _, row in frame[keep_cols].head(20).iterrows():
        out.append({str(col): _fmt_cell(row.get(col)) for col in keep_cols})
    return out


def _fmt_cell(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"0\d+[A-Z]?", text, flags=re.IGNORECASE):
            return text.upper()
        if re.fullmatch(r"\^\w+", text, flags=re.IGNORECASE):
            return text.upper()
    number = _safe_number(value)
    if number is not None and math.isfinite(number):
        if abs(number) >= 1000:
            return f"{number:,.2f}"
        return f"{number:.2f}"
    return str(value or "").strip()


def _render_html_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<p>本組目前無可呈表格資料。</p>"
    headers = list(rows[0].keys())
    head_html = "".join(f"<th>{escape(col)}</th>" for col in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{escape(str(row.get(col) or ''))}</td>" for col in headers) + "</tr>")
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _render_extra_gallery(extra_outputs: dict[str, object]) -> str:
    tiles: list[str] = []
    for key, value in dict(extra_outputs or {}).items():
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("paths"), list) and value.get("paths"):
            first_path = str(value["paths"][0]).strip()
            tiles.append(
                f"<div class='tile'><img src='{escape(_rel_chart(first_path))}' alt='{escape(key)}'><div class='caption'>{escape(_gallery_caption(key, value))}</div></div>"
            )
        elif str(value.get("path") or "").strip():
            tiles.append(
                f"<div class='tile'><img src='{escape(_rel_chart(value.get('path')))}' alt='{escape(key)}'><div class='caption'>{escape(_gallery_caption(key, value))}</div></div>"
            )
    if not tiles:
        return "<p>臣本欲另附旁圖，然此輪可用圖不足，故先以主圖為重。</p>"
    return "<div class='gallery'>" + "".join(tiles) + "</div>"


def _gallery_caption(key: str, value: dict[str, object]) -> str:
    if key == "tech_split":
        return "科技型逐檔拆看，可辨誰是真能打，誰只是借勢。"
    if key == "active_split":
        return "主動式逐檔拆看，可辨誰有紀律，誰只是倚題材。"
    if key == "top_tech_single":
        return "科技型領頭單檔，利於看其趨勢是否仍穩。"
    if key == "top_active_single":
        return "主動式領頭單檔，利於看其攻守節奏是否失衡。"
    return "臣另附此圖，以補主圖未盡之處。"


def _rel_chart(value: object) -> str:
    if isinstance(value, dict):
        path_value = value.get("path")
        if str(path_value or "").strip():
            return _rel_chart(path_value)
        paths = value.get("paths")
        if isinstance(paths, list) and paths:
            return _rel_chart(paths[0])
        result_payload = value.get("result")
        if isinstance(result_payload, dict):
            paths = result_payload.get("paths")
            if isinstance(paths, list) and paths:
                return _rel_chart(paths[0])
            items = result_payload.get("items")
            if isinstance(items, list) and items and isinstance(items[0], dict):
                return _rel_chart(items[0].get("path"))
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if "charts" in path.parts:
        idx = path.parts.index("charts")
        return "/".join(path.parts[idx:])
    return path.name


def _safe_number(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str):
        text = value.replace('="', "").replace('"', "").replace(",", "").strip()
        if not text or text in {"-", "—"}:
            return None
        try:
            return float(text)
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _mean_pct(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _dispersion_pct(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.max() - clean.min())


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value
