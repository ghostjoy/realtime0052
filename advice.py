from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Profile:
    horizon: str  # 短線/中期/長期
    risk: str  # 保守/一般/積極
    style: str  # 定期定額/波段/趨勢


def _safe(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return float(v)
    except Exception:
        return None


def score(df: pd.DataFrame) -> Tuple[int, Dict[str, str]]:
    latest = df.iloc[-1]
    notes: Dict[str, str] = {}
    points = 0

    close = _safe(latest.get("close"))
    sma20 = _safe(latest.get("sma_20"))
    sma60 = _safe(latest.get("sma_60"))
    rsi14 = _safe(latest.get("rsi_14"))
    macd_hist = _safe(latest.get("macd_hist"))
    bb_upper = _safe(latest.get("bb_upper"))
    bb_lower = _safe(latest.get("bb_lower"))
    atr14 = _safe(latest.get("atr_14"))
    stoch_k = _safe(latest.get("stoch_k"))
    mfi14 = _safe(latest.get("mfi_14"))

    if close is not None and sma20 is not None:
        if close >= sma20:
            points += 1
            notes["趨勢(短)"] = "價在SMA20之上"
        else:
            points -= 1
            notes["趨勢(短)"] = "價在SMA20之下"

    if sma20 is not None and sma60 is not None:
        if sma20 >= sma60:
            points += 1
            notes["趨勢(中)"] = "SMA20≥SMA60"
        else:
            points -= 1
            notes["趨勢(中)"] = "SMA20<SMA60"

    if rsi14 is not None:
        if rsi14 >= 70:
            points -= 1
            notes["動能(RSI)"] = f"RSI={rsi14:.1f}（偏熱）"
        elif rsi14 <= 30:
            points += 1
            notes["動能(RSI)"] = f"RSI={rsi14:.1f}（偏冷）"
        elif rsi14 >= 55:
            points += 1
            notes["動能(RSI)"] = f"RSI={rsi14:.1f}（偏強）"
        elif rsi14 <= 45:
            points -= 1
            notes["動能(RSI)"] = f"RSI={rsi14:.1f}（偏弱）"
        else:
            notes["動能(RSI)"] = f"RSI={rsi14:.1f}（中性）"

    if macd_hist is not None:
        if macd_hist >= 0:
            points += 1
            notes["動能(MACD)"] = "柱狀體≥0"
        else:
            points -= 1
            notes["動能(MACD)"] = "柱狀體<0"

    if stoch_k is not None:
        if stoch_k >= 80:
            points -= 1
            notes["動能(KD)"] = f"K={stoch_k:.1f}（偏熱）"
        elif stoch_k <= 20:
            points += 1
            notes["動能(KD)"] = f"K={stoch_k:.1f}（偏冷）"
        else:
            notes["動能(KD)"] = f"K={stoch_k:.1f}（中性）"

    if mfi14 is not None:
        if mfi14 >= 80:
            points -= 1
            notes["資金流(MFI)"] = f"MFI={mfi14:.1f}（偏熱）"
        elif mfi14 <= 20:
            points += 1
            notes["資金流(MFI)"] = f"MFI={mfi14:.1f}（偏冷）"
        else:
            notes["資金流(MFI)"] = f"MFI={mfi14:.1f}（中性）"

    if close is not None and bb_upper is not None and bb_lower is not None:
        if close >= bb_upper:
            points -= 1
            notes["波動(BB)"] = "觸及上軌（追價風險↑）"
        elif close <= bb_lower:
            points += 1
            notes["波動(BB)"] = "觸及下軌（反彈機率↑）"
        else:
            notes["波動(BB)"] = "位於布林帶內"

    if atr14 is not None:
        notes["風險(ATR)"] = f"ATR14≈{atr14:.2f}"

    return points, notes


def render_advice(df: pd.DataFrame, profile: Profile) -> str:
    points, notes = score(df)

    if points >= 2:
        bias = "偏多"
    elif points <= -2:
        bias = "偏空"
    else:
        bias = "觀望"

    if profile.horizon == "長期":
        base = "以長期為主：偏向用定期定額/分批，避免用短線訊號追高殺低。"
    elif profile.horizon == "短線":
        base = "以短線為主：重視進出場與停損，不追價；若反向，務必小部位。"
    else:
        base = "以中期為主：用趨勢與風險控管，分批進出，避免一次梭哈。"

    risk = {
        "保守": "保守：以小部位/分批為優先，設定最大回撤或停損條件。",
        "一般": "一般：可分批布局，但仍建議事先規劃停損/停利。",
        "積極": "積極：可提高操作頻率，但仍需嚴格風控與槓桿節制。",
    }.get(profile.risk, "")

    style = {
        "定期定額": "策略：以固定週期投入為主，短期波動視為成本攤平。",
        "波段": "策略：以支撐/壓力與均線交叉做分批進出。",
        "趨勢": "策略：以『站上/跌破關鍵均線』作為加減碼依據。",
    }.get(profile.style, "")

    lines = [
        f"綜合評分：{points:+d} → 目前偏向：{bias}",
        base,
        risk,
        style,
        "",
        "依據（技術面/風險）：",
    ]
    for k, v in notes.items():
        lines.append(f"- {k}：{v}")

    lines += [
        "",
        "提醒：本工具以公開行情與技術指標推論，僅供教育/研究，非投資建議；請自行評估風險與資金承受度。",
    ]
    return "\n".join([l for l in lines if l != ""])


def _fmt_pct(v: Any) -> str:
    try:
        if v is None:
            return "—"
        v = float(v)
        if np.isnan(v):
            return "—"
        return f"{v*100:.1f}%"
    except Exception:
        return "—"


def _fmt_big(v: Any) -> str:
    try:
        if v is None:
            return "—"
        v = float(v)
        if np.isnan(v):
            return "—"
        # 粗略縮寫
        abs_v = abs(v)
        if abs_v >= 1e12:
            return f"{v/1e12:.2f}T"
        if abs_v >= 1e9:
            return f"{v/1e9:.2f}B"
        if abs_v >= 1e6:
            return f"{v/1e6:.2f}M"
        return f"{v:,.0f}"
    except Exception:
        return str(v)


def _position_range(points: int, risk: str) -> tuple[int, int]:
    # Base allocation suggestion by technical score (non-leveraged, indicative only).
    if points <= -3:
        lo, hi = 0, 10
    elif points == -2:
        lo, hi = 5, 20
    elif points == -1:
        lo, hi = 15, 35
    elif points == 0:
        lo, hi = 25, 45
    elif points == 1:
        lo, hi = 40, 60
    elif points == 2:
        lo, hi = 55, 75
    else:
        lo, hi = 70, 90

    risk_adj = {"保守": -10, "一般": 0, "積極": 10}.get(str(risk), 0)
    lo = max(0, min(100, lo + risk_adj))
    hi = max(lo, min(100, hi + risk_adj))
    return lo, hi


def render_advice_scai_style(
    df: pd.DataFrame,
    profile: Profile,
    symbol: str,
    fundamentals: Optional[Dict[str, Any]] = None,
) -> str:
    """
    注意：此為「以常見投資節目/社群語境的心法」整理成的框架化輸出，
    不是任何特定人士的原話或保證能代表其觀點；僅供教育/研究。
    """
    points, notes = score(df)
    latest = df.iloc[-1]
    close = float(latest.get("close"))
    sma20 = latest.get("sma_20")
    sma60 = latest.get("sma_60")
    rsi14 = latest.get("rsi_14")
    atr14 = latest.get("atr_14")

    if points >= 2:
        bias = "偏多，但不等於立刻追"
    elif points <= -2:
        bias = "偏空，優先控風險"
    else:
        bias = "中性，等訊號/等價格"

    # 基本面快照（若資料不足就改成問題清單）
    f = fundamentals or {}
    f_lines = []
    if fundamentals:
        f_lines = [
            f"- 市值：{_fmt_big(f.get('marketCap'))}（來源：Yahoo）",
            f"- 估值：PE(trailing)={f.get('trailingPE') or '—'} / PE(forward)={f.get('forwardPE') or '—'} / P/B={f.get('priceToBook') or '—'}",
            f"- 成長：營收YoY={_fmt_pct(f.get('revenueGrowth'))} / 獲利YoY={_fmt_pct(f.get('earningsGrowth'))}",
            f"- 利潤：毛利={_fmt_pct(f.get('grossMargins'))} / 營業={_fmt_pct(f.get('operatingMargins'))} / 淨利={_fmt_pct(f.get('profitMargins'))}",
            f"- 現金流/負債：FCF={_fmt_big(f.get('freeCashflow'))} / Cash={_fmt_big(f.get('totalCash'))} / Debt={_fmt_big(f.get('totalDebt'))}",
        ]
    else:
        f_lines = [
            "- 基本面資料目前抓不到（可能被限流/資料源缺漏）。建議你自己補：營收趨勢、毛利/淨利、FCF、負債、競爭優勢與產業景氣。",
        ]

    # 技術面行動建議（把它寫成「節奏」）
    t_lines = []
    if sma20 is not None and sma60 is not None:
        try:
            t_lines.append(f"- 趨勢：價={close:.2f}；SMA20={float(sma20):.2f}；SMA60={float(sma60):.2f}")
        except Exception:
            pass
    if rsi14 is not None:
        try:
            t_lines.append(f"- 動能：RSI14={float(rsi14):.1f}（70↑偏熱 / 30↓偏冷）")
        except Exception:
            pass
    if atr14 is not None:
        try:
            t_lines.append(f"- 波動：ATR14≈{float(atr14):.2f}（用來抓停損距離/部位大小）")
        except Exception:
            pass

    # 風控：用 profile 把話說清楚
    if profile.risk == "保守":
        risk_line = "風控：保守 → 小部位/分批；若跌破關鍵均線或超出可承受回撤就先撤退。"
    elif profile.risk == "積極":
        risk_line = "風控：積極 → 可以做節奏，但停損/減碼要更硬（錯了就認）。"
    else:
        risk_line = "風控：一般 → 分批進出，先定義『錯了』的條件（停損/時間停損/趨勢破壞）。"

    pos_lo, pos_hi = _position_range(points, profile.risk)
    position_line = f"倉位建議：{pos_lo}%~{pos_hi}%（現金/短債 = {100-pos_hi}%~{100-pos_lo}%）"

    # 行動：用評分決定「做/不做/怎麼做」
    if points >= 2:
        action = "行動：偏多 → 不追價，等回檔靠近關鍵均線分批；若拉開太遠就用小部位試單。"
    elif points <= -2:
        action = "行動：偏空 → 先減碼、留現金；若要逆勢，只允許極小部位試單並設定明確停損。"
    else:
        action = "行動：中性 → 可小倉位試單，等『站上/跌破』關鍵位再加減碼，避免在盤整中重倉來回挨打。"

    lines = [
        f"{symbol}｜綜合評分 {points:+d} → {bias}",
        "",
        "基本面（先看能不能長拿）：",
        *f_lines,
        "",
        "技術面（再決定節奏與部位）：",
        *t_lines,
        "",
        "依據（技術面/風險）：",
        *[f"- {k}：{v}" for k, v in notes.items()],
        "",
        risk_line,
        position_line,
        action,
        "",
        "提醒：以上為框架化整理，非任何特定人士的原話，亦非投資建議；請自行評估風險。",
    ]
    return "\n".join([l for l in lines if l != ""])
