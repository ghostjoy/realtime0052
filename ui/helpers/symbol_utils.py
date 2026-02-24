"""共用輔助函數 - 從 app.py 重構而出"""

import re


def normalize_heatmap_etf_code(value: object) -> str:
    """正規化 ETF 代碼"""
    code = str(value or "").strip().upper()
    if not code:
        return ""
    return code if re.fullmatch(r"\d{4,6}[A-Z]?", code) else ""


def clean_heatmap_name_for_query(value: object) -> str:
    """清理 ETF 名稱用於 URL query"""
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    return text.replace("&", "＆").replace("?", "").replace("#", "")


def normalize_market_tag_for_drill(value: object) -> str:
    """正規化市場標籤"""
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


def strip_symbol_label_token(text: str) -> str:
    """去除標籤，只留代碼"""
    token = str(text or "").strip().upper()
    if not token:
        return ""
    token = re.split(r"\s+|　", token, maxsplit=1)[0].strip()
    if "／" in token:
        token = token.split("／", 1)[0].strip()
    if "/" in token:
        token = token.split("/", 1)[0].strip()
    return token


def parse_drill_symbol(value: object) -> tuple[str, str]:
    """解析 drilldown 符號"""
    raw = str(value or "").strip()
    if not raw:
        return "", ""
    token = strip_symbol_label_token(raw)
    if not token or token in {"—", "-", "NAN", "NONE", "NULL"}:
        return "", ""
    tw_match = re.fullmatch(r"(\d{4,6}[A-Z]?)\.(TW|TWO)", token, flags=re.IGNORECASE)
    if tw_match:
        symbol = str(tw_match.group(1)).upper()
        market = "OTC" if str(tw_match.group(2)).upper() == "TWO" else "TW"
        return symbol, market
    if re.fullmatch(r"\d{4,6}[A-Z]?", token):
        return token, "TW"
    if token == "^TWII":
        return token, "TW"
    if re.fullmatch(r"\^[A-Z0-9.\-]{2,10}", token):
        return token, "US"
    if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", token):
        return token, "US"
    return "", ""


def looks_like_tw_symbol(symbol: str) -> bool:
    """檢查是否像台股代碼"""
    return bool(re.fullmatch(r"\d{4,6}[A-Z]?", symbol.strip().upper()))


def looks_like_us_symbol(symbol: str) -> bool:
    """檢查是否像美股代碼"""
    token = str(symbol or "").strip().upper()
    if not token:
        return False
    if token.endswith(".TW") or token.endswith(".TWO"):
        return False
    if looks_like_tw_symbol(token):
        return False
    return bool(re.fullmatch(r"\^?[A-Z][A-Z0-9.\-]{0,11}", token))
