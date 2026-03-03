"""URL 建構工具 - 從 app.py 重構而出"""

from urllib.parse import quote, urlencode


def build_backtest_drill_url(symbol: str, market: str) -> str:
    """建構回測 drilldown URL"""
    symbol_text = str(symbol).strip().upper()
    market_text = str(market).strip().upper()
    label = symbol_text
    if market_text in {"TW", "OTC"} and "." not in symbol_text and not symbol_text.startswith("^"):
        label = f"{symbol_text}.tw"
    return "?" + urlencode(
        {
            "bt_symbol": symbol_text,
            "bt_market": market_text,
            "bt_label": label,
            "bt_autorun": "1",
            "bt_src": "table",
        }
    )


def build_heatmap_drill_url(etf_code: str, etf_name: str, *, src: str = "all_types_table") -> str:
    """建構熱力圖 drilldown URL"""
    from ui.helpers.symbol_utils import (
        clean_heatmap_name_for_query,
        normalize_heatmap_etf_code,
    )

    code = normalize_heatmap_etf_code(etf_code)
    if not code:
        return ""
    label = clean_heatmap_name_for_query(etf_name) or code
    name_encoded = quote(str(etf_name or "").strip(), safe="")
    return (
        f"?hm_etf={code}&hm_name={name_encoded}"
        f"&hm_label={label}&hm_open=1&hm_src={str(src or 'all_types_table').strip()}"
    )
