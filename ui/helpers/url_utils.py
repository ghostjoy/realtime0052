"""URL 建構工具 - 從 app.py 重構而出"""

from urllib.parse import quote, urlencode


def build_backtest_drill_url(symbol: str, market: str) -> str:
    """建構回測 drilldown URL"""
    return "?" + urlencode(
        {
            "bt_symbol": str(symbol).strip().upper(),
            "bt_market": str(market).strip().upper(),
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
