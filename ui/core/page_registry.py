from __future__ import annotations

from typing import Callable

import streamlit as st

def runtime_page_cards(base_cards: list[dict[str, str]], dynamic_cards_fn: Callable[[], list[dict[str, str]]]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = [dict(item) for item in base_cards]
    existing_keys = {str(item.get("key", "")).strip() for item in cards}
    for item in dynamic_cards_fn():
        key = str(item.get("key", "")).strip()
        if not key or key in existing_keys:
            continue
        cards.append(item)
        existing_keys.add(key)
    return cards

def render_page_cards_nav(*, cards: list[dict[str, str]], default_active_page: str) -> str:
    page_options = [item["key"] for item in cards]
    default_page = default_active_page if default_active_page in page_options else page_options[0]
    active_page = str(st.session_state.get("active_page", default_page))
    if active_page not in page_options:
        active_page = default_page
    st.session_state["active_page"] = active_page

    st.markdown("#### 功能卡片")
    cols = st.columns(5, gap="small")
    for idx, item in enumerate(cards):
        key = item["key"]
        desc = item["desc"]
        is_active = key == active_page
        with cols[idx % 5]:
            st.markdown((
                f"<div class='page-nav-card{' active' if is_active else ''}'>"
                f"<div class='page-card-title'>{key}</div>"
                f"<div class='page-card-desc'>{desc}</div>"
                "</div>"
            ), unsafe_allow_html=True)
            clicked = st.button(
                "已開啟" if is_active else "開啟",
                key=f"page-card:{key}",
                width="stretch",
                type="primary" if is_active else "secondary",
            )
            if clicked and not is_active:
                st.session_state["active_page"] = key
                st.rerun()

    return str(st.session_state.get("active_page", default_page))

__all__ = ["runtime_page_cards", "render_page_cards_nav"]
