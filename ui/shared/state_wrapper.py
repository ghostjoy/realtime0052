"""Session state wrapper for unified access to Streamlit session_state."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st

T = TypeVar("T")


def get_state(key: str, default: T | None = None) -> T | None:
    """Get value from session_state with optional default.

    Args:
        key: The session_state key to retrieve.
        default: Default value if key doesn't exist.

    Returns:
        The value from session_state or default.
    """
    return st.session_state.get(key, default)  # type: ignore[no-any-return]


def set_state(key: str, value: Any) -> None:
    """Set value in session_state.

    Args:
        key: The session_state key to set.
        value: The value to store.
    """
    st.session_state[key] = value


def pop_state(key: str, default: T | None = None) -> T | None:
    """Remove key from session_state and return its value.

    Args:
        key: The session_state key to remove.
        default: Default value if key doesn't exist.

    Returns:
        The removed value or default.
    """
    return st.session_state.pop(key, default)  # type: ignore[no-any-return]


def has_state(key: str) -> bool:
    """Check if key exists in session_state.

    Args:
        key: The session_state key to check.

    Returns:
        True if key exists, False otherwise.
    """
    return key in st.session_state


def get_or_set(key: str, factory: Callable[[], T]) -> T:
    """Get value from session_state or set it using factory.

    Args:
        key: The session_state key.
        factory: Callable that produces the default value.

    Returns:
        The existing value or newly created value.
    """
    if key not in st.session_state:
        st.session_state[key] = factory()
    return st.session_state[key]  # type: ignore[no-any-return]


__all__ = [
    "get_state",
    "set_state",
    "pop_state",
    "has_state",
    "get_or_set",
]
