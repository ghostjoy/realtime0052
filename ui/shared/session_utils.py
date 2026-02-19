from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def ensure_defaults(state: MutableMapping[str, Any], defaults: dict[str, Any]):
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


__all__ = ["ensure_defaults"]
