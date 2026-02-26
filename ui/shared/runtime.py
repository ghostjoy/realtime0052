from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def configure_module_runtime(
    module_globals: dict[str, Any],
    required_names: tuple[str, ...],
    values: Mapping[str, Any],
    *,
    module_name: str,
) -> None:
    """Inject required runtime symbols into a module namespace.

    This keeps page/core modules explicit about what they consume from `app.py`,
    without relying on `ctx=globals()` bridge injection.
    """
    missing = [name for name in required_names if name not in values]
    if missing:
        missing_text = ", ".join(missing[:8])
        extra = " ..." if len(missing) > 8 else ""
        raise KeyError(f"{module_name} missing runtime symbols: {missing_text}{extra}")
    for name in required_names:
        module_globals[name] = values[name]


__all__ = ["configure_module_runtime"]
