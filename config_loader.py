from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover
    compose = None  # type: ignore[assignment]
    initialize_config_dir = None  # type: ignore[assignment]
    GlobalHydra = None  # type: ignore[assignment]
    DictConfig = object  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]


def _config_dir() -> Path:
    return Path(__file__).resolve().parent / "conf"


@lru_cache(maxsize=1)
def _load_hydra_config() -> DictConfig | None:
    if compose is None or initialize_config_dir is None or OmegaConf is None:
        return None
    conf_dir = _config_dir()
    if not conf_dir.exists():
        return None
    if GlobalHydra is not None and GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        return compose(config_name="config")


def _deep_get(cfg: DictConfig | None, path: str, default: Any = None) -> Any:
    if cfg is None or OmegaConf is None:
        return default
    val = OmegaConf.select(cfg, path)
    return default if val is None else val


def get_config_source() -> str:
    env_src = str(os.getenv("REALTIME0052_CONFIG_SOURCE", "")).strip().lower()
    if env_src in {"legacy_env", "hydra"}:
        return env_src
    cfg = _load_hydra_config()
    src = str(_deep_get(cfg, "features.config_source", "hydra") or "hydra").strip().lower()
    return src if src in {"legacy_env", "hydra"} else "hydra"


def cfg_get(path: str, default: Any = None) -> Any:
    if get_config_source() != "hydra":
        return default
    cfg = _load_hydra_config()
    return _deep_get(cfg, path, default)


def cfg_or_env(path: str, env_var: str, default: Any = None, cast: Any = None) -> Any:
    raw = os.getenv(env_var)
    if raw is not None and str(raw).strip() != "":
        val = raw
    else:
        val = cfg_get(path, None)
        if val is None:
            val = default
    if cast is None:
        return val
    try:
        return cast(val)
    except Exception:
        return default


def cfg_or_env_bool(path: str, env_var: str, default: bool = False) -> bool:
    raw = str(os.getenv(env_var, "")).strip().lower()
    if raw != "":
        return raw in {"1", "true", "yes", "on"}
    val = cfg_get(path, None)
    if val is None:
        return bool(default)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def cfg_or_env_str(path: str, env_var: str, default: str) -> str:
    raw = str(os.getenv(env_var, "")).strip()
    if raw:
        return raw
    val = cfg_get(path, None)
    if val is None:
        return default
    text = str(val).strip()
    return text if text else default
