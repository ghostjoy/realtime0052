from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConsistencyMode(str, Enum):
    FAST_FIRST = "fast_first"
    STRICT_2WAY = "strict_2way"
    PREFERRED_WITH_FALLBACK = "preferred_with_fallback"


@dataclass(frozen=True)
class ProviderPolicy:
    timeout_sec: float
    max_retries: int
    base_backoff_ms: int
    max_backoff_ms: int
    jitter_ratio: float
    breaker_fail_threshold: int
    breaker_cooldown_sec: int
    consistency_mode: ConsistencyMode = ConsistencyMode.PREFERRED_WITH_FALLBACK


def default_provider_policy(*, operation: str) -> ProviderPolicy:
    op = str(operation or "").strip().lower()
    timeout_sec = 2.0 if op == "quote" else 3.0
    return ProviderPolicy(
        timeout_sec=timeout_sec,
        max_retries=2,
        base_backoff_ms=150,
        max_backoff_ms=1200,
        jitter_ratio=0.2,
        breaker_fail_threshold=5,
        breaker_cooldown_sec=60,
    )


def provider_quality_base(provider_name: str) -> float:
    token = str(provider_name or "").strip().lower()
    if token in {"tw_fugle_ws", "tw_fugle_rest"}:
        return 0.95
    if token in {"tw_openapi", "tw_tpex", "tw_mis"}:
        return 0.90
    if token == "us_twelve":
        return 0.88
    if token in {"yahoo", "us_stooq"}:
        return 0.75
    return 0.70

