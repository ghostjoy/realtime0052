from __future__ import annotations

import json
import math
import random
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import pandas as pd

from market_data_types import OhlcvSnapshot, QuoteSnapshot
from providers.base import ProviderError, ProviderErrorKind, ProviderRequest
from services.provider_policies import (
    ProviderPolicy,
    default_provider_policy,
    provider_quality_base,
)


@dataclass
class _BreakerState:
    fail_count: int = 0
    open_until_monotonic: float = 0.0
    state: str = "closed"


class ProviderGateway:
    def __init__(self):
        self._breaker_states: dict[str, _BreakerState] = {}
        self._breaker_lock = threading.Lock()

    def execute_quote_candidates(
        self,
        candidates: list[tuple[object, ProviderRequest]],
        *,
        policy: ProviderPolicy | None = None,
    ) -> tuple[QuoteSnapshot, list[str], str | None, int]:
        quote_policy = policy or default_provider_policy(operation="quote")
        errors: list[str] = []
        chain_names = [str(getattr(provider, "name", "unknown")) for provider, _ in candidates]
        for idx, (provider, request) in enumerate(candidates):
            provider_name = str(getattr(provider, "name", "unknown"))
            breaker_key = self._breaker_key(
                provider_name=provider_name,
                operation="quote",
                market=str(getattr(request, "market", "")),
            )
            if self._breaker_is_open(breaker_key):
                errors.append(f"[{provider_name}:circuit] open")
                continue
            try:
                quote = self._call_with_retries(
                    provider=provider,
                    request=request,
                    operation="quote",
                    policy=quote_policy,
                )
                self._breaker_mark_success(breaker_key)
                return (
                    self._decorate_quote_snapshot(
                        quote=quote,
                        fallback_depth=idx,
                        previous_errors=errors,
                    ),
                    chain_names,
                    errors[-1] if errors else None,
                    idx,
                )
            except Exception as exc:
                errors.append(str(exc))
                self._breaker_mark_failure(breaker_key, quote_policy)
        raise RuntimeError("; ".join(errors) if errors else "no provider available")

    def execute_ohlcv_candidates(
        self,
        candidates: list[tuple[object, ProviderRequest]],
        *,
        policy: ProviderPolicy | None = None,
    ) -> OhlcvSnapshot:
        ohlcv_policy = policy or default_provider_policy(operation="ohlcv")
        errors: list[str] = []
        for idx, (provider, request) in enumerate(candidates):
            provider_name = str(getattr(provider, "name", "unknown"))
            breaker_key = self._breaker_key(
                provider_name=provider_name,
                operation="ohlcv",
                market=str(getattr(request, "market", "")),
            )
            if self._breaker_is_open(breaker_key):
                errors.append(f"[{provider_name}:circuit] open")
                continue
            try:
                snap = self._call_with_retries(
                    provider=provider,
                    request=request,
                    operation="ohlcv",
                    policy=ohlcv_policy,
                )
                self._breaker_mark_success(breaker_key)
                return self._decorate_ohlcv_snapshot(
                    snap=snap,
                    fallback_depth=idx,
                    previous_errors=errors,
                )
            except Exception as exc:
                errors.append(str(exc))
                self._breaker_mark_failure(breaker_key, ohlcv_policy)
        raise RuntimeError("; ".join(errors) if errors else "no provider available")

    @staticmethod
    def _breaker_key(*, provider_name: str, operation: str, market: str) -> str:
        return (
            f"{str(provider_name).strip().lower()}:"
            f"{str(operation).strip().lower()}:"
            f"{str(market).strip().upper()}"
        )

    def _breaker_is_open(self, key: str) -> bool:
        now = time.monotonic()
        with self._breaker_lock:
            state = self._breaker_states.setdefault(key, _BreakerState())
            if state.open_until_monotonic > now:
                state.state = "open"
                return True
            if state.state == "open":
                state.state = "half_open"
                return False
            return False

    def _breaker_mark_success(self, key: str) -> None:
        with self._breaker_lock:
            state = self._breaker_states.setdefault(key, _BreakerState())
            state.fail_count = 0
            state.open_until_monotonic = 0.0
            state.state = "closed"

    def _breaker_mark_failure(self, key: str, policy: ProviderPolicy) -> None:
        with self._breaker_lock:
            state = self._breaker_states.setdefault(key, _BreakerState())
            if state.state == "half_open":
                state.fail_count = int(policy.breaker_fail_threshold)
            else:
                state.fail_count += 1
            if state.fail_count >= int(policy.breaker_fail_threshold):
                state.open_until_monotonic = time.monotonic() + max(
                    1, int(policy.breaker_cooldown_sec)
                )
                state.state = "open"
            else:
                state.state = "closed"

    def _call_with_retries(
        self,
        *,
        provider: object,
        request: ProviderRequest,
        operation: str,
        policy: ProviderPolicy,
    ):
        attempts = max(1, int(policy.max_retries) + 1)
        provider_name = str(getattr(provider, "name", "unknown"))
        for attempt in range(attempts):
            try:
                return self._call_provider_once(
                    provider=provider,
                    request=request,
                    operation=operation,
                    timeout_sec=float(policy.timeout_sec),
                )
            except Exception as exc:
                if attempt >= attempts - 1 or not self._is_retryable_error(exc):
                    raise RuntimeError(f"[{provider_name}] {exc}") from exc
                sleep_sec = self._compute_backoff_sleep(policy=policy, attempt=attempt)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
        raise RuntimeError(f"[{provider_name}] exhausted retries")

    @staticmethod
    def _call_provider_once(
        *,
        provider: object,
        request: ProviderRequest,
        operation: str,
        timeout_sec: float,
    ):
        method_name = "quote" if str(operation).strip().lower() == "quote" else "ohlcv"
        method = getattr(provider, method_name)
        has_timeout = hasattr(provider, "timeout_sec")
        old_timeout = None
        if has_timeout:
            try:
                old_timeout = provider.timeout_sec  # type: ignore[attr-defined]
                provider.timeout_sec = max(1.0, float(timeout_sec))  # type: ignore[attr-defined]
            except Exception:
                has_timeout = False
        try:
            return method(request)
        finally:
            if has_timeout:
                try:
                    provider.timeout_sec = old_timeout  # type: ignore[attr-defined]
                except Exception:
                    pass

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, ProviderError):
            if exc.kind in {ProviderErrorKind.NETWORK, ProviderErrorKind.RATE_LIMIT}:
                return True
            text = str(exc).lower()
            return ("timeout" in text) or ("timed out" in text) or ("http 5" in text)
        text = str(exc).lower()
        retry_tokens = (
            "timeout",
            "timed out",
            "429",
            "too many requests",
            "http 5",
            "connection",
        )
        return any(token in text for token in retry_tokens)

    @staticmethod
    def _compute_backoff_sleep(*, policy: ProviderPolicy, attempt: int) -> float:
        base = max(1, int(policy.base_backoff_ms))
        cap = max(base, int(policy.max_backoff_ms))
        step_ms = min(cap, base * (2 ** max(0, int(attempt))))
        jitter = random.uniform(0.0, max(0.0, float(policy.jitter_ratio))) * step_ms
        return max(0.0, float(min(cap, step_ms + jitter)) / 1000.0)

    @staticmethod
    def _score_with_fallback(*, provider_name: str, fallback_depth: int) -> float:
        score = float(provider_quality_base(provider_name)) - (0.05 * max(0, int(fallback_depth)))
        return max(0.30, min(1.0, score))

    def _decorate_quote_snapshot(
        self,
        *,
        quote: QuoteSnapshot,
        fallback_depth: int,
        previous_errors: list[str],
    ) -> QuoteSnapshot:
        source = str(getattr(quote, "source", "") or "unknown")
        asof = quote.asof or quote.ts
        quality_score = (
            float(quote.quality_score)
            if quote.quality_score is not None and math.isfinite(float(quote.quality_score))
            else self._score_with_fallback(provider_name=source, fallback_depth=fallback_depth)
        )
        raw_json = quote.raw_json
        if raw_json is None:
            raw_json = {
                "provider": source,
                "fallback_depth": max(0, int(fallback_depth)),
                "errors_before_success": list(previous_errors[-3:]),
            }
        return replace(quote, asof=asof, quality_score=quality_score, raw_json=raw_json)

    def _decorate_ohlcv_snapshot(
        self,
        *,
        snap: OhlcvSnapshot,
        fallback_depth: int,
        previous_errors: list[str],
    ) -> OhlcvSnapshot:
        source = str(getattr(snap, "source", "") or "unknown")
        asof = snap.asof
        if asof is None:
            try:
                if isinstance(snap.df, pd.DataFrame) and (not snap.df.empty):
                    asof_ts = pd.Timestamp(snap.df.index.max())
                    if asof_ts.tzinfo is None:
                        asof_ts = asof_ts.tz_localize("UTC")
                    else:
                        asof_ts = asof_ts.tz_convert("UTC")
                    asof = asof_ts.to_pydatetime()
            except Exception:
                asof = None
        if asof is None:
            asof = datetime.now(tz=timezone.utc)
        quality_score = (
            float(snap.quality_score)
            if snap.quality_score is not None and math.isfinite(float(snap.quality_score))
            else self._score_with_fallback(provider_name=source, fallback_depth=fallback_depth)
        )
        raw_json = snap.raw_json
        if raw_json is None:
            raw_json = {
                "provider": source,
                "fallback_depth": max(0, int(fallback_depth)),
                "errors_before_success": list(previous_errors[-3:]),
            }
        try:
            json.dumps(raw_json, ensure_ascii=False)
        except Exception:
            raw_json = {"provider": source}
        return replace(snap, asof=asof, quality_score=quality_score, raw_json=raw_json)
