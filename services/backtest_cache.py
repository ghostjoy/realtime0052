from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any


def stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_source_hash(payload: dict[str, object]) -> str:
    raw = stable_json_dumps(payload)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_backtest_run_key(
    *,
    market: str,
    symbols: list[str],
    strategy: str,
    start_date: date,
    end_date: date,
    enable_wf: bool,
    train_ratio: float,
    objective: str,
    initial_capital: float,
    strategy_params: dict[str, Any],
    fee_rate: float,
    sell_tax: float,
    slippage: float,
    use_split_adjustment: bool,
    auto_detect_split: bool,
    use_total_return_adjustment: bool,
    invest_start_mode: str,
    invest_start_date: date | None,
    invest_start_k: int,
) -> str:
    strategy_token = stable_json_dumps(strategy_params if isinstance(strategy_params, dict) else {})
    objective_token = str(objective or "") if bool(enable_wf) else ""
    return (
        f"bt_result:{str(market).strip().upper()}:{','.join([str(s).strip().upper() for s in symbols])}:"
        f"{str(strategy).strip()}:{str(start_date)}:{str(end_date)}:{int(bool(enable_wf))}:{float(train_ratio)}:{objective_token}:"
        f"{int(float(initial_capital))}:{strategy_token}:{float(fee_rate):.6f}:{float(sell_tax):.6f}:{float(slippage):.6f}:"
        f"{int(bool(use_split_adjustment))}:{int(bool(auto_detect_split))}:{int(bool(use_total_return_adjustment))}:"
        f"{str(invest_start_mode)}:{str(invest_start_date) if invest_start_date is not None else ''}:{int(invest_start_k)}"
    )


def build_backtest_run_params_base(
    *,
    market: str,
    mode: str,
    symbols: list[str],
    strategy: str,
    strategy_params: dict[str, Any],
    start_date: date,
    end_date: date,
    enable_walk_forward: bool,
    train_ratio: float,
    objective: str,
    initial_capital: float,
    fee_rate: float,
    sell_tax_rate: float,
    slippage_rate: float,
    use_split_adjustment: bool,
    auto_detect_split: bool,
    use_total_return_adjustment: bool,
    invest_start_mode: str,
    invest_start_date: date | None,
    invest_start_k: int,
) -> dict[str, object]:
    return {
        "market": str(market).strip().upper(),
        "mode": str(mode).strip(),
        "symbols": [str(s).strip().upper() for s in symbols if str(s).strip()],
        "strategy": str(strategy).strip(),
        "strategy_params": strategy_params if isinstance(strategy_params, dict) else {},
        "start_date": str(start_date),
        "end_date": str(end_date),
        "enable_walk_forward": bool(enable_walk_forward),
        "train_ratio": float(train_ratio),
        "objective": str(objective or "") if bool(enable_walk_forward) else "",
        "initial_capital": float(initial_capital),
        "fee_rate": float(fee_rate),
        "sell_tax_rate": float(sell_tax_rate),
        "slippage_rate": float(slippage_rate),
        "use_split_adjustment": bool(use_split_adjustment),
        "auto_detect_split": bool(auto_detect_split),
        "use_total_return_adjustment": bool(use_total_return_adjustment),
        "invest_start_mode": str(invest_start_mode),
        "invest_start_date": str(invest_start_date) if invest_start_date is not None else "",
        "invest_start_k": int(invest_start_k),
    }


def build_replay_params_with_signature(
    *,
    params_base: dict[str, object],
    schema_version: int,
) -> tuple[dict[str, object], str]:
    source_hash = build_source_hash(params_base)
    params = {
        **params_base,
        "schema_version": int(schema_version),
        "source_hash": source_hash,
    }
    return params, source_hash


__all__ = [
    "stable_json_dumps",
    "build_source_hash",
    "build_backtest_run_key",
    "build_backtest_run_params_base",
    "build_replay_params_with_signature",
]
