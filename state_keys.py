from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class BacktestStateKeys:
    market: str = "bt_market"
    mode: str = "bt_mode"
    symbol: str = "bt_symbol"
    strategy: str = "bt_strategy"
    start_date: str = "bt_start_date"
    end_date: str = "bt_end_date"
    benchmark_choice: str = "bt_benchmark_choice"
    auto_cost: str = "bt_auto_cost"
    fee_rate: str = "bt_fee_rate"
    sell_tax: str = "bt_sell_tax"
    slippage: str = "bt_slippage"
    cost_profile: str = "bt_cost_profile"
    enable_wf: str = "bt_enable_wf"
    train_ratio: str = "bt_train_ratio"
    objective: str = "bt_objective"
    use_split_adjustment: str = "bt_use_split_adjustment"
    auto_detect_split: str = "bt_auto_detect_split"
    use_total_return_adjustment: str = "bt_use_total_return_adjustment"
    invest_start_mode: str = "bt_invest_start_mode"
    invest_start_date: str = "bt_invest_start_date"
    invest_start_k: str = "bt_invest_start_k"
    sync_parallel: str = "bt_sync_parallel"
    auto_sync: str = "bt_auto_sync"
    auto_fill_gaps: str = "bt_auto_fill_gaps"


BT_KEYS = BacktestStateKeys()


def backtest_key_values() -> list[str]:
    return [str(getattr(BT_KEYS, item.name)) for item in fields(BT_KEYS)]


__all__ = ["BacktestStateKeys", "BT_KEYS", "backtest_key_values"]
