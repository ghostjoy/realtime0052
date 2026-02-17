from __future__ import annotations

from datetime import date
import unittest

from services.backtest_cache import (
    build_backtest_run_key,
    build_backtest_run_params_base,
    build_replay_params_with_signature,
    build_source_hash,
    stable_json_dumps,
)


class BacktestCacheTests(unittest.TestCase):
    def _base_kwargs(self) -> dict[str, object]:
        return {
            "market": "TW",
            "symbols": ["0050", "0056"],
            "strategy": "buy_hold",
            "start_date": date(2026, 1, 1),
            "end_date": date(2026, 2, 1),
            "enable_wf": False,
            "train_ratio": 0.7,
            "objective": "sharpe",
            "initial_capital": 1_000_000.0,
            "strategy_params": {},
            "fee_rate": 0.001425,
            "sell_tax": 0.001,
            "slippage": 0.0005,
            "use_split_adjustment": True,
            "auto_detect_split": True,
            "use_total_return_adjustment": False,
            "invest_start_mode": "keep",
            "invest_start_date": date(2026, 1, 1),
            "invest_start_k": 0,
        }


    def test_stable_json_dumps_and_source_hash_are_stable(self):
        payload_a = {"b": 2, "a": 1}
        payload_b = {"a": 1, "b": 2}
        self.assertEqual(stable_json_dumps(payload_a), stable_json_dumps(payload_b))
        self.assertEqual(build_source_hash(payload_a), build_source_hash(payload_b))
        self.assertNotEqual(build_source_hash({"a": 1, "b": 3}), build_source_hash(payload_a))


    def test_build_backtest_run_key_is_deterministic_and_sensitive_to_params(self):
        kwargs = self._base_kwargs()
        key_1 = build_backtest_run_key(**kwargs)
        key_2 = build_backtest_run_key(**kwargs)
        self.assertEqual(key_1, key_2)

        kwargs_mutated = dict(kwargs)
        kwargs_mutated["fee_rate"] = 0.002
        key_3 = build_backtest_run_key(**kwargs_mutated)
        self.assertNotEqual(key_3, key_1)


    def test_build_backtest_run_params_and_replay_signature(self):
        params_base = build_backtest_run_params_base(
            market="TW",
            mode="單一標的",
            symbols=["0050"],
            strategy="buy_hold",
            strategy_params={},
            start_date=date(2026, 1, 1),
            end_date=date(2026, 2, 1),
            enable_walk_forward=False,
            train_ratio=0.7,
            objective="sharpe",
            initial_capital=1_000_000.0,
            fee_rate=0.001425,
            sell_tax_rate=0.001,
            slippage_rate=0.0005,
            use_split_adjustment=True,
            auto_detect_split=True,
            use_total_return_adjustment=False,
            invest_start_mode="keep",
            invest_start_date=date(2026, 1, 1),
            invest_start_k=0,
        )
        self.assertEqual(params_base["objective"], "")
        replay_params, source_hash = build_replay_params_with_signature(params_base=params_base, schema_version=2)
        self.assertEqual(replay_params["schema_version"], 2)
        self.assertEqual(replay_params["source_hash"], source_hash)
        self.assertEqual(source_hash, build_source_hash(params_base))


if __name__ == "__main__":
    unittest.main()
