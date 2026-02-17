from __future__ import annotations

import unittest

from state_keys import BT_KEYS, backtest_key_values


class StateKeysTests(unittest.TestCase):
    def test_backtest_state_keys_unique(self):
        values = backtest_key_values()
        self.assertEqual(len(values), len(set(values)))

    def test_backtest_state_keys_backward_compatible(self):
        self.assertEqual(BT_KEYS.market, "bt_market")
        self.assertEqual(BT_KEYS.mode, "bt_mode")
        self.assertEqual(BT_KEYS.symbol, "bt_symbol")
        self.assertEqual(BT_KEYS.auto_cost, "bt_auto_cost")
        self.assertEqual(BT_KEYS.fee_rate, "bt_fee_rate")
        self.assertEqual(BT_KEYS.sell_tax, "bt_sell_tax")
        self.assertEqual(BT_KEYS.slippage, "bt_slippage")
        self.assertEqual(BT_KEYS.cost_profile, "bt_cost_profile")
        self.assertEqual(BT_KEYS.invest_start_mode, "bt_invest_start_mode")
        self.assertEqual(BT_KEYS.invest_start_date, "bt_invest_start_date")
        self.assertEqual(BT_KEYS.invest_start_k, "bt_invest_start_k")


if __name__ == "__main__":
    unittest.main()
