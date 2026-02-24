from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest import CostModel
from services.backtest_runner import (
    BacktestExecutionInput,
    default_cost_params,
    execute_backtest_run,
    load_and_prepare_symbol_bars,
    load_benchmark_from_store,
    parse_symbols,
    queue_benchmark_writeback,
)


def _bars(n: int = 220, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.002, n))
    out = pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(n, 1000.0),
            "source": ["unit"] * n,
        },
        index=idx,
    )
    return out


class _FakeStore:
    def __init__(self, bars_map: dict[str, pd.DataFrame]):
        self._bars_map = {k.upper(): v for k, v in bars_map.items()}
        self.sync_calls: list[str] = []

    def load_daily_bars(self, symbol, market, start=None, end=None):
        return self._bars_map.get(str(symbol).upper(), pd.DataFrame())

    def sync_symbol_history(self, symbol, market, start=None, end=None):
        self.sync_calls.append(str(symbol).upper())
        return SimpleNamespace(error="", rows_upserted=1, source="unit", fallback_depth=0)


class BacktestRunnerTests(unittest.TestCase):
    def test_parse_symbols_and_default_cost(self):
        self.assertEqual(parse_symbols(" 0050, 2330,0050，8069 "), ["0050", "2330", "8069"])
        self.assertEqual(default_cost_params("TW", ["0050"]), (0.001425, 0.001, 0.0005))
        self.assertEqual(default_cost_params("TW", ["2330"]), (0.001425, 0.003, 0.0005))
        self.assertEqual(default_cost_params("US", ["AAPL"]), (0.0005, 0.0, 0.001))

    def test_load_benchmark_from_store(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 3, 1, tzinfo=timezone.utc)
        store = _FakeStore({"^GSPC": _bars(80, seed=11)})
        benchmark = load_benchmark_from_store(
            store=store,
            market_code="US",
            start=start,
            end=end,
            choice="gspc",
        )
        self.assertFalse(benchmark.empty)
        self.assertEqual(str(benchmark.attrs.get("symbol", "")), "^GSPC")
        self.assertIn("close", benchmark.columns)

    def test_load_and_prepare_symbol_bars(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 3, 1, tzinfo=timezone.utc)
        store = _FakeStore({"0050": _bars(80, seed=13), "2330": _bars(80, seed=17)})
        prepared = load_and_prepare_symbol_bars(
            store=store,
            market_code="TW",
            symbols=["0050", "2330"],
            start=start,
            end=end,
            use_total_return_adjustment=False,
            use_split_adjustment=False,
            auto_detect_split=False,
            apply_total_return_adjustment=lambda bars: (bars, {"applied": False}),
        )
        self.assertEqual(set(prepared.bars_by_symbol.keys()), {"0050", "2330"})
        self.assertEqual(len(prepared.availability_rows), 2)
        self.assertEqual({row["status"] for row in prepared.availability_rows}, {"OK"})

    def test_queue_benchmark_writeback(self):
        class _WritebackStore:
            def __init__(self):
                self.calls: list[dict[str, object]] = []

            def queue_daily_bars_writeback(self, **kwargs):
                self.calls.append(kwargs)
                return True

        idx = pd.date_range("2025-01-01", periods=3, freq="B", tz="UTC")
        benchmark = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
        benchmark.attrs["symbol"] = "^GSPC"
        benchmark.attrs["source"] = "yfinance"
        store = _WritebackStore()

        queued = queue_benchmark_writeback(store=store, market_code="US", benchmark=benchmark)

        self.assertTrue(queued)
        self.assertEqual(len(store.calls), 1)
        self.assertEqual(store.calls[0]["symbol"], "^GSPC")
        self.assertEqual(store.calls[0]["market"], "US")

    def test_execute_backtest_run_single_and_portfolio(self):
        bars_map = {"0050": _bars(220, seed=21), "2330": _bars(220, seed=22)}
        config_single = BacktestExecutionInput(
            mode="單一標的",
            strategy="buy_hold",
            strategy_params={},
            enable_walk_forward=False,
            train_ratio=0.7,
            objective="sharpe",
            initial_capital=1_000_000.0,
            cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
        )
        single_payload = execute_backtest_run(
            bars_by_symbol={"0050": bars_map["0050"]}, config=config_single
        )
        self.assertEqual(single_payload["mode"], "single")
        self.assertIn("result", single_payload)

        config_portfolio = BacktestExecutionInput(
            mode="投組(多標的)",
            strategy="buy_hold",
            strategy_params={},
            enable_walk_forward=False,
            train_ratio=0.7,
            objective="sharpe",
            initial_capital=1_000_000.0,
            cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
        )
        portfolio_payload = execute_backtest_run(bars_by_symbol=bars_map, config=config_portfolio)
        self.assertEqual(portfolio_payload["mode"], "portfolio")
        self.assertIn("result", portfolio_payload)


if __name__ == "__main__":
    unittest.main()
