from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest import CostModel
from services.heatmap_runner import compute_heatmap_rows, prepare_heatmap_bars


def _bars(n: int = 60, *, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.002, n))
    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


class _FakeStore:
    def __init__(
        self, bars_map: dict[str, pd.DataFrame], sync_map: dict[str, pd.DataFrame] | None = None
    ):
        self._bars_map = {k.upper(): v for k, v in bars_map.items()}
        self._sync_map = {k.upper(): v for k, v in (sync_map or {}).items()}
        self.sync_calls: list[str] = []

    def load_daily_bars(self, symbol, market, start=None, end=None):
        return self._bars_map.get(str(symbol).upper(), pd.DataFrame())

    def sync_symbol_history(self, symbol, market, start=None, end=None):
        token = str(symbol).upper()
        self.sync_calls.append(token)
        if token in self._sync_map:
            self._bars_map[token] = self._sync_map[token]
        return SimpleNamespace(error="", rows_upserted=1, source="unit", fallback_depth=0)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return frame


class HeatmapRunnerTests(unittest.TestCase):
    def test_prepare_heatmap_bars_lazy_sync(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 3, 1, tzinfo=timezone.utc)
        store = _FakeStore(
            bars_map={"0050": _bars(5, seed=1), "2330": pd.DataFrame()},
            sync_map={"2330": _bars(40, seed=2)},
        )
        prepared = prepare_heatmap_bars(
            store=store,
            symbols=["0050", "2330"],
            start_dt=start,
            end_dt=end,
            min_required=20,
            sync_before_run=False,
            parallel_sync=False,
            normalize_ohlcv_frame=_normalize,
        )
        self.assertIn("0050", prepared.bars_cache)
        self.assertIn("2330", prepared.bars_cache)
        self.assertGreaterEqual(len(prepared.bars_cache["2330"]), 20)
        self.assertEqual(store.sync_calls, ["0050", "2330"])

    def test_prepare_heatmap_bars_can_disable_lazy_sync(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 3, 1, tzinfo=timezone.utc)
        store = _FakeStore(
            bars_map={"0050": _bars(5, seed=11), "2330": pd.DataFrame()},
            sync_map={"2330": _bars(40, seed=12)},
        )
        prepared = prepare_heatmap_bars(
            store=store,
            symbols=["0050", "2330"],
            start_dt=start,
            end_dt=end,
            min_required=20,
            sync_before_run=False,
            parallel_sync=False,
            lazy_sync_on_insufficient=False,
            normalize_ohlcv_frame=_normalize,
        )
        self.assertIn("0050", prepared.bars_cache)
        self.assertIn("2330", prepared.bars_cache)
        self.assertEqual(len(prepared.bars_cache["2330"]), 0)
        self.assertEqual(store.sync_calls, [])

    def test_compute_heatmap_rows(self):
        bars_0050 = _bars(80, seed=3)
        bars_2330 = _bars(80, seed=4)
        benchmark_close = pd.to_numeric(bars_0050["close"], errors="coerce")
        rows = compute_heatmap_rows(
            run_symbols=["0050", "2330"],
            bars_cache={"0050": bars_0050, "2330": bars_2330},
            benchmark_close=benchmark_close,
            strategy="buy_hold",
            strategy_params={},
            cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
            name_map={"0050": "台灣50", "2330": "台積電"},
            min_required=2,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["symbol"] for row in rows}, {"0050", "2330"})
        self.assertIn("excess_pct", rows[0])

    def test_compute_heatmap_rows_parallel_keeps_symbol_coverage(self):
        bars_0050 = _bars(100, seed=21)
        bars_2330 = _bars(100, seed=22)
        bars_2317 = _bars(100, seed=23)
        benchmark_close = pd.to_numeric(bars_0050["close"], errors="coerce")
        rows = compute_heatmap_rows(
            run_symbols=["0050", "2330", "2317"],
            bars_cache={"0050": bars_0050, "2330": bars_2330, "2317": bars_2317},
            benchmark_close=benchmark_close,
            strategy="buy_hold",
            strategy_params={},
            cost_model=CostModel(fee_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0),
            name_map={"0050": "台灣50", "2330": "台積電", "2317": "鴻海"},
            min_required=2,
            max_workers=3,
        )
        self.assertEqual({row["symbol"] for row in rows}, {"0050", "2330", "2317"})


if __name__ == "__main__":
    unittest.main()
