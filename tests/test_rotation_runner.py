from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from services.rotation_runner import (
    build_rotation_holding_rank,
    build_rotation_payload,
    prepare_rotation_bars,
)


def _bars(n: int = 140, *, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.002, n))
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


class RotationRunnerTests(unittest.TestCase):
    def test_prepare_rotation_bars(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 1, tzinfo=timezone.utc)
        store = _FakeStore(
            bars_map={"0050": _bars(80, seed=1), "0052": pd.DataFrame()},
            sync_map={"0052": _bars(130, seed=2)},
        )
        prepared = prepare_rotation_bars(
            store=store,
            symbols=["0050", "0052"],
            start_dt=start,
            end_dt=end,
            sync_before_run=False,
            parallel_sync=False,
            normalize_ohlcv_frame=_normalize,
            min_required=120,
        )
        self.assertIn("0052", prepared.bars_by_symbol)
        self.assertIn("0050", prepared.skipped_symbols)
        self.assertEqual(store.sync_calls, ["0050", "0052"])

    def test_build_rotation_holding_rank(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="B", tz="UTC")
        weights = pd.DataFrame(
            {
                "0050": [1.0, 1.0, 0.0, 0.0, 0.0],
                "0052": [0.0, 0.0, 1.0, 1.0, 0.0],
            },
            index=idx,
        )
        rows = build_rotation_holding_rank(
            weights_df=weights,
            selected_symbol_lists=[["0050"], ["0052"], ["0052"]],
            universe_symbols=["0050", "0052", "00935"],
            name_map={"0050": "台灣50", "0052": "科技ETF"},
        )
        self.assertEqual([r["symbol"] for r in rows], ["0052", "0050"])
        self.assertEqual(int(rows[0]["selected_months"]), 2)

    def test_build_rotation_payload(self):
        idx = pd.date_range("2025-01-01", periods=3, freq="B", tz="UTC")
        equity = pd.Series([1_000_000.0, 1_010_000.0, 1_020_000.0], index=idx)
        payload = build_rotation_payload(
            run_key="rk",
            benchmark_symbol="^TWII",
            universe_symbols=["0050", "0052"],
            bars_by_symbol={"0050": _bars(3, seed=11)},
            skipped_symbols=["0052"],
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc).date(),
            end_date=datetime(2025, 1, 10, tzinfo=timezone.utc).date(),
            top_n=1,
            initial_capital=1_000_000.0,
            metrics={"total_return": 0.02},
            equity_series=equity,
            benchmark_equity=equity,
            buy_hold_equity=equity,
            rebalance_records=[],
            trades_df=pd.DataFrame(),
            holding_rank=[],
        )
        self.assertEqual(payload["run_key"], "rk")
        self.assertEqual(payload["benchmark_symbol"], "^TWII")
        self.assertEqual(len(payload["equity_curve"]), 3)


if __name__ == "__main__":
    unittest.main()
