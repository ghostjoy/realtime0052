from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

import pandas as pd

from app import (
    _build_replay_source_hash,
    _deserialize_backtest_run_payload,
    _load_cached_backtest_payload,
    _serialize_backtest_run_payload,
    _serialize_backtest_run_payload_v2,
)
from backtest.types import BacktestMetrics, BacktestResult, Trade


def _build_single_run_payload() -> dict[str, object]:
    idx = pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1200.0],
        },
        index=idx,
    )
    equity = pd.DataFrame({"equity": [1_000_000.0, 1_010_000.0]}, index=idx)
    drawdown = pd.Series([0.0, -0.01], index=idx)
    signals = pd.Series([0, 1], index=idx)
    trade = Trade(
        entry_date=idx[0],
        entry_price=100.0,
        exit_date=idx[1],
        exit_price=101.0,
        qty=1.0,
        fee=0.0,
        tax=0.0,
        slippage=0.0,
        pnl=1.0,
        pnl_pct=0.01,
    )
    metrics = BacktestMetrics(
        total_return=0.01,
        cagr=0.12,
        max_drawdown=-0.01,
        sharpe=1.2,
        win_rate=1.0,
        avg_win=1.0,
        avg_loss=0.0,
        trades=1,
    )
    result = BacktestResult(
        equity_curve=equity,
        trades=[trade],
        metrics=metrics,
        drawdown_series=drawdown,
        yearly_returns={"2026": 0.01},
        signals=signals,
    )
    return {
        "mode": "single",
        "walk_forward": False,
        "initial_capital": 1_000_000.0,
        "symbol": "0050",
        "bars_by_symbol": {"0050": bars},
        "result": result,
    }


class BacktestReplayPayloadTests(unittest.TestCase):
    def test_serialize_v3_and_deserialize_roundtrip(self):
        run_payload = _build_single_run_payload()
        payload_v3 = _serialize_backtest_run_payload(run_payload)
        self.assertEqual(int(payload_v3.get("format_version", 0) or 0), 3)
        restored = _deserialize_backtest_run_payload(payload_v3)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.get("mode"), "single")
        self.assertIn("0050", restored.get("bars_by_symbol", {}))

    def test_deserialize_legacy_v2_payload(self):
        run_payload = _build_single_run_payload()
        payload_v2 = _serialize_backtest_run_payload_v2(run_payload)
        restored = _deserialize_backtest_run_payload(payload_v2)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.get("mode"), "single")
        self.assertIn("0050", restored.get("bars_by_symbol", {}))

    def test_load_cached_backtest_payload_accepts_schema2_when_hash_matches(self):
        run_payload = _build_single_run_payload()
        payload_v2 = _serialize_backtest_run_payload_v2(run_payload)
        expected_hash = _build_replay_source_hash({"k": "v"})
        cached = SimpleNamespace(
            params={"schema_version": 2, "source_hash": expected_hash},
            payload=payload_v2,
        )

        class _FakeStore:
            def load_latest_backtest_replay_run(self, run_key):
                return cached

        restored, message = _load_cached_backtest_payload(
            store=_FakeStore(),
            run_key="bt:tw:0050",
            expected_schema=3,
            expected_hash=expected_hash,
        )
        self.assertIsNotNone(restored)
        self.assertIn("已載入", message)


if __name__ == "__main__":
    unittest.main()
