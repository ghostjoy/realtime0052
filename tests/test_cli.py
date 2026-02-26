from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

import cli


class CliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("cli._resolve_store")
    @patch("cli.sync_symbols_if_needed")
    def test_sync_defaults_to_market_seed_symbols(self, sync_mock, resolve_store_mock):
        resolve_store_mock.return_value = object()
        sync_mock.return_value = (
            {},
            SimpleNamespace(synced_symbols=[], skipped_symbols=["0050", "0052"], issues=[]),
        )

        result = self.runner.invoke(cli.cli, ["sync", "--market", "TW", "--days", "5"])

        self.assertEqual(result.exit_code, 0)
        kwargs = sync_mock.call_args.kwargs
        self.assertEqual(kwargs["market"], "TW")
        self.assertEqual(kwargs["symbols"], ["0050", "0052"])

    @patch("cli._resolve_store")
    @patch("cli.sync_symbols_if_needed")
    @patch("cli.load_and_prepare_symbol_bars")
    @patch("cli.execute_backtest_run")
    def test_backtest_outputs_metrics(
        self,
        execute_mock,
        prepare_mock,
        sync_mock,
        resolve_store_mock,
    ):
        resolve_store_mock.return_value = object()
        sync_mock.return_value = (
            {},
            SimpleNamespace(synced_symbols=["0050"], skipped_symbols=[], issues=[]),
        )
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1100.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
        )
        prepare_mock.return_value = SimpleNamespace(
            bars_by_symbol={"0050": bars},
            availability_rows=[{"symbol": "0050", "status": "OK"}],
        )
        execute_mock.return_value = {
            "mode": "single",
            "result": SimpleNamespace(
                metrics=SimpleNamespace(
                    total_return=0.1,
                    cagr=0.08,
                    max_drawdown=-0.12,
                    sharpe=1.23,
                    trades=5,
                )
            ),
        }

        result = self.runner.invoke(cli.cli, ["backtest", "--symbol", "0050", "--market", "TW"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("total_return=10.00%", result.output)
        self.assertIn("sharpe=1.230", result.output)

    @patch("cli._resolve_store")
    @patch("cli.run_market_data_bootstrap")
    def test_bootstrap_reports_summary(self, bootstrap_mock, resolve_store_mock):
        resolve_store_mock.return_value = object()
        bootstrap_mock.return_value = {
            "scope": "both",
            "years": 5,
            "metadata_upserted": 120,
            "total_symbols": 25,
            "synced_success": 24,
            "failed_symbols": 1,
            "issue_count": 1,
            "issues": ["TW:0050:timeout"],
        }

        result = self.runner.invoke(cli.cli, ["bootstrap", "--scope", "both"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("scope=both", result.output)
        self.assertIn("! TW:0050:timeout", result.output)


if __name__ == "__main__":
    unittest.main()
