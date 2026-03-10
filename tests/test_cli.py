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
    @patch("cli.sync_twse_etf_daily_market")
    def test_sync_twse_etf_daily_outputs_summary(self, sync_mock, resolve_store_mock):
        resolve_store_mock.return_value = object()
        sync_mock.return_value = {
            "start_date": "2026-03-01",
            "end_date": "2026-03-07",
            "latest_date": "2026-03-06",
            "requested_days": 7,
            "synced_days": 3,
            "skipped_days": 2,
            "empty_days": 2,
            "saved_rows": 600,
            "issues": ["2026-03-02: timeout"],
        }

        result = self.runner.invoke(cli.cli, ["sync-twse-etf-daily", "--start", "2026-03-01"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("latest=2026-03-06", result.output)
        self.assertIn("saved_rows=600", result.output)
        self.assertIn("! 2026-03-02: timeout", result.output)

    @patch("cli._resolve_store")
    @patch("cli.sync_twse_etf_mis_daily")
    def test_sync_twse_etf_mis_outputs_summary(self, sync_mock, resolve_store_mock):
        resolve_store_mock.return_value = object()
        sync_mock.return_value = {
            "start_date": "2026-03-06",
            "end_date": "2026-03-06",
            "latest_date": "2026-03-06",
            "requested_days": 1,
            "synced_days": 1,
            "skipped_days": 0,
            "empty_days": 0,
            "saved_rows": 120,
            "issues": ["latest available date 2026-03-05 is outside requested range"],
        }

        result = self.runner.invoke(cli.cli, ["sync-twse-etf-mis", "--end", "2026-03-06"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("latest=2026-03-06", result.output)
        self.assertIn("saved_rows=120", result.output)
        self.assertIn(
            "! latest available date 2026-03-05 is outside requested range", result.output
        )

    @patch("cli._resolve_store")
    @patch("cli.export_tw_etf_super_table_artifact")
    def test_export_tw_etf_super_table_outputs_summary(self, export_mock, resolve_store_mock):
        fake_store = object()
        resolve_store_mock.return_value = fake_store
        export_mock.return_value = {
            "run_id": "tw_etf_super_export:20260310T090000000000",
            "output_path": "/tmp/tw_etf_super_export_20260310.csv",
            "trade_date_anchor": "20260310",
            "row_count": 123,
            "column_count": 27,
            "ytd_start": "20260101",
            "ytd_end": "20260310",
            "compare_start": "20250101",
            "compare_end": "20251231",
            "refresh_summary": {
                "main": {"status": "synced", "used_trade_date": "20260310"},
                "daily_market": {"synced_days": 2, "saved_rows": 240},
                "mis": {"synced_days": 1, "saved_rows": 120},
            },
            "issues": ["daily_market: skipped holiday 2026-03-08"],
        }

        result = self.runner.invoke(
            cli.cli,
            [
                "export-tw-etf-super-table",
                "--out",
                "/tmp/custom.csv",
                "--ytd-start",
                "20260101",
                "--ytd-end",
                "20260310",
                "--compare-start",
                "20250101",
                "--compare-end",
                "20251231",
                "--daily-lookback-days",
                "21",
                "--force",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        export_mock.assert_called_once_with(
            store=fake_store,
            out="/tmp/custom.csv",
            ytd_start="20260101",
            ytd_end="20260310",
            compare_start="20250101",
            compare_end="20251231",
            force=True,
            daily_lookback_days=21,
        )
        self.assertIn("trade_date=20260310", result.output)
        self.assertIn("path=/tmp/tw_etf_super_export_20260310.csv", result.output)
        self.assertIn("rows=123 cols=27", result.output)
        self.assertIn("refresh main=synced(20260310)", result.output)
        self.assertIn("daily=synced:2/saved:240", result.output)
        self.assertIn("mis=synced:1/saved:120", result.output)
        self.assertIn("! daily_market: skipped holiday 2026-03-08", result.output)

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
