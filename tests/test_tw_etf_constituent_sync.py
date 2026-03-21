from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from services.tw_etf_constituent_sync import (
    TW_ETF_CONSTITUENTS_DATASET_KEY,
    load_latest_tw_etf_constituent_snapshot,
    sync_tw_etf_constituent_snapshots,
)


class _FakeStore:
    def __init__(self):
        self.snapshots: dict[tuple[str, str, str, str], dict[str, object]] = {}
        self.saved_calls: list[dict[str, object]] = []

    def save_market_snapshot(self, **kwargs):
        key = (
            str(kwargs.get("dataset_key") or ""),
            str(kwargs.get("market") or ""),
            str(kwargs.get("symbol") or ""),
            str(kwargs.get("interval") or ""),
        )
        payload = dict(kwargs)
        payload["payload"] = dict(kwargs.get("payload") or {})
        self.snapshots[key] = payload
        self.saved_calls.append(payload)
        return 1

    def load_latest_market_snapshot(self, *, dataset_key: str, market: str = "", symbol: str = "", interval: str = ""):
        key = (dataset_key, market, symbol, interval)
        row = self.snapshots.get(key)
        if row is None:
            return None
        return {
            "payload": dict(row.get("payload") or {}),
            "source": str(row.get("source") or ""),
            "asof": row.get("asof"),
        }


class TwEtfConstituentSyncTests(unittest.TestCase):
    def test_sync_without_symbols_uses_official_profile_map(self):
        store = _FakeStore()
        fake_app = SimpleNamespace(
            _resolve_latest_tw_trade_day_token=lambda anchor_yyyymmdd=None: "20260322",
            _load_tw_etf_official_profile_map=lambda target_yyyymmdd="": {
                "0050": {"stock_name": "元大台灣50"},
                "0052": {"stock_name": "富邦科技"},
            },
            _load_etf_constituents_rows=lambda service, etf_code, force_refresh_constituents=False: (
                [
                    {
                        "rank": 1,
                        "symbol": "2330.TW",
                        "tw_code": "2330",
                        "name": "台積電",
                        "market": "TW",
                        "weight_pct": 55.5,
                    }
                ],
                "moneydj_basic0007b_full",
                "",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("services.tw_etf_constituent_sync._load_app_module", return_value=fake_app):
                result = sync_tw_etf_constituent_snapshots(
                    store=store,
                    symbols=None,
                    force=False,
                    max_workers=1,
                    log_dir=tmpdir,
                )

        self.assertEqual(result["etf_count"], 2)
        self.assertEqual(result["updated_count"], 2)
        self.assertEqual(result["symbols"], ["0050", "0052"])

    def test_sync_saves_updated_snapshot_and_logs(self):
        store = _FakeStore()
        fake_app = SimpleNamespace(
            _resolve_latest_tw_trade_day_token=lambda anchor_yyyymmdd=None: "20260320",
            _build_tw_etf_all_types_performance_table=lambda **kwargs: (
                pd.DataFrame([{"代碼": "0052", "ETF": "富邦科技"}]),
                {},
            ),
            _load_etf_constituents_rows=lambda service, etf_code, force_refresh_constituents=False: (
                [
                    {
                        "rank": 1,
                        "symbol": "2330.TW",
                        "tw_code": "2330",
                        "name": "台積電",
                        "market": "TW",
                        "weight_pct": 55.5,
                    }
                ],
                "moneydj_basic0007b_full",
                "",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("services.tw_etf_constituent_sync._load_app_module", return_value=fake_app):
                result = sync_tw_etf_constituent_snapshots(
                    store=store,
                    symbols=None,
                    force=False,
                    max_workers=1,
                    log_dir=tmpdir,
                )

            self.assertEqual(result["updated_count"], 1)
            self.assertEqual(result["unchanged_count"], 0)
            self.assertTrue(Path(str(result["json_log_path"])).exists())
            self.assertTrue(Path(str(result["markdown_log_path"])).exists())
            snapshot = load_latest_tw_etf_constituent_snapshot(store=store, symbol="0052")
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot["source"], "moneydj_basic0007b_full")
            self.assertEqual(snapshot["rows"][0]["tw_code"], "2330")
            self.assertEqual(
                store.saved_calls[0]["dataset_key"],
                TW_ETF_CONSTITUENTS_DATASET_KEY,
            )

    def test_sync_marks_unchanged_when_payload_matches_latest_snapshot(self):
        store = _FakeStore()
        store.save_market_snapshot(
            dataset_key=TW_ETF_CONSTITUENTS_DATASET_KEY,
            market="TW",
            symbol="0052",
            interval="constituents",
            source="moneydj_basic0007b_full",
            payload={
                "rows": [
                    {
                        "rank": 1,
                        "symbol": "2330.TW",
                        "tw_code": "2330",
                        "name": "台積電",
                        "market": "TW",
                        "weight_pct": 55.5,
                    }
                ],
                "source": "moneydj_basic0007b_full",
            },
        )
        fake_app = SimpleNamespace(
            _load_etf_constituents_rows=lambda service, etf_code, force_refresh_constituents=False: (
                [
                    {
                        "rank": 1,
                        "symbol": "2330.TW",
                        "tw_code": "2330",
                        "name": "台積電",
                        "market": "TW",
                        "weight_pct": 55.5,
                    }
                ],
                "moneydj_basic0007b_full",
                "",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("services.tw_etf_constituent_sync._load_app_module", return_value=fake_app):
                result = sync_tw_etf_constituent_snapshots(
                    store=store,
                    symbols=["0052"],
                    force=False,
                    max_workers=1,
                    log_dir=tmpdir,
                )

        self.assertEqual(result["updated_count"], 0)
        self.assertEqual(result["unchanged_count"], 1)
        self.assertEqual(len(store.saved_calls), 1)


if __name__ == "__main__":
    unittest.main()
