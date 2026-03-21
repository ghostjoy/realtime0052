from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from services.tw_etf_constituent_sync import TW_ETF_CONSTITUENTS_DATASET_KEY
from services.tw_etf_report import export_tw_etf_report_artifact


class _FakeStore:
    def __init__(self):
        self.snapshots: dict[tuple[str, str, str, str], dict[str, object]] = {}

    def save_market_snapshot(self, **kwargs):
        key = (
            str(kwargs.get("dataset_key") or ""),
            str(kwargs.get("market") or ""),
            str(kwargs.get("symbol") or ""),
            str(kwargs.get("interval") or ""),
        )
        self.snapshots[key] = {
            "payload": dict(kwargs.get("payload") or {}),
            "source": str(kwargs.get("source") or ""),
            "asof": kwargs.get("asof"),
        }
        return 1

    def load_latest_market_snapshot(self, *, dataset_key: str, market: str = "", symbol: str = "", interval: str = ""):
        return self.snapshots.get((dataset_key, market, symbol, interval))

    def load_tw_etf_aum_history(self, *, etf_codes=(), keep_days=0):
        return pd.DataFrame(
            [
                {
                    "etf_code": "0052",
                    "etf_name": "富邦科技",
                    "trade_date": "2026-03-19",
                    "aum_billion": 88.0,
                },
                {
                    "etf_code": "0052",
                    "etf_name": "富邦科技",
                    "trade_date": "2026-03-20",
                    "aum_billion": 90.5,
                },
            ]
        )

    def load_daily_bars(self, *, symbol: str, market: str, start: datetime, end: datetime):
        return pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 102.5, 103.5],
                "volume": [1000.0, 1100.0, 1050.0, 1200.0],
            },
            index=pd.to_datetime(
                ["2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20"], utc=True
            ),
        )

    def sync_symbol_history(self, symbol: str, market: str, start: datetime, end: datetime):
        return SimpleNamespace(error=None)

    def load_symbol_metadata(self, symbols: list[str], market: str):
        return {"0052": {"name": "富邦科技"}}


class TwEtfReportTests(unittest.TestCase):
    def test_export_report_bundle_creates_symbol_prefixed_files(self):
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
            _resolve_latest_tw_trade_day_token=lambda anchor_yyyymmdd=None: str(anchor_yyyymmdd or "20260320"),
            _build_tw_etf_all_types_performance_table=lambda **kwargs: (
                pd.DataFrame(
                    [
                        {
                            "編號": "001",
                            "代碼": "0052",
                            "ETF": "富邦科技",
                            "類型": "科技型",
                            "YTD績效(%)": 12.34,
                            "ETF規模(億)": 90.5,
                            "收盤": 103.5,
                            "今日漲幅": 1.2,
                        }
                    ]
                ),
                {"market_ytd_return": 8.0},
            ),
            _build_tw_etf_aum_export_frame=lambda history_df: pd.DataFrame(
                [
                    {
                        "代碼": "0052",
                        "ETF": "富邦科技",
                        "基金規模最新(億)": 90.5,
                    }
                ]
            ),
            _build_tw_etf_daily_market_overview=lambda **kwargs: (
                pd.DataFrame([{"代碼": "0052", "ETF": "富邦科技", "成交金額(億)": 12.0}]),
                {},
            ),
            _build_tw_etf_margin_overview=lambda **kwargs: (
                pd.DataFrame([{"代碼": "0052", "ETF": "富邦科技", "融資餘額": 1000}]),
                {},
            ),
            _build_tw_etf_mis_overview=lambda **kwargs: (
                pd.DataFrame([{"代碼": "0052", "ETF": "富邦科技", "折溢價(%)": 0.1}]),
                {},
            ),
            _build_tw_etf_three_investors_overview=lambda **kwargs: (
                pd.DataFrame([{"代碼": "0052", "ETF": "富邦科技", "三大法人買賣超(張)": 25}]),
                {},
            ),
            _load_etf_constituents_rows=lambda service, etf_code, force_refresh_constituents=False: (
                [],
                "moneydj_basic0007b_full",
                "",
            ),
            _market_service=lambda: object(),
        )

        def _fake_chart_export(**kwargs):
            out_path = Path(str(kwargs["out"]))
            out_path.write_bytes(b"png")
            return {"items": [{"symbol": "0052", "path": str(out_path)}], "exported_count": 1}

        def _fake_aum_chart(frame, output_path, symbol):
            Path(output_path).write_bytes(b"png")

        def _fake_heatmap(**kwargs):
            Path(str(kwargs["output_path"])).write_bytes(b"png")
            return True

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("services.tw_etf_report._load_app_module", return_value=fake_app),
                patch(
                    "services.tw_etf_report._sync_tw_etf_super_export_sources",
                    return_value={
                        "main": {"status": "synced", "used_trade_date": "20260320"},
                        "daily_market": {"saved_rows": 10, "synced_days": 1, "latest_date": "2026-03-20"},
                        "margin": {"saved_rows": 10, "synced_days": 1, "latest_date": "2026-03-20"},
                        "mis": {"saved_rows": 10, "synced_days": 1, "latest_date": "2026-03-20"},
                        "three_investors": {"status": "synced", "used_trade_date": "2026-03-20", "row_count": 1},
                        "aum_track": {"status": "synced", "updated": 2, "trade_date": "2026-03-20"},
                        "issues": [],
                    },
                ),
                patch("services.tw_etf_report.sync_symbols_if_needed", return_value=({}, SimpleNamespace())),
                patch("services.tw_etf_report.export_backtest_chart_artifact", side_effect=_fake_chart_export),
                patch("services.tw_etf_report._write_aum_chart", side_effect=_fake_aum_chart),
                patch("services.tw_etf_report._write_constituent_heatmap", side_effect=_fake_heatmap),
            ):
                result = export_tw_etf_report_artifact(
                    store=store,
                    symbol="0052",
                    out=tmpdir,
                    sync_constituents=False,
                    backtest_start="2026-03-17",
                    backtest_end="2026-03-20",
                )

            report_dir = Path(str(result["report_dir"]))
            self.assertTrue((report_dir / "0052_summary.md").exists())
            self.assertTrue((report_dir / "0052_overview.csv").exists())
            self.assertTrue((report_dir / "0052_backtest.png").exists())
            self.assertTrue((report_dir / "0052_aum_track.png").exists())
            self.assertTrue((report_dir / "0052_constituent_heatmap.png").exists())
            self.assertTrue((report_dir / "0052_indicators_snapshot.csv").exists())
            self.assertTrue((report_dir / "0052_indicators_timeseries.csv").exists())
            self.assertTrue((report_dir / "0052_indicators_summary.md").exists())
            self.assertTrue((report_dir / "0052_sync_log.json").exists())
            self.assertTrue((report_dir / "0052_sync_log.md").exists())
            self.assertGreaterEqual(int(result["file_count"]), 10)


if __name__ == "__main__":
    unittest.main()
