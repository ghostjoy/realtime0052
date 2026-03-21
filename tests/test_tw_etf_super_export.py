from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from services.tw_etf_super_export import (
    _sync_tw_etf_super_export_sources,
    build_tw_etf_all_types_main_export_frame,
    export_tw_etf_super_table_artifact,
)


class _FakeStore:
    def __init__(self):
        self.saved_runs: list[dict[str, object]] = []
        self.saved_aum_snapshots: list[dict[str, object]] = []
        self.aum_history_calls: list[dict[str, object]] = []

    def save_tw_etf_super_export_run(self, **kwargs):
        self.saved_runs.append(dict(kwargs))
        return "tw_etf_super_export:test"

    def load_tw_etf_mis_daily_coverage(self, **kwargs):
        return {}

    def save_tw_etf_aum_snapshot(self, **kwargs):
        self.saved_aum_snapshots.append(dict(kwargs))
        rows = kwargs.get("rows")
        return len(rows) if isinstance(rows, list) else 0

    def load_tw_etf_aum_history(self, **kwargs):
        self.aum_history_calls.append(dict(kwargs))
        return pd.DataFrame(
            [
                {
                    "etf_code": "0050",
                    "etf_name": "元大台灣50",
                    "trade_date": "2026-03-07",
                    "aum_billion": 1234.0,
                },
                {
                    "etf_code": "0050",
                    "etf_name": "元大台灣50",
                    "trade_date": "2026-03-10",
                    "aum_billion": 1250.0,
                },
            ]
        )


class TwEtfSuperExportTests(unittest.TestCase):
    def test_build_main_export_frame_strips_official_category_columns(self):
        frame = pd.DataFrame(
            [
                {
                    "編號": 1,
                    "代碼": "0050",
                    "ETF": "元大台灣50",
                    "官方主分類": "台股ETF",
                    "官方次分類": "全市場指數",
                    "類型": "台股ETF",
                    "YTD績效(%)": 12.34,
                }
            ]
        )

        out = build_tw_etf_all_types_main_export_frame(
            table_df=frame,
            meta={
                "market_2025_return": 20.01,
                "market_ytd_return": 8.88,
                "market_daily_return": 0.45,
            },
        )

        self.assertEqual(str(out.iloc[0]["代碼"]), "^TWII")
        self.assertNotIn("官方主分類", out.columns)
        self.assertNotIn("官方次分類", out.columns)

    @patch("services.tw_etf_super_export.sync_twse_etf_mis_daily")
    @patch("services.tw_etf_super_export.sync_twse_etf_margin_daily")
    @patch("services.tw_etf_super_export.sync_twse_etf_daily_market")
    def test_sync_sources_also_updates_aum_track(
        self,
        daily_mock,
        margin_mock,
        mis_mock,
    ):
        store = _FakeStore()
        daily_mock.return_value = {"synced_days": 1, "saved_rows": 100}
        margin_mock.return_value = {"synced_days": 1, "saved_rows": 95}
        mis_mock.return_value = {"synced_days": 1, "saved_rows": 88}

        fake_app = SimpleNamespace(
            _resolve_latest_tw_trade_day_token=lambda anchor_yyyymmdd=None: str(
                anchor_yyyymmdd or "20260310"
            ),
            _resolve_latest_tw_trade_date_iso=lambda token: "2026-03-10",
            _fetch_twse_snapshot_network_single=lambda token: (
                str(token),
                pd.DataFrame([{"code": "0050", "name": "元大台灣50"}]),
            ),
            _fetch_twse_snapshot_with_fallback=lambda token, lookback_days=14: (
                str(token),
                pd.DataFrame([{"code": "0050", "name": "元大台灣50"}]),
            ),
            _fetch_twse_three_investors_with_fallback=lambda token, lookback_days=14: (
                str(token),
                pd.DataFrame([{"code": "0050", "name": "元大台灣50"}]),
            ),
            _load_tw_etf_aum_snapshot_info=lambda target_yyyymmdd="": {
                "0050": {"name": "元大台灣50", "aum_billion": 1234.0}
            },
            _build_tw_etf_aum_rows_from_snapshot_info=lambda info: [
                {
                    "etf_code": "0050",
                    "etf_name": "元大台灣50",
                    "aum_billion": 1234.0,
                }
            ],
        )

        summary = _sync_tw_etf_super_export_sources(
            store=store,
            app_module=fake_app,
            daily_lookback_days=14,
            force=False,
            target_trade_date="20260310",
        )

        self.assertEqual(summary["aum_track"]["status"], "synced")
        self.assertEqual(summary["aum_track"]["trade_date"], "2026-03-10")
        self.assertEqual(int(summary["aum_track"]["updated"]), 1)
        self.assertEqual(len(store.saved_aum_snapshots), 1)
        self.assertEqual(store.saved_aum_snapshots[0]["trade_date"], "2026-03-10")

    def test_export_passes_target_trade_date_to_historical_helpers(self):
        store = _FakeStore()
        captured: dict[str, object] = {}

        def build_performance_table(**kwargs):
            captured["performance"] = dict(kwargs)
            return pd.DataFrame([{"代碼": "0050", "ETF": "元大台灣50"}]), {
                "ytd_end_used": "2026-03-10"
            }

        def build_daily_market_overview(**kwargs):
            captured["daily"] = dict(kwargs)
            return pd.DataFrame([{"代碼": "0050", "ETF": "元大台灣50"}]), {
                "last_trade_date": "2026-03-10"
            }

        def build_margin_overview(**kwargs):
            captured["margin"] = dict(kwargs)
            return pd.DataFrame([{"代碼": "0050", "ETF": "元大台灣50"}]), {
                "last_trade_date": "2026-03-10"
            }

        def build_mis_overview(**kwargs):
            captured["mis"] = dict(kwargs)
            return pd.DataFrame([{"代碼": "0050", "ETF": "元大台灣50"}]), {
                "last_trade_date": "2026-03-10"
            }

        def build_three_investors_overview(**kwargs):
            captured["three"] = dict(kwargs)
            return pd.DataFrame([{"代碼": "0050", "ETF": "元大台灣50"}]), {
                "last_trade_date": "2026-03-10"
            }

        def build_aum_export_frame(history_df):
            captured["aum_history_df"] = history_df.copy()
            return pd.DataFrame(
                [
                    {
                        "代碼": "0050",
                        "ETF": "元大台灣50",
                        "基金規模1日變化(億)": 16.0,
                    }
                ]
            )

        fake_app = SimpleNamespace(
            _build_tw_etf_all_types_performance_table=build_performance_table,
            _build_tw_etf_aum_export_frame=build_aum_export_frame,
            _build_tw_etf_daily_market_overview=build_daily_market_overview,
            _build_tw_etf_margin_overview=build_margin_overview,
            _build_tw_etf_mis_overview=build_mis_overview,
            _build_tw_etf_three_investors_overview=build_three_investors_overview,
            _load_tw_etf_aum_snapshot_info=lambda target_yyyymmdd="": {
                "0050": {"name": "元大台灣50", "aum_billion": 1250.0}
            },
            _resolve_latest_tw_trade_day_token=lambda anchor_yyyymmdd=None: str(
                anchor_yyyymmdd or "20260310"
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("services.tw_etf_super_export._load_app_module", return_value=fake_app),
                patch(
                    "services.tw_etf_super_export._sync_tw_etf_super_export_sources",
                    return_value={},
                ) as sync_mock,
                patch(
                    "services.tw_etf_super_export.build_tw_etf_all_types_main_export_frame",
                    return_value=pd.DataFrame([{"代碼": "0050"}]),
                ),
                patch(
                    "services.tw_etf_super_export.build_tw_etf_super_export_table",
                    side_effect=lambda **kwargs: (
                        captured.__setitem__("super_export_kwargs", dict(kwargs))
                        or pd.DataFrame([{"代碼": "0050", "ETF": "元大台灣50"}])
                    ),
                ),
                patch(
                    "services.tw_etf_super_export.build_tw_etf_super_export_csv_bytes",
                    return_value=b"csv",
                ),
                patch("services.tw_etf_super_export._frame_payload", return_value={"rows": []}),
            ):
                result = export_tw_etf_super_table_artifact(
                    store=store,
                    out=tmpdir,
                    ytd_end="20260310",
                )

        sync_mock.assert_called_once()
        self.assertEqual(sync_mock.call_args.kwargs["target_trade_date"], "20260310")
        self.assertEqual(captured["performance"]["ytd_end_yyyymmdd"], "20260310")
        self.assertEqual(captured["daily"]["target_trade_date"], "20260310")
        self.assertEqual(captured["margin"]["target_trade_date"], "20260310")
        self.assertEqual(captured["mis"]["target_trade_date"], "20260310")
        self.assertEqual(captured["three"]["target_trade_date"], "20260310")
        self.assertEqual(store.aum_history_calls[0]["etf_codes"], ("0050",))
        self.assertEqual(store.aum_history_calls[0]["keep_days"], 0)
        self.assertFalse(captured["aum_history_df"].empty)
        self.assertIn("aum_frame", captured["super_export_kwargs"])
        self.assertEqual(
            float(captured["super_export_kwargs"]["aum_frame"].iloc[0]["基金規模1日變化(億)"]),
            16.0,
        )
        self.assertEqual(
            Path(str(result["output_path"])).name,
            "tw_etf_super_export_20260310.csv",
        )
        self.assertEqual(str(result["trade_date_anchor"]), "20260310")


if __name__ == "__main__":
    unittest.main()
