from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from services.tw_etf_super_export import export_tw_etf_super_table_artifact


class _FakeStore:
    def __init__(self):
        self.saved_runs: list[dict[str, object]] = []

    def save_tw_etf_super_export_run(self, **kwargs):
        self.saved_runs.append(dict(kwargs))
        return "tw_etf_super_export:test"

    def load_tw_etf_mis_daily_coverage(self, **kwargs):
        return {}


class TwEtfSuperExportTests(unittest.TestCase):
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

        fake_app = SimpleNamespace(
            _build_tw_etf_all_types_performance_table=build_performance_table,
            _build_tw_etf_daily_market_overview=build_daily_market_overview,
            _build_tw_etf_margin_overview=build_margin_overview,
            _build_tw_etf_mis_overview=build_mis_overview,
            _build_tw_etf_three_investors_overview=build_three_investors_overview,
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
                    return_value=pd.DataFrame([{"代碼": "0050", "ETF": "元大台灣50"}]),
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
        self.assertEqual(
            Path(str(result["output_path"])).name,
            "tw_etf_super_export_20260310.csv",
        )
        self.assertEqual(str(result["trade_date_anchor"]), "20260310")


if __name__ == "__main__":
    unittest.main()
