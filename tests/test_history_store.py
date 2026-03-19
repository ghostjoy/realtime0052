from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone

import pandas as pd

from market_data_types import OhlcvSnapshot
from providers.base import ProviderRequest
from storage.history_store import HistoryStore


class _Provider:
    def __init__(self, name: str):
        self.name = name


class _FakeService:
    def __init__(self, source: str = "yahoo", fail: bool = False):
        self.us_twelve = _Provider("twelvedata")
        self.yahoo = _Provider("yahoo")
        self.us_stooq = _Provider("stooq")
        self.tw_fugle_rest = _Provider("tw_fugle_rest")
        self.tw_openapi = _Provider("tw_openapi")
        self.tw_tpex = _Provider("tw_tpex")
        self.source = source
        self.fail = fail

    def _try_ohlcv_chain(self, providers, request: ProviderRequest):
        if self.fail:
            raise RuntimeError("upstream unavailable")
        idx = next(i for i, p in enumerate(providers) if p.name == self.source)
        provider = providers[idx]
        dates = pd.date_range("2024-01-01", periods=30, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            },
            index=dates,
        )
        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval="1d",
            tz="UTC",
            df=df,
            source=provider.name,
            is_delayed=provider.name != "twelvedata",
            fetched_at=datetime.now(tz=timezone.utc),
        )


class _CaptureService:
    def __init__(self):
        self.us_twelve = _Provider("twelvedata")
        self.yahoo = _Provider("yahoo")
        self.us_stooq = _Provider("stooq")
        self.tw_fugle_rest = _Provider("tw_fugle_rest")
        self.tw_fugle_rest.api_key = None
        self.tw_openapi = _Provider("tw_openapi")
        self.tw_tpex = _Provider("tw_tpex")
        self.last_request: ProviderRequest | None = None
        self.last_provider_name: str | None = None

    def _try_ohlcv_chain(self, providers, request: ProviderRequest):
        self.last_request = request
        self.last_provider_name = str(getattr(providers[0], "name", "") or "")
        start = request.start or datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = pd.date_range(start=start, periods=5, freq="B", tz="UTC")
        provider = providers[0]
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            },
            index=dates,
        )
        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval="1d",
            tz="UTC",
            df=df,
            source=provider.name,
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )


class _DupColumnService:
    def __init__(self):
        self.us_twelve = _Provider("twelvedata")
        self.yahoo = _Provider("yahoo")
        self.us_stooq = _Provider("stooq")
        self.tw_fugle_rest = _Provider("tw_fugle_rest")
        self.tw_fugle_rest.api_key = None
        self.tw_openapi = _Provider("tw_openapi")
        self.tw_tpex = _Provider("tw_tpex")

    def _try_ohlcv_chain(self, providers, request: ProviderRequest):
        dates = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
        # Simulate malformed upstream shape: duplicated "open" column.
        df = pd.DataFrame(
            data=[[100.0, 100.0, 101.0, 99.0, 100.5, 1000.0] for _ in range(len(dates))],
            columns=["open", "open", "high", "low", "close", "volume"],
            index=dates,
        )
        return OhlcvSnapshot(
            symbol=request.symbol,
            market=request.market,
            interval="1d",
            tz="UTC",
            df=df,
            source="yahoo",
            is_delayed=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )


class HistoryStoreTests(unittest.TestCase):
    def test_sync_and_load_daily_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService(source="yahoo"))
            report = store.sync_symbol_history(symbol="TSLA", market="US")
            self.assertIsNone(report.error)
            self.assertGreater(report.rows_upserted, 0)
            self.assertEqual(report.fallback_depth, 1)

            bars = store.load_daily_bars("TSLA", "US")
            self.assertFalse(bars.empty)
            self.assertTrue(
                {"open", "high", "low", "close", "volume", "vwap", "source"}.issubset(bars.columns)
            )
            self.assertAlmostEqual(float(bars["vwap"].iloc[0]), 100.1666666667, places=6)

            second = store.sync_symbol_history(symbol="TSLA", market="US")
            self.assertEqual(second.rows_upserted, 0)

    def test_incremental_stale_sync_is_non_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            service = _FakeService(source="yahoo")
            store = HistoryStore(db_path=db_path, service=service)
            first = store.sync_symbol_history(symbol="TSLA", market="US")
            self.assertIsNone(first.error)
            self.assertGreater(first.rows_upserted, 0)

            store.service = _FakeService(source="yahoo", fail=True)
            second = store.sync_symbol_history(
                symbol="TSLA",
                market="US",
                end=datetime(2024, 2, 9, tzinfo=timezone.utc),
            )
            self.assertIsNone(second.error)
            self.assertTrue(second.stale)

    def test_save_and_load_universe_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            store.save_universe_snapshot(
                universe_id="TW:00935",
                symbols=["2330", "2454", " 2317 "],
                source="unit_test",
            )
            snap = store.load_universe_snapshot("TW:00935")
            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(snap.universe_id, "TW:00935")
            self.assertEqual(snap.symbols, ["2330", "2454", "2317"])
            self.assertEqual(snap.source, "unit_test")

    def test_save_and_load_tw_etf_aum_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            dates = ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
            for idx, trade_date in enumerate(dates):
                saved = store.save_tw_etf_aum_snapshot(
                    rows=[
                        {"etf_code": "0050", "etf_name": "元大台灣50", "aum_billion": 1000 + idx},
                        {"etf_code": "0056", "etf_name": "高股息", "aum_billion": 800 + idx},
                    ],
                    trade_date=trade_date,
                    keep_days=3,
                )
                self.assertEqual(saved, 2)

            hist = store.load_tw_etf_aum_history(etf_codes=["0050", "0056"], keep_days=3)
            self.assertFalse(hist.empty)
            self.assertEqual(set(hist["etf_code"].unique()), {"0050", "0056"})
            self.assertEqual(int((hist["etf_code"] == "0050").sum()), 3)
            self.assertEqual(int((hist["etf_code"] == "0056").sum()), 3)
            self.assertNotIn("2026-01-02", set(hist["trade_date"].astype(str)))

    def test_save_tw_etf_aum_history_keep_days_zero_keeps_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            dates = ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
            for idx, trade_date in enumerate(dates):
                saved = store.save_tw_etf_aum_snapshot(
                    rows=[
                        {"etf_code": "0050", "etf_name": "元大台灣50", "aum_billion": 1000 + idx},
                    ],
                    trade_date=trade_date,
                    keep_days=0,
                )
                self.assertEqual(saved, 1)

            hist_all = store.load_tw_etf_aum_history(etf_codes=["0050"], keep_days=0)
            self.assertEqual(int((hist_all["etf_code"] == "0050").sum()), 4)
            self.assertIn("2026-01-02", set(hist_all["trade_date"].astype(str)))

    def test_tw_etf_aum_history_defaults_to_keep_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            dates = ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
            for idx, trade_date in enumerate(dates):
                saved = store.save_tw_etf_aum_snapshot(
                    rows=[
                        {"etf_code": "0050", "etf_name": "元大台灣50", "aum_billion": 1000 + idx},
                    ],
                    trade_date=trade_date,
                )
                self.assertEqual(saved, 1)

            hist_all = store.load_tw_etf_aum_history(etf_codes=["0050"])
            self.assertEqual(int((hist_all["etf_code"] == "0050").sum()), 4)
            self.assertIn("2026-01-02", set(hist_all["trade_date"].astype(str)))

    def test_clear_tw_etf_aum_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            store.save_tw_etf_aum_snapshot(
                rows=[{"etf_code": "0050", "etf_name": "元大台灣50", "aum_billion": 1000.0}],
                trade_date="2026-01-05",
                keep_days=0,
            )
            removed = store.clear_tw_etf_aum_history()
            self.assertGreaterEqual(int(removed), 1)
            hist = store.load_tw_etf_aum_history(etf_codes=["0050"], keep_days=0)
            self.assertTrue(hist.empty)

    def test_save_and_load_tw_etf_daily_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            saved = store.save_tw_etf_daily_market(
                rows=[
                    {
                        "trade_date": "2026-03-06",
                        "etf_code": "0050",
                        "etf_name": "元大台灣50",
                        "trade_value": 100000000.0,
                        "trade_volume": 123456.0,
                        "trade_count": 789,
                        "open": 75.0,
                        "high": 76.0,
                        "low": 74.5,
                        "close": 75.5,
                        "change": -0.5,
                        "source": "unit_test",
                    },
                    {
                        "trade_date": "2026-03-06",
                        "etf_code": "0056",
                        "etf_name": "元大高股息",
                        "trade_value": 50000000.0,
                        "trade_volume": 654321.0,
                        "trade_count": 321,
                        "open": 38.0,
                        "high": 38.5,
                        "low": 37.8,
                        "close": 38.2,
                        "change": 0.2,
                        "source": "unit_test",
                    },
                ],
                trade_date="2026-03-06",
            )
            self.assertEqual(saved, 2)

            coverage = store.load_tw_etf_daily_market_coverage()
            self.assertEqual(int(coverage["row_count"]), 2)
            self.assertEqual(int(coverage["trade_date_count"]), 1)
            self.assertEqual(int(coverage["symbol_count"]), 2)

            frame = store.load_tw_etf_daily_market(start="2026-03-06", end="2026-03-06")
            self.assertEqual(len(frame), 2)
            self.assertEqual(float(frame.loc[frame["etf_code"] == "0050", "close"].iloc[0]), 75.5)

    def test_save_and_load_tw_etf_mis_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            saved = store.save_tw_etf_mis_daily(
                rows=[
                    {
                        "trade_date": "2026-03-06",
                        "etf_code": "0050",
                        "etf_name": "元大台灣50",
                        "issued_units": 1350000000.0,
                        "creation_redemption_diff": 1000000.0,
                        "market_price": 182.1,
                        "estimated_nav": 181.62,
                        "premium_discount_pct": 0.26,
                        "previous_nav": 180.55,
                        "reference_url": "https://example.com/nav",
                        "updated_at": "2026-03-06T14:30:00+08:00",
                        "source": "unit_test",
                    },
                    {
                        "trade_date": "2026-03-06",
                        "etf_code": "0056",
                        "etf_name": "元大高股息",
                        "issued_units": 2460000000.0,
                        "creation_redemption_diff": -500000.0,
                        "market_price": 40.2,
                        "estimated_nav": 40.05,
                        "premium_discount_pct": 0.37,
                        "previous_nav": 39.88,
                        "reference_url": "https://example.com/nav2",
                        "updated_at": "2026-03-06T14:30:00+08:00",
                        "source": "unit_test",
                    },
                ],
                trade_date="2026-03-06",
            )
            self.assertEqual(saved, 2)

            coverage = store.load_tw_etf_mis_daily_coverage()
            self.assertEqual(int(coverage["row_count"]), 2)
            self.assertEqual(int(coverage["trade_date_count"]), 1)
            self.assertEqual(int(coverage["symbol_count"]), 2)

            frame = store.load_tw_etf_mis_daily(start="2026-03-06", end="2026-03-06")
            self.assertEqual(len(frame), 2)
            self.assertEqual(
                float(frame.loc[frame["etf_code"] == "0050", "estimated_nav"].iloc[0]),
                181.62,
            )

    def test_save_and_load_tw_etf_margin_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            saved = store.save_tw_etf_margin_daily(
                rows=[
                    {
                        "trade_date": "2026-03-18",
                        "etf_code": "0050",
                        "etf_name": "元大台灣50",
                        "margin_buy": 826,
                        "margin_sell": 945,
                        "margin_cash_redemption": 4,
                        "margin_prev_balance": 12753,
                        "margin_balance": 12630,
                        "margin_next_limit": 4480625,
                        "short_buy": 22,
                        "short_sell": 89,
                        "short_stock_redemption": 0,
                        "short_prev_balance": 745,
                        "short_balance": 812,
                        "short_next_limit": 4480625,
                        "offset_amount": 21,
                        "note": "X",
                        "source": "unit_test",
                    },
                    {
                        "trade_date": "2026-03-18",
                        "etf_code": "0056",
                        "etf_name": "元大高股息",
                        "margin_buy": 139,
                        "margin_sell": 46,
                        "margin_cash_redemption": 57,
                        "margin_prev_balance": 3669,
                        "margin_balance": 3705,
                        "margin_next_limit": 3444008,
                        "short_buy": 0,
                        "short_sell": 12,
                        "short_stock_redemption": 0,
                        "short_prev_balance": 184,
                        "short_balance": 196,
                        "short_next_limit": 3444008,
                        "offset_amount": 0,
                        "note": "",
                        "source": "unit_test",
                    },
                ],
                trade_date="2026-03-18",
            )
            self.assertEqual(saved, 2)

            coverage = store.load_tw_etf_margin_daily_coverage()
            self.assertEqual(int(coverage["row_count"]), 2)
            self.assertEqual(int(coverage["trade_date_count"]), 1)
            self.assertEqual(int(coverage["symbol_count"]), 2)

            frame = store.load_tw_etf_margin_daily(start="2026-03-18", end="2026-03-18")
            self.assertEqual(len(frame), 2)
            self.assertEqual(
                int(frame.loc[frame["etf_code"] == "0050", "margin_balance"].iloc[0]),
                12630,
            )
            self.assertEqual(
                int(frame.loc[frame["etf_code"] == "0050", "short_balance"].iloc[0]),
                812,
            )

    def test_save_and_load_latest_heatmap_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            payload_old = {
                "rows": [{"symbol": "2330", "excess_pct": 1.2}],
                "generated_at": "2026-01-01T00:00:00+00:00",
            }
            payload_new = {
                "rows": [{"symbol": "2454", "excess_pct": 2.3}],
                "generated_at": "2026-01-02T00:00:00+00:00",
            }
            store.save_heatmap_run(universe_id="TW:00935", payload=payload_old)
            store.save_heatmap_run(universe_id="TW:00935", payload=payload_new)
            latest = store.load_latest_heatmap_run("TW:00935")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.universe_id, "TW:00935")
            self.assertEqual(latest.payload.get("generated_at"), "2026-01-02T00:00:00+00:00")

    def test_upsert_and_list_heatmap_hub_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            store.upsert_heatmap_hub_entry(
                etf_code="00935", etf_name="野村臺灣新科技50", opened=True
            )
            store.upsert_heatmap_hub_entry(
                etf_code="00935", etf_name="野村臺灣新科技50", opened=True, pin_as_card=True
            )
            store.upsert_heatmap_hub_entry(etf_code="0050", etf_name="元大台灣50", opened=False)

            all_rows = store.list_heatmap_hub_entries()
            self.assertEqual(len(all_rows), 2)
            top = all_rows[0]
            self.assertEqual(top.etf_code, "00935")
            self.assertTrue(top.pin_as_card)
            self.assertEqual(top.open_count, 2)

            pinned_rows = store.list_heatmap_hub_entries(pinned_only=True)
            self.assertEqual(len(pinned_rows), 1)
            self.assertEqual(pinned_rows[0].etf_code, "00935")

            ok = store.set_heatmap_hub_pin(etf_code="0050", pin_as_card=True)
            self.assertTrue(ok)
            pinned_rows = store.list_heatmap_hub_entries(pinned_only=True)
            self.assertEqual({row.etf_code for row in pinned_rows}, {"0050", "00935"})

    def test_save_and_load_latest_rotation_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            payload_old = {"generated_at": "2026-01-01T00:00:00+00:00", "top_n": 2}
            payload_new = {"generated_at": "2026-01-02T00:00:00+00:00", "top_n": 3}
            store.save_rotation_run(
                universe_id="TW:ROTATION:CORE6",
                run_key="run:old",
                params={"top_n": 2},
                payload=payload_old,
            )
            store.save_rotation_run(
                universe_id="TW:ROTATION:CORE6",
                run_key="run:new",
                params={"top_n": 3},
                payload=payload_new,
            )
            latest = store.load_latest_rotation_run("TW:ROTATION:CORE6")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.universe_id, "TW:ROTATION:CORE6")
            self.assertEqual(latest.run_key, "run:new")
            self.assertEqual(latest.params.get("top_n"), 3)
            self.assertEqual(latest.payload.get("generated_at"), "2026-01-02T00:00:00+00:00")

    def test_save_and_load_latest_backtest_replay_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            store.save_backtest_replay_run(
                run_key="bt:tw:0050",
                params={"strategy": "buy_hold"},
                payload={"generated_at": "2026-02-17T00:00:00+00:00", "mode": "single"},
            )
            store.save_backtest_replay_run(
                run_key="bt:tw:0050",
                params={"strategy": "sma_cross"},
                payload={"generated_at": "2026-02-18T00:00:00+00:00", "mode": "single"},
            )
            latest = store.load_latest_backtest_replay_run("bt:tw:0050")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.run_key, "bt:tw:0050")
            self.assertEqual(latest.params.get("strategy"), "sma_cross")
            self.assertEqual(latest.payload.get("generated_at"), "2026-02-18T00:00:00+00:00")

    def test_save_and_load_latest_tw_etf_super_export_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            first_run_id = store.save_tw_etf_super_export_run(
                ytd_start="20260101",
                ytd_end="20260309",
                compare_start="20250101",
                compare_end="20251231",
                trade_date_anchor="20260309",
                output_path="/tmp/tw_etf_super_export_20260309.csv",
                row_count=100,
                column_count=27,
                payload={"frame": {"columns": ["代碼"], "rows": [{"代碼": "0050"}]}},
            )
            second_run_id = store.save_tw_etf_super_export_run(
                ytd_start="20260101",
                ytd_end="20260310",
                compare_start="20250101",
                compare_end="20251231",
                trade_date_anchor="20260310",
                output_path="/tmp/tw_etf_super_export_20260310.csv",
                row_count=120,
                column_count=29,
                payload={"csv_sha256": "abc123", "frame": {"rows": [{"代碼": "0052"}]}},
            )

            self.assertNotEqual(first_run_id, second_run_id)
            latest = store.load_latest_tw_etf_super_export_run()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.run_id, second_run_id)
            self.assertEqual(latest.trade_date_anchor, "20260310")
            self.assertEqual(latest.output_path, "/tmp/tw_etf_super_export_20260310.csv")
            self.assertEqual(latest.row_count, 120)
            self.assertEqual(latest.column_count, 29)
            self.assertEqual(latest.payload.get("csv_sha256"), "abc123")

    def test_sync_history_backfills_when_start_before_local_first_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            service = _CaptureService()
            store = HistoryStore(db_path=db_path, service=service)
            first_start = datetime(2024, 1, 10, tzinfo=timezone.utc)
            end = datetime(2024, 1, 31, tzinfo=timezone.utc)

            store.sync_symbol_history(symbol="2330", market="TW", start=first_start, end=end)
            assert service.last_request is not None
            self.assertEqual(service.last_request.start.date().isoformat(), "2024-01-10")

            backfill_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            store.sync_symbol_history(symbol="2330", market="TW", start=backfill_start, end=end)
            assert service.last_request is not None
            self.assertEqual(service.last_request.start.date().isoformat(), "2024-01-01")

    def test_sync_history_prefers_fugle_for_tw_when_key_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            service = _CaptureService()
            service.tw_fugle_rest.api_key = "fake-key"
            store = HistoryStore(db_path=db_path, service=service)
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            end = datetime(2024, 1, 31, tzinfo=timezone.utc)

            report = store.sync_symbol_history(symbol="0050", market="TW", start=start, end=end)
            self.assertEqual(report.source, "tw_fugle_rest")
            self.assertEqual(service.last_provider_name, "tw_fugle_rest")

    def test_sync_history_tw_index_uses_yahoo_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            service = _CaptureService()
            service.tw_fugle_rest.api_key = "fake-key"
            store = HistoryStore(db_path=db_path, service=service)
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            end = datetime(2024, 1, 31, tzinfo=timezone.utc)

            report = store.sync_symbol_history(symbol="^TWII", market="TW", start=start, end=end)
            self.assertEqual(report.source, "yahoo")
            self.assertEqual(service.last_provider_name, "yahoo")
            assert service.last_request is not None
            self.assertEqual(service.last_request.symbol, "^TWII")

    def test_sync_history_normalizes_duplicate_ohlcv_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_DupColumnService())
            report = store.sync_symbol_history(symbol="^N225", market="US")
            self.assertIsNone(report.error)
            self.assertGreater(report.rows_upserted, 0)
            bars = store.load_daily_bars("^N225", "US")
            self.assertFalse(bars.empty)
            self.assertTrue({"open", "high", "low", "close", "volume"}.issubset(bars.columns))

    def test_symbol_metadata_upsert_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            written = store.upsert_symbol_metadata(
                [
                    {
                        "symbol": "2330",
                        "market": "TW",
                        "name": "台積電",
                        "exchange": "TW",
                        "industry": "24",
                    },
                    {
                        "symbol": "AAPL",
                        "market": "US",
                        "name": "Apple",
                        "exchange": "US",
                        "currency": "USD",
                    },
                ]
            )
            self.assertEqual(written, 2)

            tw_meta = store.load_symbol_metadata(["2330", "2454"], market="TW")
            self.assertEqual(tw_meta["2330"]["name"], "台積電")
            self.assertEqual(tw_meta["2330"]["industry"], "24")
            self.assertNotIn("2454", tw_meta)

            us_symbols = store.list_symbols("US")
            self.assertIn("AAPL", us_symbols)

    def test_bootstrap_run_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            run_id = store.start_bootstrap_run(
                scope="manual:both",
                params={"years": 5, "parallel": True},
            )
            store.finish_bootstrap_run(
                run_id,
                status="completed",
                total_symbols=10,
                synced_symbols=8,
                failed_symbols=2,
                summary={"scope": "both"},
                error=None,
            )
            latest = store.load_latest_bootstrap_run()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.run_id, run_id)
            self.assertEqual(latest.status, "completed")
            self.assertEqual(latest.total_symbols, 10)
            self.assertEqual(latest.synced_symbols, 8)
            self.assertEqual(latest.failed_symbols, 2)
            self.assertEqual(latest.summary.get("scope"), "both")

    def test_save_and_load_intraday_ticks_with_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.sqlite3"
            store = HistoryStore(db_path=db_path, service=_FakeService())
            now = datetime.now(tz=timezone.utc)
            old_ts = now - pd.Timedelta(days=5)
            rows = [
                {
                    "ts": old_ts.isoformat(),
                    "price": 100.0,
                    "cum_volume": 10.0,
                    "source": "fugle_ws",
                },
                {"ts": now.isoformat(), "price": 110.0, "cum_volume": 20.0, "source": "fugle_ws"},
            ]
            written = store.save_intraday_ticks("2330", "TW", rows, retain_days=2)
            self.assertEqual(written, 2)
            loaded = store.load_intraday_ticks(
                "2330",
                "TW",
                start=now - pd.Timedelta(days=30),
                end=now + pd.Timedelta(days=1),
            )
            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(float(loaded["price"].iloc[0]), 110.0)
            self.assertEqual(str(loaded["source"].iloc[0]), "fugle_ws")


if __name__ == "__main__":
    unittest.main()
