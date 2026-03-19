from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from services.tw_etf_margin_sync import (
    fetch_twse_etf_margin_report,
    sync_twse_etf_margin_daily,
)


class _FakeResponse:
    def __init__(self, payload: object):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeStore:
    def __init__(self):
        self.saved: dict[str, list[dict[str, object]]] = {}

    def load_tw_etf_margin_daily_coverage(self, start=None, end=None, etf_codes=None):
        if start is not None and end is not None:
            token = pd.Timestamp(start).date().isoformat()
            rows = self.saved.get(token, [])
            return {
                "row_count": len(rows),
                "first_date": pd.Timestamp(token).to_pydatetime() if rows else None,
                "last_date": pd.Timestamp(token).to_pydatetime() if rows else None,
                "trade_date_count": 1 if rows else 0,
                "symbol_count": len(rows),
            }
        dates = sorted(self.saved)
        if not dates:
            return {
                "row_count": 0,
                "first_date": None,
                "last_date": None,
                "trade_date_count": 0,
                "symbol_count": 0,
            }
        all_rows = [row for items in self.saved.values() for row in items]
        return {
            "row_count": len(all_rows),
            "first_date": pd.Timestamp(dates[0]).to_pydatetime(),
            "last_date": pd.Timestamp(dates[-1]).to_pydatetime(),
            "trade_date_count": len(dates),
            "symbol_count": len({str(row.get("etf_code")) for row in all_rows}),
        }

    def save_tw_etf_margin_daily(self, *, rows, trade_date, source="twse_margin_mi_margn"):
        self.saved[str(trade_date)] = list(rows)
        return len(rows)


class TwEtfMarginSyncTests(unittest.TestCase):
    @patch("services.tw_etf_margin_sync.requests.get")
    def test_fetch_twse_etf_margin_report_parses_etf_rows(self, get_mock):
        get_mock.return_value = _FakeResponse(
            {
                "stat": "OK",
                "date": "20260318",
                "tables": [
                    {
                        "title": "信用交易統計",
                        "fields": ["項目", "買進"],
                        "data": [["融資(交易單位)", "123"]],
                    },
                    {
                        "title": "融資融券彙總 (全部)",
                        "fields": [
                            "代號",
                            "名稱",
                            "買進",
                            "賣出",
                            "現金償還",
                            "前日餘額",
                            "今日餘額",
                            "次一營業日限額",
                            "買進",
                            "賣出",
                            "現券償還",
                            "前日餘額",
                            "今日餘額",
                            "次一營業日限額",
                            "資券互抵",
                            "註記",
                        ],
                        "data": [
                            [
                                "2330",
                                "台積電",
                                "1",
                                "2",
                                "3",
                                "4",
                                "5",
                                "6",
                                "7",
                                "8",
                                "9",
                                "10",
                                "11",
                                "12",
                                "13",
                                "",
                            ],
                            [
                                "0050",
                                "元大台灣50",
                                "826",
                                "945",
                                "4",
                                "12,753",
                                "12,630",
                                "4,480,625",
                                "22",
                                "89",
                                "0",
                                "745",
                                "812",
                                "4,480,625",
                                "21",
                                "X ",
                            ],
                        ],
                    },
                ],
            }
        )

        trade_date, frame, meta = fetch_twse_etf_margin_report("2026-03-18")

        self.assertEqual(trade_date, "2026-03-18")
        self.assertEqual(len(frame), 1)
        self.assertEqual(str(frame.iloc[0]["etf_code"]), "0050")
        self.assertEqual(int(frame.iloc[0]["margin_balance"]), 12630)
        self.assertEqual(int(frame.iloc[0]["short_balance"]), 812)
        self.assertEqual(str(frame.iloc[0]["note"]), "X")
        self.assertEqual(int(meta["row_count"]), 1)

    @patch("services.tw_etf_margin_sync.requests.get")
    def test_fetch_twse_etf_margin_report_returns_empty_frame_for_no_data_stat(self, get_mock):
        get_mock.return_value = _FakeResponse(
            {
                "stat": "很抱歉，沒有符合條件的資料",
                "date": "20260319",
                "tables": None,
            }
        )

        trade_date, frame, meta = fetch_twse_etf_margin_report("2026-03-19")

        self.assertEqual(trade_date, "2026-03-19")
        self.assertTrue(frame.empty)
        self.assertEqual(str(meta.get("stat") or ""), "很抱歉，沒有符合條件的資料")

    @patch("services.tw_etf_margin_sync.fetch_twse_etf_margin_report")
    def test_sync_twse_etf_margin_daily_skips_existing_dates(self, fetch_mock):
        store = _FakeStore()
        store.save_tw_etf_margin_daily(
            rows=[
                {
                    "trade_date": "2026-03-18",
                    "etf_code": "0050",
                    "etf_name": "元大台灣50",
                    "margin_buy": 1,
                    "margin_sell": 2,
                    "margin_cash_redemption": 0,
                    "margin_prev_balance": 10,
                    "margin_balance": 9,
                    "margin_next_limit": 100,
                    "short_buy": 0,
                    "short_sell": 1,
                    "short_stock_redemption": 0,
                    "short_prev_balance": 2,
                    "short_balance": 3,
                    "short_next_limit": 100,
                    "offset_amount": 0,
                    "note": "",
                    "source": "unit",
                }
            ],
            trade_date="2026-03-18",
        )
        fetch_mock.return_value = (
            "2026-03-19",
            pd.DataFrame(
                [
                    {
                        "trade_date": "2026-03-19",
                        "etf_code": "0052",
                        "etf_name": "富邦科技",
                        "margin_buy": 3,
                        "margin_sell": 4,
                        "margin_cash_redemption": 0,
                        "margin_prev_balance": 12,
                        "margin_balance": 11,
                        "margin_next_limit": 200,
                        "short_buy": 1,
                        "short_sell": 2,
                        "short_stock_redemption": 0,
                        "short_prev_balance": 5,
                        "short_balance": 6,
                        "short_next_limit": 200,
                        "offset_amount": 1,
                        "note": "",
                        "source": "twse_margin_mi_margn",
                    }
                ]
            ),
            {"row_count": 1},
        )

        summary = sync_twse_etf_margin_daily(
            store=store,
            start="2026-03-18",
            end="2026-03-19",
        )

        self.assertEqual(int(summary["skipped_days"]), 1)
        self.assertEqual(int(summary["synced_days"]), 1)
        self.assertEqual(int(summary["saved_rows"]), 1)
        self.assertIn("2026-03-19", store.saved)


if __name__ == "__main__":
    unittest.main()
