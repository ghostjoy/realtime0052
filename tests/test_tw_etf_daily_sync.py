from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from services.tw_etf_daily_sync import fetch_twse_etf_daily_report, sync_twse_etf_daily_market


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

    def load_tw_etf_daily_market_coverage(self, start=None, end=None, etf_codes=None):
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

    def save_tw_etf_daily_market(self, *, rows, trade_date, source="twse_etf_daily"):
        self.saved[str(trade_date)] = list(rows)
        return len(rows)


class TwEtfDailySyncTests(unittest.TestCase):
    @patch("services.tw_etf_daily_sync.requests.get")
    def test_fetch_twse_etf_daily_report_parses_rows(self, get_mock):
        get_mock.return_value = _FakeResponse(
            {
                "stat": "OK",
                "date": "20260306",
                "fields": [
                    "證券代號",
                    "證券名稱",
                    "成交金額",
                    "成交股數",
                    "成交筆數",
                    "開盤價",
                    "最高價",
                    "最低價",
                    "收盤價",
                    "漲跌價差",
                ],
                "data": [
                    [
                        "0050",
                        "元大台灣50",
                        "11,638,160,681",
                        "151,611,764",
                        "179,651",
                        "76.80",
                        "77.30",
                        "76.30",
                        "76.85",
                        "-0.55",
                    ]
                ],
            }
        )

        trade_date, frame, meta = fetch_twse_etf_daily_report("2026-03-06")

        self.assertEqual(trade_date, "2026-03-06")
        self.assertFalse(frame.empty)
        self.assertEqual(str(frame.iloc[0]["etf_code"]), "0050")
        self.assertEqual(float(frame.iloc[0]["trade_value"]), 11638160681.0)
        self.assertEqual(int(frame.iloc[0]["trade_count"]), 179651)
        self.assertEqual(meta["row_count"], 1)

    @patch("services.tw_etf_daily_sync.fetch_twse_etf_daily_report")
    def test_sync_twse_etf_daily_market_skips_existing_dates(self, fetch_mock):
        store = _FakeStore()
        store.save_tw_etf_daily_market(
            rows=[
                {
                    "trade_date": "2026-03-06",
                    "etf_code": "0050",
                    "etf_name": "元大台灣50",
                    "trade_value": 1.0,
                    "trade_volume": 1.0,
                    "trade_count": 1,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "change": 0.0,
                    "source": "unit",
                }
            ],
            trade_date="2026-03-06",
        )
        fetch_mock.return_value = (
            "2026-03-07",
            pd.DataFrame(
                [
                    {
                        "trade_date": "2026-03-07",
                        "etf_code": "0052",
                        "etf_name": "富邦科技",
                        "trade_value": 2.0,
                        "trade_volume": 3.0,
                        "trade_count": 4,
                        "open": 5.0,
                        "high": 6.0,
                        "low": 4.0,
                        "close": 5.5,
                        "change": 0.1,
                        "source": "twse_etf_daily",
                    }
                ]
            ),
            {"row_count": 1},
        )

        summary = sync_twse_etf_daily_market(
            store=store,
            start="2026-03-06",
            end="2026-03-07",
        )

        self.assertEqual(int(summary["skipped_days"]), 1)
        self.assertEqual(int(summary["synced_days"]), 1)
        self.assertEqual(int(summary["saved_rows"]), 1)
        self.assertIn("2026-03-07", store.saved)


if __name__ == "__main__":
    unittest.main()
