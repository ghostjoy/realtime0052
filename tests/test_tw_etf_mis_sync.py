from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from services.tw_etf_mis_sync import fetch_twse_etf_mis_report, sync_twse_etf_mis_daily


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

    def load_tw_etf_mis_daily_coverage(self, start=None, end=None, etf_codes=None):
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

    def save_tw_etf_mis_daily(self, *, rows, trade_date, source="twse_mis_etf_indicator"):
        self.saved[str(trade_date)] = list(rows)
        return len(rows)


class TwEtfMisSyncTests(unittest.TestCase):
    @patch("services.tw_etf_mis_sync.requests.get")
    def test_fetch_twse_etf_mis_report_parses_rows(self, get_mock):
        get_mock.side_effect = [
            _FakeResponse(
                {
                    "msgArray": [
                        {
                            "ex": "tse",
                            "ch": "0050.tw",
                            "nf": "元大台灣50",
                            "n": "元大台灣50",
                        }
                    ]
                }
            ),
            _FakeResponse(
                {
                    "a1": [
                        {
                            "msgArray": [
                                {
                                    "a": "0050",
                                    "b": "元大台灣50",
                                    "c": "1,350,000,000",
                                    "d": "1000000",
                                    "e": "182.10",
                                    "f": "181.62",
                                    "g": "0.26",
                                    "h": "180.55",
                                    "i": "20260306",
                                    "j": "14:30:00",
                                    "k": "1",
                                }
                            ],
                            "refURL": "https://example.com/nav",
                        }
                    ]
                }
            ),
        ]

        trade_date, frame, meta = fetch_twse_etf_mis_report()

        self.assertEqual(trade_date, "2026-03-06")
        self.assertFalse(frame.empty)
        self.assertEqual(str(frame.iloc[0]["etf_code"]), "0050")
        self.assertEqual(float(frame.iloc[0]["estimated_nav"]), 181.62)
        self.assertEqual(float(frame.iloc[0]["premium_discount_pct"]), 0.26)
        self.assertEqual(str(frame.iloc[0]["reference_url"]), "https://example.com/nav")
        self.assertEqual(meta["row_count"], 1)

    @patch("services.tw_etf_mis_sync.fetch_twse_etf_mis_report")
    def test_sync_twse_etf_mis_daily_skips_existing_latest_date(self, fetch_mock):
        store = _FakeStore()
        store.save_tw_etf_mis_daily(
            rows=[
                {
                    "trade_date": "2026-03-06",
                    "etf_code": "0050",
                    "etf_name": "元大台灣50",
                    "issued_units": 1.0,
                    "creation_redemption_diff": 0.0,
                    "market_price": 180.0,
                    "estimated_nav": 179.5,
                    "premium_discount_pct": 0.28,
                    "previous_nav": 179.0,
                    "reference_url": "unit",
                    "updated_at": "2026-03-06T14:30:00+08:00",
                    "source": "unit",
                }
            ],
            trade_date="2026-03-06",
        )
        fetch_mock.return_value = (
            "2026-03-06",
            pd.DataFrame(
                [
                    {
                        "trade_date": "2026-03-06",
                        "etf_code": "0052",
                        "etf_name": "富邦科技",
                        "issued_units": 2.0,
                        "creation_redemption_diff": 1.0,
                        "market_price": 100.0,
                        "estimated_nav": 99.5,
                        "premium_discount_pct": 0.5,
                        "previous_nav": 99.0,
                        "reference_url": "unit",
                        "updated_at": "2026-03-06T14:30:00+08:00",
                        "source": "twse_mis_etf_indicator",
                    }
                ]
            ),
            {"row_count": 1},
        )

        summary = sync_twse_etf_mis_daily(store=store)

        self.assertEqual(int(summary["skipped_days"]), 1)
        self.assertEqual(int(summary["synced_days"]), 0)
        self.assertEqual(int(summary["saved_rows"]), 0)
        self.assertEqual(list(store.saved), ["2026-03-06"])


if __name__ == "__main__":
    unittest.main()
