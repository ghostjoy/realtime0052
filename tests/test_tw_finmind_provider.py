from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from providers.base import ProviderErrorKind
from providers.tw_finmind import TwFinMindClient


class TwFinMindClientTests(unittest.TestCase):
    def test_resolve_api_key_from_default_key_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = os.path.join(tmp, "finmindkey")
            with open(key_file, "w", encoding="utf-8") as fh:
                fh.write(" finmind-file-key \n")
            with patch.object(TwFinMindClient, "default_key_file", key_file):
                client = TwFinMindClient(api_key=None)
            self.assertEqual(client.api_key, "finmind-file-key")

    @patch("requests.get")
    def test_fetch_month_revenue_uses_authorization_header(self, mock_get):
        resp = unittest.mock.MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": 200,
            "msg": "success",
            "data": [{"date": "2026-02-01", "stock_id": "2330", "revenue": 123456789}],
        }
        mock_get.return_value = resp

        client = TwFinMindClient(api_key="finmind-token")
        rows = client.fetch_month_revenue("2330", start_date="2025-01-01")

        self.assertEqual(len(rows), 1)
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer finmind-token")
        self.assertEqual(kwargs["params"]["dataset"], "TaiwanStockMonthRevenue")
        self.assertEqual(kwargs["params"]["data_id"], "2330")

    @patch("requests.get")
    def test_rate_limit_maps_to_provider_error(self, mock_get):
        resp = unittest.mock.MagicMock()
        resp.status_code = 429
        resp.json.return_value = {"status": 429, "msg": "too many requests", "data": []}
        mock_get.return_value = resp

        client = TwFinMindClient(api_key="finmind-token")
        with self.assertRaisesRegex(Exception, "too many requests") as ctx:
            client.fetch_stock_news("2330", start_date="2026-03-14")
        self.assertEqual(ctx.exception.kind, ProviderErrorKind.RATE_LIMIT)
