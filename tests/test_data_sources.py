from __future__ import annotations

from unittest.mock import MagicMock, patch
import unittest

import pandas as pd

from data_sources import fetch_yf_ohlcv


class DataSourcesTests(unittest.TestCase):
    @patch("yfinance.Ticker")
    def test_fetch_yf_ohlcv_uses_ticker_history_and_normalizes_columns(self, mock_ticker):
        idx = pd.date_range("2025-01-02", periods=3, freq="D", tz="UTC")
        frame = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0],
                "High": [11.0, 12.0, 13.0],
                "Low": [9.0, 10.0, 11.0],
                "Close": [10.5, 11.5, 12.5],
                "Volume": [100, 200, 300],
            },
            index=idx,
        )
        ticker = MagicMock()
        ticker.history.return_value = frame
        mock_ticker.return_value = ticker

        out = fetch_yf_ohlcv("2330.TW", period="10y", interval="1d")

        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(out), 3)
        self.assertIsNotNone(out.index.tz)
        ticker.history.assert_called_once_with(period="10y", interval="1d", auto_adjust=False)

    @patch("yfinance.Ticker")
    def test_fetch_yf_ohlcv_returns_empty_when_history_fails(self, mock_ticker):
        ticker = MagicMock()
        ticker.history.side_effect = RuntimeError("rate limit")
        mock_ticker.return_value = ticker

        out = fetch_yf_ohlcv("6913.TW", period="10y", interval="1d")

        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume"])


if __name__ == "__main__":
    unittest.main()
