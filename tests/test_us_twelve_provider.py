from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from providers.us_twelve import UsTwelveDataProvider


class UsTwelveProviderTests(unittest.TestCase):
    def test_api_key_from_key_file_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "twelvedatakey"
            key_file.write_text("  td-file-key-123  \n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "TWELVE_DATA_API_KEY_FILE": str(key_file),
                    "TWELVE_DATA_API_KEY": "",
                },
                clear=True,
            ):
                provider = UsTwelveDataProvider(api_key=None)
        self.assertEqual(provider.api_key, "td-file-key-123")

    def test_api_key_from_default_key_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "twelvedatakey"
            key_file.write_text("default-td-key", encoding="utf-8")
            with patch.object(UsTwelveDataProvider, "default_key_file", key_file):
                with patch.dict(os.environ, {}, clear=True):
                    provider = UsTwelveDataProvider(api_key=None)
        self.assertEqual(provider.api_key, "default-td-key")


if __name__ == "__main__":
    unittest.main()
