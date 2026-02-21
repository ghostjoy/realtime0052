from __future__ import annotations

import os
import unittest

import config_loader


class ConfigLoaderTests(unittest.TestCase):
    def setUp(self):
        self._old_env = dict(os.environ)
        config_loader._load_hydra_config.cache_clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)
        config_loader._load_hydra_config.cache_clear()

    def test_default_source_is_hydra(self):
        os.environ.pop("REALTIME0052_CONFIG_SOURCE", None)
        self.assertEqual(config_loader.get_config_source(), "hydra")
        self.assertEqual(config_loader.cfg_get("features.storage_backend", "sqlite"), "duckdb")
        os.environ["REALTIME0052_SYNC_YEARS"] = "7"
        years = config_loader.cfg_or_env("sync.years", "REALTIME0052_SYNC_YEARS", default=5, cast=int)
        self.assertEqual(years, 7)

    def test_hydra_source_via_env(self):
        os.environ["REALTIME0052_CONFIG_SOURCE"] = "hydra"
        self.assertEqual(config_loader.get_config_source(), "hydra")
        self.assertEqual(config_loader.cfg_get("features.storage_backend", "sqlite"), "duckdb")
        self.assertEqual(config_loader.cfg_get("features.kline_renderer_replay", "plotly"), "plotly")

    def test_env_overrides_hydra_value(self):
        os.environ["REALTIME0052_CONFIG_SOURCE"] = "hydra"
        os.environ["REALTIME0052_STORAGE_BACKEND"] = "duckdb"
        self.assertEqual(
            config_loader.cfg_or_env_str("features.storage_backend", "REALTIME0052_STORAGE_BACKEND", "duckdb"),
            "duckdb",
        )


if __name__ == "__main__":
    unittest.main()
