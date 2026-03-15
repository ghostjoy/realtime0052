from __future__ import annotations

import os
import unittest
from unittest.mock import patch

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
        years = config_loader.cfg_or_env(
            "sync.years", "REALTIME0052_SYNC_YEARS", default=5, cast=int
        )
        self.assertEqual(years, 7)

    def test_hydra_source_via_env(self):
        os.environ["REALTIME0052_CONFIG_SOURCE"] = "hydra"
        self.assertEqual(config_loader.get_config_source(), "hydra")
        self.assertEqual(config_loader.cfg_get("features.storage_backend", "sqlite"), "duckdb")
        self.assertEqual(
            config_loader.cfg_get("features.kline_renderer_replay", "plotly"), "plotly"
        )

    def test_env_overrides_hydra_value(self):
        os.environ["REALTIME0052_CONFIG_SOURCE"] = "hydra"
        os.environ["REALTIME0052_STORAGE_BACKEND"] = "duckdb"
        self.assertEqual(
            config_loader.cfg_or_env_str(
                "features.storage_backend", "REALTIME0052_STORAGE_BACKEND", "duckdb"
            ),
            "duckdb",
        )

    def test_reuses_initialized_global_hydra_without_reinitializing(self):
        fake_cfg = object()
        fake_instance = type("FakeHydraInstance", (), {"is_initialized": lambda self: True})()
        with (
            patch.object(config_loader, "compose", return_value=fake_cfg) as compose_mock,
            patch.object(config_loader, "initialize_config_dir") as init_mock,
            patch.object(
                config_loader.GlobalHydra,
                "instance",
                return_value=fake_instance,
            ),
        ):
            config_loader._load_hydra_config.cache_clear()
            cfg = config_loader._load_hydra_config()

        self.assertIs(cfg, fake_cfg)
        compose_mock.assert_called_once_with(config_name="config")
        init_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
