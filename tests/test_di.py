from __future__ import annotations

import unittest
from unittest.mock import patch

import di
from services.market_data_service import MarketDataService
from storage.history_store import HistoryStore
from ui.loaders import BarDataLoader, get_registry


class DiTests(unittest.TestCase):
    def test_register_all_services_wires_market_and_store(self):
        fake_service = object()
        fake_store = object()

        with patch("di.create_market_service", return_value=fake_service) as service_factory:
            with patch("di.create_history_store", return_value=fake_store) as store_factory:
                di.register_all_services(force=True)
                market = di.get_market_service()
                store = di.get_history_store()

        self.assertIs(market, fake_service)
        self.assertIs(store, fake_store)
        service_factory.assert_called_once_with()
        store_factory.assert_called_once_with(service=fake_service)

    def test_loader_registry_is_bootstrapped(self):
        di.register_all_services(force=True)
        registry = get_registry()
        self.assertTrue(registry.get_enabled_sources(BarDataLoader))

    def test_container_resolves_expected_types(self):
        di.register_all_services(force=True)
        market = di.get_container().resolve(MarketDataService)
        store = di.get_container().resolve(HistoryStore)
        self.assertIsNotNone(market)
        self.assertIsNotNone(store)


if __name__ == "__main__":
    unittest.main()
