"""Dependency injection helpers for service/store construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from config_loader import cfg_or_env, cfg_or_env_bool, cfg_or_env_str
from services.market_data_service import MarketDataService
from storage.history_store import HistoryStore
from ui.loaders import (
    BarDataLoader,
    DataSourceConfig,
    MetadataLoader,
    QuoteDataLoader,
    get_registry,
)

T = TypeVar("T")


class Container:
    """Simple dependency injection container."""

    def __init__(self) -> None:
        self._services: dict[type, Any] = {}
        self._factories: dict[type, Callable[[], Any]] = {}

    def register_singleton(self, service_type: type[T], instance: T) -> None:
        self._services[service_type] = instance

    def register_factory(self, service_type: type[T], factory: Callable[[], T]) -> None:
        self._factories[service_type] = factory

    def resolve(self, service_type: type[T]) -> T:
        if service_type in self._services:
            return self._services[service_type]
        if service_type in self._factories:
            instance = self._factories[service_type]()
            self._services[service_type] = instance
            return instance
        raise KeyError(f"No service registered for type: {service_type}")

    def clear(self) -> None:
        self._services.clear()
        self._factories.clear()


_container = Container()
_bootstrapped = False
_registry_bootstrapped = False


def get_container() -> Container:
    return _container


def create_market_service() -> MarketDataService:
    return MarketDataService()


def create_history_store(*, service: MarketDataService | None = None) -> HistoryStore:
    svc = service or create_market_service()
    backend = (
        cfg_or_env_str("features.storage_backend", "REALTIME0052_STORAGE_BACKEND", "duckdb")
        .strip()
        .lower()
    )
    if backend not in {"duckdb", "sqlite"}:
        backend = "duckdb"

    if backend == "duckdb":
        from storage.duck_store import DuckHistoryStore

        duck_db_path = cfg_or_env(
            "storage.duckdb.db_path", "REALTIME0052_DUCKDB_PATH", default=None
        )
        parquet_root = cfg_or_env(
            "storage.duckdb.parquet_root", "REALTIME0052_PARQUET_ROOT", default=None
        )
        retain_days = cfg_or_env(
            "storage.duckdb.intraday_retain_days",
            "REALTIME0052_INTRADAY_RETAIN_DAYS",
            default=1095,
            cast=int,
        )
        legacy_sqlite_path = cfg_or_env(
            "storage.duckdb.legacy_sqlite_path", "REALTIME0052_DB_PATH", default=None
        )
        auto_migrate = cfg_or_env_bool(
            "storage.duckdb.auto_migrate_legacy_sqlite",
            "REALTIME0052_AUTO_MIGRATE_LEGACY_SQLITE",
            default=True,
        )
        return DuckHistoryStore(
            service=svc,
            db_path=duck_db_path,
            parquet_root=parquet_root,
            intraday_retain_days=int(retain_days),
            legacy_sqlite_path=legacy_sqlite_path,
            auto_migrate_legacy_sqlite=bool(auto_migrate),
        )

    sqlite_db_path = cfg_or_env("storage.sqlite.db_path", "REALTIME0052_DB_PATH", default=None)
    return HistoryStore(db_path=sqlite_db_path, service=svc)


def register_market_service() -> None:
    _container.register_factory(MarketDataService, create_market_service)


def register_history_store() -> None:
    def _factory() -> HistoryStore:
        service = _container.resolve(MarketDataService)
        return create_history_store(service=service)

    _container.register_factory(HistoryStore, _factory)


def register_all_services(*, force: bool = False) -> None:
    global _bootstrapped, _registry_bootstrapped
    if _bootstrapped and not force:
        return
    _container.clear()
    register_market_service()
    register_history_store()
    if not _registry_bootstrapped:
        registry = get_registry()
        registry.register(
            BarDataLoader,
            DataSourceConfig(name="history_store", priority=0, enabled=True, cache_ttl=3600),
        )
        registry.register(
            QuoteDataLoader,
            DataSourceConfig(name="market_data_service", priority=0, enabled=True, cache_ttl=120),
        )
        registry.register(
            MetadataLoader,
            DataSourceConfig(name="market_data_service", priority=0, enabled=True, cache_ttl=21600),
        )
        _registry_bootstrapped = True
    _bootstrapped = True


def get_market_service() -> MarketDataService:
    register_all_services()
    return _container.resolve(MarketDataService)


def get_history_store() -> HistoryStore:
    register_all_services()
    return _container.resolve(HistoryStore)


__all__ = [
    "Container",
    "create_history_store",
    "create_market_service",
    "get_container",
    "get_history_store",
    "get_market_service",
    "register_all_services",
    "register_history_store",
    "register_market_service",
]
