"""Unified data loader interfaces for the application.

This module defines protocols and interfaces for data loading,
ensuring consistent patterns across the application.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BarDataLoader(Protocol):
    """Protocol for bar data loaders (daily/intraday)."""

    def load_bars(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Load OHLCV bars for a symbol.

        Args:
            symbol: Stock/ETF symbol code.
            start: Start date.
            end: End date.

        Returns:
            DataFrame with OHLCV columns.
        """
        ...


@runtime_checkable
class QuoteDataLoader(Protocol):
    """Protocol for quote data loaders."""

    def load_quote(self, symbol: str) -> dict:
        """Load current quote for a symbol.

        Args:
            symbol: Stock/ETF symbol code.

        Returns:
            Dict with quote data (price, volume, etc.).
        """
        ...


@runtime_checkable
class MetadataLoader(Protocol):
    """Protocol for metadata loaders (names, industries, etc.)."""

    def load_symbol_names(self, symbols: list[str]) -> dict[str, str]:
        """Load symbol to name mapping.

        Args:
            symbols: List of symbol codes.

        Returns:
            Dict mapping symbol -> name.
        """
        ...

    def load_symbol_industries(self, symbols: list[str]) -> dict[str, str]:
        """Load symbol to industry mapping.

        Args:
            symbols: List of symbol codes.

        Returns:
            Dict mapping symbol -> industry code.
        """
        ...


@runtime_checkable
class ConstituentsLoader(Protocol):
    """Protocol for ETF constituents loaders."""

    def load_constituents(
        self,
        etf_code: str,
        *,
        force_refresh: bool = False,
    ) -> list[dict]:
        """Load ETF constituents.

        Args:
            etf_code: ETF symbol code.
            force_refresh: Force refresh from source.

        Returns:
            List of constituent records.
        """
        ...


class DataSourceConfig:
    """Configuration for a data source."""

    def __init__(
        self,
        name: str,
        priority: int = 0,
        enabled: bool = True,
        cache_ttl: int = 3600,
    ):
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self.cache_ttl = cache_ttl


class LoaderRegistry:
    """Registry for data loaders with fallback support."""

    def __init__(self):
        self._loaders: dict[type, list[DataSourceConfig]] = {}

    def register(
        self,
        loader_type: type,
        config: DataSourceConfig,
    ) -> None:
        """Register a data source for a loader type."""
        if loader_type not in self._loaders:
            self._loaders[loader_type] = []
        self._loaders[loader_type].append(config)
        self._loaders[loader_type].sort(key=lambda x: x.priority)

    def get_enabled_sources(self, loader_type: type) -> list[DataSourceConfig]:
        """Get enabled sources for a loader type, sorted by priority."""
        sources = self._loaders.get(loader_type, [])
        return [s for s in sources if s.enabled]


# Global registry instance
_registry = LoaderRegistry()


def get_registry() -> LoaderRegistry:
    """Get the global loader registry."""
    return _registry


__all__ = [
    "BarDataLoader",
    "QuoteDataLoader",
    "MetadataLoader",
    "ConstituentsLoader",
    "DataSourceConfig",
    "LoaderRegistry",
    "get_registry",
]
