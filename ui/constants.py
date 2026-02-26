"""UI and caching constants for the application."""

# Cache TTL values (in seconds)
CACHE_TTL_LONG = 21600  # 6 hours - for static data like ETF lists
CACHE_TTL_MEDIUM = 3600  # 1 hour - for daily data like AUM
CACHE_TTL_SHORT = 600  # 10 minutes - for frequently updated data
CACHE_TTL_REALTIME = 120  # 2 minutes - for near-realtime data
CACHE_TTL_HOUR = 3600  # 1 hour alias
CACHE_TTL_DAY = 86400  # 24 hours alias

# UI refresh intervals (in seconds)
UI_REFRESH_LIVE = 30  # Live data refresh
UI_REFRESH_SNAPSHOT = 300  # Snapshot refresh (5 min)
UI_REFRESH_DAILY = 3600  # Daily data refresh (1 hour)

# Thresholds
MIN_BARS_FOR_BACKTEST = 2  # Minimum bars for buy_hold
RECOMMENDED_BARS = 40  # Recommended bars for strategies
WALKFORWARD_MIN_BARS = 80  # Minimum bars for walk-forward

# Number formatting
PRICE_DECIMAL_PLACES = 2
PERCENTAGE_DECIMAL_PLACES = 2
VOLUME_SCALE = 1000  # Volume in thousands

# Chart dimensions
DEFAULT_CHART_HEIGHT = 500
COMPACT_CHART_HEIGHT = 340
INDICATOR_CHART_HEIGHT = 430

# Table dimensions
TABLE_MIN_HEIGHT = 220
TABLE_MAX_HEIGHT = 3200
TABLE_DEFAULT_HEIGHT = 600

# Heatmap settings
HEATMAP_MIN_ABS = 0.15  # Minimum value for heatmap color scaling

# ETF Universe
TW_ETF_ROTATION_UNIVERSE = ["0050", "0052", "00935", "0056", "00878", "00919"]

# Provider fallback order
TW_REALTIME_PROVIDER_CHAIN = ["Fugle WS", "TW MIS", "TW OpenAPI", "TPEx OpenAPI"]
TW_DAILY_PROVIDER_CHAIN = ["Fugle Historical", "TW OpenAPI", "TPEx OpenAPI", "Yahoo"]
US_REALTIME_PROVIDER_CHAIN = ["Twelve Data", "Yahoo", "Stooq"]

# Benchmark defaults
DEFAULT_TW_BENCHMARK = "^TWII"
DEFAULT_US_BENCHMARK = "^GSPC"
FALLBACK_TW_BENCHMARKS = ["0050", "006208"]
FALLBACK_US_BENCHMARKS = ["SPY", "QQQ", "DIA"]

__all__ = [
    # Cache TTL
    "CACHE_TTL_LONG",
    "CACHE_TTL_MEDIUM",
    "CACHE_TTL_SHORT",
    "CACHE_TTL_REALTIME",
    "CACHE_TTL_HOUR",
    "CACHE_TTL_DAY",
    # UI refresh
    "UI_REFRESH_LIVE",
    "UI_REFRESH_SNAPSHOT",
    "UI_REFRESH_DAILY",
    # Thresholds
    "MIN_BARS_FOR_BACKTEST",
    "RECOMMENDED_BARS",
    "WALKFORWARD_MIN_BARS",
    # Formatting
    "PRICE_DECIMAL_PLACES",
    "PERCENTAGE_DECIMAL_PLACES",
    "VOLUME_SCALE",
    # Chart/Table
    "DEFAULT_CHART_HEIGHT",
    "COMPACT_CHART_HEIGHT",
    "INDICATOR_CHART_HEIGHT",
    "TABLE_MIN_HEIGHT",
    "TABLE_MAX_HEIGHT",
    "TABLE_DEFAULT_HEIGHT",
    "HEATMAP_MIN_ABS",
    # ETF
    "TW_ETF_ROTATION_UNIVERSE",
    # Provider chains
    "TW_REALTIME_PROVIDER_CHAIN",
    "TW_DAILY_PROVIDER_CHAIN",
    "US_REALTIME_PROVIDER_CHAIN",
    # Benchmarks
    "DEFAULT_TW_BENCHMARK",
    "DEFAULT_US_BENCHMARK",
    "FALLBACK_TW_BENCHMARKS",
    "FALLBACK_US_BENCHMARKS",
]
