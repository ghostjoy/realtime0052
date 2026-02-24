#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services import MarketDataService
from services.bootstrap_loader import run_market_data_bootstrap
from storage import HistoryStore


def _parse_symbols_csv(raw: str) -> list[str]:
    parts = [token.strip().upper() for token in str(raw or "").split(",")]
    return [token for token in parts if token]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prebuild TW/US market baseline into SQLite")
    parser.add_argument(
        "--scope", choices=["both", "tw", "us"], default="both", help="Bootstrap market scope"
    )
    parser.add_argument(
        "--years", type=int, default=5, help="Lookback years to backfill daily bars"
    )
    parser.add_argument("--max-workers", type=int, default=6, help="Parallel worker count")
    parser.add_argument("--serial", action="store_true", help="Disable parallel sync")
    parser.add_argument(
        "--tw-limit", type=int, default=None, help="Limit TW symbols for trial runs"
    )
    parser.add_argument(
        "--us-symbols",
        type=str,
        default="",
        help="Comma-separated custom US symbols (empty means default core list)",
    )
    parser.add_argument(
        "--sync-mode",
        choices=["min_rows", "backfill", "all"],
        default="min_rows",
        help="Sync mode for orchestrator",
    )
    parser.add_argument(
        "--min-rows", type=int, default=None, help="Required rows when sync-mode=min_rows"
    )
    args = parser.parse_args()

    service = MarketDataService()
    store = HistoryStore(service=service)

    us_symbols = _parse_symbols_csv(args.us_symbols)
    started = datetime.now().isoformat()
    result = run_market_data_bootstrap(
        store=store,
        scope=args.scope,
        years=max(1, int(args.years)),
        parallel=not bool(args.serial),
        max_workers=max(1, int(args.max_workers)),
        tw_limit=args.tw_limit,
        us_symbols=us_symbols or None,
        sync_mode=args.sync_mode,
        min_rows=args.min_rows,
    )
    finished = datetime.now().isoformat()

    print("[bootstrap] started=", started)
    print("[bootstrap] finished=", finished)
    print("[bootstrap] summary=")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
