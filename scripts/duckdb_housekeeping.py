#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.duck_store import DuckHistoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DuckDB market_snapshots housekeeping (purge + optional checkpoint/vacuum)"
    )
    parser.add_argument("--db-path", default="", help="DuckDB file path (optional)")
    parser.add_argument("--parquet-root", default="", help="Parquet root path (optional)")
    parser.add_argument(
        "--default-keep-days",
        type=int,
        default=30,
        help="Default retention days for datasets not listed in --keep (default: 30)",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Retention override, format: dataset_key=days (can pass multiple times)",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Skip market_snapshots purge and run only maintenance actions",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Run DuckDB CHECKPOINT after purge",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run DuckDB VACUUM after purge (implies checkpoint)",
    )
    return parser


def _parse_keep_overrides(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        token = str(item or "").strip()
        if not token or "=" not in token:
            raise SystemExit(f"Invalid --keep format: {item} (expected dataset_key=days)")
        key, raw_days = token.split("=", 1)
        dataset_key = str(key or "").strip()
        if not dataset_key:
            raise SystemExit(f"Invalid --keep dataset key: {item}")
        try:
            days = max(1, int(raw_days))
        except Exception as exc:
            raise SystemExit(f"Invalid --keep days: {item}") from exc
        out[dataset_key] = days
    return out


def main() -> int:
    args = build_parser().parse_args()
    keep_overrides = _parse_keep_overrides(list(args.keep or []))

    store = DuckHistoryStore(
        db_path=(args.db_path or None),
        parquet_root=(args.parquet_root or None),
        auto_migrate_legacy_sqlite=False,
    )
    report: dict[str, object] = {
        "db_path": str(store.db_path),
        "parquet_root": str(store.parquet_root),
    }
    if not bool(args.no_purge):
        report["snapshot_housekeeping"] = store.run_market_snapshot_housekeeping(
            policy=keep_overrides,
            default_keep_days=max(1, int(args.default_keep_days)),
        )
    if bool(args.checkpoint) or bool(args.vacuum):
        report["duckdb_maintenance"] = store.run_duckdb_maintenance(
            checkpoint=True,
            vacuum=bool(args.vacuum),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
