#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is importable when executed as:
# `python scripts/migrate_sqlite_to_duckdb.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from storage.duck_store import DuckHistoryStore

ICLOUD_CODEXAPP_DIR = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "codexapp"


def _default_sqlite_path() -> str:
    env = str(os.getenv("REALTIME0052_DB_PATH", "")).strip()
    if env:
        return env
    if ICLOUD_CODEXAPP_DIR.exists():
        return str(ICLOUD_CODEXAPP_DIR / "market_history.sqlite3")
    return "market_history.sqlite3"


def _default_duckdb_path() -> str:
    env = str(os.getenv("REALTIME0052_DUCKDB_PATH", "")).strip()
    if env:
        return env
    if ICLOUD_CODEXAPP_DIR.exists():
        return str(ICLOUD_CODEXAPP_DIR / "market_history.duckdb")
    return "market_history.duckdb"


def _count_sqlite_daily_rows(sqlite_path: Path) -> int:
    import sqlite3

    conn = sqlite3.connect(str(sqlite_path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()
        return int(row[0] or 0)
    except Exception:
        return 0
    finally:
        conn.close()


def _count_parquet_daily_rows_and_symbols(parquet_root: Path) -> tuple[int, int]:
    import duckdb

    pattern = parquet_root / "daily_bars" / "market=*" / "symbol=*" / "bars.parquet"
    if not any((parquet_root / "daily_bars").rglob("bars.parquet")):
        return 0, 0
    conn = duckdb.connect(":memory:")
    try:
        rows = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}')").fetchone()
        syms = conn.execute(
            f"SELECT COUNT(DISTINCT market || '|' || symbol) FROM read_parquet('{pattern}', hive_partitioning=1)"
        ).fetchone()
        return int(rows[0] or 0), int(syms[0] or 0)
    except Exception:
        return 0, 0
    finally:
        conn.close()


def _backup_target(duckdb_path: Path, parquet_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = duckdb_path.parent / f"migration_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if duckdb_path.exists():
        shutil.copy2(duckdb_path, backup_dir / duckdb_path.name)
    if parquet_root.exists():
        shutil.copytree(parquet_root, backup_dir / parquet_root.name, dirs_exist_ok=True)
    return backup_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot migration from SQLite history DB to DuckDB + Parquet hybrid.")
    parser.add_argument("--sqlite-path", default=_default_sqlite_path(), help="Legacy SQLite path.")
    parser.add_argument("--duckdb-path", default=_default_duckdb_path(), help="DuckDB destination path.")
    parser.add_argument("--parquet-root", default=None, help="Parquet root directory. Default: <duckdb_dir>/parquet")
    parser.add_argument(
        "--no-auto-migrate",
        action="store_true",
        help="Create DuckDB store without triggering automatic SQLite migration.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force migration even if destination already has data.",
    )
    parser.add_argument(
        "--reset-target",
        action="store_true",
        help="Delete existing DuckDB + Parquet target before migration (recommended with --backup).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing DuckDB + Parquet target before migrating.",
    )
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_path).expanduser()
    duckdb_path = Path(args.duckdb_path).expanduser()
    parquet_root = (
        Path(args.parquet_root).expanduser()
        if args.parquet_root
        else DuckHistoryStore.resolve_parquet_root(None, db_path=duckdb_path)
    )

    if not sqlite_path.exists() and not args.no_auto_migrate:
        print(f"[error] SQLite source not found: {sqlite_path}", file=sys.stderr)
        return 1

    if args.backup:
        backup_dir = _backup_target(duckdb_path=duckdb_path, parquet_root=parquet_root)
        print(f"[ok] backup saved: {backup_dir}")

    if args.reset_target:
        if duckdb_path.exists():
            duckdb_path.unlink()
            print(f"[ok] removed old duckdb: {duckdb_path}")
        if parquet_root.exists():
            shutil.rmtree(parquet_root)
            print(f"[ok] removed old parquet root: {parquet_root}")

    store = DuckHistoryStore(
        db_path=str(duckdb_path),
        parquet_root=str(parquet_root),
        legacy_sqlite_path=str(sqlite_path),
        auto_migrate_legacy_sqlite=(not args.no_auto_migrate) and (not args.force),
    )
    if (not args.no_auto_migrate) and args.force:
        store._migrate_from_sqlite(sqlite_path)  # noqa: SLF001

    tw_count = len(store.list_symbols("TW"))
    us_count = len(store.list_symbols("US"))
    sqlite_daily_rows = _count_sqlite_daily_rows(sqlite_path) if sqlite_path.exists() else 0
    parquet_daily_rows, parquet_daily_symbols = _count_parquet_daily_rows_and_symbols(store.parquet_root)
    print("[ok] DuckDB store is ready")
    print(f"  duckdb_path : {store.db_path}")
    print(f"  parquet_root: {store.parquet_root}")
    print(f"  symbols(TW) : {tw_count}")
    print(f"  symbols(US) : {us_count}")
    print(f"  sqlite daily_bars rows : {sqlite_daily_rows}")
    print(f"  parquet daily rows     : {parquet_daily_rows}")
    print(f"  parquet daily symbols  : {parquet_daily_symbols}")
    if args.no_auto_migrate:
        print("  note        : auto migration disabled (--no-auto-migrate).")
    else:
        print(f"  source      : {sqlite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
