#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.duck_store import DuckHistoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restore DuckDB + Parquet snapshot")
    parser.add_argument("snapshot_dir", help="Snapshot directory created by backup_duckdb_snapshot.py")
    parser.add_argument("--db-path", default="", help="Restore target DuckDB path (optional)")
    parser.add_argument("--parquet-root", default="", help="Restore target parquet root (optional)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target DuckDB/parquet paths",
    )
    return parser


def _pick_snapshot_db_file(snapshot_dir: Path) -> Path:
    candidates = [p for p in snapshot_dir.glob("*.duckdb") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"Snapshot DuckDB not found under: {snapshot_dir}")
    return sorted(candidates)[0]


def _remove_target(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    if not snapshot_dir.exists() or not snapshot_dir.is_dir():
        raise SystemExit(f"Snapshot directory not found: {snapshot_dir}")

    snapshot_db = _pick_snapshot_db_file(snapshot_dir)
    snapshot_parquet = snapshot_dir / "parquet"
    if not snapshot_parquet.exists() or not snapshot_parquet.is_dir():
        raise SystemExit(f"Snapshot parquet directory not found: {snapshot_parquet}")

    target_db = DuckHistoryStore.resolve_history_db_path(args.db_path or None)
    target_parquet = DuckHistoryStore.resolve_parquet_root(args.parquet_root or None, db_path=target_db)

    if target_db.exists() and not args.force:
        raise SystemExit(f"Target DuckDB exists (use --force): {target_db}")
    if target_parquet.exists() and not args.force:
        raise SystemExit(f"Target parquet root exists (use --force): {target_parquet}")

    if target_db.exists():
        _remove_target(target_db)
    if target_parquet.exists():
        _remove_target(target_parquet)

    target_db.parent.mkdir(parents=True, exist_ok=True)
    target_parquet.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot_db, target_db)
    shutil.copytree(snapshot_parquet, target_parquet)

    print(f"restored_db={target_db}")
    print(f"restored_parquet={target_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
