#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.duck_store import DuckHistoryStore


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _default_backup_root(db_path: Path) -> Path:
    return db_path.parent / "backups"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backup DuckDB + Parquet snapshot")
    parser.add_argument("--db-path", default="", help="DuckDB file path (optional)")
    parser.add_argument("--parquet-root", default="", help="Parquet root path (optional)")
    parser.add_argument("--backup-root", default="", help="Backup root directory (optional)")
    parser.add_argument(
        "--name-prefix",
        default="snapshot",
        help="Backup folder name prefix (default: snapshot)",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA256 calculation to speed up large backups",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    db_path = DuckHistoryStore.resolve_history_db_path(args.db_path or None)
    parquet_root = DuckHistoryStore.resolve_parquet_root(
        args.parquet_root or None, db_path=db_path
    )
    backup_root = (
        Path(args.backup_root).expanduser()
        if str(args.backup_root).strip()
        else _default_backup_root(db_path)
    )
    backup_root.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise SystemExit(f"DuckDB file not found: {db_path}")
    if not parquet_root.exists():
        raise SystemExit(f"Parquet root not found: {parquet_root}")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = str(args.name_prefix or "snapshot").strip() or "snapshot"
    snapshot_dir = backup_root / f"{prefix}_{ts}"
    if snapshot_dir.exists():
        raise SystemExit(f"Backup target already exists: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    db_target = snapshot_dir / db_path.name
    parquet_target = snapshot_dir / "parquet"
    shutil.copy2(db_path, db_target)
    shutil.copytree(parquet_root, parquet_target)

    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source": {
            "db_path": str(db_path),
            "parquet_root": str(parquet_root),
        },
        "snapshot": {
            "dir": str(snapshot_dir),
            "db_file": db_target.name,
            "parquet_dir": "parquet",
        },
        "sizes": {
            "db_bytes": db_target.stat().st_size,
            "parquet_bytes": sum(p.stat().st_size for p in parquet_target.rglob("*") if p.is_file()),
            "parquet_files": sum(1 for p in parquet_target.rglob("*") if p.is_file()),
        },
    }
    if not bool(args.no_hash):
        manifest["hashes"] = {
            "db_sha256": _sha256(db_target),
        }

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(snapshot_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
