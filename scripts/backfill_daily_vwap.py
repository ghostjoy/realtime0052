#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from storage.duck_store import DuckHistoryStore
from storage.history_store import HistoryStore


def _compute_daily_vwap_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or not isinstance(frame, pd.DataFrame):
        return pd.Series(dtype=float)
    if frame.empty:
        return pd.Series(index=frame.index, dtype=float)

    high = pd.to_numeric(frame.get("high"), errors="coerce")
    low = pd.to_numeric(frame.get("low"), errors="coerce")
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0)
    typical = (high + low + close) / 3.0
    cum_volume = volume.cumsum().replace(0, float("nan"))
    return (typical * volume).cumsum() / cum_volume


def _iter_duck_symbols(store: DuckHistoryStore, markets: list[str] | None) -> list[tuple[str, str]]:
    market_filter = {m.strip().upper() for m in (markets or []) if m.strip()}
    out: list[tuple[str, str]] = []
    daily_root = store.parquet_root / "daily_bars"
    if not daily_root.exists():
        return out
    for market_dir in sorted([p for p in daily_root.glob("market=*") if p.is_dir()]):
        market_token = market_dir.name.split("=", 1)[-1].strip().upper()
        if not market_token:
            continue
        if market_filter and market_token not in market_filter:
            continue
        for symbol_dir in sorted([p for p in market_dir.glob("symbol=*") if p.is_dir()]):
            symbol_token = symbol_dir.name.split("=", 1)[-1].strip().upper()
            if not symbol_token:
                continue
            out.append((market_token, symbol_token))
    return out


def _load_duck_raw_frame(store: DuckHistoryStore, *, symbol: str, market: str) -> pd.DataFrame:
    sources = store._daily_parquet_sources(symbol, market)  # noqa: SLF001
    if not sources:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    expected = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "adj_close",
        "source",
        "fetched_at",
        "asof",
        "quality_score",
        "raw_json",
    ]
    for path in sources:
        frame = pd.read_parquet(path)
        for col in expected:
            if col not in frame.columns:
                frame[col] = None
        frames.append(frame[expected])
    if not frames:
        return pd.DataFrame(columns=expected)
    raw = frames[0].copy() if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date"])
    if raw.empty:
        return pd.DataFrame(columns=expected)
    raw["fetched_ts"] = pd.to_datetime(raw.get("fetched_at"), utc=True, errors="coerce")
    raw = raw.sort_values(["date", "fetched_ts", "fetched_at"], ascending=[True, True, True])
    raw = raw.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return raw[expected]


def _materialize_duck_bars(store: DuckHistoryStore, raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    normalized = store._normalize_daily_bars_frame(frame)  # noqa: SLF001
    if normalized.empty:
        return normalized
    meta_cols = [c for c in ["source", "fetched_at"] if c in frame.columns]
    if meta_cols:
        meta = frame[meta_cols].copy()
        meta = meta.loc[~meta.index.duplicated(keep="last")]
        normalized = normalized.join(meta, how="left")
    normalized = store._ensure_daily_vwap_column(normalized)  # noqa: SLF001
    if "source" not in normalized.columns:
        normalized["source"] = "unknown"
    else:
        normalized["source"] = normalized["source"].astype(str).replace({"": "unknown"})
    if "fetched_at" not in normalized.columns:
        normalized["fetched_at"] = datetime.now(tz=timezone.utc).isoformat()
    else:
        normalized["fetched_at"] = normalized["fetched_at"].fillna(
            datetime.now(tz=timezone.utc).isoformat()
        ).astype(str)
    if "asof" in normalized.columns:
        normalized["asof"] = pd.to_datetime(normalized["asof"], utc=True, errors="coerce")
        normalized["asof"] = normalized["asof"].where(
            normalized["asof"].notna(),
            pd.Series(pd.to_datetime(normalized.index, utc=True), index=normalized.index),
        )
    else:
        normalized["asof"] = pd.Series(
            pd.to_datetime(normalized.index, utc=True),
            index=normalized.index,
        )
    if "quality_score" in normalized.columns:
        normalized["quality_score"] = pd.to_numeric(normalized["quality_score"], errors="coerce")
    else:
        normalized["quality_score"] = float("nan")
    if "raw_json" not in normalized.columns:
        normalized["raw_json"] = ""
    else:
        normalized["raw_json"] = normalized["raw_json"].fillna("").astype(str)
    return normalized


def _write_duck_delta(store: DuckHistoryStore, *, symbol: str, market: str, bars: pd.DataFrame) -> int:
    if bars.empty:
        return 0
    payload = pd.DataFrame(
        {
            "date": pd.to_datetime(bars.index, utc=True, errors="coerce").strftime("%Y-%m-%d"),
            "open": pd.to_numeric(bars["open"], errors="coerce"),
            "high": pd.to_numeric(bars["high"], errors="coerce"),
            "low": pd.to_numeric(bars["low"], errors="coerce"),
            "close": pd.to_numeric(bars["close"], errors="coerce"),
            "volume": pd.to_numeric(bars.get("volume", 0.0), errors="coerce").fillna(0.0),
            "vwap": pd.to_numeric(bars.get("vwap"), errors="coerce"),
            "adj_close": pd.to_numeric(bars.get("adj_close"), errors="coerce"),
            "source": bars.get("source", pd.Series(index=bars.index, data="unknown")).astype(str),
            "fetched_at": bars.get(
                "fetched_at",
                pd.Series(index=bars.index, data=datetime.now(tz=timezone.utc).isoformat()),
            ).astype(str),
            "asof": pd.to_datetime(
                bars.get("asof", pd.Series(index=bars.index, data=bars.index)),
                utc=True,
                errors="coerce",
            )
            .astype("datetime64[ns, UTC]")
            .astype(str),
            "quality_score": pd.to_numeric(bars.get("quality_score"), errors="coerce"),
            "raw_json": bars.get("raw_json", pd.Series(index=bars.index, data="")).astype(str),
        }
    )
    payload = payload.reset_index(drop=True)
    payload = payload.dropna(subset=["date", "open", "high", "low", "close"])
    if payload.empty:
        return 0
    payload = payload.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    delta_dir = store._daily_delta_dir(symbol, market)  # noqa: SLF001
    delta_dir.mkdir(parents=True, exist_ok=True)
    delta_path = store._next_delta_parquet_path(delta_dir, prefix="bars_delta")  # noqa: SLF001
    payload.to_parquet(delta_path, index=False)
    return int(len(payload))


def _backfill_duckdb(
    *,
    store: DuckHistoryStore,
    markets: list[str] | None,
    dry_run: bool,
    rewrite_all: bool,
    progress_every: int,
) -> dict[str, int]:
    stats = {
        "symbols_scanned": 0,
        "symbols_rewritten": 0,
        "rows_rewritten": 0,
        "rows_missing": 0,
        "errors": 0,
    }
    for market, symbol in _iter_duck_symbols(store, markets):
        stats["symbols_scanned"] += 1
        if progress_every > 0 and stats["symbols_scanned"] % progress_every == 0:
            print(
                f"[progress] scanned={stats['symbols_scanned']} rewritten={stats['symbols_rewritten']} "
                f"missing_rows={stats['rows_missing']} errors={stats['errors']}"
            )
        try:
            raw = _load_duck_raw_frame(store, symbol=symbol, market=market)
        except Exception as exc:
            stats["errors"] += 1
            print(f"[warn] skip {market}:{symbol} while loading raw bars: {exc}")
            continue
        if raw.empty:
            continue
        if "vwap" not in raw.columns:
            raw["vwap"] = pd.Series(index=raw.index, dtype=float)
        missing_mask = pd.to_numeric(raw["vwap"], errors="coerce").isna()
        raw_missing_count = int(missing_mask.sum())
        if (not rewrite_all) and raw_missing_count <= 0:
            continue
        try:
            bars = _materialize_duck_bars(store, raw)
        except Exception as exc:
            stats["errors"] += 1
            print(f"[warn] skip {market}:{symbol} while materializing bars: {exc}")
            continue
        if bars.empty:
            continue
        fillable_missing_count = raw_missing_count
        if not rewrite_all and raw_missing_count > 0:
            missing_dates = pd.to_datetime(raw.loc[missing_mask, "date"], utc=True, errors="coerce")
            missing_dates = missing_dates.dropna()
            fillable_missing_count = 0
            if len(missing_dates) > 0:
                aligned = bars.reindex(missing_dates)
                if "vwap" in aligned.columns:
                    fillable_missing_count = int(pd.to_numeric(aligned["vwap"], errors="coerce").notna().sum())
            if fillable_missing_count <= 0:
                continue
        stats["rows_missing"] += int(fillable_missing_count)
        stats["symbols_rewritten"] += 1
        if dry_run:
            stats["rows_rewritten"] += int(len(bars))
            continue
        try:
            written = _write_duck_delta(store, symbol=symbol, market=market, bars=bars)
            stats["rows_rewritten"] += int(written)
        except Exception as exc:
            stats["errors"] += 1
            print(f"[warn] skip {market}:{symbol} while writing bars: {exc}")
    return stats


def _ensure_sqlite_vwap_column(conn: sqlite3.Connection) -> None:
    cols = {
        str(row[1]).strip().lower()
        for row in conn.execute("PRAGMA table_info(daily_bars)").fetchall()
    }
    if "vwap" not in cols:
        conn.execute("ALTER TABLE daily_bars ADD COLUMN vwap REAL")


def _backfill_sqlite(
    *,
    db_path: Path,
    markets: list[str] | None,
    dry_run: bool,
    rewrite_all: bool,
    progress_every: int,
) -> dict[str, int]:
    stats = {
        "symbols_scanned": 0,
        "symbols_rewritten": 0,
        "rows_rewritten": 0,
        "rows_missing": 0,
        "errors": 0,
    }
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_sqlite_vwap_column(conn)
        params: list[object] = []
        market_filter = [m.strip().upper() for m in (markets or []) if m.strip()]
        if market_filter:
            params.extend(market_filter)
        rows = conn.execute(
            f"""
            SELECT i.id, i.market, i.symbol
            FROM instruments i
            WHERE EXISTS (
                SELECT 1 FROM daily_bars d WHERE d.instrument_id = i.id
            )
            {f"AND i.market IN ({', '.join('?' for _ in market_filter)})" if market_filter else ""}
            ORDER BY i.market, i.symbol
            """,
            params,
        ).fetchall()
        for instrument_id, market, symbol in rows:
            stats["symbols_scanned"] += 1
            if progress_every > 0 and stats["symbols_scanned"] % progress_every == 0:
                print(
                    f"[progress] scanned={stats['symbols_scanned']} rewritten={stats['symbols_rewritten']} "
                    f"missing_rows={stats['rows_missing']} errors={stats['errors']}"
                )
            try:
                bars = pd.read_sql_query(
                    """
                    SELECT date, high, low, close, volume, vwap
                    FROM daily_bars
                    WHERE instrument_id=?
                    ORDER BY date ASC
                    """,
                    conn,
                    params=[int(instrument_id)],
                )
            except Exception as exc:
                stats["errors"] += 1
                print(f"[warn] skip {market}:{symbol} while loading sqlite bars: {exc}")
                continue
            if bars.empty:
                continue
            bars["date"] = pd.to_datetime(bars["date"], utc=True, errors="coerce")
            bars = bars.dropna(subset=["date"]).set_index("date").sort_index()
            computed = _compute_daily_vwap_series(bars)
            existing = pd.to_numeric(bars.get("vwap"), errors="coerce")
            missing_mask = existing.isna() if isinstance(existing, pd.Series) else pd.Series(True, index=bars.index)
            stats["rows_missing"] += int(missing_mask.sum())
            if rewrite_all:
                update_mask = computed.notna()
            else:
                update_mask = missing_mask & computed.notna()
            if not bool(update_mask.any()):
                continue
            payload = [
                (float(computed.loc[ts]), int(instrument_id), pd.Timestamp(ts).date().isoformat())
                for ts in bars.index[update_mask]
            ]
            stats["symbols_rewritten"] += 1
            stats["rows_rewritten"] += len(payload)
            if dry_run or not payload:
                continue
            conn.executemany(
                "UPDATE daily_bars SET vwap=? WHERE instrument_id=? AND date=?",
                payload,
            )
        if not dry_run:
            conn.commit()
    finally:
        conn.close()
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot backfill for persisted daily_bars VWAP.")
    parser.add_argument(
        "--backend",
        choices=["duckdb", "sqlite"],
        default="duckdb",
        help="Which storage backend to backfill. Default: current DuckDB store.",
    )
    parser.add_argument(
        "--duckdb-path",
        default=None,
        help="DuckDB path. Default uses REALTIME0052_DUCKDB_PATH or repo default.",
    )
    parser.add_argument(
        "--parquet-root",
        default=None,
        help="DuckDB parquet root. Default uses <duckdb_dir>/parquet.",
    )
    parser.add_argument(
        "--sqlite-path",
        default=None,
        help="Legacy SQLite path. Default uses REALTIME0052_DB_PATH or repo default.",
    )
    parser.add_argument(
        "--markets",
        default="",
        help="Comma-separated markets to process, e.g. TW,US,OTC. Default: all markets found.",
    )
    parser.add_argument(
        "--rewrite-all",
        action="store_true",
        help="Rewrite all symbols even if raw storage already has vwap values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; do not write changes.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N symbols. Default: 200.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    markets = [part.strip().upper() for part in str(args.markets or "").split(",") if part.strip()]
    if args.backend == "duckdb":
        store = DuckHistoryStore(
            db_path=args.duckdb_path,
            parquet_root=args.parquet_root,
        )
        stats = _backfill_duckdb(
            store=store,
            markets=markets,
            dry_run=bool(args.dry_run),
            rewrite_all=bool(args.rewrite_all),
            progress_every=max(0, int(args.progress_every)),
        )
        target_desc = f"duckdb={store.db_path} parquet={store.parquet_root}"
    else:
        store = HistoryStore(db_path=args.sqlite_path)
        stats = _backfill_sqlite(
            db_path=store.db_path,
            markets=markets,
            dry_run=bool(args.dry_run),
            rewrite_all=bool(args.rewrite_all),
            progress_every=max(0, int(args.progress_every)),
        )
        target_desc = f"sqlite={store.db_path}"

    print(f"[ok] target: {target_desc}")
    print(f"  symbols_scanned   : {stats['symbols_scanned']}")
    print(f"  symbols_rewritten : {stats['symbols_rewritten']}")
    print(f"  rows_missing      : {stats['rows_missing']}")
    print(f"  rows_rewritten    : {stats['rows_rewritten']}")
    print(f"  errors            : {stats['errors']}")
    print(f"  dry_run           : {'yes' if args.dry_run else 'no'}")
    print(f"  rewrite_all       : {'yes' if args.rewrite_all else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
