from __future__ import annotations

from contextlib import contextmanager

import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

from providers.base import ProviderRequest
from services.market_data_service import MarketDataService
from storage.history_store import (
    BacktestReplayRun,
    BootstrapRun,
    HeatmapHubEntry,
    HeatmapRun,
    RotationRun,
    SyncReport,
    UniverseSnapshot,
)

DEFAULT_DB_FILENAME = "market_history.duckdb"
DEFAULT_LEGACY_SQLITE_FILENAME = "market_history.sqlite3"
DEFAULT_INTRADAY_RETAIN_DAYS = 365 * 3
ICLOUD_DOCS_ROOT = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
DEFAULT_ICLOUD_DB_PATH = ICLOUD_DOCS_ROOT / "codexapp" / DEFAULT_DB_FILENAME
DEFAULT_LEGACY_ICLOUD_DB_PATH = ICLOUD_DOCS_ROOT / "codexapp" / DEFAULT_LEGACY_SQLITE_FILENAME


class DuckHistoryStore:
    backend_name = "duckdb"

    def __init__(
        self,
        db_path: Optional[str] = None,
        parquet_root: Optional[str] = None,
        service: Optional[MarketDataService] = None,
        intraday_retain_days: Optional[int] = None,
        legacy_sqlite_path: Optional[str] = None,
        auto_migrate_legacy_sqlite: bool = True,
    ):
        self.db_path = self.resolve_history_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_root = self.resolve_parquet_root(parquet_root, db_path=self.db_path)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self.intraday_retain_days = self.resolve_intraday_retain_days(intraday_retain_days)
        self.service = service or MarketDataService()
        self._writeback_lock = threading.Lock()
        self._writeback_inflight: set[tuple[str, str]] = set()
        self._writeback_executor: Optional[ThreadPoolExecutor] = None
        self._init_db()

        set_metadata_store = getattr(self.service, "set_metadata_store", None)
        if callable(set_metadata_store):
            try:
                set_metadata_store(self)
            except Exception:
                pass

        if auto_migrate_legacy_sqlite:
            src = Path(legacy_sqlite_path).expanduser() if legacy_sqlite_path else self._default_legacy_sqlite_path()
            self._maybe_migrate_from_legacy_sqlite(src)

    @staticmethod
    def resolve_history_db_path(db_path: Optional[str] = None) -> Path:
        if db_path:
            return Path(db_path).expanduser()
        if ICLOUD_DOCS_ROOT.exists():
            return DEFAULT_ICLOUD_DB_PATH
        return Path(DEFAULT_DB_FILENAME)

    @staticmethod
    def resolve_parquet_root(parquet_root: Optional[str] = None, *, db_path: Optional[Path] = None) -> Path:
        if parquet_root:
            return Path(parquet_root).expanduser()
        base = db_path if db_path is not None else DuckHistoryStore.resolve_history_db_path()
        return base.parent / "parquet"

    @staticmethod
    def resolve_intraday_retain_days(days: Optional[int] = None) -> int:
        if days is None:
            return DEFAULT_INTRADAY_RETAIN_DAYS
        try:
            return max(1, int(days))
        except Exception:
            return DEFAULT_INTRADAY_RETAIN_DAYS

    @staticmethod
    def _normalize_symbol_token(value: object) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalize_market_token(value: object) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalize_text(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _is_tw_local_security_symbol(symbol: str) -> bool:
        token = str(symbol or "").strip().upper()
        return bool(re.fullmatch(r"\d{4,6}[A-Z]?", token))

    @staticmethod
    def _parse_iso_datetime(value: object) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @contextmanager
    def _connect_ctx(self):
        db_path = str(self.db_path)
        is_memory = db_path == ":memory:" or db_path.startswith("file::")
        
        if is_memory and hasattr(self, '_memory_conn') and self._memory_conn:
            yield self._memory_conn
            return
        
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if not is_memory:
                conn.close()
            else:
                self._memory_conn = conn

    def _connect(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(str(self.db_path))
        conn.execute("PRAGMA threads=4")
        return conn

    def _next_id(self, conn: duckdb.DuckDBPyConnection, table: str) -> int:
        row = conn.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table}").fetchone()
        return int(row[0] or 1)

    def _init_db(self):
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS instruments (
                    id BIGINT,
                    symbol VARCHAR NOT NULL,
                    market VARCHAR NOT NULL,
                    name VARCHAR,
                    currency VARCHAR,
                    timezone VARCHAR,
                    active INTEGER DEFAULT 1,
                    UNIQUE(symbol, market)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_state (
                    instrument_id BIGINT,
                    last_success_date VARCHAR,
                    last_source VARCHAR,
                    last_error VARCHAR,
                    updated_at VARCHAR
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbol_metadata (
                    symbol VARCHAR,
                    market VARCHAR,
                    name VARCHAR,
                    exchange VARCHAR,
                    industry VARCHAR,
                    asset_type VARCHAR,
                    currency VARCHAR,
                    source VARCHAR,
                    updated_at VARCHAR,
                    UNIQUE(symbol, market)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bootstrap_runs (
                    id BIGINT,
                    run_id VARCHAR,
                    scope VARCHAR,
                    status VARCHAR,
                    started_at VARCHAR,
                    finished_at VARCHAR,
                    total_symbols BIGINT,
                    synced_symbols BIGINT,
                    failed_symbols BIGINT,
                    params_json VARCHAR,
                    summary_json VARCHAR,
                    error VARCHAR,
                    UNIQUE(run_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id BIGINT,
                    created_at VARCHAR,
                    symbol VARCHAR,
                    market VARCHAR,
                    strategy VARCHAR,
                    params_json VARCHAR,
                    cost_json VARCHAR,
                    result_json VARCHAR
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_replay_runs (
                    id BIGINT,
                    run_key VARCHAR,
                    created_at VARCHAR,
                    params_json VARCHAR,
                    payload_json VARCHAR
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_snapshots (
                    universe_id VARCHAR,
                    symbols_json VARCHAR,
                    source VARCHAR,
                    fetched_at VARCHAR,
                    updated_at VARCHAR,
                    UNIQUE(universe_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS heatmap_runs (
                    id BIGINT,
                    universe_id VARCHAR,
                    created_at VARCHAR,
                    payload_json VARCHAR
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS heatmap_hub_entries (
                    etf_code VARCHAR,
                    etf_name VARCHAR,
                    pin_as_card INTEGER DEFAULT 0,
                    open_count BIGINT DEFAULT 0,
                    last_opened_at VARCHAR,
                    created_at VARCHAR,
                    updated_at VARCHAR,
                    UNIQUE(etf_code)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rotation_runs (
                    id BIGINT,
                    universe_id VARCHAR,
                    run_key VARCHAR,
                    created_at VARCHAR,
                    params_json VARCHAR,
                    payload_json VARCHAR
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS migration_meta (
                    key VARCHAR,
                    value VARCHAR,
                    updated_at VARCHAR,
                    UNIQUE(key)
                )
                """
            )
        finally:
            conn.close()

    def _default_legacy_sqlite_path(self) -> Path:
        env = str(os.getenv("REALTIME0052_DB_PATH", "")).strip()
        if env:
            return Path(env).expanduser()
        if str(self.db_path).startswith(str(ICLOUD_DOCS_ROOT)) and DEFAULT_LEGACY_ICLOUD_DB_PATH.exists():
            return DEFAULT_LEGACY_ICLOUD_DB_PATH
        local = Path(DEFAULT_LEGACY_SQLITE_FILENAME)
        if local.exists():
            return local
        return DEFAULT_LEGACY_ICLOUD_DB_PATH

    def _has_any_parquet(self) -> bool:
        return any(self.parquet_root.rglob("*.parquet"))

    def _is_effectively_empty(self) -> bool:
        if self._has_any_parquet():
            return False
        conn = self._connect()
        try:
            checks = [
                "SELECT COUNT(*) FROM symbol_metadata",
                "SELECT COUNT(*) FROM backtest_replay_runs",
                "SELECT COUNT(*) FROM heatmap_runs",
                "SELECT COUNT(*) FROM heatmap_hub_entries",
                "SELECT COUNT(*) FROM rotation_runs",
                "SELECT COUNT(*) FROM instruments",
            ]
            for sql in checks:
                row = conn.execute(sql).fetchone()
                if int(row[0] or 0) > 0:
                    return False
            return True
        finally:
            conn.close()

    def _maybe_migrate_from_legacy_sqlite(self, source_path: Path):
        if not source_path.exists() or source_path.resolve() == self.db_path.resolve():
            return
        if not self._is_effectively_empty():
            return
        self._migrate_from_sqlite(source_path)

    def _migrate_from_sqlite(self, source_path: Path):
        try:
            src = sqlite3.connect(str(source_path))
        except Exception:
            return

        def _table_exists(name: str) -> bool:
            row = src.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            ).fetchone()
            return row is not None

        conn = self._connect()
        try:
            if _table_exists("instruments"):
                rows = src.execute("SELECT id, symbol, market, name, currency, timezone, active FROM instruments").fetchall()
                for _, symbol, market, name, currency, tz, active in rows:
                    inst_id = self._get_or_create_instrument(str(symbol), str(market), name=str(name or ""))
                    conn.execute(
                        "UPDATE instruments SET currency=?, timezone=?, active=? WHERE id=?",
                        (str(currency or ""), str(tz or "UTC"), int(active or 1), int(inst_id)),
                    )

            if _table_exists("daily_bars") and _table_exists("instruments"):
                rows = src.execute("SELECT id, symbol, market FROM instruments").fetchall()
                for old_id, symbol, market in rows:
                    df = pd.read_sql_query(
                        "SELECT date, open, high, low, close, volume, adj_close, source, fetched_at FROM daily_bars WHERE instrument_id=? ORDER BY date ASC",
                        src,
                        params=[int(old_id)],
                    )
                    if df.empty:
                        continue
                    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
                    df = df.dropna(subset=["date"]).set_index("date")
                    self._upsert_daily_bars(symbol=str(symbol), market=str(market), bars=df)

            if _table_exists("intraday_ticks") and _table_exists("instruments"):
                rows = src.execute("SELECT id, symbol, market FROM instruments").fetchall()
                for old_id, symbol, market in rows:
                    df = pd.read_sql_query(
                        "SELECT ts_utc, price, cum_volume, source FROM intraday_ticks WHERE instrument_id=? ORDER BY ts_utc ASC",
                        src,
                        params=[int(old_id)],
                    )
                    if df.empty:
                        continue
                    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
                    df = df.dropna(subset=["ts_utc"]).set_index("ts_utc")
                    self._upsert_intraday_ticks(symbol=str(symbol), market=str(market), ticks=df)

            if _table_exists("sync_state") and _table_exists("instruments"):
                rows = src.execute(
                    "SELECT s.instrument_id, i.symbol, i.market, s.last_success_date, s.last_source, s.last_error, s.updated_at "
                    "FROM sync_state s JOIN instruments i ON i.id = s.instrument_id"
                ).fetchall()
                for _, symbol, market, last_date, last_source, last_error, updated_at in rows:
                    inst_id = self._get_or_create_instrument(str(symbol), str(market), name=None)
                    conn.execute("DELETE FROM sync_state WHERE instrument_id=?", (inst_id,))
                    conn.execute(
                        "INSERT INTO sync_state(instrument_id, last_success_date, last_source, last_error, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            int(inst_id),
                            str(last_date or ""),
                            str(last_source or ""),
                            str(last_error or ""),
                            str(updated_at or datetime.now(tz=timezone.utc).isoformat()),
                        ),
                    )

            for table in [
                "symbol_metadata",
                "bootstrap_runs",
                "backtest_runs",
                "backtest_replay_runs",
                "universe_snapshots",
                "heatmap_runs",
                "heatmap_hub_entries",
                "rotation_runs",
            ]:
                if not _table_exists(table):
                    continue
                df = pd.read_sql_query(f"SELECT * FROM {table}", src)
                if df.empty:
                    continue
                conn.register("tmp_migrate_df", df)
                conn.execute(f"DELETE FROM {table}")
                cols = ", ".join(df.columns.astype(str).tolist())
                conn.execute(f"INSERT INTO {table}({cols}) SELECT {cols} FROM tmp_migrate_df")
                conn.unregister("tmp_migrate_df")

            now_iso = datetime.now(tz=timezone.utc).isoformat()
            conn.execute("DELETE FROM migration_meta WHERE key='legacy_sqlite_path'")
            conn.execute(
                "INSERT INTO migration_meta(key, value, updated_at) VALUES (?, ?, ?)",
                ("legacy_sqlite_path", str(source_path), now_iso),
            )
            conn.execute("DELETE FROM migration_meta WHERE key='legacy_migrated_at'")
            conn.execute(
                "INSERT INTO migration_meta(key, value, updated_at) VALUES (?, ?, ?)",
                ("legacy_migrated_at", now_iso, now_iso),
            )
        finally:
            conn.close()
            src.close()

    def _daily_symbol_path(self, symbol: str, market: str) -> Path:
        return self.parquet_root / "daily_bars" / f"market={self._normalize_market_token(market)}" / f"symbol={self._normalize_symbol_token(symbol)}" / "bars.parquet"

    def _intraday_symbol_path(self, symbol: str, market: str) -> Path:
        return self.parquet_root / "intraday_ticks" / f"market={self._normalize_market_token(market)}" / f"symbol={self._normalize_symbol_token(symbol)}" / "ticks.parquet"

    @staticmethod
    def _normalize_daily_bars_frame(df: pd.DataFrame) -> pd.DataFrame:
        base_cols = ["open", "high", "low", "close", "volume"]
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=base_cols)

        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            renamed: list[str] = []
            for col in out.columns:
                parts = [str(part).strip().lower() for part in col if str(part).strip()]
                candidate = ""
                for item in reversed(parts):
                    if item in {"open", "high", "low", "close", "adj close", "adj_close", "volume", "price"}:
                        candidate = item
                        break
                renamed.append(candidate or (parts[-1] if parts else ""))
            out.columns = renamed
        else:
            out.columns = [str(col).strip().lower() for col in out.columns]

        if "adj close" in out.columns and "adj_close" not in out.columns:
            out = out.rename(columns={"adj close": "adj_close"})

        if "price" in out.columns and "close" not in out.columns:
            out["close"] = out["price"]

        def _extract_numeric_col(name: str) -> pd.Series:
            if name not in out.columns:
                return pd.Series(index=out.index, dtype=float)
            selected = out.loc[:, out.columns == name]
            if isinstance(selected, pd.Series):
                return pd.to_numeric(selected, errors="coerce")
            if selected.shape[1] == 1:
                return pd.to_numeric(selected.iloc[:, 0], errors="coerce")
            return pd.to_numeric(selected.bfill(axis=1).iloc[:, 0], errors="coerce")

        close = _extract_numeric_col("close")
        if close.empty:
            return pd.DataFrame(columns=base_cols)

        norm = pd.DataFrame(index=out.index)
        norm["close"] = close
        for col in ["open", "high", "low"]:
            series = _extract_numeric_col(col)
            norm[col] = series.fillna(close)
        volume = _extract_numeric_col("volume")
        norm["volume"] = volume.fillna(0.0)
        if "adj_close" in out.columns:
            norm["adj_close"] = _extract_numeric_col("adj_close")

        idx = pd.to_datetime(norm.index, utc=True, errors="coerce")
        norm.index = idx
        norm = norm[~norm.index.isna()]
        keep_cols = [c for c in ["open", "high", "low", "close", "volume", "adj_close"] if c in norm.columns]
        norm = norm[keep_cols]
        norm = norm.dropna(subset=["open", "high", "low", "close"], how="any").sort_index()
        if "volume" in norm.columns:
            norm["volume"] = pd.to_numeric(norm["volume"], errors="coerce").fillna(0.0)
        return norm

    def _load_daily_frame_raw(self, symbol: str, market: str) -> pd.DataFrame:
        path = self._daily_symbol_path(symbol, market)
        if not path.exists():
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "adj_close", "source", "fetched_at"])
        df = pd.read_parquet(path)
        if df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "adj_close", "source", "fetched_at"])
        expected = ["date", "open", "high", "low", "close", "volume", "adj_close", "source", "fetched_at"]
        for col in expected:
            if col not in df.columns:
                df[col] = None
        return df[expected]

    def _upsert_daily_bars(self, symbol: str, market: str, bars: pd.DataFrame):
        if bars is None or bars.empty:
            return 0
        frame = bars.copy().sort_index()
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame[~frame.index.isna()]
        if frame.empty:
            return 0

        payload = pd.DataFrame(
            {
                "date": frame.index.tz_convert("UTC").strftime("%Y-%m-%d"),
                "open": pd.to_numeric(frame["open"], errors="coerce"),
                "high": pd.to_numeric(frame["high"], errors="coerce"),
                "low": pd.to_numeric(frame["low"], errors="coerce"),
                "close": pd.to_numeric(frame["close"], errors="coerce"),
                "volume": pd.to_numeric(frame.get("volume", 0.0), errors="coerce").fillna(0.0),
                "adj_close": pd.to_numeric(frame.get("adj_close"), errors="coerce") if "adj_close" in frame.columns else None,
                "source": frame.get("source", pd.Series(index=frame.index, data="unknown")).astype(str),
                "fetched_at": frame.get(
                    "fetched_at",
                    pd.Series(index=frame.index, data=datetime.now(tz=timezone.utc).isoformat()),
                ).astype(str),
            }
        )
        payload = payload.dropna(subset=["open", "high", "low", "close"])  # type: ignore[arg-type]
        payload = payload.reset_index(drop=True)
        if payload.empty:
            return 0

        existing = self._load_daily_frame_raw(symbol, market)
        if existing.empty:
            merged = payload.copy()
        else:
            merged = pd.concat([existing, payload], axis=0, ignore_index=True)
        merged = merged.reset_index(drop=True)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        merged = merged.dropna(subset=["date"])  # type: ignore[arg-type]
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        out_path = self._daily_symbol_path(symbol, market)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path, index=False)
        return int(len(payload))

    def _get_writeback_executor(self) -> ThreadPoolExecutor:
        with self._writeback_lock:
            if self._writeback_executor is None:
                self._writeback_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duck-writeback")
            return self._writeback_executor

    def _persist_daily_bars_writeback(
        self,
        *,
        symbol: str,
        market: str,
        bars: pd.DataFrame,
        source: Optional[str],
    ) -> int:
        normalized = self._normalize_daily_bars_frame(bars)
        if normalized.empty:
            return 0

        source_text = self._normalize_text(source)
        if "source" not in normalized.columns:
            normalized["source"] = source_text or "writeback"
        else:
            normalized["source"] = normalized["source"].astype(str).replace({"": None})
            normalized["source"] = normalized["source"].fillna(source_text or "writeback").astype(str)
        normalized["fetched_at"] = datetime.now(tz=timezone.utc).isoformat()

        rows_upserted = self._upsert_daily_bars(symbol=symbol, market=market, bars=normalized)
        if rows_upserted <= 0:
            return 0

        instrument_id = self._get_or_create_instrument(symbol, market)
        last_synced = pd.Timestamp(normalized.index.max()).to_pydatetime().replace(tzinfo=timezone.utc)
        current_last = self._load_last_success_date(instrument_id)
        merged_last = last_synced if current_last is None else max(current_last, last_synced)
        self._save_sync_state(instrument_id, merged_last, source_text or "writeback", None)
        return rows_upserted

    def queue_daily_bars_writeback(
        self,
        *,
        symbol: str,
        market: str,
        bars: pd.DataFrame,
        source: Optional[str] = None,
    ) -> bool:
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            return False

        symbol_token = self._normalize_symbol_token(symbol)
        market_token = self._normalize_market_token(market)
        if not symbol_token or not market_token:
            return False

        payload = bars.copy()
        source_text = self._normalize_text(source)
        key = (market_token, symbol_token)
        with self._writeback_lock:
            if key in self._writeback_inflight:
                return False
            self._writeback_inflight.add(key)

        def _worker():
            try:
                self._persist_daily_bars_writeback(
                    symbol=symbol_token,
                    market=market_token,
                    bars=payload,
                    source=source_text,
                )
            except Exception:
                pass
            finally:
                with self._writeback_lock:
                    self._writeback_inflight.discard(key)

        try:
            executor = self._get_writeback_executor()
            executor.submit(_worker)
            return True
        except Exception:
            with self._writeback_lock:
                self._writeback_inflight.discard(key)
            return False

    def flush_writeback_queue(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            with self._writeback_lock:
                pending = bool(self._writeback_inflight)
            if not pending:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def _load_intraday_frame_raw(self, symbol: str, market: str) -> pd.DataFrame:
        path = self._intraday_symbol_path(symbol, market)
        if not path.exists():
            return pd.DataFrame(columns=["ts_utc", "price", "cum_volume", "source", "fetched_at"])
        df = pd.read_parquet(path)
        if df.empty:
            return pd.DataFrame(columns=["ts_utc", "price", "cum_volume", "source", "fetched_at"])
        expected = ["ts_utc", "price", "cum_volume", "source", "fetched_at"]
        for col in expected:
            if col not in df.columns:
                df[col] = None
        return df[expected]

    def _upsert_intraday_ticks(self, symbol: str, market: str, ticks: pd.DataFrame) -> int:
        if ticks is None or ticks.empty:
            return 0
        frame = ticks.copy().sort_index()
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame[~frame.index.isna()]
        if frame.empty:
            return 0

        payload = pd.DataFrame(
            {
                "ts_utc": frame.index.tz_convert("UTC").astype("datetime64[ns, UTC]").astype(str),
                "price": pd.to_numeric(frame["price"], errors="coerce"),
                "cum_volume": pd.to_numeric(frame.get("cum_volume", 0.0), errors="coerce").fillna(0.0),
                "source": frame.get("source", pd.Series(index=frame.index, data="unknown")).astype(str),
                "fetched_at": frame.get(
                    "fetched_at",
                    pd.Series(index=frame.index, data=datetime.now(tz=timezone.utc).isoformat()),
                ).astype(str),
            }
        )
        payload = payload.dropna(subset=["price"])  # type: ignore[arg-type]
        if payload.empty:
            return 0

        existing = self._load_intraday_frame_raw(symbol, market)
        if existing.empty:
            merged = payload.copy()
        else:
            merged = pd.concat([existing, payload], axis=0, ignore_index=True)
        merged["ts_utc"] = pd.to_datetime(merged["ts_utc"], utc=True, errors="coerce")
        merged = merged.dropna(subset=["ts_utc"])  # type: ignore[arg-type]
        merged = merged.reset_index(drop=True)
        merged = merged.drop_duplicates(subset=["ts_utc"], keep="last").sort_values("ts_utc")
        cutoff = datetime.now(tz=timezone.utc) - pd.Timedelta(days=self.intraday_retain_days)
        merged = merged[merged["ts_utc"] >= pd.Timestamp(cutoff)]
        merged["ts_utc"] = merged["ts_utc"].astype(str)
        out_path = self._intraday_symbol_path(symbol, market)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path, index=False)
        return int(len(payload))

    def _get_or_create_instrument(self, symbol: str, market: str, name: Optional[str] = None) -> int:
        symbol = self._normalize_symbol_token(symbol)
        market = self._normalize_market_token(market)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM instruments WHERE symbol=? AND market=?",
                (symbol, market),
            ).fetchone()
            if row is not None:
                inst_id = int(row[0])
                if name:
                    conn.execute("UPDATE instruments SET name=? WHERE id=?", (str(name), inst_id))
                return inst_id

            inst_id = self._next_id(conn, "instruments")
            conn.execute(
                "INSERT INTO instruments(id, symbol, market, name, currency, timezone, active) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (inst_id, symbol, market, str(name or ""), "", "UTC", 1),
            )
            return inst_id
        finally:
            conn.close()

    def _load_first_bar_date(self, symbol: str, market: str) -> Optional[datetime]:
        bars = self.load_daily_bars(symbol=symbol, market=market)
        if bars.empty:
            return None
        return pd.Timestamp(bars.index.min()).to_pydatetime().replace(tzinfo=timezone.utc)

    def _load_last_success_date(self, instrument_id: int) -> Optional[datetime]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT last_success_date FROM sync_state WHERE instrument_id=?",
                (int(instrument_id),),
            ).fetchone()
            if row is None or not row[0]:
                return None
            return datetime.fromisoformat(str(row[0])).replace(tzinfo=timezone.utc)
        except Exception:
            return None
        finally:
            conn.close()

    def _save_sync_state(
        self,
        instrument_id: int,
        last_success_date: Optional[datetime],
        source: Optional[str],
        error: Optional[str],
    ):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM sync_state WHERE instrument_id=?", (int(instrument_id),))
            conn.execute(
                "INSERT INTO sync_state(instrument_id, last_success_date, last_source, last_error, updated_at) VALUES(?, ?, ?, ?, ?)",
                (
                    int(instrument_id),
                    last_success_date.isoformat() if last_success_date else None,
                    str(source or ""),
                    str(error or "") if error else None,
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )
        finally:
            conn.close()

    def sync_symbol_history(
        self,
        symbol: str,
        market: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> SyncReport:
        symbol = self._normalize_symbol_token(symbol)
        market = self._normalize_market_token(market)
        started_at = datetime.now(tz=timezone.utc)
        end = end or started_at

        instrument_id = self._get_or_create_instrument(symbol, market)
        last_success = self._load_last_success_date(instrument_id)
        first_bar_date = self._load_first_bar_date(symbol, market)
        if start is None:
            fetch_start = last_success or datetime(end.year - 5, 1, 1, tzinfo=timezone.utc)
            if last_success:
                fetch_start = max(fetch_start, last_success + pd.Timedelta(days=1))
        else:
            fetch_start = start
            if last_success and first_bar_date is not None and start >= first_bar_date and start <= last_success:
                fetch_start = max(fetch_start, last_success + pd.Timedelta(days=1))

        if fetch_start.date() > end.date():
            return SyncReport(
                symbol=symbol,
                market=market,
                source="cache",
                rows_upserted=0,
                started_at=started_at,
                finished_at=datetime.now(tz=timezone.utc),
                fallback_depth=0,
                stale=last_success is not None,
                error=None,
            )

        request = ProviderRequest(symbol=symbol, market=market, interval="1d", start=fetch_start, end=end)
        is_tw_local_symbol = self._is_tw_local_security_symbol(symbol)
        if market == "US":
            providers = [self.service.us_twelve, self.service.yahoo, self.service.us_stooq]
        elif market == "OTC":
            if is_tw_local_symbol:
                providers = []
                fugle_rest = getattr(self.service, "tw_fugle_rest", None)
                if fugle_rest is not None and getattr(fugle_rest, "api_key", None):
                    providers.append(fugle_rest)
                providers.extend([self.service.tw_tpex, self.service.tw_openapi, self.service.yahoo])
            else:
                providers = [self.service.yahoo]
        else:
            if is_tw_local_symbol:
                providers = []
                fugle_rest = getattr(self.service, "tw_fugle_rest", None)
                if fugle_rest is not None and getattr(fugle_rest, "api_key", None):
                    providers.append(fugle_rest)
                providers.extend([self.service.tw_openapi, self.service.yahoo])
            else:
                providers = [self.service.yahoo]

        source = "unknown"
        fallback_depth = 0
        stale = False
        try:
            snap = self.service._try_ohlcv_chain(providers, request)  # noqa: SLF001
            source = snap.source
            fallback_depth = max(0, providers.index(next(p for p in providers if p.name == source)))
            df = self._normalize_daily_bars_frame(snap.df)
            df = df[(df.index >= fetch_start) & (df.index <= end)]
            if df.empty:
                self._save_sync_state(instrument_id, last_success, source, None)
                return SyncReport(
                    symbol=symbol,
                    market=market,
                    source=source,
                    rows_upserted=0,
                    started_at=started_at,
                    finished_at=datetime.now(tz=timezone.utc),
                    fallback_depth=fallback_depth,
                    stale=last_success is not None,
                )
            df["source"] = source
            df["fetched_at"] = datetime.now(tz=timezone.utc).isoformat()
        except Exception as exc:
            if last_success is not None and fetch_start.date() >= end.date():
                self._save_sync_state(instrument_id, last_success, source, None)
                return SyncReport(
                    symbol=symbol,
                    market=market,
                    source=source,
                    rows_upserted=0,
                    started_at=started_at,
                    finished_at=datetime.now(tz=timezone.utc),
                    fallback_depth=fallback_depth,
                    stale=True,
                    error=None,
                )
            stale = last_success is not None
            self._save_sync_state(instrument_id, last_success, source, str(exc))
            return SyncReport(
                symbol=symbol,
                market=market,
                source=source,
                rows_upserted=0,
                started_at=started_at,
                finished_at=datetime.now(tz=timezone.utc),
                fallback_depth=fallback_depth,
                stale=stale,
                error=str(exc),
            )

        rows_upserted = self._upsert_daily_bars(symbol=symbol, market=market, bars=df)
        last_synced = pd.Timestamp(df.index.max()).to_pydatetime().replace(tzinfo=timezone.utc)
        self._save_sync_state(instrument_id, last_synced, source, None)
        return SyncReport(
            symbol=symbol,
            market=market,
            source=source,
            rows_upserted=rows_upserted,
            started_at=started_at,
            finished_at=datetime.now(tz=timezone.utc),
            fallback_depth=fallback_depth,
            stale=stale,
        )

    def load_daily_bars(
        self,
        symbol: str,
        market: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        symbol = self._normalize_symbol_token(symbol)
        market = self._normalize_market_token(market)
        df = self._load_daily_frame_raw(symbol, market)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "adj_close", "source"])
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        for col in ["open", "high", "low", "close", "volume", "adj_close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "source" not in df.columns:
            df["source"] = "unknown"
        return df[[c for c in ["open", "high", "low", "close", "volume", "adj_close", "source"] if c in df.columns]]

    def save_intraday_ticks(
        self,
        symbol: str,
        market: str,
        ticks: list[dict[str, object]],
        retain_days: Optional[int] = None,
    ) -> int:
        if not ticks:
            return 0
        rows: dict[str, tuple[float, float, str, str]] = {}
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        for row in ticks:
            ts_value = row.get("ts") or row.get("ts_utc")
            if ts_value is None:
                continue
            ts = pd.Timestamp(ts_value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            price_val = pd.to_numeric(row.get("price"), errors="coerce")
            if pd.isna(price_val):
                continue
            cum_val = pd.to_numeric(row.get("cum_volume", 0.0), errors="coerce")
            rows[ts.isoformat()] = (
                float(price_val),
                0.0 if pd.isna(cum_val) else float(cum_val),
                str(row.get("source", "unknown") or "unknown"),
                str(row.get("fetched_at", now_iso) or now_iso),
            )

        if not rows:
            return 0

        frame = pd.DataFrame(
            {
                "ts_utc": list(rows.keys()),
                "price": [v[0] for v in rows.values()],
                "cum_volume": [v[1] for v in rows.values()],
                "source": [v[2] for v in rows.values()],
                "fetched_at": [v[3] for v in rows.values()],
            }
        )
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["ts_utc"]).set_index("ts_utc").sort_index()

        retain = self.resolve_intraday_retain_days(self.intraday_retain_days if retain_days is None else retain_days)
        self.intraday_retain_days = retain
        return self._upsert_intraday_ticks(symbol=symbol, market=market, ticks=frame)

    def load_intraday_ticks(
        self,
        symbol: str,
        market: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        symbol = self._normalize_symbol_token(symbol)
        market = self._normalize_market_token(market)
        df = self._load_intraday_frame_raw(symbol, market)
        if df.empty:
            return pd.DataFrame(columns=["price", "cum_volume", "source"])
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts_utc"]).set_index("ts_utc").sort_index()
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        for col in ["price", "cum_volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "source" not in df.columns:
            df["source"] = "unknown"
        return df[[c for c in ["price", "cum_volume", "source"] if c in df.columns]]

    def upsert_symbol_metadata(self, rows: list[Dict[str, object]]) -> int:
        payload: list[tuple[str, str, str, str, str, str, str, str, str]] = []
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        normalized: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = self._normalize_symbol_token(row.get("symbol"))
            market = self._normalize_market_token(row.get("market"))
            if not symbol or not market:
                continue
            normalized.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "name": self._normalize_text(row.get("name")),
                    "exchange": self._normalize_text(row.get("exchange")),
                    "industry": self._normalize_text(row.get("industry")),
                    "asset_type": self._normalize_text(row.get("asset_type")),
                    "currency": self._normalize_text(row.get("currency")),
                    "source": self._normalize_text(row.get("source")),
                }
            )

        if not normalized:
            return 0

        existing = self.load_symbol_metadata([r["symbol"] for r in normalized], normalized[0]["market"]) if normalized else {}
        for row in normalized:
            base = existing.get(row["symbol"], {})
            payload.append(
                (
                    row["symbol"],
                    row["market"],
                    row["name"] or str(base.get("name", "") or ""),
                    row["exchange"] or str(base.get("exchange", "") or ""),
                    row["industry"] or str(base.get("industry", "") or ""),
                    row["asset_type"] or str(base.get("asset_type", "") or ""),
                    row["currency"] or str(base.get("currency", "") or ""),
                    row["source"] or str(base.get("source", "") or ""),
                    now_iso,
                )
            )

        conn = self._connect()
        try:
            for p in payload:
                conn.execute("DELETE FROM symbol_metadata WHERE symbol=? AND market=?", (p[0], p[1]))
                conn.execute(
                    "INSERT INTO symbol_metadata(symbol, market, name, exchange, industry, asset_type, currency, source, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    p,
                )
        finally:
            conn.close()
        return len(payload)

    def load_symbol_metadata(self, symbols: list[str], market: str) -> dict[str, dict[str, object]]:
        normalized_symbols: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            token = self._normalize_symbol_token(symbol)
            if not token or token in seen:
                continue
            normalized_symbols.append(token)
            seen.add(token)
        market_token = self._normalize_market_token(market)
        if not normalized_symbols or not market_token:
            return {}

        placeholders = ", ".join(["?"] * len(normalized_symbols))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT symbol, market, name, exchange, industry, asset_type, currency, source, updated_at FROM symbol_metadata WHERE market=? AND symbol IN ({placeholders})",
                [market_token, *normalized_symbols],
            ).fetchall()
        finally:
            conn.close()

        out: dict[str, dict[str, object]] = {}
        for row in rows:
            symbol = self._normalize_symbol_token(row[0])
            if not symbol:
                continue
            out[symbol] = {
                "symbol": symbol,
                "market": self._normalize_market_token(row[1]),
                "name": self._normalize_text(row[2]),
                "exchange": self._normalize_text(row[3]),
                "industry": self._normalize_text(row[4]),
                "asset_type": self._normalize_text(row[5]),
                "currency": self._normalize_text(row[6]),
                "source": self._normalize_text(row[7]),
                "updated_at": self._parse_iso_datetime(row[8]),
            }
        return out

    def list_symbols(self, market: str, limit: Optional[int] = None) -> list[str]:
        market_token = self._normalize_market_token(market)
        if not market_token:
            return []
        limit_sql = ""
        params: list[object] = [market_token]
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(1, int(limit)))

        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT symbol FROM symbol_metadata WHERE market=? ORDER BY symbol ASC{limit_sql}",
                params,
            ).fetchall()
            symbols = [self._normalize_symbol_token(r[0]) for r in rows if self._normalize_symbol_token(r[0])]
            if symbols:
                return symbols

            rows = conn.execute(
                f"SELECT symbol FROM instruments WHERE market=? ORDER BY symbol ASC{limit_sql}",
                params,
            ).fetchall()
            symbols = [self._normalize_symbol_token(r[0]) for r in rows if self._normalize_symbol_token(r[0])]
            if symbols:
                return symbols
        finally:
            conn.close()

        daily_root = self.parquet_root / "daily_bars" / f"market={market_token}"
        if not daily_root.exists():
            return []
        all_symbols = sorted(
            [p.name.split("symbol=")[-1].upper() for p in daily_root.glob("symbol=*") if p.is_dir()]
        )
        if limit is not None:
            return all_symbols[: max(1, int(limit))]
        return all_symbols

    def start_bootstrap_run(self, scope: str, params: Dict[str, object]) -> str:
        run_id = f"bootstrap:{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        conn = self._connect()
        try:
            rid = self._next_id(conn, "bootstrap_runs")
            conn.execute(
                "INSERT INTO bootstrap_runs(id, run_id, scope, status, started_at, params_json, total_symbols, synced_symbols, failed_symbols, summary_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rid,
                    run_id,
                    self._normalize_text(scope) or "unknown",
                    "running",
                    datetime.now(tz=timezone.utc).isoformat(),
                    json.dumps(params, ensure_ascii=False),
                    0,
                    0,
                    0,
                    "{}",
                ),
            )
        finally:
            conn.close()
        return run_id

    def finish_bootstrap_run(
        self,
        run_id: str,
        *,
        status: str,
        total_symbols: int,
        synced_symbols: int,
        failed_symbols: int,
        summary: Optional[Dict[str, object]] = None,
        error: Optional[str] = None,
    ):
        key = self._normalize_text(run_id)
        if not key:
            return
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE bootstrap_runs SET status=?, finished_at=?, total_symbols=?, synced_symbols=?, failed_symbols=?, summary_json=?, error=? WHERE run_id=?",
                (
                    self._normalize_text(status) or "unknown",
                    datetime.now(tz=timezone.utc).isoformat(),
                    max(0, int(total_symbols)),
                    max(0, int(synced_symbols)),
                    max(0, int(failed_symbols)),
                    json.dumps(summary or {}, ensure_ascii=False),
                    self._normalize_text(error) or None,
                    key,
                ),
            )
        finally:
            conn.close()

    def load_latest_bootstrap_run(self) -> Optional[BootstrapRun]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT run_id, scope, status, started_at, finished_at, total_symbols, synced_symbols, failed_symbols, params_json, summary_json, error FROM bootstrap_runs ORDER BY started_at DESC, id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None

        params_obj: Dict[str, Any] = {}
        summary_obj: Dict[str, Any] = {}
        try:
            loaded = json.loads(str(row[8] or "{}"))
            if isinstance(loaded, dict):
                params_obj = loaded
        except Exception:
            params_obj = {}
        try:
            loaded = json.loads(str(row[9] or "{}"))
            if isinstance(loaded, dict):
                summary_obj = loaded
        except Exception:
            summary_obj = {}

        started_at = self._parse_iso_datetime(row[3]) or datetime.now(tz=timezone.utc)
        finished_at = self._parse_iso_datetime(row[4])
        return BootstrapRun(
            run_id=self._normalize_text(row[0]),
            scope=self._normalize_text(row[1]),
            status=self._normalize_text(row[2]),
            started_at=started_at,
            finished_at=finished_at,
            total_symbols=max(0, int(row[5] or 0)),
            synced_symbols=max(0, int(row[6] or 0)),
            failed_symbols=max(0, int(row[7] or 0)),
            params=params_obj,
            summary=summary_obj,
            error=self._normalize_text(row[10]) or None,
        )

    def save_backtest_run(
        self,
        symbol: str,
        market: str,
        strategy: str,
        params: Dict[str, object],
        cost: Dict[str, object],
        result: Dict[str, object],
    ) -> int:
        conn = self._connect()
        try:
            rid = self._next_id(conn, "backtest_runs")
            conn.execute(
                "INSERT INTO backtest_runs(id, created_at, symbol, market, strategy, params_json, cost_json, result_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rid,
                    datetime.now(tz=timezone.utc).isoformat(),
                    symbol,
                    market,
                    strategy,
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(cost, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
            return int(rid)
        finally:
            conn.close()

    def save_backtest_replay_run(
        self,
        run_key: str,
        params: Dict[str, object],
        payload: Dict[str, object],
    ) -> int:
        key = str(run_key or "").strip()
        if not key:
            raise ValueError("run_key is required")
        conn = self._connect()
        try:
            rid = self._next_id(conn, "backtest_replay_runs")
            conn.execute(
                "INSERT INTO backtest_replay_runs(id, run_key, created_at, params_json, payload_json) VALUES(?, ?, ?, ?, ?)",
                (
                    rid,
                    key,
                    datetime.now(tz=timezone.utc).isoformat(),
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            return int(rid)
        finally:
            conn.close()

    def load_latest_backtest_replay_run(self, run_key: str) -> Optional[BacktestReplayRun]:
        key = str(run_key or "").strip()
        if not key:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT run_key, created_at, params_json, payload_json FROM backtest_replay_runs WHERE run_key=? ORDER BY created_at DESC, id DESC LIMIT 1",
                (key,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None

        created_at = self._parse_iso_datetime(row[1]) or datetime.now(tz=timezone.utc)
        params: Dict[str, object] = {}
        payload: Dict[str, object] = {}
        try:
            obj = json.loads(str(row[2] or "{}"))
            if isinstance(obj, dict):
                params = obj
        except Exception:
            params = {}
        try:
            obj = json.loads(str(row[3] or "{}"))
            if isinstance(obj, dict):
                payload = obj
        except Exception:
            payload = {}

        return BacktestReplayRun(
            run_key=str(row[0] or ""),
            params=params,
            payload=payload,
            created_at=created_at,
        )

    def save_universe_snapshot(
        self,
        universe_id: str,
        symbols: list[str],
        source: str,
    ):
        norm_symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        payload = json.dumps(norm_symbols, ensure_ascii=False)
        now = datetime.now(tz=timezone.utc).isoformat()
        key = str(universe_id or "").strip().upper()
        conn = self._connect()
        try:
            conn.execute("DELETE FROM universe_snapshots WHERE universe_id=?", (key,))
            conn.execute(
                "INSERT INTO universe_snapshots(universe_id, symbols_json, source, fetched_at, updated_at) VALUES(?, ?, ?, ?, ?)",
                (key, payload, source, now, now),
            )
        finally:
            conn.close()

    def load_universe_snapshot(self, universe_id: str) -> Optional[UniverseSnapshot]:
        key = str(universe_id or "").strip().upper()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT universe_id, symbols_json, source, fetched_at FROM universe_snapshots WHERE universe_id=?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None

        try:
            symbols_obj = json.loads(str(row[1] or "[]"))
        except Exception:
            symbols_obj = []
        symbols = [str(s).strip().upper() for s in symbols_obj if str(s).strip()]
        fetched = self._parse_iso_datetime(row[3]) or datetime.now(tz=timezone.utc)

        return UniverseSnapshot(
            universe_id=str(row[0]),
            symbols=symbols,
            source=str(row[2] or "unknown"),
            fetched_at=fetched,
        )

    def save_heatmap_run(self, universe_id: str, payload: Dict[str, object]) -> int:
        key = str(universe_id or "").strip().upper()
        conn = self._connect()
        try:
            rid = self._next_id(conn, "heatmap_runs")
            conn.execute(
                "INSERT INTO heatmap_runs(id, universe_id, created_at, payload_json) VALUES(?, ?, ?, ?)",
                (rid, key, datetime.now(tz=timezone.utc).isoformat(), json.dumps(payload, ensure_ascii=False)),
            )
            return int(rid)
        finally:
            conn.close()

    def load_latest_heatmap_run(self, universe_id: str) -> Optional[HeatmapRun]:
        universe_key = str(universe_id or "").strip().upper()
        if not universe_key:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT universe_id, created_at, payload_json FROM heatmap_runs WHERE universe_id=? ORDER BY created_at DESC, id DESC LIMIT 1",
                (universe_key,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None

        created_at = self._parse_iso_datetime(row[1]) or datetime.now(tz=timezone.utc)
        payload: Dict[str, object] = {}
        try:
            obj = json.loads(str(row[2] or "{}"))
            if isinstance(obj, dict):
                payload = obj
        except Exception:
            payload = {}

        return HeatmapRun(
            universe_id=str(row[0]),
            payload=payload,
            created_at=created_at,
        )

    def upsert_heatmap_hub_entry(
        self,
        *,
        etf_code: str,
        etf_name: str,
        opened: bool = False,
        pin_as_card: Optional[bool] = None,
    ) -> None:
        code = self._normalize_symbol_token(etf_code)
        if not code:
            return
        name = self._normalize_text(etf_name) or code
        now = datetime.now(tz=timezone.utc).isoformat()
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT etf_name, pin_as_card, open_count, last_opened_at, created_at
                FROM heatmap_hub_entries
                WHERE etf_code=?
                """,
                (code,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO heatmap_hub_entries(
                        etf_code, etf_name, pin_as_card, open_count, last_opened_at, created_at, updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        code,
                        name,
                        1 if bool(pin_as_card) else 0,
                        1 if bool(opened) else 0,
                        now if bool(opened) else "",
                        now,
                        now,
                    ),
                )
                return

            existing_name = self._normalize_text(row[0]) or code
            existing_pin = 1 if int(row[1] or 0) else 0
            existing_open = int(row[2] or 0)
            existing_last = self._normalize_text(row[3])
            created_at = self._normalize_text(row[4]) or now
            next_pin = existing_pin if pin_as_card is None else (1 if bool(pin_as_card) else 0)
            next_open = existing_open + (1 if bool(opened) else 0)
            next_last = now if bool(opened) else existing_last
            next_name = name or existing_name
            conn.execute(
                """
                UPDATE heatmap_hub_entries
                SET etf_name=?, pin_as_card=?, open_count=?, last_opened_at=?, created_at=?, updated_at=?
                WHERE etf_code=?
                """,
                (next_name, next_pin, next_open, next_last, created_at, now, code),
            )
        finally:
            conn.close()

    def list_heatmap_hub_entries(self, *, pinned_only: bool = False) -> list[HeatmapHubEntry]:
        where_sql = "WHERE pin_as_card=1" if bool(pinned_only) else ""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT etf_code, etf_name, pin_as_card, open_count, last_opened_at, created_at, updated_at
                FROM heatmap_hub_entries
                {where_sql}
                ORDER BY pin_as_card DESC, last_opened_at DESC, updated_at DESC, etf_code ASC
                """
            ).fetchall()
        finally:
            conn.close()
        out: list[HeatmapHubEntry] = []
        now = datetime.now(tz=timezone.utc)
        for row in rows:
            last_opened_at = self._parse_iso_datetime(row[4]) or now
            created_at = self._parse_iso_datetime(row[5]) or now
            updated_at = self._parse_iso_datetime(row[6]) or now
            out.append(
                HeatmapHubEntry(
                    etf_code=self._normalize_symbol_token(row[0]),
                    etf_name=self._normalize_text(row[1]) or self._normalize_symbol_token(row[0]),
                    pin_as_card=bool(int(row[2] or 0)),
                    open_count=max(0, int(row[3] or 0)),
                    last_opened_at=last_opened_at,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return out

    def set_heatmap_hub_pin(self, *, etf_code: str, pin_as_card: bool) -> bool:
        code = self._normalize_symbol_token(etf_code)
        if not code:
            return False
        now = datetime.now(tz=timezone.utc).isoformat()
        conn = self._connect()
        try:
            row = conn.execute("SELECT 1 FROM heatmap_hub_entries WHERE etf_code=? LIMIT 1", (code,)).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE heatmap_hub_entries SET pin_as_card=?, updated_at=? WHERE etf_code=?",
                (1 if bool(pin_as_card) else 0, now, code),
            )
            return True
        finally:
            conn.close()

    def save_rotation_run(
        self,
        universe_id: str,
        run_key: str,
        params: Dict[str, object],
        payload: Dict[str, object],
    ) -> int:
        universe_key = str(universe_id or "").strip().upper()
        if not universe_key:
            raise ValueError("universe_id is required")
        conn = self._connect()
        try:
            rid = self._next_id(conn, "rotation_runs")
            conn.execute(
                "INSERT INTO rotation_runs(id, universe_id, run_key, created_at, params_json, payload_json) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    rid,
                    universe_key,
                    str(run_key or ""),
                    datetime.now(tz=timezone.utc).isoformat(),
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            return int(rid)
        finally:
            conn.close()

    def load_latest_rotation_run(self, universe_id: str) -> Optional[RotationRun]:
        universe_key = str(universe_id or "").strip().upper()
        if not universe_key:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT universe_id, run_key, created_at, params_json, payload_json FROM rotation_runs WHERE universe_id=? ORDER BY created_at DESC, id DESC LIMIT 1",
                (universe_key,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None

        created_at = self._parse_iso_datetime(row[2]) or datetime.now(tz=timezone.utc)
        params: Dict[str, object] = {}
        payload: Dict[str, object] = {}
        try:
            obj = json.loads(str(row[3] or "{}"))
            if isinstance(obj, dict):
                params = obj
        except Exception:
            params = {}
        try:
            obj = json.loads(str(row[4] or "{}"))
            if isinstance(obj, dict):
                payload = obj
        except Exception:
            payload = {}

        return RotationRun(
            universe_id=str(row[0]),
            run_key=str(row[1] or ""),
            params=params,
            payload=payload,
            created_at=created_at,
        )


__all__ = ["DuckHistoryStore"]
