from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from providers.base import ProviderRequest
from services.market_data_service import MarketDataService
from storage.history_store import (
    BacktestReplayRun,
    BootstrapRun,
    ClientVisit,
    HeatmapHubEntry,
    HeatmapRun,
    MessageBoardEntry,
    NotebookEntry,
    RotationRun,
    SuperExportRun,
    SyncReport,
    UniverseSnapshot,
)

DEFAULT_DB_FILENAME = "market_history.duckdb"
DEFAULT_LEGACY_SQLITE_FILENAME = "market_history.sqlite3"
DEFAULT_INTRADAY_RETAIN_DAYS = 365 * 3
DEFAULT_WRITEBACK_WORKERS = 2
MAX_WRITEBACK_WORKERS = 8
DEFAULT_DAILY_DELTA_COMPACT_THRESHOLD = 24
DEFAULT_INTRADAY_DELTA_COMPACT_THRESHOLD = 48
ICLOUD_DOCS_ROOT = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
DEFAULT_ICLOUD_DB_PATH = ICLOUD_DOCS_ROOT / "codexapp" / DEFAULT_DB_FILENAME
DEFAULT_LEGACY_ICLOUD_DB_PATH = ICLOUD_DOCS_ROOT / "codexapp" / DEFAULT_LEGACY_SQLITE_FILENAME


class _LockedDuckConnection:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        lock: threading.RLock,
    ):
        self._conn = conn
        self._lock = lock
        self._closed = False

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        finally:
            self._lock.release()


class DuckHistoryStore:
    backend_name = "duckdb"

    @staticmethod
    def _compute_daily_vwap_series(frame: pd.DataFrame) -> pd.Series:
        if frame is None or not isinstance(frame, pd.DataFrame):
            return pd.Series(dtype=float)
        if frame.empty:
            return pd.Series(index=frame.index, dtype=float)

        def _numeric_col(name: str) -> pd.Series:
            if name not in frame.columns:
                return pd.Series(index=frame.index, dtype=float)
            return pd.to_numeric(frame[name], errors="coerce")

        high = _numeric_col("high")
        low = _numeric_col("low")
        close = _numeric_col("close")
        volume = _numeric_col("volume").fillna(0.0)
        typical = (high + low + close) / 3.0
        cum_volume = volume.cumsum().replace(0, float("nan"))
        return (typical * volume).cumsum() / cum_volume

    @classmethod
    def _ensure_daily_vwap_column(cls, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return frame
        out = frame.copy()
        computed = cls._compute_daily_vwap_series(out)
        if "vwap" in out.columns:
            existing = pd.to_numeric(out["vwap"], errors="coerce")
            out["vwap"] = existing.where(existing.notna(), computed)
        else:
            out["vwap"] = computed
        return out

    def __init__(
        self,
        db_path: str | None = None,
        parquet_root: str | None = None,
        service: MarketDataService | None = None,
        intraday_retain_days: int | None = None,
        legacy_sqlite_path: str | None = None,
        auto_migrate_legacy_sqlite: bool = False,
    ):
        self.db_path = self.resolve_history_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_root = self.resolve_parquet_root(parquet_root, db_path=self.db_path)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self.intraday_retain_days = self.resolve_intraday_retain_days(intraday_retain_days)
        self.service = service or MarketDataService()
        self._duck_conn_lock = threading.RLock()
        self._writeback_lock = threading.Lock()
        self._writeback_inflight: set[tuple[str, str]] = set()
        self._writeback_pending: dict[tuple[str, str], tuple[pd.DataFrame, str]] = {}
        self._writeback_executor: ThreadPoolExecutor | None = None
        self._writeback_workers = self.resolve_writeback_workers()
        self._daily_delta_compact_threshold = self.resolve_daily_delta_compact_threshold()
        self._intraday_delta_compact_threshold = self.resolve_intraday_delta_compact_threshold()
        self._init_db()

        set_metadata_store = getattr(self.service, "set_metadata_store", None)
        if callable(set_metadata_store):
            try:
                set_metadata_store(self)
            except Exception:
                pass

        if auto_migrate_legacy_sqlite:
            src = (
                Path(legacy_sqlite_path).expanduser()
                if legacy_sqlite_path
                else self._default_legacy_sqlite_path()
            )
            self._maybe_migrate_from_legacy_sqlite(src)

    @staticmethod
    def resolve_history_db_path(db_path: str | None = None) -> Path:
        if db_path:
            return Path(db_path).expanduser()
        if ICLOUD_DOCS_ROOT.exists():
            return DEFAULT_ICLOUD_DB_PATH
        return Path(DEFAULT_DB_FILENAME)

    @staticmethod
    def resolve_parquet_root(
        parquet_root: str | None = None, *, db_path: Path | None = None
    ) -> Path:
        if parquet_root:
            return Path(parquet_root).expanduser()
        base = db_path if db_path is not None else DuckHistoryStore.resolve_history_db_path()
        return base.parent / "parquet"

    @staticmethod
    def resolve_intraday_retain_days(days: int | None = None) -> int:
        if days is None:
            return DEFAULT_INTRADAY_RETAIN_DAYS
        try:
            return max(1, int(days))
        except Exception:
            return DEFAULT_INTRADAY_RETAIN_DAYS

    @staticmethod
    def resolve_writeback_workers(workers: int | None = None) -> int:
        if workers is None:
            raw = str(os.getenv("REALTIME0052_DUCK_WRITEBACK_WORKERS", "")).strip()
            if raw:
                try:
                    workers = int(raw)
                except Exception:
                    workers = DEFAULT_WRITEBACK_WORKERS
            else:
                workers = DEFAULT_WRITEBACK_WORKERS
        try:
            return max(1, min(int(workers), MAX_WRITEBACK_WORKERS))
        except Exception:
            return DEFAULT_WRITEBACK_WORKERS

    @staticmethod
    def resolve_daily_delta_compact_threshold(value: int | None = None) -> int:
        if value is None:
            raw = str(os.getenv("REALTIME0052_DAILY_DELTA_COMPACT_THRESHOLD", "")).strip()
            if raw:
                try:
                    value = int(raw)
                except Exception:
                    value = DEFAULT_DAILY_DELTA_COMPACT_THRESHOLD
            else:
                value = DEFAULT_DAILY_DELTA_COMPACT_THRESHOLD
        try:
            return max(4, int(value))
        except Exception:
            return DEFAULT_DAILY_DELTA_COMPACT_THRESHOLD

    @staticmethod
    def resolve_intraday_delta_compact_threshold(value: int | None = None) -> int:
        if value is None:
            raw = str(os.getenv("REALTIME0052_INTRADAY_DELTA_COMPACT_THRESHOLD", "")).strip()
            if raw:
                try:
                    value = int(raw)
                except Exception:
                    value = DEFAULT_INTRADAY_DELTA_COMPACT_THRESHOLD
            else:
                value = DEFAULT_INTRADAY_DELTA_COMPACT_THRESHOLD
        try:
            return max(8, int(value))
        except Exception:
            return DEFAULT_INTRADAY_DELTA_COMPACT_THRESHOLD

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
    def _normalize_raw_json_text(value: object) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return ""
        except Exception:
            pass
        if isinstance(value, str):
            return value.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value).strip()

    @staticmethod
    def _is_tw_local_security_symbol(symbol: str) -> bool:
        token = str(symbol or "").strip().upper()
        return bool(re.fullmatch(r"\d{4,6}[A-Z]?", token))

    @staticmethod
    def _parse_iso_datetime(value: object) -> datetime | None:
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

        if is_memory and hasattr(self, "_memory_conn") and self._memory_conn:
            yield self._memory_conn
            return

        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception as rollback_exc:
                # DuckDB may report "no transaction is active" on rollback when
                # the failing statement already ended the transaction context.
                # Do not mask the original failure with this secondary error.
                rollback_msg = str(rollback_exc or "").strip().lower()
                if "no transaction is active" not in rollback_msg:
                    pass
            raise
        finally:
            if not is_memory:
                conn.close()
            else:
                self._memory_conn = conn

    def _connect(self) -> duckdb.DuckDBPyConnection:
        self._duck_conn_lock.acquire()
        try:
            conn = duckdb.connect(str(self.db_path))
            conn.execute("PRAGMA threads=4")
            return _LockedDuckConnection(conn, self._duck_conn_lock)
        except Exception:
            self._duck_conn_lock.release()
            raise

    def _next_id(self, conn: duckdb.DuckDBPyConnection, table: str) -> int:
        row = conn.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table}").fetchone()
        return int(row[0] or 1)

    def _table_has_column(self, conn: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        target = str(column or "").strip().lower()
        return any(str(row[1] or "").strip().lower() == target for row in rows)

    def _ensure_notebook_schema(self, conn: duckdb.DuckDBPyConnection) -> None:
        if not self._table_has_column(conn, "notebook_entries", "title"):
            conn.execute("ALTER TABLE notebook_entries ADD COLUMN title VARCHAR")
            conn.execute(
                "UPDATE notebook_entries SET title='未命名筆記' WHERE COALESCE(title, '') = ''"
            )

    def _init_db(self):
        with self._connect_ctx() as conn:
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
                CREATE TABLE IF NOT EXISTS tw_etf_super_export_runs (
                    id BIGINT,
                    run_id VARCHAR,
                    created_at VARCHAR,
                    ytd_start VARCHAR,
                    ytd_end VARCHAR,
                    compare_start VARCHAR,
                    compare_end VARCHAR,
                    trade_date_anchor VARCHAR,
                    output_path VARCHAR,
                    row_count BIGINT,
                    column_count BIGINT,
                    payload_json VARCHAR,
                    UNIQUE(run_id)
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
                CREATE TABLE IF NOT EXISTS client_visits (
                    session_id VARCHAR,
                    ip_address VARCHAR,
                    forwarded_for VARCHAR,
                    user_agent VARCHAR,
                    last_page VARCHAR,
                    visit_count BIGINT DEFAULT 0,
                    first_seen_at VARCHAR,
                    last_seen_at VARCHAR,
                    UNIQUE(session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_board_entries (
                    id BIGINT,
                    message_id VARCHAR,
                    parent_message_id VARCHAR,
                    author_name VARCHAR,
                    body VARCHAR,
                    ip_address VARCHAR,
                    user_agent VARCHAR,
                    created_at VARCHAR,
                    updated_at VARCHAR,
                    UNIQUE(message_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notebook_entries (
                    id BIGINT,
                    note_id VARCHAR,
                    title VARCHAR,
                    body VARCHAR,
                    created_at VARCHAR,
                    updated_at VARCHAR,
                    UNIQUE(note_id)
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
                CREATE TABLE IF NOT EXISTS tw_etf_aum_history (
                    etf_code VARCHAR,
                    etf_name VARCHAR,
                    trade_date VARCHAR,
                    aum_billion DOUBLE,
                    updated_at VARCHAR,
                    UNIQUE(etf_code, trade_date)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tw_etf_daily_market (
                    trade_date VARCHAR,
                    etf_code VARCHAR,
                    etf_name VARCHAR,
                    trade_value DOUBLE,
                    trade_volume DOUBLE,
                    trade_count BIGINT,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    change DOUBLE,
                    source VARCHAR,
                    fetched_at VARCHAR,
                    UNIQUE(etf_code, trade_date)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tw_etf_mis_daily (
                    trade_date VARCHAR,
                    etf_code VARCHAR,
                    etf_name VARCHAR,
                    issued_units DOUBLE,
                    creation_redemption_diff DOUBLE,
                    market_price DOUBLE,
                    estimated_nav DOUBLE,
                    premium_discount_pct DOUBLE,
                    previous_nav DOUBLE,
                    reference_url VARCHAR,
                    updated_at VARCHAR,
                    source VARCHAR,
                    fetched_at VARCHAR,
                    UNIQUE(etf_code, trade_date)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id BIGINT,
                    dataset_key VARCHAR,
                    market VARCHAR,
                    symbol VARCHAR,
                    interval VARCHAR,
                    source VARCHAR,
                    as_of VARCHAR,
                    fetched_at VARCHAR,
                    freshness_sec BIGINT,
                    quality_score DOUBLE,
                    stale INTEGER DEFAULT 0,
                    payload_json VARCHAR,
                    raw_json VARCHAR,
                    UNIQUE(dataset_key, market, symbol, interval, as_of)
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_instruments_market_symbol ON instruments(market, symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sync_state_instrument_id ON sync_state(instrument_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sync_state_updated_at ON sync_state(updated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbol_metadata_market_symbol ON symbol_metadata(market, symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bootstrap_runs_started_at ON bootstrap_runs(started_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_backtest_replay_runs_key_created_at ON backtest_replay_runs(run_key, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tw_etf_super_export_runs_created_at ON tw_etf_super_export_runs(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_heatmap_runs_universe_created_at ON heatmap_runs(universe_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_snapshots_lookup ON market_snapshots(dataset_key, market, symbol, interval, fetched_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_snapshots_as_of ON market_snapshots(as_of)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_heatmap_hub_entries_pin_updated ON heatmap_hub_entries(pin_as_card, updated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_client_visits_last_seen ON client_visits(last_seen_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_message_board_parent_created ON message_board_entries(parent_message_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notebook_entries_updated_at ON notebook_entries(updated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rotation_runs_universe_created_at ON rotation_runs(universe_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tw_etf_aum_history_trade_date ON tw_etf_aum_history(trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tw_etf_aum_history_code_date ON tw_etf_aum_history(etf_code, trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tw_etf_daily_market_trade_date ON tw_etf_daily_market(trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tw_etf_daily_market_code_date ON tw_etf_daily_market(etf_code, trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tw_etf_mis_daily_trade_date ON tw_etf_mis_daily(trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tw_etf_mis_daily_code_date ON tw_etf_mis_daily(etf_code, trade_date)"
            )
            self._ensure_notebook_schema(conn)

    def _default_legacy_sqlite_path(self) -> Path:
        env = str(os.getenv("REALTIME0052_DB_PATH", "")).strip()
        if env:
            return Path(env).expanduser()
        if (
            str(self.db_path).startswith(str(ICLOUD_DOCS_ROOT))
            and DEFAULT_LEGACY_ICLOUD_DB_PATH.exists()
        ):
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
                "SELECT COUNT(*) FROM tw_etf_aum_history",
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
                rows = src.execute(
                    "SELECT id, symbol, market, name, currency, timezone, active FROM instruments"
                ).fetchall()
                for _, symbol, market, name, currency, tz, active in rows:
                    inst_id = self._get_or_create_instrument(
                        str(symbol), str(market), name=str(name or "")
                    )
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
                "client_visits",
                "message_board_entries",
                "rotation_runs",
                "tw_etf_aum_history",
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
        return self._daily_symbol_dir(symbol, market) / "bars.parquet"

    def _intraday_symbol_path(self, symbol: str, market: str) -> Path:
        return self._intraday_symbol_dir(symbol, market) / "ticks.parquet"

    def _daily_symbol_dir(self, symbol: str, market: str) -> Path:
        return (
            self.parquet_root
            / "daily_bars"
            / f"market={self._normalize_market_token(market)}"
            / f"symbol={self._normalize_symbol_token(symbol)}"
        )

    def _intraday_symbol_dir(self, symbol: str, market: str) -> Path:
        return (
            self.parquet_root
            / "intraday_ticks"
            / f"market={self._normalize_market_token(market)}"
            / f"symbol={self._normalize_symbol_token(symbol)}"
        )

    def _daily_delta_dir(self, symbol: str, market: str) -> Path:
        return self._daily_symbol_dir(symbol, market) / "_delta"

    def _intraday_delta_dir(self, symbol: str, market: str) -> Path:
        return self._intraday_symbol_dir(symbol, market) / "_delta"

    @staticmethod
    def _next_delta_parquet_path(delta_dir: Path, *, prefix: str) -> Path:
        stamp = time.time_ns()
        return delta_dir / f"{prefix}_{stamp}.parquet"

    @staticmethod
    def _duck_read_parquet_expr(paths: list[Path]) -> str:
        quoted = [str(path.resolve()).replace("'", "''") for path in paths]
        if len(quoted) == 1:
            return f"read_parquet('{quoted[0]}', union_by_name=true)"
        items = ", ".join(f"'{item}'" for item in quoted)
        return f"read_parquet([{items}], union_by_name=true)"

    def _daily_parquet_sources(self, symbol: str, market: str) -> list[Path]:
        base = self._daily_symbol_path(symbol, market)
        delta_dir = self._daily_delta_dir(symbol, market)
        out: list[Path] = []
        if base.exists():
            out.append(base)
        if delta_dir.exists():
            out.extend(sorted([p for p in delta_dir.glob("*.parquet") if p.is_file()]))
        return out

    def _intraday_parquet_sources(self, symbol: str, market: str) -> list[Path]:
        base = self._intraday_symbol_path(symbol, market)
        delta_dir = self._intraday_delta_dir(symbol, market)
        out: list[Path] = []
        if base.exists():
            out.append(base)
        if delta_dir.exists():
            out.extend(sorted([p for p in delta_dir.glob("*.parquet") if p.is_file()]))
        return out

    @staticmethod
    def _normalize_daily_bars_frame(df: pd.DataFrame) -> pd.DataFrame:
        base_cols = ["open", "high", "low", "close", "volume", "vwap"]
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=base_cols)

        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            renamed: list[str] = []
            for col in out.columns:
                parts = [str(part).strip().lower() for part in col if str(part).strip()]
                candidate = ""
                for item in reversed(parts):
                    if item in {
                        "open",
                        "high",
                        "low",
                        "close",
                        "adj close",
                        "adj_close",
                        "volume",
                        "price",
                    }:
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
        computed_vwap = DuckHistoryStore._compute_daily_vwap_series(norm)
        if "vwap" in out.columns:
            supplied_vwap = _extract_numeric_col("vwap")
            norm["vwap"] = supplied_vwap.where(supplied_vwap.notna(), computed_vwap)
        else:
            norm["vwap"] = computed_vwap
        if "adj_close" in out.columns:
            norm["adj_close"] = _extract_numeric_col("adj_close")
        if "asof" in out.columns:
            norm["asof"] = pd.to_datetime(out["asof"], utc=True, errors="coerce")
        if "quality_score" in out.columns:
            norm["quality_score"] = pd.to_numeric(out["quality_score"], errors="coerce")
        if "raw_json" in out.columns:
            norm["raw_json"] = out["raw_json"].map(DuckHistoryStore._normalize_raw_json_text)

        idx = pd.to_datetime(norm.index, utc=True, errors="coerce")
        norm.index = idx
        norm = norm[~norm.index.isna()]
        keep_cols = [
            c
            for c in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "adj_close",
                "asof",
                "quality_score",
                "raw_json",
            ]
            if c in norm.columns
        ]
        norm = norm[keep_cols]
        norm = norm.dropna(subset=["open", "high", "low", "close"], how="any").sort_index()
        if "volume" in norm.columns:
            norm["volume"] = pd.to_numeric(norm["volume"], errors="coerce").fillna(0.0)
        return norm

    def _load_daily_frame_raw(self, symbol: str, market: str) -> pd.DataFrame:
        return self._load_daily_frame_window(symbol=symbol, market=market, start=None, end=None)

    def _load_daily_frame_window(
        self,
        symbol: str,
        market: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
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
        sources = self._daily_parquet_sources(symbol, market)
        if not sources:
            return pd.DataFrame(columns=expected)

        read_expr = self._duck_read_parquet_expr(sources)
        where: list[str] = ["rn=1"]
        params: list[object] = []
        if start is not None:
            where.append("d >= CAST(? AS DATE)")
            params.append(pd.Timestamp(start).date().isoformat())
        if end is not None:
            where.append("d <= CAST(? AS DATE)")
            params.append(pd.Timestamp(end).date().isoformat())

        sql = f"""
            WITH raw AS (
                SELECT
                    TRY_CAST(date AS DATE) AS d,
                    TRY_CAST(open AS DOUBLE) AS open,
                    TRY_CAST(high AS DOUBLE) AS high,
                    TRY_CAST(low AS DOUBLE) AS low,
                    TRY_CAST(close AS DOUBLE) AS close,
                    TRY_CAST(volume AS DOUBLE) AS volume,
                    TRY_CAST(vwap AS DOUBLE) AS vwap,
                    TRY_CAST(adj_close AS DOUBLE) AS adj_close,
                    NULLIF(TRIM(CAST(source AS VARCHAR)), '') AS source,
                    CAST(fetched_at AS VARCHAR) AS fetched_at_text,
                    TRY_CAST(fetched_at AS TIMESTAMP) AS fetched_ts,
                    CAST(asof AS VARCHAR) AS asof_text,
                    TRY_CAST(quality_score AS DOUBLE) AS quality_score,
                    CAST(raw_json AS VARCHAR) AS raw_json_text
                FROM {read_expr}
            ),
            ranked AS (
                SELECT
                    d, open, high, low, close, volume, vwap, adj_close,
                    COALESCE(source, 'unknown') AS source,
                    fetched_at_text,
                    asof_text,
                    quality_score,
                    raw_json_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY d
                        ORDER BY fetched_ts DESC NULLS LAST, fetched_at_text DESC NULLS LAST
                    ) AS rn
                FROM raw
                WHERE d IS NOT NULL
            )
            SELECT
                STRFTIME(d, '%Y-%m-%d') AS date,
                open, high, low, close, volume, vwap, adj_close, source,
                COALESCE(fetched_at_text, '') AS fetched_at,
                COALESCE(asof_text, fetched_at_text, '') AS asof,
                quality_score,
                COALESCE(raw_json_text, '') AS raw_json
            FROM ranked
            WHERE {" AND ".join(where)}
            ORDER BY d ASC
        """
        legacy_sql = f"""
            WITH raw AS (
                SELECT
                    TRY_CAST(date AS DATE) AS d,
                    TRY_CAST(open AS DOUBLE) AS open,
                    TRY_CAST(high AS DOUBLE) AS high,
                    TRY_CAST(low AS DOUBLE) AS low,
                    TRY_CAST(close AS DOUBLE) AS close,
                    TRY_CAST(volume AS DOUBLE) AS volume,
                    TRY_CAST(adj_close AS DOUBLE) AS adj_close,
                    NULLIF(TRIM(CAST(source AS VARCHAR)), '') AS source,
                    CAST(fetched_at AS VARCHAR) AS fetched_at_text,
                    TRY_CAST(fetched_at AS TIMESTAMP) AS fetched_ts
                FROM {read_expr}
            ),
            ranked AS (
                SELECT
                    d, open, high, low, close, volume, adj_close,
                    COALESCE(source, 'unknown') AS source,
                    fetched_at_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY d
                        ORDER BY fetched_ts DESC NULLS LAST, fetched_at_text DESC NULLS LAST
                    ) AS rn
                FROM raw
                WHERE d IS NOT NULL
            )
            SELECT
                STRFTIME(d, '%Y-%m-%d') AS date,
                open, high, low, close, volume, CAST(NULL AS DOUBLE) AS vwap, adj_close, source,
                COALESCE(fetched_at_text, '') AS fetched_at,
                COALESCE(fetched_at_text, '') AS asof,
                CAST(NULL AS DOUBLE) AS quality_score,
                '' AS raw_json
            FROM ranked
            WHERE {" AND ".join(where)}
            ORDER BY d ASC
        """
        conn = self._connect()
        try:
            out = conn.execute(sql, params).df()
        except Exception:
            try:
                out = conn.execute(legacy_sql, params).df()
            except Exception:
                return pd.DataFrame(columns=expected)
        finally:
            conn.close()

        if out.empty:
            return pd.DataFrame(columns=expected)
        for col in expected:
            if col not in out.columns:
                out[col] = None
        return out[expected]

    def _upsert_daily_bars(self, symbol: str, market: str, bars: pd.DataFrame):
        if bars is None or bars.empty:
            return 0
        frame = bars.copy().sort_index()
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame[~frame.index.isna()]
        if frame.empty:
            return 0
        frame = self._ensure_daily_vwap_column(frame)

        payload = pd.DataFrame(
            {
                "date": frame.index.tz_convert("UTC").strftime("%Y-%m-%d"),
                "open": pd.to_numeric(frame["open"], errors="coerce"),
                "high": pd.to_numeric(frame["high"], errors="coerce"),
                "low": pd.to_numeric(frame["low"], errors="coerce"),
                "close": pd.to_numeric(frame["close"], errors="coerce"),
                "volume": pd.to_numeric(frame.get("volume", 0.0), errors="coerce").fillna(0.0),
                "vwap": pd.to_numeric(frame.get("vwap"), errors="coerce"),
                "adj_close": pd.to_numeric(frame.get("adj_close"), errors="coerce")
                if "adj_close" in frame.columns
                else None,
                "source": frame.get("source", pd.Series(index=frame.index, data="unknown")).astype(
                    str
                ),
                "fetched_at": frame.get(
                    "fetched_at",
                    pd.Series(index=frame.index, data=datetime.now(tz=timezone.utc).isoformat()),
                ).astype(str),
                "asof": pd.to_datetime(
                    frame.get("asof", pd.Series(index=frame.index, data=frame.index)),
                    utc=True,
                    errors="coerce",
                )
                .astype("datetime64[ns, UTC]")
                .astype(str),
                "quality_score": pd.to_numeric(frame.get("quality_score"), errors="coerce"),
                "raw_json": frame.get(
                    "raw_json",
                    pd.Series(index=frame.index, data=""),
                ).map(self._normalize_raw_json_text),
            }
        )
        payload = payload.dropna(subset=["open", "high", "low", "close"])  # type: ignore[arg-type]
        payload = payload.reset_index(drop=True)
        if payload.empty:
            return 0
        payload["date"] = pd.to_datetime(payload["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        payload = payload.dropna(subset=["date"])  # type: ignore[arg-type]
        payload = payload.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        if payload.empty:
            return 0

        delta_dir = self._daily_delta_dir(symbol, market)
        delta_dir.mkdir(parents=True, exist_ok=True)
        delta_path = self._next_delta_parquet_path(delta_dir, prefix="bars_delta")
        payload.to_parquet(delta_path, index=False)
        self._compact_daily_bars_if_needed(symbol=symbol, market=market)
        return int(len(payload))

    def _compact_daily_bars_if_needed(self, *, symbol: str, market: str) -> None:
        delta_dir = self._daily_delta_dir(symbol, market)
        if not delta_dir.exists():
            return
        delta_files = sorted([p for p in delta_dir.glob("*.parquet") if p.is_file()])
        if len(delta_files) < int(self._daily_delta_compact_threshold):
            return

        sources = self._daily_parquet_sources(symbol, market)
        if not sources:
            return
        read_expr = self._duck_read_parquet_expr(sources)
        sql = f"""
            WITH raw AS (
                SELECT
                    TRY_CAST(date AS DATE) AS d,
                    TRY_CAST(open AS DOUBLE) AS open,
                    TRY_CAST(high AS DOUBLE) AS high,
                    TRY_CAST(low AS DOUBLE) AS low,
                    TRY_CAST(close AS DOUBLE) AS close,
                    TRY_CAST(volume AS DOUBLE) AS volume,
                    TRY_CAST(vwap AS DOUBLE) AS vwap,
                    TRY_CAST(adj_close AS DOUBLE) AS adj_close,
                    NULLIF(TRIM(CAST(source AS VARCHAR)), '') AS source,
                    CAST(fetched_at AS VARCHAR) AS fetched_at_text,
                    TRY_CAST(fetched_at AS TIMESTAMP) AS fetched_ts,
                    CAST(asof AS VARCHAR) AS asof_text,
                    TRY_CAST(quality_score AS DOUBLE) AS quality_score,
                    CAST(raw_json AS VARCHAR) AS raw_json_text
                FROM {read_expr}
            ),
            ranked AS (
                SELECT
                    d, open, high, low, close, volume, vwap, adj_close,
                    COALESCE(source, 'unknown') AS source,
                    fetched_at_text,
                    asof_text,
                    quality_score,
                    raw_json_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY d
                        ORDER BY fetched_ts DESC NULLS LAST, fetched_at_text DESC NULLS LAST
                    ) AS rn
                FROM raw
                WHERE d IS NOT NULL
            )
            SELECT
                STRFTIME(d, '%Y-%m-%d') AS date,
                open, high, low, close, volume, vwap, adj_close, source,
                COALESCE(fetched_at_text, '') AS fetched_at,
                COALESCE(asof_text, fetched_at_text, '') AS asof,
                quality_score,
                COALESCE(raw_json_text, '') AS raw_json
            FROM ranked
            WHERE rn=1
            ORDER BY d ASC
        """
        legacy_sql = f"""
            WITH raw AS (
                SELECT
                    TRY_CAST(date AS DATE) AS d,
                    TRY_CAST(open AS DOUBLE) AS open,
                    TRY_CAST(high AS DOUBLE) AS high,
                    TRY_CAST(low AS DOUBLE) AS low,
                    TRY_CAST(close AS DOUBLE) AS close,
                    TRY_CAST(volume AS DOUBLE) AS volume,
                    TRY_CAST(adj_close AS DOUBLE) AS adj_close,
                    NULLIF(TRIM(CAST(source AS VARCHAR)), '') AS source,
                    CAST(fetched_at AS VARCHAR) AS fetched_at_text,
                    TRY_CAST(fetched_at AS TIMESTAMP) AS fetched_ts
                FROM {read_expr}
            ),
            ranked AS (
                SELECT
                    d, open, high, low, close, volume, adj_close,
                    COALESCE(source, 'unknown') AS source,
                    fetched_at_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY d
                        ORDER BY fetched_ts DESC NULLS LAST, fetched_at_text DESC NULLS LAST
                    ) AS rn
                FROM raw
                WHERE d IS NOT NULL
            )
            SELECT
                STRFTIME(d, '%Y-%m-%d') AS date,
                open, high, low, close, volume, CAST(NULL AS DOUBLE) AS vwap, adj_close, source,
                COALESCE(fetched_at_text, '') AS fetched_at,
                COALESCE(fetched_at_text, '') AS asof,
                CAST(NULL AS DOUBLE) AS quality_score,
                '' AS raw_json
            FROM ranked
            WHERE rn=1
            ORDER BY d ASC
        """
        conn = self._connect()
        try:
            compacted = conn.execute(sql).df()
        except Exception:
            try:
                compacted = conn.execute(legacy_sql).df()
            except Exception:
                return
        finally:
            conn.close()

        out_path = self._daily_symbol_path(symbol, market)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".compact.parquet")
        if compacted.empty:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            compacted["date"] = pd.to_datetime(compacted["date"], utc=True, errors="coerce")
            compacted = compacted.dropna(subset=["date"]).sort_values("date")
            compacted = self._ensure_daily_vwap_column(compacted.set_index("date")).reset_index()
            compacted["date"] = compacted["date"].dt.strftime("%Y-%m-%d")
            compacted.to_parquet(tmp_path, index=False)
            tmp_path.replace(out_path)
        for delta in delta_files:
            try:
                delta.unlink(missing_ok=True)
            except Exception:
                pass

    def _get_writeback_executor(self) -> ThreadPoolExecutor:
        with self._writeback_lock:
            if self._writeback_executor is None:
                self._writeback_executor = ThreadPoolExecutor(
                    max_workers=self._writeback_workers, thread_name_prefix="duck-writeback"
                )
            return self._writeback_executor

    @staticmethod
    def _merge_writeback_payload_frames(
        base: pd.DataFrame | None,
        incoming: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if incoming is None or not isinstance(incoming, pd.DataFrame) or incoming.empty:
            if base is None or not isinstance(base, pd.DataFrame):
                return pd.DataFrame()
            return base.copy()
        if base is None or not isinstance(base, pd.DataFrame) or base.empty:
            return incoming.copy()

        merged = pd.concat([base, incoming], axis=0, sort=False)
        if merged.empty:
            return merged
        idx = pd.to_datetime(merged.index, utc=True, errors="coerce")
        if idx.notna().any():
            mask = ~idx.isna()
            merged = merged.loc[mask].copy()
            merged.index = idx[mask]
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return merged

    def _persist_daily_bars_writeback(
        self,
        *,
        symbol: str,
        market: str,
        bars: pd.DataFrame,
        source: str | None,
    ) -> int:
        normalized = self._normalize_daily_bars_frame(bars)
        if normalized.empty:
            return 0

        source_text = self._normalize_text(source)
        if "source" not in normalized.columns:
            normalized["source"] = source_text or "writeback"
        else:
            normalized["source"] = normalized["source"].astype(str).replace({"": None})
            normalized["source"] = (
                normalized["source"].fillna(source_text or "writeback").astype(str)
            )
        normalized["fetched_at"] = datetime.now(tz=timezone.utc).isoformat()
        if "asof" in normalized.columns:
            normalized["asof"] = pd.to_datetime(normalized["asof"], utc=True, errors="coerce")
            normalized["asof"] = normalized["asof"].fillna(
                pd.to_datetime(normalized.index, utc=True, errors="coerce")
            )
        else:
            normalized["asof"] = pd.to_datetime(normalized.index, utc=True, errors="coerce")
        if "quality_score" in normalized.columns:
            normalized["quality_score"] = pd.to_numeric(
                normalized["quality_score"], errors="coerce"
            )
        else:
            normalized["quality_score"] = float("nan")
        if "raw_json" in normalized.columns:
            normalized["raw_json"] = normalized["raw_json"].map(self._normalize_raw_json_text)
        else:
            normalized["raw_json"] = ""

        rows_upserted = self._upsert_daily_bars(symbol=symbol, market=market, bars=normalized)
        if rows_upserted <= 0:
            return 0

        instrument_id = self._get_or_create_instrument(symbol, market)
        last_synced = (
            pd.Timestamp(normalized.index.max()).to_pydatetime().replace(tzinfo=timezone.utc)
        )
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
        source: str | None = None,
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
            pending = self._writeback_pending.get(key)
            if pending is None:
                merged_payload = payload
                merged_source = source_text
            else:
                prev_bars, prev_source = pending
                merged_payload = self._merge_writeback_payload_frames(prev_bars, payload)
                merged_source = source_text or prev_source
            self._writeback_pending[key] = (merged_payload, merged_source)
            should_submit = key not in self._writeback_inflight
            if should_submit:
                self._writeback_inflight.add(key)

        if not should_submit:
            return True

        def _worker() -> None:
            while True:
                with self._writeback_lock:
                    pending_payload = self._writeback_pending.pop(key, None)
                    if pending_payload is None:
                        self._writeback_inflight.discard(key)
                        return
                bars_payload, source_payload = pending_payload
                try:
                    self._persist_daily_bars_writeback(
                        symbol=symbol_token,
                        market=market_token,
                        bars=bars_payload,
                        source=source_payload,
                    )
                except Exception:
                    continue

        try:
            executor = self._get_writeback_executor()
            executor.submit(_worker)
            return True
        except Exception:
            with self._writeback_lock:
                self._writeback_pending.pop(key, None)
                self._writeback_inflight.discard(key)
            return False

    def flush_writeback_queue(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            with self._writeback_lock:
                pending = bool(self._writeback_inflight) or bool(self._writeback_pending)
            if not pending:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def _load_intraday_frame_raw(self, symbol: str, market: str) -> pd.DataFrame:
        expected = [
            "ts_utc",
            "price",
            "cum_volume",
            "source",
            "fetched_at",
            "asof",
            "quality_score",
            "raw_json",
        ]
        sources = self._intraday_parquet_sources(symbol, market)
        if not sources:
            return pd.DataFrame(columns=expected)
        read_expr = self._duck_read_parquet_expr(sources)
        sql = f"""
            WITH raw AS (
                SELECT
                    TRY_CAST(ts_utc AS TIMESTAMP) AS ts_utc,
                    TRY_CAST(price AS DOUBLE) AS price,
                    TRY_CAST(cum_volume AS DOUBLE) AS cum_volume,
                    NULLIF(TRIM(CAST(source AS VARCHAR)), '') AS source,
                    CAST(fetched_at AS VARCHAR) AS fetched_at_text,
                    TRY_CAST(fetched_at AS TIMESTAMP) AS fetched_ts,
                    CAST(asof AS VARCHAR) AS asof_text,
                    TRY_CAST(quality_score AS DOUBLE) AS quality_score,
                    CAST(raw_json AS VARCHAR) AS raw_json_text
                FROM {read_expr}
            ),
            ranked AS (
                SELECT
                    ts_utc, price, cum_volume,
                    COALESCE(source, 'unknown') AS source,
                    fetched_at_text,
                    asof_text,
                    quality_score,
                    raw_json_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY ts_utc
                        ORDER BY fetched_ts DESC NULLS LAST, fetched_at_text DESC NULLS LAST
                    ) AS rn
                FROM raw
                WHERE ts_utc IS NOT NULL
            )
            SELECT
                CAST(ts_utc AS VARCHAR) AS ts_utc,
                price,
                cum_volume,
                source,
                COALESCE(fetched_at_text, '') AS fetched_at,
                COALESCE(asof_text, fetched_at_text, '') AS asof,
                quality_score,
                COALESCE(raw_json_text, '') AS raw_json
            FROM ranked
            WHERE rn=1
            ORDER BY ts_utc ASC
        """
        legacy_sql = f"""
            WITH raw AS (
                SELECT
                    TRY_CAST(ts_utc AS TIMESTAMP) AS ts_utc,
                    TRY_CAST(price AS DOUBLE) AS price,
                    TRY_CAST(cum_volume AS DOUBLE) AS cum_volume,
                    NULLIF(TRIM(CAST(source AS VARCHAR)), '') AS source,
                    CAST(fetched_at AS VARCHAR) AS fetched_at_text,
                    TRY_CAST(fetched_at AS TIMESTAMP) AS fetched_ts
                FROM {read_expr}
            ),
            ranked AS (
                SELECT
                    ts_utc, price, cum_volume,
                    COALESCE(source, 'unknown') AS source,
                    fetched_at_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY ts_utc
                        ORDER BY fetched_ts DESC NULLS LAST, fetched_at_text DESC NULLS LAST
                    ) AS rn
                FROM raw
                WHERE ts_utc IS NOT NULL
            )
            SELECT
                CAST(ts_utc AS VARCHAR) AS ts_utc,
                price,
                cum_volume,
                source,
                COALESCE(fetched_at_text, '') AS fetched_at,
                COALESCE(fetched_at_text, '') AS asof,
                CAST(NULL AS DOUBLE) AS quality_score,
                '' AS raw_json
            FROM ranked
            WHERE rn=1
            ORDER BY ts_utc ASC
        """
        conn = self._connect()
        try:
            out = conn.execute(sql).df()
        except Exception:
            try:
                out = conn.execute(legacy_sql).df()
            except Exception:
                return pd.DataFrame(columns=expected)
        finally:
            conn.close()
        if out.empty:
            return pd.DataFrame(columns=expected)
        for col in expected:
            if col not in out.columns:
                out[col] = None
        return out[expected]

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
                "cum_volume": pd.to_numeric(frame.get("cum_volume", 0.0), errors="coerce").fillna(
                    0.0
                ),
                "source": frame.get("source", pd.Series(index=frame.index, data="unknown")).astype(
                    str
                ),
                "fetched_at": frame.get(
                    "fetched_at",
                    pd.Series(index=frame.index, data=datetime.now(tz=timezone.utc).isoformat()),
                ).astype(str),
                "asof": pd.to_datetime(
                    frame.get("asof", pd.Series(index=frame.index, data=frame.index)),
                    utc=True,
                    errors="coerce",
                )
                .astype("datetime64[ns, UTC]")
                .astype(str),
                "quality_score": pd.to_numeric(frame.get("quality_score"), errors="coerce"),
                "raw_json": frame.get(
                    "raw_json",
                    pd.Series(index=frame.index, data=""),
                ).map(self._normalize_raw_json_text),
            }
        )
        payload = payload.dropna(subset=["price"])  # type: ignore[arg-type]
        if payload.empty:
            return 0
        payload["ts_utc"] = pd.to_datetime(payload["ts_utc"], utc=True, errors="coerce")
        payload = payload.dropna(subset=["ts_utc"])  # type: ignore[arg-type]
        payload = payload.reset_index(drop=True)
        cutoff = datetime.now(tz=timezone.utc) - pd.Timedelta(days=self.intraday_retain_days)
        payload = payload[payload["ts_utc"] >= pd.Timestamp(cutoff)]
        payload = payload.drop_duplicates(subset=["ts_utc"], keep="last").sort_values("ts_utc")
        if payload.empty:
            return 0
        payload["ts_utc"] = payload["ts_utc"].astype(str)

        delta_dir = self._intraday_delta_dir(symbol, market)
        delta_dir.mkdir(parents=True, exist_ok=True)
        delta_path = self._next_delta_parquet_path(delta_dir, prefix="ticks_delta")
        payload.to_parquet(delta_path, index=False)
        self._compact_intraday_ticks_if_needed(symbol=symbol, market=market)
        return int(len(payload))

    def _compact_intraday_ticks_if_needed(self, *, symbol: str, market: str) -> None:
        delta_dir = self._intraday_delta_dir(symbol, market)
        if not delta_dir.exists():
            return
        delta_files = sorted([p for p in delta_dir.glob("*.parquet") if p.is_file()])
        if len(delta_files) < int(self._intraday_delta_compact_threshold):
            return

        sources = self._intraday_parquet_sources(symbol, market)
        if not sources:
            return
        read_expr = self._duck_read_parquet_expr(sources)
        cutoff_iso = (
            datetime.now(tz=timezone.utc) - pd.Timedelta(days=self.intraday_retain_days)
        ).isoformat()
        sql = f"""
            WITH raw AS (
                SELECT
                    TRY_CAST(ts_utc AS TIMESTAMP) AS ts_utc,
                    TRY_CAST(price AS DOUBLE) AS price,
                    TRY_CAST(cum_volume AS DOUBLE) AS cum_volume,
                    NULLIF(TRIM(CAST(source AS VARCHAR)), '') AS source,
                    CAST(fetched_at AS VARCHAR) AS fetched_at_text,
                    TRY_CAST(fetched_at AS TIMESTAMP) AS fetched_ts,
                    CAST(asof AS VARCHAR) AS asof_text,
                    TRY_CAST(quality_score AS DOUBLE) AS quality_score,
                    CAST(raw_json AS VARCHAR) AS raw_json_text
                FROM {read_expr}
            ),
            ranked AS (
                SELECT
                    ts_utc, price, cum_volume,
                    COALESCE(source, 'unknown') AS source,
                    fetched_at_text,
                    asof_text,
                    quality_score,
                    raw_json_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY ts_utc
                        ORDER BY fetched_ts DESC NULLS LAST, fetched_at_text DESC NULLS LAST
                    ) AS rn
                FROM raw
                WHERE ts_utc IS NOT NULL
            )
            SELECT
                CAST(ts_utc AS VARCHAR) AS ts_utc,
                price,
                cum_volume,
                source,
                COALESCE(fetched_at_text, '') AS fetched_at,
                COALESCE(asof_text, fetched_at_text, '') AS asof,
                quality_score,
                COALESCE(raw_json_text, '') AS raw_json
            FROM ranked
            WHERE rn=1 AND ts_utc >= CAST(? AS TIMESTAMP)
            ORDER BY ts_utc ASC
        """
        legacy_sql = f"""
            WITH raw AS (
                SELECT
                    TRY_CAST(ts_utc AS TIMESTAMP) AS ts_utc,
                    TRY_CAST(price AS DOUBLE) AS price,
                    TRY_CAST(cum_volume AS DOUBLE) AS cum_volume,
                    NULLIF(TRIM(CAST(source AS VARCHAR)), '') AS source,
                    CAST(fetched_at AS VARCHAR) AS fetched_at_text,
                    TRY_CAST(fetched_at AS TIMESTAMP) AS fetched_ts
                FROM {read_expr}
            ),
            ranked AS (
                SELECT
                    ts_utc, price, cum_volume,
                    COALESCE(source, 'unknown') AS source,
                    fetched_at_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY ts_utc
                        ORDER BY fetched_ts DESC NULLS LAST, fetched_at_text DESC NULLS LAST
                    ) AS rn
                FROM raw
                WHERE ts_utc IS NOT NULL
            )
            SELECT
                CAST(ts_utc AS VARCHAR) AS ts_utc,
                price,
                cum_volume,
                source,
                COALESCE(fetched_at_text, '') AS fetched_at,
                COALESCE(fetched_at_text, '') AS asof,
                CAST(NULL AS DOUBLE) AS quality_score,
                '' AS raw_json
            FROM ranked
            WHERE rn=1 AND ts_utc >= CAST(? AS TIMESTAMP)
            ORDER BY ts_utc ASC
        """
        conn = self._connect()
        try:
            compacted = conn.execute(sql, [cutoff_iso]).df()
        except Exception:
            try:
                compacted = conn.execute(legacy_sql, [cutoff_iso]).df()
            except Exception:
                return
        finally:
            conn.close()

        out_path = self._intraday_symbol_path(symbol, market)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".compact.parquet")
        if compacted.empty:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            compacted.to_parquet(tmp_path, index=False)
            tmp_path.replace(out_path)
        for delta in delta_files:
            try:
                delta.unlink(missing_ok=True)
            except Exception:
                pass

    def _get_or_create_instrument(self, symbol: str, market: str, name: str | None = None) -> int:
        symbol = self._normalize_symbol_token(symbol)
        market = self._normalize_market_token(market)
        with self._connect_ctx() as conn:
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

    def _load_first_bar_date(self, symbol: str, market: str) -> datetime | None:
        bars = self.load_daily_bars(symbol=symbol, market=market)
        if bars.empty:
            return None
        return pd.Timestamp(bars.index.min()).to_pydatetime().replace(tzinfo=timezone.utc)

    def _load_last_success_date(self, instrument_id: int) -> datetime | None:
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

    def load_sync_state(self, symbol: str, market: str) -> dict[str, object] | None:
        symbol_token = self._normalize_symbol_token(symbol)
        market_token = self._normalize_market_token(market)
        if not symbol_token or not market_token:
            return None

        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT s.last_success_date, s.last_source, s.last_error, s.updated_at
                FROM sync_state s
                JOIN instruments i ON i.id = s.instrument_id
                WHERE i.symbol=? AND i.market=?
                ORDER BY s.updated_at DESC
                LIMIT 1
                """,
                (symbol_token, market_token),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return {
            "last_success_date": self._parse_iso_datetime(row[0]),
            "last_source": self._normalize_text(row[1]),
            "last_error": self._normalize_text(row[2]),
            "updated_at": self._parse_iso_datetime(row[3]),
        }

    def load_daily_coverage(
        self,
        symbol: str,
        market: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, object]:
        symbol_token = self._normalize_symbol_token(symbol)
        market_token = self._normalize_market_token(market)
        sources = self._daily_parquet_sources(symbol_token, market_token)
        empty = {"row_count": 0, "first_date": None, "last_date": None}
        if not sources:
            return empty

        read_expr = self._duck_read_parquet_expr(sources)
        where: list[str] = ["rn=1"]
        params: list[object] = []
        if start is not None:
            where.append("d >= CAST(? AS DATE)")
            params.append(pd.Timestamp(start).date().isoformat())
        if end is not None:
            where.append("d <= CAST(? AS DATE)")
            params.append(pd.Timestamp(end).date().isoformat())

        sql = f"""
            WITH raw AS (
                SELECT
                    TRY_CAST(date AS DATE) AS d,
                    CAST(fetched_at AS VARCHAR) AS fetched_at_text,
                    TRY_CAST(fetched_at AS TIMESTAMP) AS fetched_ts
                FROM {read_expr}
            ),
            ranked AS (
                SELECT
                    d,
                    ROW_NUMBER() OVER (
                        PARTITION BY d
                        ORDER BY fetched_ts DESC NULLS LAST, fetched_at_text DESC NULLS LAST
                    ) AS rn
                FROM raw
                WHERE d IS NOT NULL
            )
            SELECT COUNT(*) AS row_count, MIN(d) AS first_date, MAX(d) AS last_date
            FROM ranked
            WHERE {" AND ".join(where)}
        """
        conn = self._connect()
        try:
            row = conn.execute(sql, params).fetchone()
        except Exception:
            return empty
        finally:
            conn.close()

        if row is None:
            return empty
        first = pd.Timestamp(row[1], tz="UTC").to_pydatetime() if row[1] is not None else None
        last = pd.Timestamp(row[2], tz="UTC").to_pydatetime() if row[2] is not None else None
        return {
            "row_count": max(0, int(row[0] or 0)),
            "first_date": first,
            "last_date": last,
        }

    def _save_sync_state(
        self,
        instrument_id: int,
        last_success_date: datetime | None,
        source: str | None,
        error: str | None,
    ):
        with self._connect_ctx() as conn:
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

    def sync_symbol_history(
        self,
        symbol: str,
        market: str,
        start: datetime | None = None,
        end: datetime | None = None,
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
            if (
                last_success
                and first_bar_date is not None
                and start >= first_bar_date
                and start <= last_success
            ):
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

        request = ProviderRequest(
            symbol=symbol, market=market, interval="1d", start=fetch_start, end=end
        )
        is_tw_local_symbol = self._is_tw_local_security_symbol(symbol)
        if market == "US":
            providers = [self.service.us_twelve, self.service.yahoo, self.service.us_stooq]
        elif market == "OTC":
            if is_tw_local_symbol:
                providers = []
                fugle_rest = getattr(self.service, "tw_fugle_rest", None)
                if fugle_rest is not None and getattr(fugle_rest, "api_key", None):
                    providers.append(fugle_rest)
                providers.extend(
                    [self.service.tw_tpex, self.service.tw_openapi, self.service.yahoo]
                )
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
            fetched_at = pd.Timestamp(getattr(snap, "fetched_at", datetime.now(tz=timezone.utc)))
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.tz_localize("UTC")
            else:
                fetched_at = fetched_at.tz_convert("UTC")
            asof_ts = pd.Timestamp(getattr(snap, "asof", None) or df.index.max())
            if asof_ts.tzinfo is None:
                asof_ts = asof_ts.tz_localize("UTC")
            else:
                asof_ts = asof_ts.tz_convert("UTC")
            quality_raw = pd.to_numeric(getattr(snap, "quality_score", None), errors="coerce")
            df["fetched_at"] = fetched_at.isoformat()
            df["asof"] = asof_ts.isoformat()
            df["quality_score"] = float(quality_raw) if pd.notna(quality_raw) else float("nan")
            df["raw_json"] = self._normalize_raw_json_text(getattr(snap, "raw_json", ""))
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
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        symbol = self._normalize_symbol_token(symbol)
        market = self._normalize_market_token(market)
        df = self._load_daily_frame_window(symbol, market, start=start, end=end)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "adj_close",
                    "source",
                    "asof",
                    "quality_score",
                    "raw_json",
                ]
            )
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        for col in ["open", "high", "low", "close", "volume", "vwap", "adj_close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = self._ensure_daily_vwap_column(df)
        if "source" not in df.columns:
            df["source"] = "unknown"
        if "asof" in df.columns:
            df["asof"] = pd.to_datetime(df["asof"], utc=True, errors="coerce")
        else:
            df["asof"] = pd.to_datetime(df.index, utc=True, errors="coerce")
        if "quality_score" in df.columns:
            df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
        else:
            df["quality_score"] = float("nan")
        if "raw_json" in df.columns:
            df["raw_json"] = df["raw_json"].map(self._normalize_raw_json_text)
        else:
            df["raw_json"] = ""
        return df[
            [
                c
                for c in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "adj_close",
                    "source",
                    "asof",
                    "quality_score",
                    "raw_json",
                ]
                if c in df.columns
            ]
        ]

    def save_intraday_ticks(
        self,
        symbol: str,
        market: str,
        ticks: list[dict[str, object]],
        retain_days: int | None = None,
    ) -> int:
        if not ticks:
            return 0
        rows: dict[str, tuple[float, float, str, str, str, float, str]] = {}
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
            quality_raw = pd.to_numeric(row.get("quality_score"), errors="coerce")
            rows[ts.isoformat()] = (
                float(price_val),
                0.0 if pd.isna(cum_val) else float(cum_val),
                str(row.get("source", "unknown") or "unknown"),
                str(row.get("fetched_at", now_iso) or now_iso),
                str(row.get("asof", ts.isoformat()) or ts.isoformat()),
                float(quality_raw) if pd.notna(quality_raw) else float("nan"),
                self._normalize_raw_json_text(row.get("raw_json", "")),
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
                "asof": [v[4] for v in rows.values()],
                "quality_score": [v[5] for v in rows.values()],
                "raw_json": [v[6] for v in rows.values()],
            }
        )
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["ts_utc"]).set_index("ts_utc").sort_index()

        retain = self.resolve_intraday_retain_days(
            self.intraday_retain_days if retain_days is None else retain_days
        )
        self.intraday_retain_days = retain
        return self._upsert_intraday_ticks(symbol=symbol, market=market, ticks=frame)

    def load_intraday_ticks(
        self,
        symbol: str,
        market: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        symbol = self._normalize_symbol_token(symbol)
        market = self._normalize_market_token(market)
        df = self._load_intraday_frame_raw(symbol, market)
        if df.empty:
            return pd.DataFrame(
                columns=["price", "cum_volume", "source", "asof", "quality_score", "raw_json"]
            )
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
        if "asof" in df.columns:
            df["asof"] = pd.to_datetime(df["asof"], utc=True, errors="coerce")
        else:
            df["asof"] = pd.to_datetime(df.index, utc=True, errors="coerce")
        if "quality_score" in df.columns:
            df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
        else:
            df["quality_score"] = float("nan")
        if "raw_json" in df.columns:
            df["raw_json"] = df["raw_json"].map(self._normalize_raw_json_text)
        else:
            df["raw_json"] = ""
        return df[
            [
                c
                for c in ["price", "cum_volume", "source", "asof", "quality_score", "raw_json"]
                if c in df.columns
            ]
        ]

    def upsert_symbol_metadata(self, rows: list[dict[str, object]]) -> int:
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

        existing = (
            self.load_symbol_metadata([r["symbol"] for r in normalized], normalized[0]["market"])
            if normalized
            else {}
        )
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

        with self._connect_ctx() as conn:
            for p in payload:
                conn.execute(
                    "DELETE FROM symbol_metadata WHERE symbol=? AND market=?", (p[0], p[1])
                )
                conn.execute(
                    "INSERT INTO symbol_metadata(symbol, market, name, exchange, industry, asset_type, currency, source, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    p,
                )
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

    def list_symbols(self, market: str, limit: int | None = None) -> list[str]:
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
            symbols = [
                self._normalize_symbol_token(r[0])
                for r in rows
                if self._normalize_symbol_token(r[0])
            ]
            if symbols:
                return symbols

            rows = conn.execute(
                f"SELECT symbol FROM instruments WHERE market=? ORDER BY symbol ASC{limit_sql}",
                params,
            ).fetchall()
            symbols = [
                self._normalize_symbol_token(r[0])
                for r in rows
                if self._normalize_symbol_token(r[0])
            ]
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

    def start_bootstrap_run(self, scope: str, params: dict[str, object]) -> str:
        run_id = f"bootstrap:{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        with self._connect_ctx() as conn:
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
        return run_id

    def finish_bootstrap_run(
        self,
        run_id: str,
        *,
        status: str,
        total_symbols: int,
        synced_symbols: int,
        failed_symbols: int,
        summary: dict[str, object] | None = None,
        error: str | None = None,
    ):
        key = self._normalize_text(run_id)
        if not key:
            return
        with self._connect_ctx() as conn:
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

    def load_latest_bootstrap_run(self) -> BootstrapRun | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT run_id, scope, status, started_at, finished_at, total_symbols, synced_symbols, failed_symbols, params_json, summary_json, error FROM bootstrap_runs ORDER BY started_at DESC, id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None

        params_obj: dict[str, Any] = {}
        summary_obj: dict[str, Any] = {}
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
        params: dict[str, object],
        cost: dict[str, object],
        result: dict[str, object],
    ) -> int:
        with self._connect_ctx() as conn:
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

    def save_backtest_replay_run(
        self,
        run_key: str,
        params: dict[str, object],
        payload: dict[str, object],
    ) -> int:
        key = str(run_key or "").strip()
        if not key:
            raise ValueError("run_key is required")
        with self._connect_ctx() as conn:
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

    def load_latest_backtest_replay_run(self, run_key: str) -> BacktestReplayRun | None:
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
        params: dict[str, object] = {}
        payload: dict[str, object] = {}
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

    def save_tw_etf_super_export_run(
        self,
        *,
        ytd_start: str,
        ytd_end: str,
        compare_start: str,
        compare_end: str,
        trade_date_anchor: str,
        output_path: str,
        row_count: int,
        column_count: int,
        payload: dict[str, object],
    ) -> str:
        now = datetime.now(tz=timezone.utc)
        run_id = f"tw_etf_super_export:{now.strftime('%Y%m%dT%H%M%S%f')}"
        with self._connect_ctx() as conn:
            rid = self._next_id(conn, "tw_etf_super_export_runs")
            conn.execute(
                """
                INSERT INTO tw_etf_super_export_runs(
                    id,
                    run_id,
                    created_at,
                    ytd_start,
                    ytd_end,
                    compare_start,
                    compare_end,
                    trade_date_anchor,
                    output_path,
                    row_count,
                    column_count,
                    payload_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    run_id,
                    now.isoformat(),
                    self._normalize_text(ytd_start),
                    self._normalize_text(ytd_end),
                    self._normalize_text(compare_start),
                    self._normalize_text(compare_end),
                    self._normalize_text(trade_date_anchor),
                    self._normalize_text(output_path),
                    max(0, int(row_count)),
                    max(0, int(column_count)),
                    json.dumps(payload or {}, ensure_ascii=False, default=str),
                ),
            )
        return run_id

    def load_latest_tw_etf_super_export_run(self) -> SuperExportRun | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT
                    run_id,
                    created_at,
                    ytd_start,
                    ytd_end,
                    compare_start,
                    compare_end,
                    trade_date_anchor,
                    output_path,
                    row_count,
                    column_count,
                    payload_json
                FROM tw_etf_super_export_runs
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None

        created_at = self._parse_iso_datetime(row[1]) or datetime.now(tz=timezone.utc)
        payload_obj: dict[str, object] = {}
        try:
            obj = json.loads(str(row[10] or "{}"))
            if isinstance(obj, dict):
                payload_obj = obj
        except Exception:
            payload_obj = {}

        return SuperExportRun(
            run_id=self._normalize_text(row[0]),
            created_at=created_at,
            ytd_start=self._normalize_text(row[2]),
            ytd_end=self._normalize_text(row[3]),
            compare_start=self._normalize_text(row[4]),
            compare_end=self._normalize_text(row[5]),
            trade_date_anchor=self._normalize_text(row[6]),
            output_path=self._normalize_text(row[7]),
            row_count=max(0, int(row[8] or 0)),
            column_count=max(0, int(row[9] or 0)),
            payload=payload_obj,
        )

    def save_market_snapshot(
        self,
        *,
        dataset_key: str,
        market: str = "",
        symbol: str = "",
        interval: str = "",
        source: str = "",
        asof: datetime | str | None = None,
        payload: object = None,
        freshness_sec: int | None = None,
        quality_score: float | None = None,
        stale: bool = False,
        raw_json: object = None,
    ) -> int:
        key = self._normalize_text(dataset_key)
        if not key:
            return 0
        market_token = self._normalize_market_token(market) if market else ""
        symbol_token = self._normalize_symbol_token(symbol) if symbol else ""
        interval_token = self._normalize_text(interval)
        source_token = self._normalize_text(source)

        now = datetime.now(tz=timezone.utc)
        asof_dt: datetime | None = None
        if isinstance(asof, datetime):
            asof_dt = asof if asof.tzinfo is not None else asof.replace(tzinfo=timezone.utc)
        else:
            asof_text = self._normalize_text(asof)
            if asof_text:
                if re.fullmatch(r"\d{8}", asof_text):
                    try:
                        asof_dt = datetime.strptime(asof_text, "%Y%m%d").replace(
                            tzinfo=timezone.utc
                        )
                    except Exception:
                        asof_dt = None
                else:
                    asof_dt = self._parse_iso_datetime(asof_text)
        if asof_dt is None:
            asof_dt = now
        asof_iso = asof_dt.astimezone(timezone.utc).isoformat()
        fetched_at_iso = now.isoformat()

        try:
            payload_json = json.dumps(payload, ensure_ascii=False)
        except Exception:
            payload_json = json.dumps({}, ensure_ascii=False)
        raw_json_text = self._normalize_raw_json_text(raw_json if raw_json is not None else {})
        freshness_value = None if freshness_sec is None else max(0, int(freshness_sec))
        quality_value = None
        if quality_score is not None:
            try:
                quality_value = float(quality_score)
            except Exception:
                quality_value = None

        with self._connect_ctx() as conn:
            conn.execute(
                """
                DELETE FROM market_snapshots
                WHERE dataset_key=? AND market=? AND symbol=? AND interval=? AND as_of=?
                """,
                (key, market_token, symbol_token, interval_token, asof_iso),
            )
            row_id = self._next_id(conn, "market_snapshots")
            conn.execute(
                """
                INSERT INTO market_snapshots(
                    id, dataset_key, market, symbol, interval, source, as_of, fetched_at,
                    freshness_sec, quality_score, stale, payload_json, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    key,
                    market_token,
                    symbol_token,
                    interval_token,
                    source_token,
                    asof_iso,
                    fetched_at_iso,
                    freshness_value,
                    quality_value,
                    1 if bool(stale) else 0,
                    payload_json,
                    raw_json_text,
                ),
            )
        return 1

    def _load_market_snapshot_rows(
        self,
        *,
        dataset_key: str,
        market: str = "",
        symbol: str = "",
        interval: str = "",
        asof_start: datetime | str | None = None,
        asof_end: datetime | str | None = None,
        limit: int = 1,
    ) -> list[tuple[Any, ...]]:
        key = self._normalize_text(dataset_key)
        if not key:
            return []
        where = ["dataset_key=?"]
        params: list[object] = [key]

        market_token = self._normalize_market_token(market) if market else ""
        symbol_token = self._normalize_symbol_token(symbol) if symbol else ""
        interval_token = self._normalize_text(interval)
        if market_token:
            where.append("market=?")
            params.append(market_token)
        if symbol_token:
            where.append("symbol=?")
            params.append(symbol_token)
        if interval_token:
            where.append("interval=?")
            params.append(interval_token)

        def _asof_iso(value: datetime | str | None) -> str | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            text = self._normalize_text(value)
            if not text:
                return None
            if re.fullmatch(r"\d{8}", text):
                try:
                    return (
                        datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc).isoformat()
                    )
                except Exception:
                    return None
            dt = self._parse_iso_datetime(text)
            return dt.isoformat() if dt is not None else None

        start_iso = _asof_iso(asof_start)
        end_iso = _asof_iso(asof_end)
        if start_iso:
            where.append("as_of>=?")
            params.append(start_iso)
        if end_iso:
            where.append("as_of<=?")
            params.append(end_iso)

        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT dataset_key, market, symbol, interval, source, as_of, fetched_at,
                       freshness_sec, quality_score, stale, payload_json, raw_json
                FROM market_snapshots
                WHERE {" AND ".join(where)}
                ORDER BY as_of DESC, fetched_at DESC, id DESC
                LIMIT ?
                """,
                [*params, max(1, int(limit))],
            ).fetchall()
        finally:
            conn.close()
        return rows

    def _decode_market_snapshot_row(self, row: tuple[Any, ...]) -> dict[str, object]:
        payload_obj: object
        raw_obj: object
        try:
            payload_obj = json.loads(str(row[10] or "null"))
        except Exception:
            payload_obj = None
        try:
            raw_obj = json.loads(str(row[11] or "null"))
        except Exception:
            raw_obj = self._normalize_raw_json_text(row[11])
        return {
            "dataset_key": self._normalize_text(row[0]),
            "market": self._normalize_market_token(row[1]),
            "symbol": self._normalize_symbol_token(row[2]),
            "interval": self._normalize_text(row[3]),
            "source": self._normalize_text(row[4]),
            "asof": self._parse_iso_datetime(row[5]),
            "fetched_at": self._parse_iso_datetime(row[6]),
            "freshness_sec": int(row[7]) if row[7] is not None else None,
            "quality_score": float(row[8]) if row[8] is not None else None,
            "stale": bool(int(row[9] or 0)),
            "payload": payload_obj,
            "raw_json": raw_obj,
        }

    def load_latest_market_snapshot(
        self,
        *,
        dataset_key: str,
        market: str = "",
        symbol: str = "",
        interval: str = "",
    ) -> dict[str, object] | None:
        rows = self._load_market_snapshot_rows(
            dataset_key=dataset_key,
            market=market,
            symbol=symbol,
            interval=interval,
            limit=1,
        )
        if not rows:
            return None
        return self._decode_market_snapshot_row(rows[0])

    def load_market_snapshot_window(
        self,
        *,
        dataset_key: str,
        market: str = "",
        symbol: str = "",
        interval: str = "",
        asof_start: datetime | str | None = None,
        asof_end: datetime | str | None = None,
        limit: int = 128,
    ) -> list[dict[str, object]]:
        rows = self._load_market_snapshot_rows(
            dataset_key=dataset_key,
            market=market,
            symbol=symbol,
            interval=interval,
            asof_start=asof_start,
            asof_end=asof_end,
            limit=limit,
        )
        return [self._decode_market_snapshot_row(row) for row in rows]

    def purge_market_snapshots(
        self,
        *,
        dataset_key: str | None = None,
        keep_days: int = 30,
    ) -> int:
        cutoff = datetime.now(tz=timezone.utc) - pd.Timedelta(days=max(1, int(keep_days)))
        cutoff_iso = cutoff.isoformat()
        where = ["as_of < ?"]
        params: list[object] = [cutoff_iso]
        if dataset_key:
            where.append("dataset_key=?")
            params.append(self._normalize_text(dataset_key))

        with self._connect_ctx() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM market_snapshots WHERE {' AND '.join(where)}",
                params,
            ).fetchone()
            count = int(row[0] or 0) if row is not None else 0
            if count > 0:
                conn.execute(
                    f"DELETE FROM market_snapshots WHERE {' AND '.join(where)}",
                    params,
                )
        return count

    def get_market_snapshot_stats(
        self,
        *,
        dataset_key: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        where: list[str] = []
        params: list[object] = []
        if dataset_key:
            where.append("dataset_key=?")
            params.append(self._normalize_text(dataset_key))
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT
                    dataset_key,
                    market,
                    interval,
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT symbol) AS symbol_count,
                    SUM(CASE WHEN stale THEN 1 ELSE 0 END) AS stale_rows,
                    MIN(as_of) AS min_as_of,
                    MAX(as_of) AS max_as_of,
                    MAX(fetched_at) AS last_fetched_at
                FROM market_snapshots
                {where_sql}
                GROUP BY dataset_key, market, interval
                ORDER BY row_count DESC, dataset_key ASC, market ASC, interval ASC
                LIMIT ?
                """,
                [*params, max(1, int(limit))],
            ).fetchall()
        finally:
            conn.close()
        out: list[dict[str, object]] = []
        for row in rows:
            out.append(
                {
                    "dataset_key": self._normalize_text(row[0]),
                    "market": self._normalize_market_token(row[1]),
                    "interval": self._normalize_text(row[2]),
                    "rows": int(row[3] or 0),
                    "symbols": int(row[4] or 0),
                    "stale_rows": int(row[5] or 0),
                    "min_as_of": self._parse_iso_datetime(row[6]),
                    "max_as_of": self._parse_iso_datetime(row[7]),
                    "last_fetched_at": self._parse_iso_datetime(row[8]),
                }
            )
        return out

    def run_market_snapshot_housekeeping(
        self,
        *,
        policy: dict[str, int] | None = None,
        default_keep_days: int = 30,
    ) -> dict[str, object]:
        normalized_policy: dict[str, int] = {}
        for key, value in (policy or {}).items():
            token = self._normalize_text(key)
            if not token:
                continue
            try:
                keep = max(1, int(value))
            except Exception:
                continue
            normalized_policy[token] = keep

        keep_default = max(1, int(default_keep_days))
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT dataset_key FROM market_snapshots ORDER BY dataset_key ASC"
            ).fetchall()
        finally:
            conn.close()
        dataset_keys = [
            self._normalize_text(row[0]) for row in rows if self._normalize_text(row[0])
        ]
        removed_by_dataset: dict[str, int] = {}
        removed_total = 0
        for dataset_key_value in dataset_keys:
            keep_days = normalized_policy.get(dataset_key_value, keep_default)
            removed = self.purge_market_snapshots(
                dataset_key=dataset_key_value, keep_days=keep_days
            )
            removed_by_dataset[dataset_key_value] = int(removed)
            removed_total += int(removed)
        return {
            "removed_total": int(removed_total),
            "removed_by_dataset": removed_by_dataset,
            "applied_policy": {k: int(v) for k, v in normalized_policy.items()},
            "default_keep_days": int(keep_default),
            "at": datetime.now(tz=timezone.utc).isoformat(),
        }

    def run_duckdb_maintenance(
        self,
        *,
        checkpoint: bool = True,
        vacuum: bool = False,
    ) -> dict[str, object]:
        actions: list[str] = []
        conn = self._connect()
        try:
            if checkpoint:
                conn.execute("CHECKPOINT")
                actions.append("CHECKPOINT")
            if vacuum:
                conn.execute("VACUUM")
                actions.append("VACUUM")
        finally:
            conn.close()
        return {
            "actions": actions,
            "at": datetime.now(tz=timezone.utc).isoformat(),
            "db_path": str(self.db_path),
        }

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
        with self._connect_ctx() as conn:
            conn.execute(
                """
                INSERT INTO universe_snapshots(universe_id, symbols_json, source, fetched_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(universe_id) DO UPDATE SET
                    symbols_json=excluded.symbols_json,
                    source=excluded.source,
                    fetched_at=excluded.fetched_at,
                    updated_at=excluded.updated_at
                """,
                (key, payload, source, now, now),
            )

    def save_tw_etf_aum_snapshot(
        self,
        *,
        rows: list[dict[str, object]],
        trade_date: str,
        keep_days: int = 0,
    ) -> int:
        try:
            keep = int(keep_days)
        except Exception:
            keep = 0
        try:
            trade_date_iso = pd.Timestamp(trade_date).date().isoformat()
        except Exception:
            return 0
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        dedup: dict[str, tuple[str, float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = self._normalize_symbol_token(
                row.get("etf_code") or row.get("code") or row.get("symbol")
            )
            if not code:
                continue
            raw_value = pd.to_numeric(row.get("aum_billion"), errors="coerce")
            try:
                value = float(raw_value)
            except Exception:
                continue
            if (not pd.notna(raw_value)) or value < 0:
                continue
            name = self._normalize_text(row.get("etf_name") or row.get("name")) or code
            dedup[code] = (name, value)

        if not dedup:
            return 0

        with self._connect_ctx() as conn:
            for code, (name, aum_billion) in dedup.items():
                conn.execute(
                    "DELETE FROM tw_etf_aum_history WHERE etf_code=? AND trade_date=?",
                    (code, trade_date_iso),
                )
                conn.execute(
                    "INSERT INTO tw_etf_aum_history(etf_code, etf_name, trade_date, aum_billion, updated_at) VALUES(?, ?, ?, ?, ?)",
                    (code, name, trade_date_iso, aum_billion, now_iso),
                )
                if keep > 0:
                    conn.execute(
                        """
                        DELETE FROM tw_etf_aum_history
                        WHERE etf_code=?
                          AND trade_date NOT IN (
                            SELECT trade_date
                            FROM tw_etf_aum_history
                            WHERE etf_code=?
                            ORDER BY trade_date DESC
                            LIMIT ?
                        )
                        """,
                        (code, code, keep),
                    )
        return len(dedup)

    def clear_tw_etf_aum_history(self) -> int:
        with self._connect_ctx() as conn:
            row = conn.execute("SELECT COUNT(*) FROM tw_etf_aum_history").fetchone()
            count = int(row[0]) if row and row[0] is not None else 0
            conn.execute("DELETE FROM tw_etf_aum_history")
            return count

    def load_tw_etf_aum_history(
        self,
        *,
        etf_codes: list[str],
        keep_days: int = 0,
    ) -> pd.DataFrame:
        try:
            keep = int(keep_days)
        except Exception:
            keep = 0
        keep_limit = keep if keep > 0 else None
        normalized_codes: list[str] = []
        seen: set[str] = set()
        for code in etf_codes:
            token = self._normalize_symbol_token(code)
            if not token or token in seen:
                continue
            normalized_codes.append(token)
            seen.add(token)
        if not normalized_codes:
            return pd.DataFrame(
                columns=["etf_code", "etf_name", "trade_date", "aum_billion", "updated_at"]
            )

        placeholders = ", ".join(["?"] * len(normalized_codes))
        conn = self._connect()
        try:
            df = conn.execute(
                f"""
                SELECT etf_code, etf_name, trade_date, aum_billion, updated_at
                FROM tw_etf_aum_history
                WHERE etf_code IN ({placeholders})
                ORDER BY etf_code ASC, trade_date DESC
                """,
                normalized_codes,
            ).df()
        finally:
            conn.close()
        if df.empty:
            return df

        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date.astype(str)
        df["aum_billion"] = pd.to_numeric(df["aum_billion"], errors="coerce")
        df = df.dropna(subset=["trade_date", "aum_billion"])
        if df.empty:
            return pd.DataFrame(
                columns=["etf_code", "etf_name", "trade_date", "aum_billion", "updated_at"]
            )

        df = df.sort_values(["etf_code", "trade_date"], ascending=[True, False])
        if keep_limit is not None:
            df = df.groupby("etf_code", as_index=False).head(keep_limit)
        df = df.sort_values(["etf_code", "trade_date"], ascending=[True, True]).reset_index(
            drop=True
        )
        return df[["etf_code", "etf_name", "trade_date", "aum_billion", "updated_at"]]

    def save_tw_etf_daily_market(
        self,
        *,
        rows: list[dict[str, object]],
        trade_date: str,
        source: str = "twse_etf_daily",
    ) -> int:
        try:
            trade_date_iso = pd.Timestamp(trade_date).date().isoformat()
        except Exception:
            return 0
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        source_token = self._normalize_text(source) or "twse_etf_daily"

        dedup: dict[
            str,
            tuple[
                str,
                float | None,
                float | None,
                int | None,
                float,
                float,
                float,
                float,
                float | None,
                str,
            ],
        ] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = self._normalize_symbol_token(
                row.get("etf_code") or row.get("code") or row.get("symbol")
            )
            if not code:
                continue
            name = self._normalize_text(row.get("etf_name") or row.get("name")) or code
            trade_value = pd.to_numeric(row.get("trade_value"), errors="coerce")
            trade_volume = pd.to_numeric(row.get("trade_volume"), errors="coerce")
            trade_count = pd.to_numeric(row.get("trade_count"), errors="coerce")
            open_ = pd.to_numeric(row.get("open"), errors="coerce")
            high = pd.to_numeric(row.get("high"), errors="coerce")
            low = pd.to_numeric(row.get("low"), errors="coerce")
            close = pd.to_numeric(row.get("close"), errors="coerce")
            change = pd.to_numeric(row.get("change"), errors="coerce")
            if (
                not pd.notna(open_)
                or not pd.notna(high)
                or not pd.notna(low)
                or not pd.notna(close)
            ):
                continue
            row_source = self._normalize_text(row.get("source")) or source_token
            dedup[code] = (
                name,
                float(trade_value) if pd.notna(trade_value) else None,
                float(trade_volume) if pd.notna(trade_volume) else None,
                int(trade_count) if pd.notna(trade_count) else None,
                float(open_),
                float(high),
                float(low),
                float(close),
                float(change) if pd.notna(change) else None,
                row_source,
            )

        if not dedup:
            return 0

        with self._connect_ctx() as conn:
            for code, values in dedup.items():
                conn.execute(
                    "DELETE FROM tw_etf_daily_market WHERE etf_code=? AND trade_date=?",
                    (code, trade_date_iso),
                )
                conn.execute(
                    """
                    INSERT INTO tw_etf_daily_market(
                        trade_date, etf_code, etf_name, trade_value, trade_volume, trade_count,
                        open, high, low, close, change, source, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_date_iso,
                        code,
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[8],
                        values[9],
                        now_iso,
                    ),
                )
        return len(dedup)

    def load_tw_etf_daily_market(
        self,
        *,
        start: datetime | date | str | None = None,
        end: datetime | date | str | None = None,
        etf_codes: list[str] | None = None,
    ) -> pd.DataFrame:
        where: list[str] = []
        params: list[object] = []
        if start is not None:
            where.append("trade_date>=?")
            params.append(pd.Timestamp(start).date().isoformat())
        if end is not None:
            where.append("trade_date<=?")
            params.append(pd.Timestamp(end).date().isoformat())
        normalized_codes: list[str] = []
        seen: set[str] = set()
        for code in etf_codes or []:
            token = self._normalize_symbol_token(code)
            if not token or token in seen:
                continue
            normalized_codes.append(token)
            seen.add(token)
        if normalized_codes:
            placeholders = ", ".join(["?"] * len(normalized_codes))
            where.append(f"etf_code IN ({placeholders})")
            params.extend(normalized_codes)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        conn = self._connect()
        try:
            df = conn.execute(
                f"""
                SELECT trade_date, etf_code, etf_name, trade_value, trade_volume, trade_count,
                       open, high, low, close, change, source, fetched_at
                FROM tw_etf_daily_market
                {where_sql}
                ORDER BY trade_date ASC, etf_code ASC
                """,
                params,
            ).df()
        finally:
            conn.close()
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "etf_code",
                    "etf_name",
                    "trade_value",
                    "trade_volume",
                    "trade_count",
                    "open",
                    "high",
                    "low",
                    "close",
                    "change",
                    "source",
                    "fetched_at",
                ]
            )
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date.astype(str)
        for column in [
            "trade_value",
            "trade_volume",
            "trade_count",
            "open",
            "high",
            "low",
            "close",
            "change",
        ]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def load_tw_etf_daily_market_coverage(
        self,
        *,
        start: datetime | date | str | None = None,
        end: datetime | date | str | None = None,
        etf_codes: list[str] | None = None,
    ) -> dict[str, object]:
        where: list[str] = []
        params: list[object] = []
        if start is not None:
            where.append("trade_date>=?")
            params.append(pd.Timestamp(start).date().isoformat())
        if end is not None:
            where.append("trade_date<=?")
            params.append(pd.Timestamp(end).date().isoformat())
        normalized_codes: list[str] = []
        seen: set[str] = set()
        for code in etf_codes or []:
            token = self._normalize_symbol_token(code)
            if not token or token in seen:
                continue
            normalized_codes.append(token)
            seen.add(token)
        if normalized_codes:
            placeholders = ", ".join(["?"] * len(normalized_codes))
            where.append(f"etf_code IN ({placeholders})")
            params.extend(normalized_codes)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        conn = self._connect()
        try:
            row = conn.execute(
                f"""
                SELECT COUNT(*), MIN(trade_date), MAX(trade_date),
                       COUNT(DISTINCT trade_date), COUNT(DISTINCT etf_code)
                FROM tw_etf_daily_market
                {where_sql}
                """,
                params,
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {
                "row_count": 0,
                "first_date": None,
                "last_date": None,
                "trade_date_count": 0,
                "symbol_count": 0,
            }
        return {
            "row_count": max(0, int(row[0] or 0)),
            "first_date": self._parse_iso_datetime(row[1]),
            "last_date": self._parse_iso_datetime(row[2]),
            "trade_date_count": max(0, int(row[3] or 0)),
            "symbol_count": max(0, int(row[4] or 0)),
        }

    def save_tw_etf_mis_daily(
        self,
        *,
        rows: list[dict[str, object]],
        trade_date: str,
        source: str = "twse_mis_etf_indicator",
    ) -> int:
        try:
            trade_date_iso = pd.Timestamp(trade_date).date().isoformat()
        except Exception:
            return 0
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        source_token = self._normalize_text(source) or "twse_mis_etf_indicator"

        dedup: dict[
            str,
            tuple[
                str,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
                str,
                str,
                str,
            ],
        ] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = self._normalize_symbol_token(
                row.get("etf_code") or row.get("code") or row.get("symbol")
            )
            if not code:
                continue
            name = self._normalize_text(row.get("etf_name") or row.get("name")) or code
            issued_units = pd.to_numeric(row.get("issued_units"), errors="coerce")
            creation_redemption_diff = pd.to_numeric(
                row.get("creation_redemption_diff"), errors="coerce"
            )
            market_price = pd.to_numeric(row.get("market_price"), errors="coerce")
            estimated_nav = pd.to_numeric(row.get("estimated_nav"), errors="coerce")
            premium_discount_pct = pd.to_numeric(row.get("premium_discount_pct"), errors="coerce")
            previous_nav = pd.to_numeric(row.get("previous_nav"), errors="coerce")
            reference_url = self._normalize_text(row.get("reference_url"))
            updated_at = self._normalize_text(row.get("updated_at")) or now_iso
            row_source = self._normalize_text(row.get("source")) or source_token
            dedup[code] = (
                name,
                float(issued_units) if pd.notna(issued_units) else None,
                float(creation_redemption_diff) if pd.notna(creation_redemption_diff) else None,
                float(market_price) if pd.notna(market_price) else None,
                float(estimated_nav) if pd.notna(estimated_nav) else None,
                float(premium_discount_pct) if pd.notna(premium_discount_pct) else None,
                float(previous_nav) if pd.notna(previous_nav) else None,
                reference_url,
                updated_at,
                row_source,
            )

        if not dedup:
            return 0

        with self._connect_ctx() as conn:
            for code, values in dedup.items():
                conn.execute(
                    "DELETE FROM tw_etf_mis_daily WHERE etf_code=? AND trade_date=?",
                    (code, trade_date_iso),
                )
                conn.execute(
                    """
                    INSERT INTO tw_etf_mis_daily(
                        trade_date, etf_code, etf_name, issued_units, creation_redemption_diff,
                        market_price, estimated_nav, premium_discount_pct, previous_nav,
                        reference_url, updated_at, source, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_date_iso,
                        code,
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[8],
                        values[9],
                        now_iso,
                    ),
                )
        return len(dedup)

    def load_tw_etf_mis_daily(
        self,
        *,
        start: datetime | date | str | None = None,
        end: datetime | date | str | None = None,
        etf_codes: list[str] | None = None,
    ) -> pd.DataFrame:
        where: list[str] = []
        params: list[object] = []
        if start is not None:
            where.append("trade_date>=?")
            params.append(pd.Timestamp(start).date().isoformat())
        if end is not None:
            where.append("trade_date<=?")
            params.append(pd.Timestamp(end).date().isoformat())
        normalized_codes: list[str] = []
        seen: set[str] = set()
        for code in etf_codes or []:
            token = self._normalize_symbol_token(code)
            if not token or token in seen:
                continue
            normalized_codes.append(token)
            seen.add(token)
        if normalized_codes:
            placeholders = ", ".join(["?"] * len(normalized_codes))
            where.append(f"etf_code IN ({placeholders})")
            params.extend(normalized_codes)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        with self._connect_ctx() as conn:
            df = conn.execute(
                f"""
                SELECT trade_date, etf_code, etf_name, issued_units, creation_redemption_diff,
                       market_price, estimated_nav, premium_discount_pct, previous_nav,
                       reference_url, updated_at, source, fetched_at
                FROM tw_etf_mis_daily
                {where_sql}
                ORDER BY trade_date ASC, etf_code ASC
                """,
                params,
            ).df()
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "etf_code",
                    "etf_name",
                    "issued_units",
                    "creation_redemption_diff",
                    "market_price",
                    "estimated_nav",
                    "premium_discount_pct",
                    "previous_nav",
                    "reference_url",
                    "updated_at",
                    "source",
                    "fetched_at",
                ]
            )
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date.astype(str)
        for column in [
            "issued_units",
            "creation_redemption_diff",
            "market_price",
            "estimated_nav",
            "premium_discount_pct",
            "previous_nav",
        ]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def load_tw_etf_mis_daily_coverage(
        self,
        *,
        start: datetime | date | str | None = None,
        end: datetime | date | str | None = None,
        etf_codes: list[str] | None = None,
    ) -> dict[str, object]:
        where: list[str] = []
        params: list[object] = []
        if start is not None:
            where.append("trade_date>=?")
            params.append(pd.Timestamp(start).date().isoformat())
        if end is not None:
            where.append("trade_date<=?")
            params.append(pd.Timestamp(end).date().isoformat())
        normalized_codes: list[str] = []
        seen: set[str] = set()
        for code in etf_codes or []:
            token = self._normalize_symbol_token(code)
            if not token or token in seen:
                continue
            normalized_codes.append(token)
            seen.add(token)
        if normalized_codes:
            placeholders = ", ".join(["?"] * len(normalized_codes))
            where.append(f"etf_code IN ({placeholders})")
            params.extend(normalized_codes)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        with self._connect_ctx() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*), MIN(trade_date), MAX(trade_date),
                       COUNT(DISTINCT trade_date), COUNT(DISTINCT etf_code)
                FROM tw_etf_mis_daily
                {where_sql}
                """,
                params,
            ).fetchone()
        if row is None:
            return {
                "row_count": 0,
                "first_date": None,
                "last_date": None,
                "trade_date_count": 0,
                "symbol_count": 0,
            }
        return {
            "row_count": max(0, int(row[0] or 0)),
            "first_date": self._parse_iso_datetime(row[1]),
            "last_date": self._parse_iso_datetime(row[2]),
            "trade_date_count": max(0, int(row[3] or 0)),
            "symbol_count": max(0, int(row[4] or 0)),
        }

    def load_universe_snapshot(self, universe_id: str) -> UniverseSnapshot | None:
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

    def save_heatmap_run(self, universe_id: str, payload: dict[str, object]) -> int:
        key = str(universe_id or "").strip().upper()
        with self._connect_ctx() as conn:
            rid = self._next_id(conn, "heatmap_runs")
            conn.execute(
                "INSERT INTO heatmap_runs(id, universe_id, created_at, payload_json) VALUES(?, ?, ?, ?)",
                (
                    rid,
                    key,
                    datetime.now(tz=timezone.utc).isoformat(),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            return int(rid)

    def load_latest_heatmap_run(self, universe_id: str) -> HeatmapRun | None:
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
        payload: dict[str, object] = {}
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
        pin_as_card: bool | None = None,
    ) -> None:
        code = self._normalize_symbol_token(etf_code)
        if not code:
            return
        name = self._normalize_text(etf_name) or code
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect_ctx() as conn:
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
        with self._connect_ctx() as conn:
            row = conn.execute(
                "SELECT 1 FROM heatmap_hub_entries WHERE etf_code=? LIMIT 1", (code,)
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE heatmap_hub_entries SET pin_as_card=?, updated_at=? WHERE etf_code=?",
                (1 if bool(pin_as_card) else 0, now, code),
            )
            return True

    def upsert_client_visit(
        self,
        *,
        session_id: str,
        ip_address: str,
        forwarded_for: str = "",
        user_agent: str = "",
        last_page: str = "",
    ) -> bool:
        session_key = self._normalize_text(session_id)
        if not session_key:
            return False
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with self._connect_ctx() as conn:
            row = conn.execute(
                "SELECT visit_count, first_seen_at FROM client_visits WHERE session_id=?",
                (session_key,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO client_visits(
                        session_id, ip_address, forwarded_for, user_agent, last_page,
                        visit_count, first_seen_at, last_seen_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_key,
                        self._normalize_text(ip_address),
                        self._normalize_text(forwarded_for),
                        self._normalize_text(user_agent),
                        self._normalize_text(last_page),
                        1,
                        now_iso,
                        now_iso,
                    ),
                )
                return True
            visit_count = max(0, int(row[0] or 0)) + 1
            first_seen_at = str(row[1] or now_iso)
            conn.execute(
                """
                UPDATE client_visits
                SET ip_address=?, forwarded_for=?, user_agent=?, last_page=?,
                    visit_count=?, first_seen_at=?, last_seen_at=?
                WHERE session_id=?
                """,
                (
                    self._normalize_text(ip_address),
                    self._normalize_text(forwarded_for),
                    self._normalize_text(user_agent),
                    self._normalize_text(last_page),
                    visit_count,
                    first_seen_at,
                    now_iso,
                    session_key,
                ),
            )
            return True

    def list_recent_client_visits(self, *, limit: int = 20) -> list[ClientVisit]:
        try:
            limit_value = max(1, min(int(limit), 200))
        except Exception:
            limit_value = 20
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT session_id, ip_address, forwarded_for, user_agent, last_page,
                       visit_count, first_seen_at, last_seen_at
                FROM client_visits
                ORDER BY last_seen_at DESC, first_seen_at DESC, session_id ASC
                LIMIT ?
                """,
                (limit_value,),
            ).fetchall()
        finally:
            conn.close()
        now = datetime.now(tz=timezone.utc)
        out: list[ClientVisit] = []
        for row in rows:
            out.append(
                ClientVisit(
                    session_id=self._normalize_text(row[0]),
                    ip_address=self._normalize_text(row[1]),
                    forwarded_for=self._normalize_text(row[2]),
                    user_agent=self._normalize_text(row[3]),
                    last_page=self._normalize_text(row[4]),
                    visit_count=max(0, int(row[5] or 0)),
                    first_seen_at=self._parse_iso_datetime(row[6]) or now,
                    last_seen_at=self._parse_iso_datetime(row[7]) or now,
                )
            )
        return out

    def save_message_board_entry(
        self,
        *,
        author_name: str,
        body: str,
        parent_message_id: str = "",
        ip_address: str = "",
        user_agent: str = "",
    ) -> str:
        author = self._normalize_text(author_name) or "訪客"
        text = self._normalize_text(body)
        if not text:
            raise ValueError("body is required")
        parent_id = self._normalize_text(parent_message_id)
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        message_id = f"msg_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        with self._connect_ctx() as conn:
            rid = self._next_id(conn, "message_board_entries")
            conn.execute(
                """
                INSERT INTO message_board_entries(
                    id, message_id, parent_message_id, author_name, body,
                    ip_address, user_agent, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    message_id,
                    parent_id,
                    author[:40],
                    text[:2000],
                    self._normalize_text(ip_address),
                    self._normalize_text(user_agent)[:300],
                    now_iso,
                    now_iso,
                ),
            )
        return message_id

    def list_message_board_entries(self, *, limit: int = 200) -> list[MessageBoardEntry]:
        try:
            limit_value = max(1, min(int(limit), 1000))
        except Exception:
            limit_value = 200
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT message_id, parent_message_id, author_name, body,
                       ip_address, user_agent, created_at, updated_at
                FROM message_board_entries
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit_value,),
            ).fetchall()
        finally:
            conn.close()
        now = datetime.now(tz=timezone.utc)
        out: list[MessageBoardEntry] = []
        for row in rows:
            out.append(
                MessageBoardEntry(
                    message_id=self._normalize_text(row[0]),
                    parent_message_id=self._normalize_text(row[1]),
                    author_name=self._normalize_text(row[2]) or "訪客",
                    body=self._normalize_text(row[3]),
                    ip_address=self._normalize_text(row[4]),
                    user_agent=self._normalize_text(row[5]),
                    created_at=self._parse_iso_datetime(row[6]) or now,
                    updated_at=self._parse_iso_datetime(row[7]) or now,
                )
            )
        return out

    def update_message_board_entry(self, *, message_id: str, body: str) -> bool:
        entry_id = self._normalize_text(message_id)
        text = self._normalize_text(body)
        if not entry_id:
            return False
        if not text:
            raise ValueError("body is required")
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with self._connect_ctx() as conn:
            row = conn.execute(
                "SELECT 1 FROM message_board_entries WHERE message_id=?",
                (entry_id,),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                """
                UPDATE message_board_entries
                SET body=?, updated_at=?
                WHERE message_id=?
                """,
                (text[:2000], now_iso, entry_id),
            )
        return True

    def delete_message_board_entry(self, *, message_id: str) -> int:
        entry_id = self._normalize_text(message_id)
        if not entry_id:
            return 0
        with self._connect_ctx() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM message_board_entries
                WHERE message_id=? OR parent_message_id=?
                """,
                (entry_id, entry_id),
            ).fetchone()
            delete_count = max(0, int(row[0] or 0)) if row is not None else 0
            if delete_count <= 0:
                return 0
            conn.execute(
                """
                DELETE FROM message_board_entries
                WHERE message_id=? OR parent_message_id=?
                """,
                (entry_id, entry_id),
            )
        return delete_count

    @staticmethod
    def _default_notebook_title(title: object, *, note_id: object = "") -> str:
        text = str(title or "").strip()
        if text:
            return text
        note_key = str(note_id or "").strip()
        return note_key or "未命名筆記"

    def create_notebook_entry(self, *, title: str = "未命名筆記", body: str = "") -> str:
        note_key = uuid.uuid4().hex
        title_text = self._default_notebook_title(title, note_id=note_key)
        text = str(body or "")
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with self._connect_ctx() as conn:
            rid = self._next_id(conn, "notebook_entries")
            conn.execute(
                """
                INSERT INTO notebook_entries(
                    id, note_id, title, body, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (rid, note_key, title_text, text, now_iso, now_iso),
            )
        return note_key

    def list_notebook_entries(self, *, limit: int = 200) -> list[NotebookEntry]:
        safe_limit = max(1, int(limit or 200))
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT note_id, title, body, created_at, updated_at
                FROM notebook_entries
                ORDER BY updated_at DESC, created_at DESC, note_id ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        finally:
            conn.close()
        now = datetime.now(tz=timezone.utc)
        return [
            NotebookEntry(
                note_id=self._normalize_text(row[0]) or uuid.uuid4().hex,
                title=self._default_notebook_title(row[1], note_id=row[0]),
                body=str(row[2] or ""),
                created_at=self._parse_iso_datetime(row[3]) or now,
                updated_at=self._parse_iso_datetime(row[4]) or now,
            )
            for row in rows
        ]

    def save_notebook_entry(self, *, note_id: str = "default", title: str = "", body: str) -> str:
        note_key = self._normalize_text(note_id) or "default"
        title_text = self._default_notebook_title(title, note_id=note_key)
        text = str(body or "")
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with self._connect_ctx() as conn:
            row = conn.execute(
                "SELECT id, created_at FROM notebook_entries WHERE note_id=?",
                (note_key,),
            ).fetchone()
            if row is None:
                rid = self._next_id(conn, "notebook_entries")
                conn.execute(
                    """
                    INSERT INTO notebook_entries(
                        id, note_id, title, body, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (rid, note_key, title_text, text, now_iso, now_iso),
                )
            else:
                conn.execute(
                    """
                    UPDATE notebook_entries
                    SET title=?, body=?, updated_at=?
                    WHERE note_id=?
                    """,
                    (title_text, text, now_iso, note_key),
                )
        return note_key

    def delete_notebook_entry(self, *, note_id: str = "default") -> bool:
        note_key = self._normalize_text(note_id) or "default"
        with self._connect_ctx() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM notebook_entries WHERE note_id=?",
                (note_key,),
            ).fetchone()
            if row is None or int(row[0] or 0) <= 0:
                return False
            conn.execute("DELETE FROM notebook_entries WHERE note_id=?", (note_key,))
        return True

    def load_notebook_entry(self, *, note_id: str = "default") -> NotebookEntry | None:
        note_key = self._normalize_text(note_id) or "default"
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT note_id, title, body, created_at, updated_at
                FROM notebook_entries
                WHERE note_id=?
                """,
                (note_key,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        now = datetime.now(tz=timezone.utc)
        return NotebookEntry(
            note_id=self._normalize_text(row[0]) or note_key,
            title=self._default_notebook_title(row[1], note_id=row[0]),
            body=str(row[2] or ""),
            created_at=self._parse_iso_datetime(row[3]) or now,
            updated_at=self._parse_iso_datetime(row[4]) or now,
        )

    def save_rotation_run(
        self,
        universe_id: str,
        run_key: str,
        params: dict[str, object],
        payload: dict[str, object],
    ) -> int:
        universe_key = str(universe_id or "").strip().upper()
        if not universe_key:
            raise ValueError("universe_id is required")
        with self._connect_ctx() as conn:
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

    def load_latest_rotation_run(self, universe_id: str) -> RotationRun | None:
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
        params: dict[str, object] = {}
        payload: dict[str, object] = {}
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
