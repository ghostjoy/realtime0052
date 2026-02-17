from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from providers.base import ProviderRequest
from services.market_data_service import MarketDataService

DB_PATH_ENV_VAR = "REALTIME0052_DB_PATH"
INTRADAY_RETAIN_DAYS_ENV_VAR = "REALTIME0052_INTRADAY_RETAIN_DAYS"
DEFAULT_DB_FILENAME = "market_history.sqlite3"
DEFAULT_INTRADAY_RETAIN_DAYS = 365 * 3
ICLOUD_DOCS_ROOT = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
DEFAULT_ICLOUD_DB_PATH = ICLOUD_DOCS_ROOT / "codexapp" / DEFAULT_DB_FILENAME


def resolve_history_db_path(db_path: Optional[str] = None) -> Path:
    if db_path:
        return Path(db_path).expanduser()

    env_path = str(os.getenv(DB_PATH_ENV_VAR, "")).strip()
    if env_path:
        return Path(env_path).expanduser()

    if ICLOUD_DOCS_ROOT.exists():
        return DEFAULT_ICLOUD_DB_PATH
    return Path(DEFAULT_DB_FILENAME)


def _copy_legacy_local_db_if_needed(target_path: Path):
    local_path = Path(DEFAULT_DB_FILENAME)
    if local_path == target_path:
        return
    if target_path.exists() or not local_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, target_path)


def resolve_intraday_retain_days(days: Optional[int] = None) -> int:
    if days is not None:
        try:
            return max(1, int(days))
        except Exception:
            return DEFAULT_INTRADAY_RETAIN_DAYS
    raw = str(os.getenv(INTRADAY_RETAIN_DAYS_ENV_VAR, "")).strip()
    if not raw:
        return DEFAULT_INTRADAY_RETAIN_DAYS
    try:
        return max(1, int(raw))
    except Exception:
        return DEFAULT_INTRADAY_RETAIN_DAYS


@dataclass(frozen=True)
class SyncReport:
    symbol: str
    market: str
    source: str
    rows_upserted: int
    started_at: datetime
    finished_at: datetime
    fallback_depth: int
    stale: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class UniverseSnapshot:
    universe_id: str
    symbols: list[str]
    source: str
    fetched_at: datetime


@dataclass(frozen=True)
class HeatmapRun:
    universe_id: str
    payload: Dict[str, object]
    created_at: datetime


@dataclass(frozen=True)
class RotationRun:
    universe_id: str
    run_key: str
    params: Dict[str, object]
    payload: Dict[str, object]
    created_at: datetime


@dataclass(frozen=True)
class BacktestReplayRun:
    run_key: str
    params: Dict[str, object]
    payload: Dict[str, object]
    created_at: datetime


class HistoryStore:
    def __init__(self, db_path: Optional[str] = None, service: Optional[MarketDataService] = None):
        self.db_path = resolve_history_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path is None and not str(os.getenv(DB_PATH_ENV_VAR, "")).strip():
            _copy_legacy_local_db_if_needed(self.db_path)
        self.intraday_retain_days = resolve_intraday_retain_days()
        self.service = service or MarketDataService()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS instruments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    name TEXT,
                    currency TEXT,
                    timezone TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(symbol, market)
                );

                CREATE TABLE IF NOT EXISTS daily_bars (
                    instrument_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    adj_close REAL,
                    source TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY(instrument_id, date),
                    FOREIGN KEY(instrument_id) REFERENCES instruments(id)
                );

                CREATE TABLE IF NOT EXISTS intraday_ticks (
                    instrument_id INTEGER NOT NULL,
                    ts_utc TEXT NOT NULL,
                    price REAL NOT NULL,
                    cum_volume REAL NOT NULL DEFAULT 0,
                    source TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY(instrument_id, ts_utc),
                    FOREIGN KEY(instrument_id) REFERENCES instruments(id)
                );

                CREATE TABLE IF NOT EXISTS sync_state (
                    instrument_id INTEGER PRIMARY KEY,
                    last_success_date TEXT,
                    last_source TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(instrument_id) REFERENCES instruments(id)
                );

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    cost_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_replay_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS universe_snapshots (
                    universe_id TEXT PRIMARY KEY,
                    symbols_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS heatmap_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    universe_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rotation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    universe_id TEXT NOT NULL,
                    run_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_daily_bars_market_symbol_date
                    ON daily_bars(instrument_id, date);

                CREATE INDEX IF NOT EXISTS idx_intraday_ticks_instrument_ts
                    ON intraday_ticks(instrument_id, ts_utc);

                CREATE INDEX IF NOT EXISTS idx_sync_state_updated_at
                    ON sync_state(updated_at);

                CREATE INDEX IF NOT EXISTS idx_heatmap_runs_universe_created_at
                    ON heatmap_runs(universe_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_rotation_runs_universe_created_at
                    ON rotation_runs(universe_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_backtest_replay_runs_key_created_at
                    ON backtest_replay_runs(run_key, created_at DESC);
                """
            )

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
            # Some upstream sources may produce duplicate columns for same field.
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

    def _load_first_bar_date(self, instrument_id: int) -> Optional[datetime]:
        with self._connect() as conn:
            row = conn.execute("SELECT MIN(date) FROM daily_bars WHERE instrument_id=?", (instrument_id,)).fetchone()
        if not row or not row[0]:
            return None
        return datetime.fromisoformat(str(row[0])).replace(tzinfo=timezone.utc)

    @staticmethod
    def _normalize_yahoo_symbol(symbol: str, market: str) -> str:
        if market == "TW" and "." not in symbol:
            return f"{symbol}.TW"
        return symbol

    def _get_or_create_instrument(self, symbol: str, market: str, name: Optional[str] = None) -> int:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO instruments(symbol, market, name, timezone)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(symbol, market) DO UPDATE SET name=COALESCE(excluded.name, instruments.name)
                """,
                (symbol, market, name, "UTC"),
            )
            row = conn.execute("SELECT id FROM instruments WHERE symbol=? AND market=?", (symbol, market)).fetchone()
            if row is None:
                raise RuntimeError(f"failed to load instrument id for {market}:{symbol}")
            return int(row[0])

    def _load_last_success_date(self, instrument_id: int) -> Optional[datetime]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_success_date FROM sync_state WHERE instrument_id=?",
                (instrument_id,),
            ).fetchone()
        if not row or not row[0]:
            return None
        return datetime.fromisoformat(str(row[0])).replace(tzinfo=timezone.utc)

    def _save_sync_state(
        self,
        instrument_id: int,
        last_success_date: Optional[datetime],
        source: Optional[str],
        error: Optional[str],
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sync_state(instrument_id, last_success_date, last_source, last_error, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(instrument_id) DO UPDATE SET
                    last_success_date=excluded.last_success_date,
                    last_source=excluded.last_source,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
                """,
                (
                    instrument_id,
                    last_success_date.isoformat() if last_success_date else None,
                    source,
                    error,
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )

    def sync_symbol_history(
        self,
        symbol: str,
        market: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> SyncReport:
        started_at = datetime.now(tz=timezone.utc)
        end = end or started_at
        instrument_id = self._get_or_create_instrument(symbol, market)
        last_success = self._load_last_success_date(instrument_id)
        first_bar_date = self._load_first_bar_date(instrument_id)
        if start is None:
            fetch_start = last_success or datetime(end.year - 5, 1, 1, tzinfo=timezone.utc)
            if last_success:
                fetch_start = max(fetch_start, last_success + pd.Timedelta(days=1))
        else:
            fetch_start = start
            if last_success and first_bar_date is not None and start >= first_bar_date and start <= last_success:
                # Requested start is already covered locally: keep incremental forward sync.
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

        if market == "US":
            providers = [self.service.us_twelve, self.service.yahoo, self.service.us_stooq]
        else:
            providers = []
            fugle_rest = getattr(self.service, "tw_fugle_rest", None)
            if fugle_rest is not None and getattr(fugle_rest, "api_key", None):
                providers.append(fugle_rest)
            providers.extend([self.service.tw_openapi, self.service.tw_tpex, self.service.yahoo])
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
        except Exception as exc:
            # Incremental sync may request "today" before a new daily bar exists.
            # If we already have historical data, treat this as non-fatal stale status.
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

        rows_upserted = 0
        with self._connect() as conn:
            for dt, row in df.iterrows():
                conn.execute(
                    """
                    INSERT INTO daily_bars(instrument_id, date, open, high, low, close, volume, adj_close, source, fetched_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(instrument_id, date) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume,
                        adj_close=COALESCE(excluded.adj_close, daily_bars.adj_close),
                        source=excluded.source,
                        fetched_at=excluded.fetched_at
                    """,
                    (
                        instrument_id,
                        pd.Timestamp(dt).date().isoformat(),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row.get("volume", 0.0)),
                        float(row["adj_close"]) if "adj_close" in df.columns and pd.notna(row["adj_close"]) else None,
                        source,
                        datetime.now(tz=timezone.utc).isoformat(),
                    ),
                )
                rows_upserted += 1

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
        instrument_id = self._get_or_create_instrument(symbol, market)
        where = ["instrument_id=?"]
        params: list[object] = [instrument_id]
        if start is not None:
            where.append("date>=?")
            params.append(start.date().isoformat())
        if end is not None:
            where.append("date<=?")
            params.append(end.date().isoformat())

        sql = f"""
            SELECT date, open, high, low, close, volume, adj_close, source
            FROM daily_bars
            WHERE {" AND ".join(where)}
            ORDER BY date ASC
        """
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "adj_close", "source"])
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
        return df

    def save_intraday_ticks(
        self,
        symbol: str,
        market: str,
        ticks: list[dict[str, object]],
        retain_days: Optional[int] = None,
    ) -> int:
        if not ticks:
            return 0
        instrument_id = self._get_or_create_instrument(symbol, market)
        normalized: dict[str, tuple[float, float, str]] = {}
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
            source = str(row.get("source", "unknown") or "unknown")
            normalized[ts.isoformat()] = (float(price_val), 0.0 if pd.isna(cum_val) else float(cum_val), source)

        if not normalized:
            return 0

        retain = resolve_intraday_retain_days(self.intraday_retain_days if retain_days is None else retain_days)
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        rows = sorted(normalized.items(), key=lambda item: item[0])
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO intraday_ticks(instrument_id, ts_utc, price, cum_volume, source, fetched_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(instrument_id, ts_utc) DO UPDATE SET
                    price=excluded.price,
                    cum_volume=excluded.cum_volume,
                    source=excluded.source,
                    fetched_at=excluded.fetched_at
                """,
                [
                    (
                        instrument_id,
                        ts_utc,
                        payload[0],
                        payload[1],
                        payload[2],
                        now_iso,
                    )
                    for ts_utc, payload in rows
                ],
            )
            cutoff = datetime.now(tz=timezone.utc) - pd.Timedelta(days=retain)
            conn.execute(
                "DELETE FROM intraday_ticks WHERE instrument_id=? AND ts_utc<?",
                (instrument_id, cutoff.isoformat()),
            )
        return len(rows)

    def load_intraday_ticks(
        self,
        symbol: str,
        market: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        instrument_id = self._get_or_create_instrument(symbol, market)
        where = ["instrument_id=?"]
        params: list[object] = [instrument_id]
        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            where.append("ts_utc>=?")
            params.append(start_ts.isoformat())
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            where.append("ts_utc<=?")
            params.append(end_ts.isoformat())

        sql = f"""
            SELECT ts_utc, price, cum_volume, source
            FROM intraday_ticks
            WHERE {" AND ".join(where)}
            ORDER BY ts_utc ASC
        """
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        if df.empty:
            return pd.DataFrame(columns=["price", "cum_volume", "source"])
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts_utc"]).set_index("ts_utc")
        return df

    def save_backtest_run(
        self,
        symbol: str,
        market: str,
        strategy: str,
        params: Dict[str, object],
        cost: Dict[str, object],
        result: Dict[str, object],
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO backtest_runs(created_at, symbol, market, strategy, params_json, cost_json, result_json)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(tz=timezone.utc).isoformat(),
                    symbol,
                    market,
                    strategy,
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(cost, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def save_backtest_replay_run(
        self,
        run_key: str,
        params: Dict[str, object],
        payload: Dict[str, object],
    ) -> int:
        key = str(run_key or "").strip()
        if not key:
            raise ValueError("run_key is required")
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO backtest_replay_runs(run_key, created_at, params_json, payload_json)
                VALUES(?, ?, ?, ?)
                """,
                (
                    key,
                    now,
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def load_latest_backtest_replay_run(self, run_key: str) -> Optional[BacktestReplayRun]:
        key = str(run_key or "").strip()
        if not key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_key, created_at, params_json, payload_json
                FROM backtest_replay_runs
                WHERE run_key=?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None

        created_at = datetime.now(tz=timezone.utc)
        try:
            created_at = datetime.fromisoformat(str(row[1]))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            pass

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
        with self._connect() as conn:
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
                (universe_id, payload, source, now, now),
            )

    def load_universe_snapshot(self, universe_id: str) -> Optional[UniverseSnapshot]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT universe_id, symbols_json, source, fetched_at
                FROM universe_snapshots
                WHERE universe_id=?
                """,
                (universe_id,),
            ).fetchone()
        if row is None:
            return None

        raw_symbols = row[1]
        try:
            symbols_obj = json.loads(raw_symbols or "[]")
        except Exception:
            symbols_obj = []
        symbols = [str(s).strip().upper() for s in symbols_obj if str(s).strip()]

        fetched = datetime.now(tz=timezone.utc)
        try:
            fetched = datetime.fromisoformat(str(row[3]))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
        except Exception:
            pass
        return UniverseSnapshot(
            universe_id=str(row[0]),
            symbols=symbols,
            source=str(row[2] or "unknown"),
            fetched_at=fetched,
        )

    def save_heatmap_run(self, universe_id: str, payload: Dict[str, object]) -> int:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO heatmap_runs(universe_id, created_at, payload_json)
                VALUES(?, ?, ?)
                """,
                (str(universe_id or "").strip().upper(), now, json.dumps(payload, ensure_ascii=False)),
            )
            return int(cur.lastrowid)

    def load_latest_heatmap_run(self, universe_id: str) -> Optional[HeatmapRun]:
        universe_key = str(universe_id or "").strip().upper()
        if not universe_key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT universe_id, created_at, payload_json
                FROM heatmap_runs
                WHERE universe_id=?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (universe_key,),
            ).fetchone()
        if row is None:
            return None

        created_at = datetime.now(tz=timezone.utc)
        try:
            created_at = datetime.fromisoformat(str(row[1]))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            pass

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
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO rotation_runs(universe_id, run_key, created_at, params_json, payload_json)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    universe_key,
                    str(run_key or ""),
                    now,
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def load_latest_rotation_run(self, universe_id: str) -> Optional[RotationRun]:
        universe_key = str(universe_id or "").strip().upper()
        if not universe_key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT universe_id, run_key, created_at, params_json, payload_json
                FROM rotation_runs
                WHERE universe_id=?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (universe_key,),
            ).fetchone()
        if row is None:
            return None

        created_at = datetime.now(tz=timezone.utc)
        try:
            created_at = datetime.fromisoformat(str(row[2]))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            pass

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
