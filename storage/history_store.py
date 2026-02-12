from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from providers.base import ProviderRequest
from services.market_data_service import MarketDataService


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


class HistoryStore:
    def __init__(self, db_path: str = "market_history.sqlite3", service: Optional[MarketDataService] = None):
        self.db_path = Path(db_path)
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

                CREATE INDEX IF NOT EXISTS idx_daily_bars_market_symbol_date
                    ON daily_bars(instrument_id, date);

                CREATE INDEX IF NOT EXISTS idx_sync_state_updated_at
                    ON sync_state(updated_at);
                """
            )

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
        fetch_start = start or last_success or datetime(end.year - 5, 1, 1, tzinfo=timezone.utc)
        if last_success:
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

        providers = (
            [self.service.us_twelve, self.service.yahoo, self.service.us_stooq]
            if market == "US"
            else [self.service.tw_openapi, self.service.yahoo]
        )
        source = "unknown"
        fallback_depth = 0
        stale = False

        try:
            snap = self.service._try_ohlcv_chain(providers, request)  # noqa: SLF001
            source = snap.source
            fallback_depth = max(0, providers.index(next(p for p in providers if p.name == source)))
            df = snap.df.copy()
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
