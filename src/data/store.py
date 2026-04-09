"""SQLite database storage for candles, trades, signals, and OI snapshots."""

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("store")

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    interval TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    oi INTEGER DEFAULT 0,
    UNIQUE(symbol, exchange, interval, timestamp)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    trading_symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    option_type TEXT NOT NULL,
    strike INTEGER NOT NULL,
    expiry TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    lot_size INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    exit_price REAL DEFAULT 0,
    exit_time TEXT,
    exit_reason TEXT,
    stop_loss REAL DEFAULT 0,
    target REAL DEFAULT 0,
    pnl REAL DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    strategy TEXT,
    signal_score REAL DEFAULT 0,
    order_id TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    score REAL NOT NULL,
    technical_score REAL DEFAULT 0,
    oi_score REAL DEFAULT 0,
    iv_score REAL DEFAULT 0,
    price_action_score REAL DEFAULT 0,
    strategy TEXT,
    action_taken TEXT,
    details TEXT
);

CREATE TABLE IF NOT EXISTS oi_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    expiry TEXT NOT NULL,
    strike INTEGER NOT NULL,
    ce_oi INTEGER DEFAULT 0,
    pe_oi INTEGER DEFAULT 0,
    ce_oi_change INTEGER DEFAULT 0,
    pe_oi_change INTEGER DEFAULT 0,
    ce_volume INTEGER DEFAULT 0,
    pe_volume INTEGER DEFAULT 0,
    spot_price REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    total_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    max_drawdown REAL DEFAULT 0,
    capital REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON candles(symbol, interval, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol, entry_time);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp, symbol);
CREATE INDEX IF NOT EXISTS idx_oi_ts ON oi_snapshots(timestamp, symbol);
"""


class DataStore:
    """SQLite-based data storage for the trading system."""

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with self._connect() as conn:
            conn.executescript(CREATE_TABLES_SQL)
            logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- Candle Operations ---

    def save_candles(self, symbol: str, exchange: str, interval: str, df: pd.DataFrame):
        """Save candle data to database."""
        if df.empty:
            return

        with self._connect() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    """INSERT OR REPLACE INTO candles
                    (symbol, exchange, interval, timestamp, open, high, low, close, volume, oi)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        symbol, exchange, interval,
                        str(row["timestamp"]),
                        float(row["open"]), float(row["high"]),
                        float(row["low"]), float(row["close"]),
                        int(row["volume"]),
                        int(row.get("oi", 0)),
                    ),
                )

    def get_candles(
        self,
        symbol: str,
        interval: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Retrieve candle data from database."""
        query = "SELECT * FROM candles WHERE symbol = ? AND interval = ?"
        params: list = [symbol, interval]

        if from_date:
            query += " AND timestamp >= ?"
            params.append(str(from_date))
        if to_date:
            query += " AND timestamp <= ?"
            params.append(str(to_date))

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    # --- Trade Operations ---

    def save_trade(self, trade: dict) -> int:
        """Save a trade record. Returns the trade ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                (symbol, trading_symbol, exchange, option_type, strike, expiry,
                 side, quantity, lot_size, entry_price, entry_time, exit_price,
                 exit_time, exit_reason, stop_loss, target, pnl, pnl_pct,
                 strategy, signal_score, order_id, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade["symbol"], trade["trading_symbol"], trade["exchange"],
                    trade["option_type"], trade["strike"], trade["expiry"],
                    trade["side"], trade["quantity"], trade["lot_size"],
                    trade["entry_price"], trade["entry_time"],
                    trade.get("exit_price", 0), trade.get("exit_time"),
                    trade.get("exit_reason"), trade.get("stop_loss", 0),
                    trade.get("target", 0), trade.get("pnl", 0),
                    trade.get("pnl_pct", 0), trade.get("strategy"),
                    trade.get("signal_score", 0), trade.get("order_id"),
                    trade.get("notes"),
                ),
            )
            return cursor.lastrowid

    def update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_time: str,
        exit_reason: str,
        pnl: float,
        pnl_pct: float,
    ):
        """Update a trade with exit details."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE trades SET exit_price = ?, exit_time = ?,
                exit_reason = ?, pnl = ?, pnl_pct = ? WHERE id = ?""",
                (exit_price, exit_time, exit_reason, pnl, pnl_pct, trade_id),
            )

    def get_trades(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve trade records."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []

        if from_date:
            query += " AND entry_time >= ?"
            params.append(from_date)
        if to_date:
            query += " AND entry_time <= ?"
            params.append(to_date)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        query += " ORDER BY entry_time DESC"

        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_todays_trades(self) -> pd.DataFrame:
        """Get all trades for today."""
        today = date.today().isoformat()
        return self.get_trades(from_date=today)

    def get_todays_pnl(self) -> float:
        """Calculate total P&L for today."""
        df = self.get_todays_trades()
        if df.empty:
            return 0.0
        return float(df["pnl"].sum())

    # --- Signal Operations ---

    def save_signal(self, signal: dict) -> int:
        """Save a signal record."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO signals
                (timestamp, symbol, direction, score, technical_score,
                 oi_score, iv_score, price_action_score, strategy,
                 action_taken, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal["timestamp"], signal["symbol"], signal["direction"],
                    signal["score"], signal.get("technical_score", 0),
                    signal.get("oi_score", 0), signal.get("iv_score", 0),
                    signal.get("price_action_score", 0),
                    signal.get("strategy"), signal.get("action_taken"),
                    signal.get("details"),
                ),
            )
            return cursor.lastrowid

    # --- OI Snapshot Operations ---

    def save_oi_snapshot(self, snapshot: dict):
        """Save an OI snapshot."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO oi_snapshots
                (timestamp, symbol, expiry, strike, ce_oi, pe_oi,
                 ce_oi_change, pe_oi_change, ce_volume, pe_volume, spot_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot["timestamp"], snapshot["symbol"], snapshot["expiry"],
                    snapshot["strike"], snapshot.get("ce_oi", 0),
                    snapshot.get("pe_oi", 0), snapshot.get("ce_oi_change", 0),
                    snapshot.get("pe_oi_change", 0), snapshot.get("ce_volume", 0),
                    snapshot.get("pe_volume", 0), snapshot.get("spot_price", 0),
                ),
            )

    def get_oi_snapshots(
        self,
        symbol: str,
        from_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get OI snapshots for a symbol."""
        query = "SELECT * FROM oi_snapshots WHERE symbol = ?"
        params: list = [symbol]
        if from_time:
            query += " AND timestamp >= ?"
            params.append(from_time)
        query += " ORDER BY timestamp DESC"

        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # --- Daily P&L Operations ---

    def save_daily_pnl(self, pnl: dict):
        """Save or update daily P&L summary."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_pnl
                (date, total_pnl, realized_pnl, unrealized_pnl,
                 total_trades, winning_trades, losing_trades,
                 max_drawdown, capital)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pnl["date"], pnl["total_pnl"], pnl["realized_pnl"],
                    pnl.get("unrealized_pnl", 0), pnl["total_trades"],
                    pnl["winning_trades"], pnl["losing_trades"],
                    pnl.get("max_drawdown", 0), pnl.get("capital", 0),
                ),
            )

    def get_daily_pnl_history(self, days: int = 30) -> pd.DataFrame:
        """Get daily P&L history."""
        query = "SELECT * FROM daily_pnl ORDER BY date DESC LIMIT ?"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=[days])
