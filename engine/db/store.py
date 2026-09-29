"""SQLite persistence layer for trade/run/risk-event history.

Single-file DB by default at data/trading.db -- fine for one user, one
machine. If this ever needs concurrent access (e.g. a dashboard reading
while the bot writes), consider Postgres, but SQLite's WAL mode handles
light concurrent reads fine for a solo setup.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"


@dataclass
class TradeRecord:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    entry_time: str
    exit_price: float | None = None
    exit_time: str | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    mae: float | None = None
    mfe: float | None = None
    commission: float = 0.0
    exit_reason: str | None = None


class TradeStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_PATH.read_text())
            conn.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_run(
        self,
        run_type: str,
        strategy_name: str,
        strategy_params: dict,
        symbol: str,
        timeframe: str,
        starting_balance: float,
        started_at: str,
        notes: str = "",
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO runs
                   (run_type, strategy_name, strategy_params, symbol, timeframe,
                    starting_balance, started_at, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_type,
                    strategy_name,
                    json.dumps(strategy_params),
                    symbol,
                    timeframe,
                    starting_balance,
                    started_at,
                    notes,
                ),
            )
            return cur.lastrowid

    def end_run(self, run_id: int, ended_at: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE runs SET ended_at = ? WHERE id = ?", (ended_at, run_id))

    def add_trade(self, run_id: int, trade: TradeRecord) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (run_id, symbol, side, quantity, entry_price, exit_price,
                    entry_time, exit_time, pnl, pnl_pct, mae, mfe, commission,
                    exit_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    trade.symbol,
                    trade.side,
                    trade.quantity,
                    trade.entry_price,
                    trade.exit_price,
                    trade.entry_time,
                    trade.exit_time,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.mae,
                    trade.mfe,
                    trade.commission,
                    trade.exit_reason,
                ),
            )
            return cur.lastrowid

    def add_equity_point(self, run_id: int, timestamp: str, balance: float, drawdown_pct: float = 0.0) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO equity_curve (run_id, timestamp, balance, drawdown_pct) VALUES (?, ?, ?, ?)",
                (run_id, timestamp, balance, drawdown_pct),
            )

    def add_risk_event(self, run_id: int, timestamp: str, event_type: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO risk_events (run_id, timestamp, event_type, reason) VALUES (?, ?, ?, ?)",
                (run_id, timestamp, event_type, reason),
            )

    def get_trades(self, run_id: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM trades WHERE run_id = ? ORDER BY entry_time", (run_id,)
            ).fetchall()

    def get_runs(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
