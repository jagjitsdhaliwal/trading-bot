-- Trade database schema. Used for both backtest result storage and later
-- paper/live trading history. One row per closed trade; equity_curve tracks
-- account balance over time for drawdown/performance analysis.

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL CHECK (run_type IN ('backtest', 'paper', 'live')),
    strategy_name TEXT NOT NULL,
    strategy_params TEXT,           -- JSON blob of strategy parameters
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    starting_balance REAL NOT NULL,
    started_at TEXT NOT NULL,       -- ISO8601
    ended_at TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    quantity INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    pnl REAL,
    pnl_pct REAL,
    mae REAL,                       -- max adverse excursion
    mfe REAL,                       -- max favorable excursion
    commission REAL DEFAULT 0,
    exit_reason TEXT                -- 'target', 'stop', 'signal', 'session_close', 'risk_kill_switch'
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    timestamp TEXT NOT NULL,
    balance REAL NOT NULL,
    drawdown_pct REAL
);

CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,       -- 'rejected_order', 'kill_switch', 'reset'
    reason TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_equity_curve_run_id ON equity_curve(run_id);
CREATE INDEX IF NOT EXISTS idx_risk_events_run_id ON risk_events(run_id);
