"""Runs one paper-trading tick and prints a short status report.

Meant to be run once per day (see the /schedule or cron setup that calls
this). Intentionally simple: single tick via PaperTradingLoop, then a plain
text summary printed to stdout -- whatever schedules this is responsible
for emailing/delivering that output, not this script.

Usage:
    uv run python daily_paper_run.py
"""

from __future__ import annotations

import datetime as dt

from engine.data.yfinance_source import YFinanceDataSource
from engine.paper.loop import PaperTradingLoop
from engine.risk.engine import RiskEngine, RiskLimits
from strategies.donchian_breakout import DonchianBreakoutStrategy

SYMBOL = "SPY"
STARTING_CASH = 1_000.0


def main() -> None:
    data_source = YFinanceDataSource()
    risk_engine = RiskEngine(RiskLimits(account_starting_balance=STARTING_CASH))

    loop = PaperTradingLoop(
        data_source=data_source,
        strategy_cls=DonchianBreakoutStrategy,
        symbol=SYMBOL,
        timeframe="1d",
        risk_engine=risk_engine,
        history_days=365,
    )

    loop._run_id = loop.store.create_run(
        run_type="paper",
        strategy_name=DonchianBreakoutStrategy.__name__,
        strategy_params={"dollars_per_point": 1.0},
        symbol=SYMBOL,
        timeframe="1d",
        starting_balance=STARTING_CASH,
        started_at=dt.datetime.now().isoformat(),
        notes="Daily automated paper-trading tick",
    )

    error = None
    try:
        loop.run_once()
    except Exception as e:
        error = e
    finally:
        loop.store.end_run(loop._run_id, dt.datetime.now().isoformat())

    balance = loop.broker.get_account_balance()
    positions = loop.broker.get_positions()
    all_trades = loop.store.get_trades(loop._run_id)
    kill_switch = risk_engine.state.kill_switch_triggered

    print("=== Daily Paper Trading Report ===")
    print(f"Date: {dt.date.today()}")
    print(f"Symbol: {SYMBOL}  Strategy: donchian_breakout")
    print(f"Balance: ${balance:,.2f}  (started at ${STARTING_CASH:,.2f})")
    print(f"Open positions: {len(positions)}")
    for p in positions:
        print(f"  {p.symbol}: {p.quantity} @ {p.avg_price:.2f}")
    print(f"Trades recorded this run: {len(all_trades)}")
    if kill_switch:
        print(f"KILL SWITCH TRIGGERED: {risk_engine.state.kill_switch_reason}")
    if error:
        print(f"ERROR during run: {error}")


if __name__ == "__main__":
    main()
