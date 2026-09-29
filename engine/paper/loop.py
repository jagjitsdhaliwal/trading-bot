"""Live paper-trading loop.

Polls a data source on an interval, re-runs the strategy on the accumulated
history each tick (expanding-window backtest), and if the strategy's most
recent signal implies a new position, routes the order through the risk
engine before sending it to a broker adapter (PaperBrokerAdapter by default
-- no live account needed).

Why "re-run backtest on expanding window" instead of a purpose-built live
strategy API: backtesting.py's Strategy class is designed for vectorized
historical runs, not incremental bar-by-bar state. Re-running it each tick
on the full history-so-far is wasteful CPU-wise for high-frequency loops,
but for a polling interval of minutes on daily/hourly bars it's simple,
correct, and guarantees the exact same code path as your backtests -- no
risk of the live logic silently drifting from what you validated in
Phase 1-3. Revisit this if you move to sub-minute polling.
"""

from __future__ import annotations

import datetime as dt
import signal
import time
from typing import Type

import pandas as pd
from backtesting import Strategy

from engine.broker.base import OrderSide, OrderType, Quote
from engine.broker.paper import PaperBrokerAdapter
from engine.data.base import DataSource
from engine.db.store import TradeRecord, TradeStore
from engine.risk.engine import RiskDecision, RiskEngine


class PaperTradingLoop:
    def __init__(
        self,
        data_source: DataSource,
        strategy_cls: Type[Strategy],
        symbol: str,
        timeframe: str,
        risk_engine: RiskEngine,
        broker: PaperBrokerAdapter | None = None,
        store: TradeStore | None = None,
        poll_seconds: int = 60,
        history_days: int = 365,
        tick_value: float = 5.0,  # MES: $5/point ($1.25/tick at 0.25 tick size)
    ):
        self.data_source = data_source
        self.strategy_cls = strategy_cls
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_engine = risk_engine
        self.broker = broker or PaperBrokerAdapter(
            starting_balance=risk_engine.limits.account_starting_balance
        )
        self.store = store or TradeStore()
        self.poll_seconds = poll_seconds
        self.history_days = history_days
        self.tick_value = tick_value

        self._run_id: int | None = None
        self._last_position_size = 0
        self._running = False

    def _fetch_history(self) -> pd.DataFrame:
        end = dt.datetime.now()
        start = end - dt.timedelta(days=self.history_days)
        return self.data_source.get_ohlcv(self.symbol, self.timeframe, start, end)

    def _run_strategy_on_history(self, df: pd.DataFrame):
        from backtesting import Backtest

        bt = Backtest(
            df,
            self.strategy_cls,
            cash=self.risk_engine.limits.account_starting_balance,
            commission=0.0001,
            exclusive_orders=True,
        )
        return bt.run()

    def _handle_signal(self, stats, latest_price: float, latest_time: dt.datetime) -> None:
        trades = stats.get("_trades")
        if trades is None or trades.empty:
            print(f"[{latest_time}] No signal yet (no trades generated on history so far). "
                  f"Last price: {latest_price:.2f}")
            return

        last_trade = trades.iloc[-1]
        is_open_trade = pd.isna(last_trade.get("ExitTime"))
        target_size = int(last_trade["Size"]) if is_open_trade else 0

        if target_size == self._last_position_size:
            state = "flat" if target_size == 0 else f"holding {target_size}"
            print(f"[{latest_time}] No change in position ({state}). Last price: {latest_price:.2f}")
            return  # no change in desired position

        delta = target_size - self._last_position_size
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = abs(delta)

        stop_distance_points = abs(latest_price - last_trade.get("SL", latest_price)) or 1.0
        proposed_risk_dollars = stop_distance_points * self.tick_value * quantity

        decision, reason = self.risk_engine.check_order(
            proposed_risk_dollars=proposed_risk_dollars, quantity=quantity
        )

        print(f"[{latest_time}] Signal: {side.value} {quantity} {self.symbol} @ {latest_price:.2f}"
              f" -- risk engine: {decision.value} ({reason})")

        if decision != RiskDecision.APPROVED:
            self.store.add_risk_event(
                self._run_id, latest_time.isoformat(), "rejected_order", reason
            )
            return

        self.broker.set_quote(
            self.symbol,
            Quote(timestamp=latest_time, bid=latest_price, ask=latest_price, last=latest_price),
        )
        result = self.broker.place_order(self.symbol, side, quantity, OrderType.MARKET)
        print(f"  -> Filled {result.side.value} {result.quantity} @ {result.filled_price:.2f}")

        self._last_position_size = target_size

        if not is_open_trade and "PnL" in last_trade:
            pnl = float(last_trade["PnL"])
            self.risk_engine.record_trade_result(pnl)
            self.store.add_trade(
                self._run_id,
                TradeRecord(
                    symbol=self.symbol,
                    side="LONG" if last_trade["Size"] > 0 else "SHORT",
                    quantity=abs(int(last_trade["Size"])),
                    entry_price=float(last_trade["EntryPrice"]),
                    exit_price=float(last_trade["ExitPrice"]),
                    entry_time=str(last_trade["EntryTime"]),
                    exit_time=str(last_trade["ExitTime"]),
                    pnl=pnl,
                    pnl_pct=float(last_trade["ReturnPct"]),
                    exit_reason="signal",
                ),
            )

        self.store.add_equity_point(
            self._run_id, latest_time.isoformat(), self.broker.get_account_balance()
        )

    def run_once(self) -> None:
        """Single iteration -- useful for testing without the polling loop."""
        today = dt.date.today()
        self.risk_engine.start_new_day(today)

        df = self._fetch_history()
        stats = self._run_strategy_on_history(df)

        latest_price = float(df["Close"].iloc[-1])
        latest_time = df.index[-1].to_pydatetime()

        self._handle_signal(stats, latest_price, latest_time)

    def start(self) -> None:
        self._run_id = self.store.create_run(
            run_type="paper",
            strategy_name=self.strategy_cls.__name__,
            strategy_params={},
            symbol=self.symbol,
            timeframe=self.timeframe,
            starting_balance=self.risk_engine.limits.account_starting_balance,
            started_at=dt.datetime.now().isoformat(),
            notes=f"Live paper loop, poll_seconds={self.poll_seconds}",
        )
        self._running = True
        print(f"Started paper trading run_id={self._run_id} for {self.symbol} "
              f"({self.timeframe}), polling every {self.poll_seconds}s. Ctrl+C to stop.")

        # SIGTERM (e.g. `kill <pid>`, or a process manager stopping this)
        # doesn't raise KeyboardInterrupt by default -- without this handler
        # the run would stay marked open in the database forever.
        def _handle_sigterm(signum, frame):
            raise KeyboardInterrupt

        previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)

        try:
            while self._running:
                if self.risk_engine.state.kill_switch_triggered:
                    print(f"KILL SWITCH ACTIVE: {self.risk_engine.state.kill_switch_reason}")
                    print("Manual reset required (engine.reset_kill_switch()) to resume.")
                    break

                try:
                    self.run_once()
                except Exception as e:
                    print(f"Error during tick: {e}")

                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            print("\nStopping paper trading loop...")
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            self.store.end_run(self._run_id, dt.datetime.now().isoformat())
            print(f"Run {self._run_id} ended. Final balance: "
                  f"${self.broker.get_account_balance():,.2f}")

    def stop(self) -> None:
        self._running = False
