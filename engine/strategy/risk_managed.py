"""Risk-managed Strategy mixin: routes every order through RiskEngine before
it reaches backtesting.py's real order placement, and feeds closed-trade
PnL back into the risk engine so daily-loss/consecutive-loss/trailing-
drawdown limits are enforced during backtests -- not just in the live
paper-trading loop.

Background: an adversarial stress-test pass found that engine/backtest/
runner.py and engine/backtest/walkforward.py never invoke RiskEngine at
all -- every backtest/walk-forward result in this project prior to this
file was generated with backtesting.py's default (effectively unbounded,
full-equity) position sizing, not the risk limits the project's risk
engine defines. This mixin closes that gap for any strategy that opts in
by inheriting from RiskManagedStrategy instead of backtesting.py's
Strategy.

Usage: change `class MyStrategy(Strategy):` to
`class MyStrategy(RiskManagedStrategy):`, call `super().init()` in your
own `init()` if you override it, and use `self.risk_buy(...)`/
`self.risk_sell(...)` instead of `self.buy()`/`self.sell()`. Also override
`estimate_risk_dollars(price, sl, size)` if your strategy's default
distance-from-entry-to-stop risk calculation needs adjusting (e.g. for
multi-leg or scale-in strategies).
"""

from __future__ import annotations

import datetime as dt

from backtesting import Strategy

from engine.risk.engine import RiskDecision, RiskEngine, RiskLimits


class RiskManagedStrategy(Strategy):
    """Drop-in replacement for backtesting.py's Strategy that gates every
    order through a RiskEngine. Risk limits are read from class-level
    `risk_limits` (a RiskLimits instance) so they can be overridden the
    same way backtesting.py lets you override any other strategy
    parameter via Backtest.run(risk_limits=..., ...).
    """

    risk_limits: RiskLimits = RiskLimits()

    def init(self):
        self._risk_engine = RiskEngine(self.risk_limits)
        self._last_seen_date: dt.date | None = None
        self._known_closed_trade_ids: set[int] = set()
        self.risk_rejections: list[tuple[object, str]] = []  # (bar_index, reason) -- for post-hoc inspection

    def _sync_risk_engine(self) -> None:
        """Call at the top of next() (or let risk_buy/risk_sell call it) to
        advance the risk engine's day counter and feed in any trades that
        closed since the last bar."""
        current_bar_time = self.data.index[-1]
        current_date = (
            current_bar_time.date() if hasattr(current_bar_time, "date") else current_bar_time
        )
        if current_date != self._last_seen_date:
            self._risk_engine.start_new_day(current_date)
            self._last_seen_date = current_date

        for trade in self.closed_trades:
            trade_id = id(trade)
            if trade_id not in self._known_closed_trade_ids:
                self._known_closed_trade_ids.add(trade_id)
                self._risk_engine.record_trade_result(trade.pl)

    def estimate_risk_dollars(self, price: float, sl: float | None, size: float) -> float:
        """Dollar risk if the position is stopped out at sl. If no sl is
        given, this does NOT silently assume a nominal/tiny distance (that
        exact bug -- defaulting an unset stop to 1.0 points -- was found in
        engine/paper/loop.py and let unstopped risk sail through the live
        risk engine undetected). Instead, an unstopped order is treated as
        risking its full notional value, which is the honest worst case
        and will correctly get rejected by max_trade_risk for any
        non-trivial position size.
        """
        contracts = abs(size) if size >= 1 else abs(size) * self.equity / price
        if sl is None:
            return contracts * price  # full notional -- honest worst case, not a guessed small number
        return contracts * abs(price - sl)

    def risk_buy(self, *, size=None, sl=None, tp=None, limit=None, stop=None, tag=None):
        return self._risk_order("buy", size=size, sl=sl, tp=tp, limit=limit, stop=stop, tag=tag)

    def risk_sell(self, *, size=None, sl=None, tp=None, limit=None, stop=None, tag=None):
        return self._risk_order("sell", size=size, sl=sl, tp=tp, limit=limit, stop=stop, tag=tag)

    def _risk_order(self, direction: str, *, size, sl, tp, limit, stop, tag):
        self._sync_risk_engine()

        price = self.data.Close[-1]
        effective_size = size if size is not None else 0.9999  # backtesting.py's own default
        proposed_risk_dollars = self.estimate_risk_dollars(price, sl, effective_size)
        quantity = int(effective_size) if effective_size >= 1 else 1  # fractional sizes are equity-relative, not a contract count

        decision, reason = self._risk_engine.check_order(
            proposed_risk_dollars=proposed_risk_dollars, quantity=quantity
        )

        if decision != RiskDecision.APPROVED:
            self.risk_rejections.append((self.data.index[-1], reason))
            return None

        order_fn = self.buy if direction == "buy" else self.sell
        kwargs = {"sl": sl, "tp": tp, "limit": limit, "stop": stop, "tag": tag}
        if size is not None:
            kwargs["size"] = size
        return order_fn(**{k: v for k, v in kwargs.items() if v is not None})
