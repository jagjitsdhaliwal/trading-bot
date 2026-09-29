"""Risk engine: hardcoded, non-negotiable trading limits.

This sits between the strategy and the broker. The strategy proposes trades;
this engine approves, shrinks, or rejects them. Nothing -- not the strategy,
not an AI agent, not a config the bot can rewrite at runtime -- can bypass
these checks. If you want to change a limit, you edit this file and restart
the process; the running bot cannot do it for itself.

Also models prop-firm-style rules (trailing max drawdown, daily loss limit)
so you can test whether a strategy would actually survive an evaluation,
not just whether it's profitable in the abstract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class RiskDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    KILL_SWITCH = "KILL_SWITCH"


@dataclass
class RiskLimits:
    """Edit these values directly to change risk parameters. There is no
    runtime API to modify them -- that's intentional.

    NOTE on these defaults: an earlier version had max_trade_risk=$50, which
    sounded conservative but was never checked against real instrument
    volatility -- it silently rejected every single MES trade a strategy
    ever tried to place (found when wiring RiskManagedStrategy into
    ma_crossover.py: every entry was rejected, a zero-trade backtest that
    looked like a strategy bug but was actually a miscalibrated risk
    limit).

    The fix is NOT to keep raising max_trade_risk until MES trades every
    signal regardless of volatility -- max_trailing_drawdown=$2,000 models
    a real, fixed constraint (a Topstep-style $50K Combine's drawdown
    rule), and a per-trade risk anywhere near $500+ would let a single
    trade consume 25%+ of the account's entire remaining drawdown budget,
    which defeats the point of having a drawdown limit at all. $150 is
    ~7.5% of that budget per trade -- reasonable. The honest consequence:
    on MES daily bars, a properly-sized strategy will correctly sit out
    (size=0) on higher-ATR days rather than trade a full contract's worth
    of risk it can't afford. That's the risk engine working as designed,
    not a bug -- a strategy that trades less often but never risks too
    much per trade is the point, not a defect to engineer around by
    inflating the risk budget to match whatever ATR happens to show up.
    max_trades_per_day * max_trade_risk stays comfortably under
    max_daily_loss so a full day of max-sized losses doesn't blow the
    daily limit on trade count alone.
    """

    max_daily_loss: float = 450.0
    max_trade_risk: float = 150.0
    max_trades_per_day: int = 3
    max_consecutive_losses: int = 3
    max_position_size: int = 2  # contracts

    # Prop-firm style account protection (set to match the firm you're
    # targeting, e.g. Topstep $50K Combine: profit_target=3000, max_loss=2000)
    account_starting_balance: float = 50_000.0
    max_trailing_drawdown: float | None = 2_000.0
    profit_target: float | None = 3_000.0


@dataclass
class RiskState:
    """Mutable state the engine tracks through a trading day/evaluation."""

    current_date: date | None = None
    daily_pnl: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    account_balance: float = 0.0
    peak_balance: float = 0.0
    kill_switch_triggered: bool = False
    kill_switch_reason: str = ""
    trade_history: list[float] = field(default_factory=list)


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.state = RiskState(
            account_balance=self.limits.account_starting_balance,
            peak_balance=self.limits.account_starting_balance,
        )

    def start_new_day(self, today: date) -> None:
        if self.state.current_date != today:
            self.state.current_date = today
            self.state.daily_pnl = 0.0
            self.state.trades_today = 0
            # A new day clears a daily-loss-triggered kill switch (that
            # limit is inherently per-day). It deliberately does NOT clear
            # kill switches from trailing drawdown or consecutive losses --
            # those reflect account/strategy health, not a daily counter,
            # and should require an explicit reset_kill_switch() call.
            if "Daily loss" in self.state.kill_switch_reason:
                self.reset_kill_switch()

    def check_order(self, proposed_risk_dollars: float, quantity: int) -> tuple[RiskDecision, str]:
        """Call before placing any order. Returns (decision, reason)."""
        if self.state.kill_switch_triggered:
            return RiskDecision.KILL_SWITCH, self.state.kill_switch_reason

        if quantity > self.limits.max_position_size:
            return (
                RiskDecision.REJECTED,
                f"Position size {quantity} exceeds max {self.limits.max_position_size}",
            )

        if proposed_risk_dollars > self.limits.max_trade_risk:
            return (
                RiskDecision.REJECTED,
                f"Trade risk ${proposed_risk_dollars:.2f} exceeds max "
                f"${self.limits.max_trade_risk:.2f}",
            )

        if self.state.trades_today >= self.limits.max_trades_per_day:
            return (
                RiskDecision.REJECTED,
                f"Already at max trades/day ({self.limits.max_trades_per_day})",
            )

        if self.state.daily_pnl <= -self.limits.max_daily_loss:
            self._trigger_kill_switch(
                f"Daily loss limit hit: ${self.state.daily_pnl:.2f}"
            )
            return RiskDecision.KILL_SWITCH, self.state.kill_switch_reason

        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            self._trigger_kill_switch(
                f"{self.state.consecutive_losses} consecutive losses hit"
            )
            return RiskDecision.KILL_SWITCH, self.state.kill_switch_reason

        if self.limits.max_trailing_drawdown is not None:
            drawdown = self.state.peak_balance - self.state.account_balance
            if drawdown >= self.limits.max_trailing_drawdown:
                self._trigger_kill_switch(
                    f"Trailing drawdown limit hit: ${drawdown:.2f} from peak "
                    f"${self.state.peak_balance:.2f}"
                )
                return RiskDecision.KILL_SWITCH, self.state.kill_switch_reason

        return RiskDecision.APPROVED, "OK"

    def record_trade_result(self, pnl: float) -> None:
        """Call after a trade closes to update running risk state."""
        self.state.trades_today += 1
        self.state.daily_pnl += pnl
        self.state.account_balance += pnl
        self.state.trade_history.append(pnl)
        self.state.peak_balance = max(self.state.peak_balance, self.state.account_balance)

        if pnl <= 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        if (
            self.limits.profit_target is not None
            and self.state.account_balance - self.limits.account_starting_balance
            >= self.limits.profit_target
        ):
            self._trigger_kill_switch(
                f"Profit target reached: ${self.state.account_balance:.2f} "
                "(evaluation passed -- stop trading and reassess)"
            )

    def _trigger_kill_switch(self, reason: str) -> None:
        self.state.kill_switch_triggered = True
        self.state.kill_switch_reason = reason

    def reset_kill_switch(self) -> None:
        """Manual-only. Never call this automatically from strategy code."""
        self.state.kill_switch_triggered = False
        self.state.kill_switch_reason = ""
