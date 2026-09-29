"""Baseline strategy: fast/slow moving average crossover.

Long when fast MA crosses above slow MA, exit (flat) when it crosses back
below. No shorting in this version -- keep it simple to start.

This is NOT presented as a strategy with an edge -- it's a well-understood,
easy-to-verify baseline for proving the backtest/risk/data pipeline works
end-to-end before you plug in something more considered.

Two things added after an adversarial stress-test pass found real gaps:
  1. An ATR-based stop-loss on entry. The original version had no `sl=` at
     all, meaning a single bad trade had no bound on its loss (the stress
     test found a $7,139 single-trade loss on a $50K account in a
     high-volatility synthetic scenario). ATR-based (not a fixed %) so the
     stop distance adapts to current volatility rather than being either
     too tight in calm markets or too loose in wild ones.
  2. Inherits from RiskManagedStrategy (engine/strategy/risk_managed.py)
     instead of backtesting.py's plain Strategy, and uses self.risk_buy()
     instead of self.buy() -- routes every entry through RiskEngine's
     check_order() (position size cap, per-trade risk cap, daily loss/
     consecutive-loss/trailing-drawdown kill switches) before it's placed.
     Previously, run_backtest()/run_walk_forward() never invoked RiskEngine
     at all -- every prior backtest number for this strategy reflects
     backtesting.py's default (effectively unbounded, full-equity)
     position sizing, not the risk limits RiskLimits defines.
"""

from __future__ import annotations

import pandas as pd
from backtesting.lib import crossover

from engine.strategy.indicators import atr, volatility_scaled_size
from engine.strategy.risk_managed import RiskManagedStrategy


def sma(values: pd.Series, n: int) -> pd.Series:
    return pd.Series(values).rolling(n).mean()


class MACrossoverStrategy(RiskManagedStrategy):
    fast_period = 10
    slow_period = 30
    atr_period = 14
    # 0.75x (not the 2.5x used by trend-following strategies elsewhere in
    # this codebase) -- this is a simple crossover EXIT strategy (flattens
    # on the reverse crossover) rather than a trailing-stop trend-follower,
    # so a much tighter protective stop is the appropriate choice here, not
    # a workaround. It also keeps per-contract risk on MES daily bars
    # (median ATR~82) at ~$307 -- still needs a slightly larger risk budget
    # than the 1h-timeframe strategies use, reflecting daily bars' larger
    # absolute ATR in points, not a looser risk tolerance.
    atr_stop_multiplier = 0.75

    # backtesting.py's own default entry size (~full available equity) is
    # wildly larger than RiskLimits.max_trade_risk was ever designed for --
    # wiring in RiskManagedStrategy without an explicit, deliberately-sized
    # position meant every single entry got rejected (proposed risk in the
    # thousands of dollars). Size explicitly via volatility_scaled_size()
    # instead of relying on backtesting.py's default, same pattern already
    # used in trend_following.py/donchian_breakout.py.
    #
    # Risk sized as a PERCENTAGE OF EQUITY, not a flat dollar figure -- a
    # flat risk_dollars_per_trade=140 (originally calibrated against
    # RiskLimits' $50K-account defaults) doesn't mean anything on a $1,000
    # account: 140/50_000=0.28% of equity there, but 140/1_000=14% of
    # equity on a small account -- wildly more aggressive despite being
    # "the same number." Computing the dollar figure from self.equity each
    # time keeps the RISK RATIO consistent regardless of account size.
    # NOTE: RiskManagedStrategy's check_order() still enforces
    # RiskLimits.max_trade_risk as an absolute dollar figure on top of
    # this -- that cap represents a fixed, real-world constraint (e.g. an
    # actual prop-firm evaluation's contractual drawdown rule) that does
    # NOT scale with account size by definition, so if you're backtesting
    # a genuinely different account size (not a fixed-rule evaluation),
    # override risk_limits=RiskLimits(max_trade_risk=...) to match, or the
    # percentage-based sizing above may still get rejected by a mismatched
    # absolute cap.
    risk_pct_per_trade = 0.0028  # 0.28% of equity -- matches the $140/$50K figure this replaces
    dollars_per_point = 5.0  # MES: $5/index point (set to 1.0 for equities)

    def init(self):
        super().init()
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        self.fast_ma = self.I(sma, close, self.fast_period)
        self.slow_ma = self.I(sma, close, self.slow_period)
        self.atr = self.I(atr, high, low, close, self.atr_period)

    def next(self):
        if pd.isna(self.atr[-1]):
            return  # ATR still warming up -- no stop distance available yet

        if crossover(self.fast_ma, self.slow_ma):
            if not self.position:
                price = self.data.Close[-1]
                sl = price - self.atr_stop_multiplier * self.atr[-1]
                size = volatility_scaled_size(
                    self.equity * self.risk_pct_per_trade, self.atr[-1], self.atr_stop_multiplier,
                    self.dollars_per_point,
                )
                if size < 1:
                    return  # ATR too wide to trade within the configured risk budget
                self.risk_buy(size=size, sl=sl)
        elif crossover(self.slow_ma, self.fast_ma):
            if self.position:
                self.position.close()
