"""Donchian channel breakout strategy, refined to match Andreas Clenow's
documented trend-following system from "Following the Trend" (2013) as
closely as this codebase's architecture allows.

Source (PRIMARY, followingthetrend.com/the-trading-system/trading-system-rules/):
  - Entry: new 50-day closing high (long) / 50-day closing low (short)
  - Trend filter: 50-day EMA above 100-day EMA required for longs (inverse
    for shorts) -- only trade in the direction of the intermediate trend
  - Exit: ATR trailing stop at 3x ATR from the position's peak (for longs)
    or trough (for shorts) since entry -- NOT a fixed stop from entry price,
    and NOT an opposite-direction breakout. ATR uses 100-day exponential
    smoothing per his site.
  - Position sizing: contracts sized so each position contributes a
    constant ~0.2% of account equity in daily volatility (ATR x dollars-
    per-point x contracts ~= 0.2% x equity), independent of stop distance.
    See engine/strategy/indicators.py's clenow_position_size().

This is a deliberate rewrite of an earlier, looser "Donchian breakout"
version (20-bar channel + volume/ATR filters + fixed ATR stop) toward the
specific, publicly documented Clenow ruleset -- on the theory that a
strategy with a real multi-decade track record is worth reproducing
faithfully rather than approximating loosely. Whether reproducing it
faithfully actually produces a validated edge on MES/ES 2019-2026 is an
empirical question for walk-forward testing, not assumed by this rewrite.
"""

from __future__ import annotations

import pandas as pd
from backtesting import Strategy

from engine.strategy.indicators import atr_ema, clenow_position_size, donchian_channel, ema


class DonchianBreakoutStrategy(Strategy):
    channel_period = 50  # Clenow's documented breakout lookback
    trend_fast_ema = 50
    trend_slow_ema = 100
    atr_period = 100  # Clenow's documented ATR smoothing window (exponential)
    atr_trail_multiplier = 3.0  # Clenow's documented trailing-stop distance

    daily_vol_target_pct = 0.002  # Clenow's documented 0.2% target
    # dollars_per_point is instrument-specific: $5/point for MES, but 1.0 for
    # equities (SPY, QQQ, individual stocks -- 1 point move = $1/share).
    # Clenow's original system is itself a diversified FUTURES portfolio
    # system; running it against equities is a further departure from his
    # documented rules, not just a parameter substitution.
    dollars_per_point = 5.0  # MES: $5/index point (set to 1.0 for equities)

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.channel_high, self.channel_low = self.I(donchian_channel, high, low, self.channel_period)
        self.trend_ema_fast = self.I(ema, close, self.trend_fast_ema)
        self.trend_ema_slow = self.I(ema, close, self.trend_slow_ema)
        self.atr = self.I(atr_ema, high, low, close, self.atr_period)

        self._peak_since_entry: float | None = None  # highest high (long) / lowest low (short) since entry

    def next(self):
        if pd.isna(self.trend_ema_slow[-1]) or pd.isna(self.atr[-1]) or pd.isna(self.channel_high[-1]):
            return

        price = self.data.Close[-1]
        uptrend = self.trend_ema_fast[-1] > self.trend_ema_slow[-1]
        downtrend = self.trend_ema_fast[-1] < self.trend_ema_slow[-1]

        if self.position:
            self._update_peak_and_trailing_stop()
            return

        size = clenow_position_size(
            self.equity, self.atr[-1], self.dollars_per_point, self.daily_vol_target_pct
        )
        if size < 1:
            return

        if price > self.channel_high[-1] and uptrend:
            self.buy(size=size)
            self._peak_since_entry = price
        elif price < self.channel_low[-1] and downtrend:
            self.sell(size=size)
            self._peak_since_entry = price

    def _update_peak_and_trailing_stop(self):
        """Trail the stop at 3x ATR from the trade's peak (longs) or trough
        (shorts) since entry -- Clenow's documented exit, distinct from a
        fixed stop-from-entry-price approach."""
        if not self.trades:
            return
        trade = self.trades[-1]
        price = self.data.Close[-1]
        distance = self.atr_trail_multiplier * self.atr[-1]

        if trade.is_long:
            self._peak_since_entry = max(self._peak_since_entry or price, price)
            new_sl = self._peak_since_entry - distance
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl
        else:
            self._peak_since_entry = min(self._peak_since_entry or price, price)
            new_sl = self._peak_since_entry + distance
            if trade.sl is None or new_sl < trade.sl:
                trade.sl = new_sl
