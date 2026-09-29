"""Mean-reversion / Bollinger Band deviation strategy ("Bot #3" in the
strategy-tournament design notes).

Entry when price stretches 2 standard deviations from its rolling mean with
RSI confirming exhaustion (overbought/oversold), targeting a reversion back
to the mean. Hard stop at 3 standard deviations to bail out if what looked
like a range turns into a real trend.

TREND FILTER GATE (added after walk-forward testing exposed the problem):
walk-forward validation on the original version of this strategy showed the
weakest parameter stability of any strategy tested (bb_period bounced
between 14/20/26 fold to fold with no convergence) -- a fingerprint of
noise rather than a real recurring pattern. This matches a well-known
failure mode discussed among systematic traders: mean-reversion systems
look fine until the market starts trending, then every "oversold" dip just
keeps dropping (or every "overbought" high keeps climbing) instead of
reverting. The fix per the design notes: gate entries on ADX -- only take
mean-reversion trades when ADX indicates a genuinely ranging market
(ADX <= adx_ranging_threshold). In a confirmed trend, this strategy now
sits out entirely rather than fading the trend.

Uses Bollinger Bands (rolling mean/std of Close) rather than VWAP as the
anchor -- see the note in indicators.py's rolling_vwap() if you want to
swap once you have better intraday volume data.
"""

from __future__ import annotations

import pandas as pd
from backtesting import Strategy

from engine.strategy.indicators import adx, bollinger_bands, rsi


class MeanReversionStrategy(Strategy):
    bb_period = 20
    bb_std = 2.0
    stop_std = 3.0
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30

    adx_period = 14
    adx_ranging_threshold = 20.0  # only trade mean-reversion when ADX confirms a range

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            bollinger_bands, close, self.bb_period, self.bb_std
        )
        # Recompute at stop_std for the hard-stop band (avoids adding a
        # second self.I() call with a tuple-of-tuples return shape).
        self.bb_upper_stop, _, self.bb_lower_stop = self.I(
            bollinger_bands, close, self.bb_period, self.stop_std
        )
        self.rsi = self.I(rsi, close, self.rsi_period)
        self.adx = self.I(adx, high, low, close, self.adx_period)

    def next(self):
        if pd.isna(self.bb_mid[-1]) or pd.isna(self.rsi[-1]) or pd.isna(self.adx[-1]):
            return

        price = self.data.Close[-1]

        if self.position:
            # Hard stop: price punched through the wider band -- this isn't
            # a range anymore, get out regardless of the mean-reversion thesis.
            if self.position.is_long and price <= self.bb_lower_stop[-1]:
                self.position.close()
            elif self.position.is_short and price >= self.bb_upper_stop[-1]:
                self.position.close()
            # Take profit at the mean
            elif self.position.is_long and price >= self.bb_mid[-1]:
                self.position.close()
            elif self.position.is_short and price <= self.bb_mid[-1]:
                self.position.close()
            return

        is_ranging = self.adx[-1] <= self.adx_ranging_threshold
        if not is_ranging:
            return  # ADX says we're trending -- don't fade it

        if price <= self.bb_lower[-1] and self.rsi[-1] <= self.rsi_oversold:
            sl = self.bb_lower_stop[-1]
            if sl >= price:
                # Price already gapped through the wider stop band on this
                # same bar -- a violent single-bar move, not a normal
                # "stretched but orderly" range extreme. Skip rather than
                # entering with an already-violated (or inverted) stop.
                return
            self.buy(sl=sl)
        elif price >= self.bb_upper[-1] and self.rsi[-1] >= self.rsi_overbought:
            sl = self.bb_upper_stop[-1]
            if sl <= price:
                return
            self.sell(sl=sl)
