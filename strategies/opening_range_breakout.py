"""Opening Range Breakout (ORB) strategy.

Defines a range from the first N bars after the RTH open (9:30am ET);
goes long on a break above the range high, short on a break below the range
low. Flattens at end of session. Includes an ATR-based volatility filter to
skip abnormally dead/choppy mornings, per standard ORB risk practice.

Session detection is anchored to the actual 9:30am ET cash market open
(session_id_from_rth_open), not calendar-date rollover. This matters
because MES/ES trade ~23 hours/day on Globex -- a plain date-based session
boundary falls at midnight ET, which is not a meaningful "opening range" and
was verified (in an earlier version of this file) to produce a "breakout"
trade near every midnight: -55% over 55 days of 5m bars, 4000+ trades. This
version fixes that by requiring the data's index to be tz-aware in
America/New_York (which is what yfinance returns for US futures).
"""

from __future__ import annotations

import pandas as pd
from backtesting import Strategy

from engine.strategy.indicators import atr, session_id_from_rth_open
from engine.strategy.time_filters import is_tradeable_time


class OpeningRangeBreakoutStrategy(Strategy):
    range_bars = 6  # e.g. 6 x 5-minute bars = first 30 minutes of session
    stop_ticks = 8
    target_ticks = 16
    tick_size = 0.25  # MES tick size

    atr_period = 14
    min_atr_ticks = 4  # skip the session if ATR is too low (dead/choppy morning)

    def init(self):
        index = self.data.index
        self._session_ids = session_id_from_rth_open(index)
        self._bar_in_session = self._session_ids.groupby(self._session_ids).cumcount()

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        self.atr = self.I(atr, high, low, close, self.atr_period)

        self._range_high: float | None = None
        self._range_low: float | None = None
        self._session_tradeable: bool = True
        self._traded_this_session: bool = False

    def next(self):
        i = len(self.data.Close) - 1
        bar_idx = self._bar_in_session.iloc[i]
        session_id = self._session_ids.iloc[i]
        is_new_session = bar_idx == 0

        if is_new_session or self._range_high is None:
            self._range_high = self.data.High[-1]
            self._range_low = self.data.Low[-1]
            self._session_tradeable = True
            # Reset once per session start ONLY -- must not be re-derived
            # from self.position, since a stop/target fill mid-session makes
            # self.position falsy again and would otherwise let the
            # strategy re-enter repeatedly within the same session every
            # time a trade closes (this was a real bug: caused 2800+ trades
            # in 55 days instead of ~1/session).
            self._traded_this_session = False
            return

        if bar_idx < self.range_bars:
            self._range_high = max(self._range_high, self.data.High[-1])
            self._range_low = min(self._range_low, self.data.Low[-1])
            return

        if bar_idx == self.range_bars and not pd.isna(self.atr[-1]):
            if self.atr[-1] < self.min_atr_ticks * self.tick_size:
                self._session_tradeable = False  # too quiet -- skip this session

        is_last_bar_of_session = (
            i + 1 >= len(self._session_ids) or self._session_ids.iloc[i + 1] != session_id
        )
        if is_last_bar_of_session:
            if self.position:
                self.position.close()
            return

        if self.position or self._traded_this_session or not self._session_tradeable:
            return  # one trade per session, keep it simple

        current_time = self.data.index[-1]
        if not is_tradeable_time(current_time):
            return  # inside a configured dead-zone/blackout window

        price = self.data.Close[-1]
        if price > self._range_high:
            sl = price - self.stop_ticks * self.tick_size
            tp = price + self.target_ticks * self.tick_size
            self.buy(sl=sl, tp=tp)
            self._traded_this_session = True
        elif price < self._range_low:
            sl = price + self.stop_ticks * self.tick_size
            tp = price - self.target_ticks * self.tick_size
            self.sell(sl=sl, tp=tp)
            self._traded_this_session = True
