"""Trend-following / momentum strategy ("Bot #1" in the strategy-tournament
design notes).

Standard systematic-trading structure, now with the full filter stack from
the design notes:
  - Fast/slow EMA pair sets the entry-timeframe trend direction
  - Long-term trend filter (price above/below a slower SMA) confirms the
    macro direction isn't being fought -- this is the piece that was
    missing before; without it the strategy could take a "trend" entry on
    a fast/slow EMA cross that's really just noise inside a larger opposing
    trend
  - Volume confirmation: current volume above its recent average, on the
    theory that a real trend move should show above-average participation,
    not a low-volume drift
  - ATR-based entry (pullback to fast EMA, not chasing the crossover) and
    ATR-based trailing stop, as before
  - Volatility-scaled position sizing: contracts sized so that a stop-out
    always risks roughly the same dollar amount, regardless of current ATR

Runs on 1h bars per the design notes' 15m/1h/4h suggestion -- 1h balances
enough historical bars for walk-forward validation against yfinance's
~730-day cap on hourly data (its intraday ceiling is far more generous at
1h than at 5m/15m).
"""

from __future__ import annotations

import pandas as pd
from backtesting import Strategy

from engine.strategy.indicators import atr, ema, sma, volatility_scaled_size, volume_sma


class TrendFollowingStrategy(Strategy):
    fast_ema_period = 20
    slow_ema_period = 50
    long_term_ma_period = 100  # macro trend filter
    volume_avg_period = 20
    atr_period = 14
    atr_stop_multiplier = 2.5
    pullback_tolerance_pct = 0.15  # price must be within this % of fast EMA to trigger entry

    # Risk sized as a PERCENTAGE of current equity, not a flat dollar
    # figure -- a hardcoded risk_dollars_per_trade=300 implicitly assumed a
    # ~$50K account (300 = 0.6% of equity there) and silently broke on
    # smaller accounts: at $1,000, sizing wanted ~60 shares of SPY (~$46K
    # of exposure), which is both un-affordable (margin-rejected by
    # backtesting.py) and not what a 0.6%-of-equity risk budget should
    # produce on a $1,000 account (that's $6, not $300). Computing the
    # dollar figure from self.equity each time it's needed makes sizing
    # scale correctly regardless of account size, and also as the account
    # grows/shrinks over the course of a backtest (a flat dollar figure
    # doesn't adapt to either).
    risk_pct_per_trade = 0.006  # 0.6% of current equity per trade

    # dollars_per_point is instrument-specific -- for equities (SPY, QQQ,
    # individual stocks) a 1-point move = $1/share, so set this to 1.0 when
    # backtesting against a stock symbol instead of a futures continuous
    # contract. Override via Backtest.run(dollars_per_point=1.0, ...).
    dollars_per_point = 5.0  # MES: $5/index point (set to 1.0 for equities)

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self.fast_ema = self.I(ema, close, self.fast_ema_period)
        self.slow_ema = self.I(ema, close, self.slow_ema_period)
        self.long_term_ma = self.I(sma, close, self.long_term_ma_period)
        self.volume_avg = self.I(volume_sma, volume, self.volume_avg_period)
        self.atr = self.I(atr, high, low, close, self.atr_period)

    def next(self):
        if pd.isna(self.long_term_ma[-1]) or pd.isna(self.atr[-1]) or pd.isna(self.volume_avg[-1]):
            return  # indicators still warming up

        price = self.data.Close[-1]
        volume_now = self.data.Volume[-1]

        uptrend = self.fast_ema[-1] > self.slow_ema[-1]
        downtrend = self.fast_ema[-1] < self.slow_ema[-1]
        above_long_term = price > self.long_term_ma[-1]
        below_long_term = price < self.long_term_ma[-1]
        volume_confirms = volume_now > self.volume_avg[-1]
        near_fast_ema = abs(price - self.fast_ema[-1]) / price <= self.pullback_tolerance_pct / 100

        if self.position:
            self._update_trailing_stop()
            if self.position.is_long and downtrend:
                self.position.close()
            elif self.position.is_short and uptrend:
                self.position.close()
            return

        size = volatility_scaled_size(
            self.equity * self.risk_pct_per_trade, self.atr[-1], self.atr_stop_multiplier,
            self.dollars_per_point,
        )
        if size < 1:
            return  # ATR too wide to trade within the configured risk budget

        if uptrend and above_long_term and volume_confirms and near_fast_ema:
            sl = price - self.atr_stop_multiplier * self.atr[-1]
            self.buy(size=size, sl=sl)
        elif downtrend and below_long_term and volume_confirms and near_fast_ema:
            sl = price + self.atr_stop_multiplier * self.atr[-1]
            self.sell(size=size, sl=sl)

    def _update_trailing_stop(self):
        """Ratchet the stop in the trade's favor as price moves, using ATR
        as the trailing distance. Never loosens the stop."""
        if not self.trades:
            return
        trade = self.trades[-1]
        price = self.data.Close[-1]
        distance = self.atr_stop_multiplier * self.atr[-1]

        if trade.is_long:
            new_sl = price - distance
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl
        else:
            new_sl = price + distance
            if trade.sl is None or new_sl < trade.sl:
                trade.sl = new_sl
