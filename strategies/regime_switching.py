"""Regime-switching strategy: composes trend-following and mean-reversion
logic, routed by a rule-based market regime classifier, per the design
notes' "the really interesting bot" section.

Instead of asking "what's the best strategy," this asks "what type of
market are we in right now" each bar, using engine/strategy/regime.py's
ADX+ATR classifier:

  Regime A (TRENDING): ADX >= 25 -> activate trend-following logic
    (EMA fast/slow direction, pullback entry, ATR trailing stop -- the
    same shape as strategies/trend_following.py, inlined here since
    backtesting.py strategies can't compose as separate objects mid-run)

  Regime B (RANGING): ADX <= 20 -> activate mean-reversion logic
    (Bollinger Band + RSI exhaustion entry, same shape as
    strategies/mean_reversion.py)

  Regime C (HIGH_VOLATILITY): ATR in the top decile of its trailing
    lookback -> stand down entirely (no new entries; existing positions
    still get managed/exited normally). Per the design notes: "reduce
    position size or stop trading" -- this implementation stops trading
    rather than reducing size, since reduced-size sub-minimum-tick
    contracts aren't meaningful for MES (can't size below 1 contract).

  UNKNOWN (ADX between the ranging and trending thresholds, or indicators
  still warming up): no clear regime -- sit out. Per the design notes'
  spirit ("we don't need to predict anything") -- an ambiguous regime
  isn't a signal to guess harder, it's a signal to wait.

This is the same volatility-scaled position sizing as trend_following.py
and donchian_breakout.py, so results across the "tournament" are on equal
footing risk-wise.
"""

from __future__ import annotations

import pandas as pd
from backtesting import Strategy

from engine.strategy.indicators import atr, bollinger_bands, ema, rsi, volatility_scaled_size
from engine.strategy.regime import Regime, classify_regime

# backtesting.py's self.I() casts every indicator to float internally (for
# warmup-period detection), so the regime classification -- naturally a
# string/Enum per bar -- has to be encoded as plain numbers here rather than
# passed through as Regime.value strings.
_REGIME_CODE = {
    Regime.UNKNOWN: 0,
    Regime.TRENDING: 1,
    Regime.RANGING: 2,
    Regime.HIGH_VOLATILITY: 3,
}


class RegimeSwitchingStrategy(Strategy):
    # Regime classification
    adx_period = 14
    adx_trending_threshold = 25.0
    adx_ranging_threshold = 20.0
    atr_high_vol_percentile = 0.90
    atr_lookback = 100

    # Trend-following sub-strategy params
    fast_ema_period = 20
    slow_ema_period = 50
    pullback_tolerance_pct = 0.15
    trend_atr_stop_multiplier = 2.5

    # Mean-reversion sub-strategy params
    bb_period = 20
    bb_std = 2.0
    stop_std = 3.0
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30

    atr_period = 14
    # Percentage-of-equity risk, not a flat dollar figure -- see
    # trend_following.py's risk_pct_per_trade comment for why (a flat
    # $300 implicitly assumed a ~$50K account and silently broke on
    # smaller ones).
    risk_pct_per_trade = 0.006  # 0.6% of current equity per trade
    dollars_per_point = 5.0  # MES: $5/index point (set to 1.0 for equities)

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.regime = self.I(
            lambda h, l, c: classify_regime(
                pd.Series(h), pd.Series(l), pd.Series(c),
                self.adx_period, self.atr_period,
                self.adx_trending_threshold, self.adx_ranging_threshold,
                self.atr_high_vol_percentile, self.atr_lookback,
            ).map(_REGIME_CODE).astype(float),
            high, low, close,
        )

        self.fast_ema = self.I(ema, close, self.fast_ema_period)
        self.slow_ema = self.I(ema, close, self.slow_ema_period)
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            bollinger_bands, close, self.bb_period, self.bb_std
        )
        self.bb_upper_stop, _, self.bb_lower_stop = self.I(
            bollinger_bands, close, self.bb_period, self.stop_std
        )
        self.rsi = self.I(rsi, close, self.rsi_period)
        self.atr = self.I(atr, high, low, close, self.atr_period)

        self._active_mode: str | None = None  # 'trend' or 'reversion' -- tracks which sub-strategy opened the current position

    def next(self):
        if pd.isna(self.atr[-1]) or self.regime[-1] == _REGIME_CODE[Regime.UNKNOWN]:
            return

        current_regime = self.regime[-1]
        price = self.data.Close[-1]

        if self.position:
            self._manage_open_position(current_regime, price)
            return

        if current_regime == _REGIME_CODE[Regime.HIGH_VOLATILITY]:
            return  # stand down -- no new entries in extreme volatility

        if current_regime == _REGIME_CODE[Regime.TRENDING]:
            self._try_trend_entry(price)
        elif current_regime == _REGIME_CODE[Regime.RANGING]:
            self._try_reversion_entry(price)

    def _try_trend_entry(self, price: float) -> None:
        uptrend = self.fast_ema[-1] > self.slow_ema[-1]
        downtrend = self.fast_ema[-1] < self.slow_ema[-1]
        near_fast_ema = abs(price - self.fast_ema[-1]) / price <= self.pullback_tolerance_pct / 100
        if not near_fast_ema:
            return

        size = volatility_scaled_size(
            self.equity * self.risk_pct_per_trade, self.atr[-1], self.trend_atr_stop_multiplier,
            self.dollars_per_point,
        )
        if size < 1:
            return

        if uptrend:
            sl = price - self.trend_atr_stop_multiplier * self.atr[-1]
            self.buy(size=size, sl=sl)
            self._active_mode = "trend"
        elif downtrend:
            sl = price + self.trend_atr_stop_multiplier * self.atr[-1]
            self.sell(size=size, sl=sl)
            self._active_mode = "trend"

    def _try_reversion_entry(self, price: float) -> None:
        if pd.isna(self.bb_mid[-1]) or pd.isna(self.rsi[-1]):
            return

        if price <= self.bb_lower[-1] and self.rsi[-1] <= self.rsi_oversold:
            sl = self.bb_lower_stop[-1]
            if sl >= price:
                return
            self.buy(sl=sl)
            self._active_mode = "reversion"
        elif price >= self.bb_upper[-1] and self.rsi[-1] >= self.rsi_overbought:
            sl = self.bb_upper_stop[-1]
            if sl <= price:
                return
            self.sell(sl=sl)
            self._active_mode = "reversion"

    def _manage_open_position(self, current_regime: float, price: float) -> None:
        if self._active_mode == "trend":
            self._update_trend_trailing_stop()
            uptrend = self.fast_ema[-1] > self.slow_ema[-1]
            downtrend = self.fast_ema[-1] < self.slow_ema[-1]
            if self.position.is_long and downtrend:
                self.position.close()
            elif self.position.is_short and uptrend:
                self.position.close()
        elif self._active_mode == "reversion":
            if self.position.is_long and price <= self.bb_lower_stop[-1]:
                self.position.close()
            elif self.position.is_short and price >= self.bb_upper_stop[-1]:
                self.position.close()
            elif self.position.is_long and price >= self.bb_mid[-1]:
                self.position.close()
            elif self.position.is_short and price <= self.bb_mid[-1]:
                self.position.close()

    def _update_trend_trailing_stop(self):
        if not self.trades:
            return
        trade = self.trades[-1]
        price = self.data.Close[-1]
        distance = self.trend_atr_stop_multiplier * self.atr[-1]

        if trade.is_long:
            new_sl = price - distance
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl
        else:
            new_sl = price + distance
            if trade.sl is None or new_sl < trade.sl:
                trade.sl = new_sl
