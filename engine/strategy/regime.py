"""Market regime classification: trending / ranging / high-volatility.

Per the design notes this was built from:
  Regime A -- Strong trend: ADX high, ATR elevated, EMA slope positive,
              price above long-term MA -> momentum/trend bot
  Regime B -- Range-bound: ADX low, ATR moderate, price oscillating
              around VWAP -> mean-reversion bot
  Regime C -- Extremely volatile: ATR unusually high, large candle ranges,
              volume spike -> reduce size or stand down

This is a rule-based classifier, not a statistical/ML regime model --
deliberately, to keep it as auditable and debuggable as the rest of the risk
infrastructure. ADX thresholds (25 = trending, 20 = ranging) are Wilder's
own standard cutoffs, not tuned/fit values.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

from engine.strategy.indicators import adx, atr


class Regime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"  # indicators still warming up


def classify_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    adx_period: int = 14,
    atr_period: int = 14,
    adx_trending_threshold: float = 25.0,
    adx_ranging_threshold: float = 20.0,
    atr_high_vol_percentile: float = 0.90,
    atr_lookback: int = 100,
) -> pd.Series:
    """Returns a Series of Regime values, one per bar.

    High-volatility check takes priority over trend/range classification --
    per the design notes, Regime C ("reduce size or stop trading") should
    override even a clean trend signal, since a violently volatile trending
    market is still a "stand down" situation, not a "trade the trend harder"
    one.
    """
    a = adx(high, low, close, adx_period)
    tr = atr(high, low, close, atr_period)
    atr_threshold = tr.rolling(atr_lookback).quantile(atr_high_vol_percentile)

    regimes = []
    for i in range(len(close)):
        if pd.isna(a.iloc[i]) or pd.isna(tr.iloc[i]) or pd.isna(atr_threshold.iloc[i]):
            regimes.append(Regime.UNKNOWN)
        elif tr.iloc[i] >= atr_threshold.iloc[i]:
            regimes.append(Regime.HIGH_VOLATILITY)
        elif a.iloc[i] >= adx_trending_threshold:
            regimes.append(Regime.TRENDING)
        elif a.iloc[i] <= adx_ranging_threshold:
            regimes.append(Regime.RANGING)
        else:
            regimes.append(Regime.UNKNOWN)  # in between -- no clear regime, sit out

    return pd.Series(regimes, index=close.index if hasattr(close, "index") else None)
