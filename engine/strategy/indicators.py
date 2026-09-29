"""Shared technical indicators used across strategies. Kept as plain
pandas-in/pandas-out functions so both backtesting.py's Strategy.I() wrapper
and any live/offline analysis code can call them the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(values: pd.Series, span: int) -> pd.Series:
    return pd.Series(values).ewm(span=span, adjust=False).mean()


def sma(values: pd.Series, n: int) -> pd.Series:
    return pd.Series(values).rolling(n).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(n).mean()


def atr_ema(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 100) -> pd.Series:
    """ATR with exponential (not simple rolling) smoothing -- Clenow's own
    site (followingthetrend.com) specifies a 100-day EXPONENTIALLY smoothed
    ATR for his position sizing formula, distinct from the plain
    rolling-mean atr() used elsewhere in this codebase. Clenow himself notes
    the exact smoothing method isn't critical to results, but this matches
    his documented default for fidelity when reproducing his formula.
    """
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    close = pd.Series(close)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rolling_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, session_ids: pd.Series
) -> pd.Series:
    """VWAP anchored to reset at each session boundary (session_ids groups
    bars into sessions, e.g. one integer per trading day)."""
    high, low, close, volume = (
        pd.Series(high),
        pd.Series(low),
        pd.Series(close),
        pd.Series(volume),
    )
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    session_ids = pd.Series(session_ids)

    cum_pv = pv.groupby(session_ids).cumsum()
    cum_vol = volume.groupby(session_ids).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def bollinger_bands(
    close: pd.Series, n: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = pd.Series(close)
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average Directional Index -- standard trend-strength measure (Wilder).
    High ADX (typically >25) = trending market. Low ADX (<20) = ranging.
    Used as the regime filter: gates trend-following/mean-reversion entries
    and drives the regime-switching strategy's bot selection.
    """
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = atr(high, low, close, n=1) * 1  # single-bar true range (atr(n=1) = TR itself)
    atr_n = tr.rolling(n).mean()

    plus_di = 100 * (plus_dm.rolling(n).mean() / atr_n.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr_n.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean()


def volume_sma(volume: pd.Series, n: int = 20) -> pd.Series:
    return pd.Series(volume).rolling(n).mean()


def donchian_channel(
    high: pd.Series, low: pd.Series, n: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Rolling N-bar high/low channel -- the classic breakout reference
    levels (Donchian channel / turtle-trading style breakout)."""
    high, low = pd.Series(high), pd.Series(low)
    # shift(1): breakout is measured against the PRIOR N bars, excluding the
    # current bar itself -- otherwise every bar trivially "breaks out" of a
    # channel that includes its own high/low.
    upper = high.shift(1).rolling(n).max()
    lower = low.shift(1).rolling(n).min()
    return upper, lower


def volatility_scaled_size(
    risk_dollars: float, atr_value: float, atr_stop_multiplier: float, dollars_per_point: float
) -> int:
    """Position size (in contracts) that keeps dollar risk roughly constant
    regardless of current volatility -- per the "volatility-scaled position
    sizing" principle: fewer contracts when ATR is wide, more when it's
    narrow, same dollar risk either way.

    This is the Turtle Trader style formula: risk is defined relative to
    the STOP distance (Units = 1% equity / (N x dollars-per-point), stop at
    2N -- see turtle_position_size() below for that exact original formula).
    Contrast with clenow_position_size(), which sizes off a target daily
    volatility CONTRIBUTION with no reference to a stop distance at all --
    these are two distinct, independently-documented sizing philosophies,
    not two ways of writing the same formula.

    stop_distance_points = atr_value * atr_stop_multiplier
    risk_per_contract = stop_distance_points * dollars_per_point
    contracts = risk_dollars / risk_per_contract

    NOTE: an earlier version of this function took (tick_value, tick_size)
    instead of a single dollars_per_point and computed
    stop_distance_ticks = (atr_value * multiplier) / tick_size, then
    multiplied by tick_value -- this silently overstated risk-per-contract
    by 4x on MES (tick_size=0.25) because tick_value was actually being
    passed as $5/POINT (not $/tick), so dividing by tick_size double-counted
    the point/tick conversion. Found via a similar zero-trade bug in
    donchian_breakout's Clenow-style sizing and traced back here. Passing a
    single already-correct dollars_per_point removes the ambiguity.
    """
    if atr_value <= 0 or dollars_per_point <= 0:
        return 0
    stop_distance_points = atr_value * atr_stop_multiplier
    risk_per_contract = stop_distance_points * dollars_per_point
    if risk_per_contract <= 0:
        return 0
    return max(0, int(risk_dollars // risk_per_contract))


def turtle_n(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    """Original Turtle Trading System's "N" -- a 20-day ATR (Wilder-style
    smoothed True Range), the volatility unit their whole system is built
    on. This is just atr() with the Turtles' specific lookback; kept as a
    separate named function since "N" is what their position sizing, stop,
    and pyramiding rules are all expressed in terms of, and calling it atr()
    everywhere would obscure that.

    Source: Curtis Faith, "Way of the Turtle"; corroborated across
    independent public retellings (turtletrader.com, Wikipedia).
    """
    return atr(high, low, close, n)


def turtle_position_size(account_equity: float, n_value: float, dollars_per_point: float, risk_pct: float = 0.01) -> int:
    """Turtle Trader position sizing: 1 Unit = 1% of account equity risked
    per N (their ATR unit). Since their stop is placed at 2N, sizing this
    way means each unit risks ~2% of equity if stopped out at 2N -- the 1%
    figure is deliberately expressed per-N, not per-stop-distance, which is
    numerically the same result as volatility_scaled_size() with
    atr_stop_multiplier=2 and risk_dollars=2%-of-equity, but written in the
    Turtles' own terms for direct fidelity to the documented rule.

    Units = (1% x account_equity) / (N x dollars_per_point)
    """
    if n_value <= 0 or dollars_per_point <= 0:
        return 0
    dollar_volatility_per_contract = n_value * dollars_per_point
    return max(0, int((risk_pct * account_equity) // dollar_volatility_per_contract))


def clenow_position_size(
    account_equity: float, atr_value: float, dollars_per_point: float, daily_vol_target_pct: float = 0.002
) -> int:
    """Andreas Clenow's position sizing from "Following the Trend": each
    position is sized so it contributes a constant target daily volatility
    (0.2% of account equity by default -- his stated figure) to the
    portfolio, regardless of the instrument's own volatility. Notably, this
    formula has NO reference to a stop-loss distance at all -- unlike the
    Turtle/volatility_scaled_size approach where sizing IS the risk-per-stop
    calculation, Clenow decouples "how big is this position" from "where's
    my stop" entirely. ATR is meant to be a 100-day exponentially smoothed
    average per his own site, not the default 14-20 day window used
    elsewhere in this codebase -- pass a matching atr_value.

    Contracts = (account_equity x daily_vol_target_pct) / (atr_value x dollars_per_point)

    Source: followingthetrend.com/the-trading-system/trading-system-rules/ (PRIMARY)
    """
    if atr_value <= 0 or dollars_per_point <= 0:
        return 0
    dollar_volatility_per_contract = atr_value * dollars_per_point
    return max(0, int((daily_vol_target_pct * account_equity) // dollar_volatility_per_contract))


def session_id_from_index(index: pd.DatetimeIndex) -> pd.Series:
    """Groups bars into sessions by calendar date. NOTE: for near-24-hour
    futures instruments (MES/ES trade ~23h/day) this treats midnight-ET
    rollover as a session boundary, which does NOT correspond to the 9:30am
    ET RTH open. Use session_id_from_rth_open() instead for ORB-style
    strategies that need to anchor to the actual cash market open.
    """
    dates = pd.Series(index).dt.date
    return (dates != dates.shift(1)).cumsum()


def session_id_from_rth_open(index: pd.DatetimeIndex, open_hour: int = 9, open_minute: int = 30) -> pd.Series:
    """Groups bars into sessions anchored to a specific time-of-day (default
    9:30am, matching the index's own timezone -- yfinance returns
    America/New_York-localized timestamps for US futures, so this lines up
    with RTH open as long as the index is tz-aware in that zone).
    """
    idx = pd.DatetimeIndex(index)
    times = idx.time
    open_time = pd.Timestamp(f"{open_hour:02d}:{open_minute:02d}").time()
    is_session_start = pd.Series(times, index=range(len(idx))) == open_time
    if not is_session_start.any():
        # No bar falls exactly on the open (e.g. bar granularity doesn't
        # align, like 1d bars). Fall back to date-based grouping.
        return session_id_from_index(index)
    return is_session_start.cumsum()
