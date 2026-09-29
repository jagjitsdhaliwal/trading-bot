"""Time-of-day / calendar filters shared across strategies.

Two things professional systematic traders filter on, per the design notes
this was built from:
  1. Known high-impact economic release windows (FOMC, CPI, NFP, etc.) --
     avoid holding/entering through them since a single headline can blow
     through any ATR-based stop.
  2. Low-liquidity dead zones (e.g. mid-afternoon lull before the close) --
     wider effective spreads and choppier price action with no real edge.

This does NOT fetch a live economic calendar -- that requires a paid/free
API (e.g. a data vendor's economic calendar endpoint) which isn't wired up
yet. For now, use `BLACKOUT_WINDOWS` to hardcode known recurring windows
(FOMC/CPI days change month to month, so hardcoding exact dates goes stale
fast -- this is deliberately left as a manually-maintained list rather than
faked with a fake "auto-detect" that silently does nothing).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

# Recurring low-liquidity window: late afternoon lull before the 4pm ET
# cash close, before the evening session picks back up. Adjust to taste.
DEFAULT_DEAD_ZONES: list[tuple[dt.time, dt.time]] = [
    (dt.time(14, 0), dt.time(15, 0)),  # 2:00-3:00pm ET
]

# Manually maintained list of specific known event dates/times to avoid
# trading through (FOMC announcements are 2:00pm ET, CPI/NFP are 8:30am ET).
# This list goes stale -- update it against the CME/BLS economic calendar
# before relying on it for anything beyond backtesting illustration.
BLACKOUT_WINDOWS: list[tuple[dt.datetime, dt.datetime]] = []


def is_in_dead_zone(
    timestamp: pd.Timestamp, dead_zones: list[tuple[dt.time, dt.time]] = DEFAULT_DEAD_ZONES
) -> bool:
    t = timestamp.time()
    return any(start <= t < end for start, end in dead_zones)


def is_in_blackout_window(
    timestamp: pd.Timestamp, blackout_windows: list[tuple[dt.datetime, dt.datetime]] = BLACKOUT_WINDOWS
) -> bool:
    ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
    return any(start <= ts <= end for start, end in blackout_windows)


def is_tradeable_time(
    timestamp: pd.Timestamp,
    dead_zones: list[tuple[dt.time, dt.time]] = DEFAULT_DEAD_ZONES,
    blackout_windows: list[tuple[dt.datetime, dt.datetime]] = BLACKOUT_WINDOWS,
) -> bool:
    """Combined check -- False means skip taking new entries at this bar."""
    return not is_in_dead_zone(timestamp, dead_zones) and not is_in_blackout_window(
        timestamp, blackout_windows
    )
