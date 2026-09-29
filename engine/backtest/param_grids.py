"""Parameter search grids for walk-forward optimization -- one per strategy.

Deliberately narrow, anchored to standard values used in the technical
analysis literature, not wide arbitrary search. The point is to test
"does this strategy work with parameters in the neighborhood of what
practitioners actually use," not "what's the single best-fitting
combination across all of parameter space" -- the latter is a much stronger
curve-fitting magnet and defeats the purpose of walk-forward validation.

Each strategy's grid also gets a `min_train_bars`/`min_test_bars` hint:
walk-forward test windows need enough bars for the SLOWEST parameter in the
grid to both warm up and generate a handful of trades, or out-of-sample
results are mostly zero-trade noise (this was found and fixed after an
early smoke test: 50-bar test windows against a 40-period slow MA produced
mostly 0-trade folds).
"""

from __future__ import annotations

from dataclasses import dataclass

from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.ma_crossover import MACrossoverStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from strategies.regime_switching import RegimeSwitchingStrategy
from strategies.trend_following import TrendFollowingStrategy


@dataclass
class StrategyGridConfig:
    strategy_cls: type
    param_grid: dict[str, list]
    score_metric: str = "Sharpe Ratio"
    min_train_bars: int = 250
    min_test_bars: int = 100
    # Each strategy has a "natural" timeframe/lookback it was designed and
    # sized for (e.g. daily bars for mean_reversion, 1h for trend_following)
    # -- these are the defaults walk_forward.py uses for --strategy all so a
    # single CLI invocation doesn't accidentally run every strategy on the
    # same (wrong-for-most-of-them) timeframe.
    default_timeframe: str = "1d"
    default_days: int = 2700
    default_symbol: str = "MES"


GRID_CONFIGS: dict[str, StrategyGridConfig] = {
    "ma_crossover": StrategyGridConfig(
        strategy_cls=MACrossoverStrategy,
        # Standard fast/slow SMA crossover pairs (10/30, 20/50, 50/200-style
        # ratios, scaled down for a bot trading more actively than a
        # classic "golden cross" system).
        param_grid={
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 40, 50],
        },
        min_train_bars=300,
        min_test_bars=150,
    ),
    "trend_following": StrategyGridConfig(
        strategy_cls=TrendFollowingStrategy,
        # 20/50 EMA is the textbook pair cited in the original design notes;
        # search a narrow band around it. ATR stop multiplier 2-3x is the
        # standard trailing-stop range cited across CTA/trend-following
        # literature (tighter chops you out of normal noise, looser gives
        # back too much profit).
        param_grid={
            "fast_ema_period": [15, 20, 25],
            "slow_ema_period": [40, 50, 60],
            "atr_stop_multiplier": [2.0, 2.5, 3.0],
        },
        # Now runs on 1h bars (yfinance-capped at 730 days, ~11400 bars) --
        # sized for ~10 folds across that ceiling, same as donchian_breakout
        # and regime_switching which also moved to 1h.
        min_train_bars=750,
        min_test_bars=350,
        default_timeframe="1h",
        default_days=729,
    ),
    "mean_reversion": StrategyGridConfig(
        strategy_cls=MeanReversionStrategy,
        # 20-period Bollinger Bands at 2 std dev is THE standard (this is
        # literally Bollinger's own recommended default). RSI 14/70/30 is
        # equally standard (Wilder's original parameters). Narrow search
        # around both rather than reinventing them.
        param_grid={
            "bb_period": [14, 20, 26],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_period": [10, 14, 20],
        },
        min_train_bars=300,
        min_test_bars=150,
    ),
    "donchian_breakout": StrategyGridConfig(
        strategy_cls=DonchianBreakoutStrategy,
        # Rewritten to match Andreas Clenow's documented "Following the
        # Trend" system (channel_period=50, trend_fast/slow_ema=50/100,
        # atr_trail_multiplier=3.0 are his stated defaults -- see the
        # strategy's docstring for sourcing). Search a narrow band around
        # the trailing-stop multiple, the one knob he notes as less
        # rigidly fixed in his own writing; leave channel/EMA periods at
        # his documented values rather than searching them, since the
        # whole point of reproducing a documented system is fidelity to
        # it, not re-discovering "better" numbers via search.
        param_grid={
            "atr_trail_multiplier": [2.5, 3.0, 3.5],
            "daily_vol_target_pct": [0.0015, 0.002, 0.0025],
        },
        # 1h bars, yfinance-capped at 730 days (~11400 bars) -- sized for
        # ~10 folds across that ceiling.
        min_train_bars=750,
        min_test_bars=350,
        default_timeframe="1h",
        default_days=729,
    ),
    "regime_switching": StrategyGridConfig(
        strategy_cls=RegimeSwitchingStrategy,
        # Deliberately narrow: only search the regime-classification
        # thresholds (the genuinely novel part of this strategy), not every
        # sub-strategy parameter -- that combinatorial explosion would turn
        # this into exactly the kind of wide, curve-fitting-prone search
        # walk-forward is supposed to guard against. Sub-strategy params
        # (EMA periods, BB period, etc.) stay at the values already
        # validated in trend_following/mean_reversion's own walk-forward runs.
        param_grid={
            "adx_trending_threshold": [20.0, 25.0, 30.0],
            "adx_ranging_threshold": [15.0, 20.0],
        },
        min_train_bars=750,
        min_test_bars=350,
        default_timeframe="1h",
        default_days=729,
    ),
    "orb": StrategyGridConfig(
        strategy_cls=OpeningRangeBreakoutStrategy,
        # 15/30/60-minute opening ranges are the three standard ORB
        # variants traders use. Stop/target in ticks scaled proportionally
        # (2:1 reward:risk, the standard ORB ratio).
        param_grid={
            "range_bars": [3, 6, 12],  # 15min/30min/60min at 5-min bars
            "stop_ticks": [6, 8, 12],
            "target_ticks": [12, 16, 24],
        },
        score_metric="Profit Factor",  # ORB has many small trades -- Sharpe is noisy here
        # ORB trades ~1x/session, so bar count is the wrong unit to size
        # windows by -- 1000 5-min bars is only ~4 trading sessions, nowhere
        # near enough for one-trade-per-session stats to mean anything.
        # ~275 5-min bars/session (23h Globex day) x ~40 sessions per window
        # is closer to a usable sample size for this strategy specifically.
        min_train_bars=275 * 40,
        min_test_bars=275 * 15,
        default_timeframe="5m",
        default_days=59,
    ),

    # --- Equity variants: same strategy classes, run against SPY instead
    # of MES. dollars_per_point=1.0 is baked into the grid as a fixed
    # (non-searched) param -- 1 share moving $1 = $1 P&L, no futures
    # contract multiplier. Equities give much deeper daily-bar history
    # (~10yr vs futures' ~7yr ceiling) but far fewer intraday bars per
    # --days window (RTH-only ~6.5h/day vs futures' ~23h/day Globex
    # session) -- see YFinanceDataSource's docstring for the tradeoff.
    "trend_following_spy": StrategyGridConfig(
        strategy_cls=TrendFollowingStrategy,
        param_grid={
            "fast_ema_period": [15, 20, 25],
            "slow_ema_period": [40, 50, 60],
            "atr_stop_multiplier": [2.0, 2.5, 3.0],
            "dollars_per_point": [1.0],
        },
        # SPY 1h data has ~1/3 the bars of MES 1h over the same --days
        # window (RTH-only ~6.5h/day vs futures' ~23h/day) -- fold sizes
        # scaled down proportionally rather than reusing MES's 750/350,
        # which would starve this of enough folds to say anything.
        min_train_bars=300,
        min_test_bars=150,
        default_timeframe="1h",
        default_days=729,
        default_symbol="SPY",
    ),
    "donchian_breakout_spy": StrategyGridConfig(
        strategy_cls=DonchianBreakoutStrategy,
        param_grid={
            "atr_trail_multiplier": [2.5, 3.0, 3.5],
            "daily_vol_target_pct": [0.0015, 0.002, 0.0025],
            "dollars_per_point": [1.0],
        },
        min_train_bars=300,
        min_test_bars=150,
        default_timeframe="1h",
        default_days=729,
        default_symbol="SPY",
    ),
    "regime_switching_spy": StrategyGridConfig(
        strategy_cls=RegimeSwitchingStrategy,
        param_grid={
            "adx_trending_threshold": [20.0, 25.0, 30.0],
            "adx_ranging_threshold": [15.0, 20.0],
            "dollars_per_point": [1.0],
        },
        min_train_bars=300,
        min_test_bars=150,
        default_timeframe="1h",
        default_days=729,
        default_symbol="SPY",
    ),
    "ma_crossover_spy": StrategyGridConfig(
        strategy_cls=MACrossoverStrategy,
        param_grid={
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 40, 50],
        },
        min_train_bars=300,
        min_test_bars=150,
        default_symbol="SPY",
        default_days=3650,  # equities have much deeper daily history than futures
    ),
    "mean_reversion_spy": StrategyGridConfig(
        strategy_cls=MeanReversionStrategy,
        param_grid={
            "bb_period": [14, 20, 26],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_period": [10, 14, 20],
        },
        min_train_bars=300,
        min_test_bars=150,
        default_symbol="SPY",
        default_days=3650,
    ),
}
