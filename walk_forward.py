"""Walk-forward optimization/validation CLI.

For each strategy, this:
  1. Splits historical MES/ES data into rolling train/test window pairs.
  2. Grid-searches the strategy's parameter space on EACH train window only
     (see engine/backtest/param_grids.py for the search ranges -- narrow,
     anchored to standard textbook values, not wide arbitrary search).
  3. Locks the best-scoring params and evaluates them, unmodified, on the
     immediately following unseen test window.
  4. Reports aggregate out-of-sample performance + per-parameter stability
     across folds.

Usage:
    uv run python walk_forward.py --strategy ma_crossover
    uv run python walk_forward.py --strategy trend_following
    uv run python walk_forward.py --strategy all   # runs every strategy at its own natural timeframe
"""

from __future__ import annotations

import argparse
import datetime as dt

from engine.backtest.param_grids import GRID_CONFIGS
from engine.backtest.walkforward import print_walk_forward_report, run_walk_forward
from engine.data.yfinance_source import YFinanceDataSource


MIN_FOLDS_FOR_MEANINGFUL_RESULT = 4


def run_one(
    symbol: str | None, name: str, days: int | None, timeframe: str | None, cash: float
) -> None:
    config = GRID_CONFIGS[name]
    # Fall back to each strategy's own natural timeframe/lookback/symbol
    # (e.g. 1h for trend_following, 5m for orb, SPY for the _spy variants)
    # when the user didn't explicitly override -- this matters most for
    # `--strategy all`, where a single CLI invocation would otherwise run
    # every strategy on whatever the argparse default happens to be, which
    # is wrong for most of them.
    effective_timeframe = timeframe or config.default_timeframe
    effective_days = days or config.default_days
    effective_symbol = symbol or config.default_symbol

    data_source = YFinanceDataSource()
    end = dt.datetime.now()
    start = end - dt.timedelta(days=effective_days)
    df = data_source.get_ohlcv(effective_symbol, effective_timeframe, start, end)

    bars_needed_for_min_folds = MIN_FOLDS_FOR_MEANINGFUL_RESULT * (
        config.min_train_bars + config.min_test_bars
    )
    # NOTE: this is a conservative (non-overlapping-fold) estimate; the
    # engine's default step_bars=test_bars means folds actually only need
    # min_train_bars + N*min_test_bars bars, but we intentionally require
    # more here -- a walk-forward report from 1-2 folds isn't a "parameter
    # stability" result, it's just a normal backtest wearing a costume.
    if len(df) < bars_needed_for_min_folds:
        print(
            f"\nSkipping {name}: only {len(df)} bars available for "
            f"{effective_timeframe} data over {effective_days} days. "
            f"Walk-forward needs enough data for at least "
            f"{MIN_FOLDS_FOR_MEANINGFUL_RESULT} folds "
            f"(~{bars_needed_for_min_folds} bars with this strategy's window "
            f"sizes) to say anything about parameter stability -- fewer folds "
            f"than that is just a regular backtest wearing a walk-forward "
            f"costume. "
            + (
                "yfinance intraday data is capped well below what this "
                "strategy needs (60 days for 5m bars, 730 days for 1h), "
                "which is the likely limiter here -- this strategy needs "
                "real historical intraday data (see README's data-source "
                "next steps) before walk-forward results on it can be trusted."
                if effective_timeframe != "1d"
                else "Try a larger --days value."
            )
        )
        return

    result = run_walk_forward(
        df,
        config.strategy_cls,
        param_grid=config.param_grid,
        train_bars=config.min_train_bars,
        test_bars=config.min_test_bars,
        cash=cash,
        score_metric=config.score_metric,
    )
    print_walk_forward_report(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward optimization/validation")
    parser.add_argument(
        "--symbol", default=None,
        help="Futures (MES, ES, MNQ, NQ) or any equity ticker (SPY, QQQ, ...). "
             "Defaults to each strategy's own natural symbol if omitted."
    )
    parser.add_argument(
        "--strategy",
        default="ma_crossover",
        choices=list(GRID_CONFIGS.keys()) + ["all"],
    )
    parser.add_argument(
        "--timeframe", default=None, choices=["1m", "5m", "15m", "30m", "1h", "1d"],
        help="Defaults to each strategy's own natural timeframe if omitted "
             "(see engine/backtest/param_grids.py's default_timeframe)."
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Total lookback window in days. Defaults to each strategy's own "
             "natural lookback if omitted."
    )
    parser.add_argument("--cash", type=float, default=50_000)
    args = parser.parse_args()

    names = list(GRID_CONFIGS.keys()) if args.strategy == "all" else [args.strategy]
    for name in names:
        run_one(args.symbol, name, args.days, args.timeframe, args.cash)


if __name__ == "__main__":
    main()
