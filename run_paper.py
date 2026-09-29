"""Run the live paper-trading loop against free yfinance data + local paper
broker simulator. No Tradovate API access required.

Usage:
    uv run python run_paper.py --symbol MES --strategy ma_crossover --timeframe 1d --poll-seconds 3600
"""

from __future__ import annotations

import argparse
import datetime as dt

from engine.data.yfinance_source import YFinanceDataSource
from engine.paper.loop import PaperTradingLoop
from engine.risk.engine import RiskEngine, RiskLimits
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.ma_crossover import MACrossoverStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from strategies.regime_switching import RegimeSwitchingStrategy
from strategies.trend_following import TrendFollowingStrategy

STRATEGIES = {
    "ma_crossover": MACrossoverStrategy,
    "orb": OpeningRangeBreakoutStrategy,
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "regime_switching": RegimeSwitchingStrategy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live paper-trading loop")
    parser.add_argument("--symbol", default="MES", choices=["MES", "ES", "MNQ", "NQ"])
    parser.add_argument("--strategy", default="ma_crossover", choices=list(STRATEGIES.keys()))
    parser.add_argument("--timeframe", default="1d", choices=["1m", "5m", "15m", "30m", "1h", "1d"])
    parser.add_argument("--poll-seconds", type=int, default=3600,
                         help="How often to check for a new bar/signal. "
                              "Match this to your timeframe -- no point polling "
                              "every 60s on daily bars.")
    parser.add_argument("--cash", type=float, default=50_000)
    parser.add_argument("--history-days", type=int, default=720,
                         help="How much lookback history to feed the strategy each tick. "
                              "Must be enough bars for your strategy's slowest indicator to "
                              "warm up (e.g. a 30-period MA on daily bars needs way more than "
                              "30 days of calendar time once weekends/holidays are excluded).")
    parser.add_argument("--once", action="store_true",
                         help="Run a single tick and exit, instead of looping forever")
    args = parser.parse_args()

    data_source = YFinanceDataSource()
    strategy_cls = STRATEGIES[args.strategy]
    risk_engine = RiskEngine(RiskLimits(account_starting_balance=args.cash))

    loop = PaperTradingLoop(
        data_source=data_source,
        strategy_cls=strategy_cls,
        symbol=args.symbol,
        timeframe=args.timeframe,
        risk_engine=risk_engine,
        poll_seconds=args.poll_seconds,
        history_days=args.history_days,
    )

    if args.once:
        loop._run_id = loop.store.create_run(
            run_type="paper",
            strategy_name=strategy_cls.__name__,
            strategy_params={},
            symbol=args.symbol,
            timeframe=args.timeframe,
            starting_balance=args.cash,
            started_at=dt.datetime.now().isoformat(),
            notes="Single-tick test run (--once)",
        )
        loop.run_once()
        loop.store.end_run(loop._run_id, dt.datetime.now().isoformat())
        print(f"Single tick complete. Balance: ${loop.broker.get_account_balance():,.2f}")
    else:
        loop.start()


if __name__ == "__main__":
    main()
