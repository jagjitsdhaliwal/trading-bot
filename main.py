"""Entry point: run a backtest, save results to the trade database, and
report whether the strategy would have survived a prop-firm-style
evaluation (profit target / max trailing drawdown).

Usage:
    uv run python main.py --symbol MES --strategy ma_crossover
    uv run python main.py --symbol MES --strategy trend_following
    uv run python main.py --symbol MES --strategy orb --timeframe 5m --days 55

--timeframe/--days default to each strategy's own natural window (see
engine/backtest/param_grids.py's default_timeframe/default_days) if omitted
-- several strategies here are 1h-native (trend_following, donchian_breakout,
regime_switching) and will silently produce zero trades on the wrong
timeframe/lookback combination otherwise.
"""

from __future__ import annotations

import argparse
import datetime as dt

from engine.backtest.param_grids import GRID_CONFIGS
from engine.backtest.runner import print_report, run_backtest, save_trade_log
from engine.data.yfinance_source import YFinanceDataSource
from engine.db.store import TradeRecord, TradeStore
from engine.risk.engine import RiskEngine, RiskLimits

STRATEGIES = {name: config.strategy_cls for name, config in GRID_CONFIGS.items()}


def check_prop_firm_survival(stats, limits: RiskLimits) -> None:
    """Rough post-hoc check: does the backtest's peak-to-trough equity curve
    ever breach the configured trailing drawdown, and does it ever hit the
    profit target? This is NOT the same as running the risk engine bar-by-bar
    (see engine/risk/engine.py for that), but gives a fast sanity check."""
    equity = stats["_equity_curve"]["Equity"]
    peak = equity.cummax()
    drawdown = peak - equity
    max_dd = drawdown.max()

    print("\n=== Prop-Firm Survival Check ===")
    print(f"Configured max trailing drawdown: ${limits.max_trailing_drawdown:,.2f}")
    print(f"Actual max drawdown in backtest:  ${max_dd:,.2f}")
    if limits.max_trailing_drawdown and max_dd >= limits.max_trailing_drawdown:
        breach_idx = drawdown[drawdown >= limits.max_trailing_drawdown].index[0]
        print(f"WOULD HAVE BEEN DISQUALIFIED on/around {breach_idx}")
    else:
        print("Drawdown limit never breached in this backtest.")

    final_profit = equity.iloc[-1] - equity.iloc[0]
    print(f"\nConfigured profit target: ${limits.profit_target:,.2f}")
    print(f"Final backtest profit:    ${final_profit:,.2f}")
    if limits.profit_target and final_profit >= limits.profit_target:
        print("Would have hit the profit target.")
    else:
        print("Would NOT have hit the profit target in this window.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest against historical market data")
    parser.add_argument(
        "--symbol", default=None,
        help="Futures (MES, ES, MNQ, NQ) or any equity ticker (SPY, QQQ, ...). "
             "Defaults to this strategy's own natural symbol if omitted."
    )
    parser.add_argument("--strategy", default="ma_crossover", choices=list(STRATEGIES.keys()))
    parser.add_argument(
        "--timeframe", default=None, choices=["1m", "5m", "15m", "30m", "1h", "1d"],
        help="Defaults to this strategy's own natural timeframe if omitted."
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Lookback window in days. Defaults to this strategy's own natural lookback if omitted."
    )
    parser.add_argument("--cash", type=float, default=50_000)
    args = parser.parse_args()

    config = GRID_CONFIGS[args.strategy]
    timeframe = args.timeframe or config.default_timeframe
    days = args.days or config.default_days
    symbol = args.symbol or config.default_symbol

    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)

    print(f"Fetching {symbol} {timeframe} bars from {start.date()} to {end.date()}...")
    data_source = YFinanceDataSource()
    df = data_source.get_ohlcv(symbol, timeframe, start, end)
    print(f"Loaded {len(df)} bars.")

    # Fixed (single-value) grid params -- e.g. dollars_per_point=1.0 for the
    # _spy strategy variants -- are baked into config.param_grid the same
    # way walk_forward.py's grid search treats them; pass them through here
    # too so main.py's single backtest actually uses the right contract math.
    fixed_params = {k: v[0] for k, v in config.param_grid.items() if len(v) == 1}

    strategy_cls = STRATEGIES[args.strategy]
    stats, bt = run_backtest(df, strategy_cls, cash=args.cash, **fixed_params)
    print_report(stats)

    limits = RiskLimits(account_starting_balance=args.cash)
    check_prop_firm_survival(stats, limits)

    save_trade_log(stats, f"logs/{args.strategy}_{symbol}_trades.csv")

    store = TradeStore()
    run_id = store.create_run(
        run_type="backtest",
        strategy_name=strategy_cls.__name__,
        strategy_params={},
        symbol=symbol,
        timeframe=timeframe,
        starting_balance=args.cash,
        started_at=dt.datetime.now().isoformat(),
        notes=f"CLI backtest run: {days} days lookback",
    )
    trades = stats.get("_trades")
    if trades is not None:
        for _, t in trades.iterrows():
            store.add_trade(
                run_id,
                TradeRecord(
                    symbol=symbol,
                    side="LONG" if t["Size"] > 0 else "SHORT",
                    quantity=abs(int(t["Size"])),
                    entry_price=float(t["EntryPrice"]),
                    exit_price=float(t["ExitPrice"]),
                    entry_time=str(t["EntryTime"]),
                    exit_time=str(t["ExitTime"]),
                    pnl=float(t["PnL"]),
                    pnl_pct=float(t["ReturnPct"]),
                    exit_reason="signal",
                ),
            )
    store.end_run(run_id, dt.datetime.now().isoformat())
    print(f"\nSaved to trade database as run_id={run_id} ({store.db_path})")


if __name__ == "__main__":
    main()
