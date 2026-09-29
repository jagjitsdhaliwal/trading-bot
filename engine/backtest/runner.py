"""Runs a Strategy through backtesting.py with commission + slippage modeling
and prints/saves a full performance report.

Commission/slippage defaults are ballpark futures numbers -- confirm against
your actual broker's fee schedule before trusting results:
- MES commission: ~$0.37-0.85/side depending on broker (using $0.50 default)
- Slippage: modeled as a fixed % of price; real slippage varies by liquidity
  and time of day and this is a simplification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

import pandas as pd
from backtesting import Backtest, Strategy

DEFAULT_COMMISSION_PER_CONTRACT = 0.50
DEFAULT_SLIPPAGE_PCT = 0.0001  # 1 bp


def run_backtest(
    data: pd.DataFrame,
    strategy_cls: Type[Strategy],
    cash: float = 50_000,
    commission: float = DEFAULT_SLIPPAGE_PCT,
    margin: float = 1.0,
    **strategy_params,
) -> tuple[pd.Series, Backtest]:
    """
    commission here is passed straight to backtesting.py's Backtest(), which
    models it as a fraction of trade value (not per-contract) -- simplest way
    to account for both exchange fees and slippage in one knob. For a more
    precise per-contract commission model, extend this with a custom
    Broker/Strategy accounting layer later.
    """
    bt = Backtest(
        data,
        strategy_cls,
        cash=cash,
        commission=commission,
        margin=margin,
        trade_on_close=False,
        exclusive_orders=True,
    )
    stats = bt.run(**strategy_params)
    return stats, bt


def print_report(stats: pd.Series) -> None:
    keys = [
        "Start",
        "End",
        "Duration",
        "Return [%]",
        "Buy & Hold Return [%]",
        "Max. Drawdown [%]",
        "Avg. Drawdown [%]",
        "# Trades",
        "Win Rate [%]",
        "Best Trade [%]",
        "Worst Trade [%]",
        "Avg. Trade [%]",
        "Max. Trade Duration",
        "Avg. Trade Duration",
        "Profit Factor",
        "Expectancy [%]",
        "SQN",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
    ]
    print("\n=== Backtest Report ===")
    for k in keys:
        if k in stats:
            print(f"{k:.<30}{stats[k]}")

    trades = stats.get("_trades")
    if trades is not None and not trades.empty:
        wins = trades[trades["PnL"] > 0]
        losses = trades[trades["PnL"] <= 0]
        print("\n--- Trade Breakdown ---")
        print(f"Total trades: {len(trades)}")
        print(f"Wins: {len(wins)}  Losses: {len(losses)}")
        if len(wins):
            print(f"Avg win: ${wins['PnL'].mean():.2f}")
        if len(losses):
            print(f"Avg loss: ${losses['PnL'].mean():.2f}")
        print(f"Max consecutive losses: {_max_consecutive_losses(trades)}")


def _max_consecutive_losses(trades: pd.DataFrame) -> int:
    streak = 0
    max_streak = 0
    for pnl in trades["PnL"]:
        if pnl <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def save_trade_log(stats: pd.Series, out_path: str | Path) -> None:
    trades = stats.get("_trades")
    if trades is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out_path, index=False)
        print(f"\nTrade log saved to {out_path}")
