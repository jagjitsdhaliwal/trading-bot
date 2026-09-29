"""Walk-forward optimization/validation engine.

This is the rigor step the README has flagged as "still on the next-steps
list" since the beginning of this project. The core question it answers:

    Does this strategy have a real edge, or did it just get lucky/curve-fit
    on the specific historical window we backtested it on?

How it works:
  1. Split history into N rolling (train_window, test_window) pairs, each
     window advancing forward in time (never look-ahead).
  2. On EACH train window ONLY: grid-search the strategy's parameter space,
     pick the best-scoring parameter set by a chosen metric (default:
     Sharpe Ratio).
  3. Lock those parameters and run (unmodified) on the immediately
     following, entirely unseen test window.
  4. Aggregate out-of-sample (test-window) results across all folds.

The critical discipline this enforces: the optimizer NEVER sees test-window
data when picking parameters. If it did, this would just be regular
in-sample optimization wearing a walk-forward costume, and the results would
be meaningless (guaranteed to look good, tells you nothing about the
future). This is the single most common mistake in retail algo trading.

A second signal this produces, arguably as important as the returns
themselves: PARAMETER STABILITY across folds. If the best-fit parameters
swing wildly from fold to fold (e.g. fast EMA jumping between 12 and 40),
that's a strong sign the "edge" is noise the optimizer is chasing, not a
real, persistent market inefficiency. Real edges tend to have broad, stable
optima; curve-fit noise has sharp, unstable ones.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Type

import pandas as pd
from backtesting import Backtest, Strategy


@dataclass
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_score: float
    test_stats: pd.Series


@dataclass
class WalkForwardResult:
    strategy_name: str
    folds: list[WalkForwardFold] = field(default_factory=list)

    @property
    def test_returns_pct(self) -> list[float]:
        return [f.test_stats["Return [%]"] for f in self.folds]

    @property
    def aggregate_return_pct(self) -> float:
        """Compounded return across all out-of-sample folds stitched
        together -- the honest answer to "how would this have actually
        performed if you'd traded it this way the whole time."""
        compounded = 1.0
        for r in self.test_returns_pct:
            compounded *= 1 + r / 100
        return (compounded - 1) * 100

    @property
    def pct_folds_profitable(self) -> float:
        if not self.folds:
            return 0.0
        wins = sum(1 for r in self.test_returns_pct if r > 0)
        return 100 * wins / len(self.folds)

    def param_stability_report(self) -> dict[str, list]:
        """Per-parameter list of the winning value in each fold -- eyeball
        this for stability (real edge) vs. wild swings (curve-fit noise)."""
        if not self.folds:
            return {}
        keys = self.folds[0].best_params.keys()
        return {k: [f.best_params[k] for f in self.folds] for k in keys}


def generate_folds(
    data: pd.DataFrame, train_bars: int, test_bars: int, step_bars: int | None = None
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split data into rolling (train, test) window pairs. step_bars
    defaults to test_bars (non-overlapping test windows, the standard
    setup)."""
    step_bars = step_bars or test_bars
    folds = []
    start = 0
    while True:
        train_end = start + train_bars
        test_end = train_end + test_bars
        if test_end > len(data):
            break
        train_df = data.iloc[start:train_end]
        test_df = data.iloc[train_end:test_end]
        folds.append((train_df, test_df))
        start += step_bars
    return folds


def _grid_search(
    train_df: pd.DataFrame,
    strategy_cls: Type[Strategy],
    param_grid: dict[str, list],
    cash: float,
    commission: float,
    score_metric: str,
) -> tuple[dict, float]:
    """Exhaustive grid search over param_grid, scored on score_metric.
    Runs ONLY on train_df -- caller must never pass test data here."""
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    best_params: dict = {}
    best_score = float("-inf")

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            bt = Backtest(
                train_df,
                strategy_cls,
                cash=cash,
                commission=commission,
                exclusive_orders=True,
                finalize_trades=True,
            )
            stats = bt.run(**params)
        except Exception:
            continue  # invalid param combo (e.g. fast_period >= slow_period) -- skip

        score = stats.get(score_metric)
        if score is None or pd.isna(score):
            continue
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


def run_walk_forward(
    data: pd.DataFrame,
    strategy_cls: Type[Strategy],
    param_grid: dict[str, list],
    train_bars: int,
    test_bars: int,
    cash: float = 50_000,
    commission: float = 0.0001,
    score_metric: str = "Sharpe Ratio",
    step_bars: int | None = None,
) -> WalkForwardResult:
    result = WalkForwardResult(strategy_name=strategy_cls.__name__)
    fold_pairs = generate_folds(data, train_bars, test_bars, step_bars)

    if not fold_pairs:
        raise ValueError(
            f"Not enough data for even one walk-forward fold: need at least "
            f"{train_bars + test_bars} bars, got {len(data)}."
        )

    for train_df, test_df in fold_pairs:
        best_params, train_score = _grid_search(
            train_df, strategy_cls, param_grid, cash, commission, score_metric
        )
        if not best_params:
            continue  # every param combo failed on this train window -- skip fold

        test_bt = Backtest(
            test_df,
            strategy_cls,
            cash=cash,
            commission=commission,
            exclusive_orders=True,
            finalize_trades=True,
        )
        test_stats = test_bt.run(**best_params)

        result.folds.append(
            WalkForwardFold(
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                best_params=best_params,
                train_score=train_score,
                test_stats=test_stats,
            )
        )

    return result


def print_walk_forward_report(result: WalkForwardResult) -> None:
    print(f"\n=== Walk-Forward Report: {result.strategy_name} ===")
    print(f"Folds completed: {len(result.folds)}")

    if not result.folds:
        print("No folds completed -- insufficient data or all param combos failed.")
        return

    print(f"Aggregate out-of-sample return (compounded): {result.aggregate_return_pct:.2f}%")
    print(f"Folds profitable out-of-sample: {result.pct_folds_profitable:.1f}%")

    print("\nPer-fold results:")
    for i, fold in enumerate(result.folds):
        oos_return = fold.test_stats["Return [%]"]
        oos_trades = fold.test_stats["# Trades"]
        print(
            f"  Fold {i+1}: train {fold.train_start.date()} to {fold.train_end.date()} "
            f"-> test {fold.test_start.date()} to {fold.test_end.date()} | "
            f"params={fold.best_params} | OOS return={oos_return:.2f}% | "
            f"OOS trades={oos_trades}"
        )

    print("\nParameter stability across folds (wide swings = likely curve-fit noise):")
    for param, values in result.param_stability_report().items():
        print(f"  {param}: {values}")
