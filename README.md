# Trading Bot

Systematic futures/equity trading engine: deterministic strategy + hardcoded
risk limits + backtesting + (eventually) paper/live execution. Built around
one principle from the design discussion that kicked this off: **the
strategy logic and risk engine are deterministic code, not an LLM** — an AI
agent can sit around this system to analyze results and propose ideas, but
it never places a trade directly.

## Realistic expectations (read this before anything else)

No strategy in this repo, or any strategy that could be built, is "always
net profitable" or capable of anything like 15%/month (~435%/year)
compounded. That return profile doesn't exist as a sustainable target —
Renaissance Medallion, widely considered the best real trading track record
in history, ran ~66%/year before fees, and that's a firm with resources and
data access no retail setup can replicate. A monthly figure in the
mid-teens percent range is a hallmark of trading-scam marketing, not a
planning target. Realistic reference points: 8-15% annually is solid for a
skilled retail systematic trader; the strongest systems in this repo so far
(see `ma_crossover_spy`'s walk-forward result below) show low-double-digit
annualized out-of-sample returns with real drawdowns along the way, not a
smooth line up.

The risk engine's job (`engine/risk/engine.py`) is to make sure a losing
strategy or a bad streak can't blow up the account — hard daily-loss/
trailing-drawdown circuit breakers, position size caps — not to make losses
impossible. Individual trades and even whole months can and will lose money
under any strategy tested here.

## Architecture

```
Historical/live data (yfinance, Tradovate, CSV)
        v
   Strategy (backtesting.py Strategy subclass)
        v
   Risk Engine (hardcoded limits, kill switch)
        v
   Broker Adapter (paper sim, Tradovate, IBKR)
        v
   SQLite trade database
```

Every layer is behind an interface (`engine/data/base.py`,
`engine/broker/base.py`) so you can swap yfinance for a paid data provider,
or Tradovate for IBKR, without touching strategy code.

## Setup

Everything is managed by [`uv`](https://docs.astral.sh/uv/) — no system
Python touched, no manual virtualenv activation needed.

```bash
cd trading-bot
uv sync              # installs Python 3.12 + all dependencies into .venv
uv run pytest tests/ # confirm the risk engine tests pass
```

## Running a backtest

```bash
# Each strategy defaults to its own natural timeframe/lookback --
# --timeframe/--days only need to be passed to override.
uv run python main.py --symbol MES --strategy ma_crossover       # daily bars, ~7yr
uv run python main.py --symbol MES --strategy trend_following    # 1h bars, ~2yr (yfinance cap)
uv run python main.py --symbol MES --strategy donchian_breakout  # 1h bars, ~2yr
uv run python main.py --symbol MES --strategy mean_reversion     # daily bars, ~7yr
uv run python main.py --symbol MES --strategy regime_switching   # 1h bars, ~2yr
uv run python main.py --symbol MES --strategy orb                # 5m bars, ~55 days (yfinance cap)
```

### Strategy comparison (as of this writing, MES, 720-day daily window / 55-day 5m window)

Running any strategy above prints a single-backtest report (return,
drawdown, win rate, etc. on whatever window it fetched) plus a prop-firm
survival check. **Don't stop at that single-backtest number** — it's the
"how did this do on one historical window" question, which is the wrong
question (you can always find some window a strategy looks good on). Skip
straight to the "Walk-forward optimization" section below for the number
that actually matters: does it hold up out-of-sample.

## Strategy tournament

Six strategies now exist, following a "don't bet on one shape, run a
tournament and let walk-forward pick the survivor" design:

| Strategy | Shape | Timeframe |
|---|---|---|
| `ma_crossover` | Simple fast/slow SMA crossover (original baseline) | daily |
| `trend_following` | EMA 20/50 direction + long-term MA filter + volume confirmation + ATR trailing stop + volatility-scaled size | 1h |
| `donchian_breakout` | Reproduces Andreas Clenow's documented "Following the Trend" system: 50-day breakout, 50/100 EMA trend filter, 3x ATR trailing stop from trade peak, 0.2%-daily-volatility position sizing | 1h |
| `mean_reversion` | Bollinger Band + RSI exhaustion, gated by ADX so it never fades a confirmed trend | daily |
| `regime_switching` | ADX/ATR-based regime classifier routes each bar to trend-following logic (trending), mean-reversion logic (ranging), or stands down entirely (high volatility) | 1h |
| `orb` | Opening range breakout anchored to 9:30am ET RTH open | 5m |

`trend_following` and `mean_reversion` both got upgrades directly motivated
by earlier walk-forward findings (see below) rather than by hand-tuning
against a backtest number — `mean_reversion`'s ADX gate exists specifically
because the original version's walk-forward run showed the worst parameter
stability of anything tested, a fingerprint of exactly the failure mode
("mean reversion looks fine until the market starts trending") that
systematic traders warn about.

### `donchian_breakout`: reverse-engineered from a real documented track record

There's no accessible database of actual "most profitable traders'" trade
logs — what's genuinely useful and public is well-documented *systematic
rule sets* with real multi-decade track records. `donchian_breakout` was
rewritten to reproduce Andreas Clenow's system from "Following the Trend"
(2013) as faithfully as this codebase allows, sourced from his own site
(followingthetrend.com/the-trading-system/trading-system-rules/ — PRIMARY):

- **Entry**: new 50-day closing high (long) / low (short)
- **Trend filter**: 50-day EMA above 100-day EMA required for longs (and
  the inverse for shorts) — only trade with the intermediate trend
- **Exit**: ATR trailing stop at 3x ATR (100-day exponentially smoothed)
  from the trade's peak/trough since entry — not a fixed stop from entry,
  not an opposite-breakout exit
- **Position sizing**: each position sized to contribute a constant ~0.2%
  of account equity in daily volatility — `clenow_position_size()` in
  `engine/strategy/indicators.py`. Notably this formula has no reference to
  stop distance at all, unlike the Turtle-style sizing used elsewhere in
  this codebase (`volatility_scaled_size()`, `turtle_position_size()`) —
  these are two distinct, independently-documented sizing philosophies.

The original Turtle Trading rules (Dennis/Eckhardt, 1983-1988) are also
implemented as reference functions (`turtle_n()`, `turtle_position_size()`
in `indicators.py`) — 20-day ATR-based N, 1%-of-equity-per-N sizing, 2N
stops — though not yet wired into a full strategy class the way Clenow's
system is.

**A real bug this rewrite surfaced**: implementing Clenow's sizing formula
exposed that `volatility_scaled_size()` — used by `trend_following` and
`regime_switching` since they were first built — had been silently
overstating risk-per-contract by 4x on MES. It took a `(tick_value,
tick_size)` pair and computed `stop_distance_ticks = (ATR * multiplier) /
tick_size`, then multiplied by `tick_value` — but `tick_value` was actually
being passed as $5/**point** (correct for MES), not $/tick, so dividing by
`tick_size` double-counted the point/tick conversion. This silently starved
both strategies of real position sizes: after the fix, `trend_following`'s
trade count jumped from 24 to 322 and `regime_switching`'s from 82 to 432
on the same ~2-year 1h window — meaning every walk-forward result reported
for those two strategies before this fix was measuring a strategy that was
barely trading, not the strategy as actually designed. Both were re-run
after the fix; see updated results below.

## Walk-forward optimization: does any of this hold up out-of-sample?

The single-backtest table above answers "how did this strategy do on this
one historical window" — which is exactly the wrong question, because you
can always find *some* window a strategy looks good on. Walk-forward
optimization answers a harder, more honest question: **if you'd picked
parameters using only past data at the time, then traded forward blind,
would it have worked?**

```bash
uv run python walk_forward.py --strategy trend_following
uv run python walk_forward.py --strategy all   # runs every strategy at its own natural timeframe
```

How it works: history is split into rolling (train, test) window pairs. A
small, standard-values-anchored parameter grid (see
`engine/backtest/param_grids.py` — e.g. EMA 15/20/25 and 40/50/60, not some
wide arbitrary search) is grid-searched **on the train window only**, the
winning parameters are locked, and then evaluated unmodified on the very
next, entirely unseen test window. This repeats across the whole history.
The optimizer never sees test data when picking parameters — if it did,
this would just be regular curve-fitting wearing a walk-forward costume.
Each strategy defaults to its own natural timeframe/lookback
(`default_timeframe`/`default_days` in `param_grids.py`) so `--strategy all`
doesn't run everything on the same wrong-for-most-of-them timeframe.

### Results

| Strategy | Window | Folds | Aggregate OOS return | Folds profitable | Parameter stability |
|---|---|---|---|---|---|
| `ma_crossover` | daily, ~7yr | 10 | +10.9% | 50% | Weak — `slow_period` drifts 20-50 |
| `trend_following` | 1h, ~2yr (yfinance cap) | 30 | -5.38% | 20% | Bimodal — `fast_ema_period` splits between 15 and 25 with no convergence |
| `donchian_breakout` (Clenow rules) | 1h, ~2yr | 30 | -2.26% | 27% | Weak — `atr_trail_multiplier`/`daily_vol_target_pct` scattered across the grid |
| `mean_reversion` (ADX-gated) | daily, ~7yr | 10 | -11.3% | 10% | Weak — few trades per fold once gated to only genuinely ranging days |
| `regime_switching` | 1h, ~2yr | 30 | -2.45% | 40% | `adx_trending_threshold` favors 30 in ~half the folds, otherwise scattered |
| `orb` | *(blocked — see below)* | — | — | — | — |

**Honest read: none of the six clears the bar for "real, tradeable edge."**
A few specific findings worth internalizing rather than just the top-line
numbers:

- **`trend_following` and `regime_switching`'s numbers above are corrected
  results, re-run after fixing a real position-sizing bug** (see the
  `donchian_breakout` section above for the full story) — a mislabeled
  `tick_value` was overstating risk-per-contract by 4x, which was starving
  both strategies of position size and made most out-of-sample windows
  trade barely or not at all. After the fix, `trend_following` trades ~15x
  more often (24 → 322 trades on the same window) and its result got
  *worse*, not better (was near-zero, now -5.38%) — the earlier "looks
  roughly flat" read was an artifact of the bug suppressing trades, not a
  genuine near-breakeven strategy. `regime_switching` actually improved
  (16.7% → 40% fold win rate) once it could size positions correctly,
  though it's still net negative. The lesson: a near-zero walk-forward
  result with suspiciously few trades deserves the same skepticism as a
  too-good-to-be-true one — both can indicate a bug rather than a real
  finding.
- **`mean_reversion`'s ADX gate made walk-forward numbers *worse*
  (+3.1% → -11.3%)**, which was the opposite of the hoped-for effect and
  worth investigating before accepting: it turned out only ~17% of days in
  this ~7-year window (2019-2026, a persistently trend-dominated period)
  ever classify as "ranging" by ADX. The gate is working exactly as
  designed — it's revealing that this specific historical period simply
  hasn't offered many genuine mean-reversion opportunities, and the few it
  did offer, this version still lost on. That's a more honest number than
  the ungated version's better-looking result, which was mostly noise from
  trading during trends it shouldn't have.
- **`donchian_breakout`, now reproducing Clenow's exact documented rules**
  (see above), still shows no parameter convergence and a negative result.
  Faithfully reproducing a real documented track record's rules doesn't
  guarantee those rules produce an edge on a different instrument (MES
  futures) over a different period (2024-2026) than the system was
  originally built and proven on — a strategy with a genuine multi-decade
  track record on a diversified futures portfolio isn't guaranteed to work
  on one instrument over one ~2-year window, and this result doesn't
  contradict Clenow's own track record, it just means this narrow
  reproduction hasn't demonstrated an edge here.
- **`regime_switching` doesn't yet beat its ungated components.** Composing
  trend-following and mean-reversion behind a regime switch is
  conceptually appealing but this implementation isn't outperforming
  either sub-strategy in isolation — worth remembering when reaching for
  "combine strategies" as a fix; composition isn't automatically better
  than a single well-validated strategy.

**`orb` walk-forward is currently blocked, and the CLI says so rather than
faking a result**: it needs enough 5-minute sessions per fold to mean
anything (see `min_train_bars`/`min_test_bars` in `param_grids.py`), and
yfinance's intraday history is capped at roughly the last 60 days
regardless of what `--days` you pass. Getting a real ORB walk-forward
result requires the paid intraday data source noted in "Next steps" below.

**Takeaway for what to do next**: none of these six strategies has
demonstrated a walk-forward-validated edge yet. That's a legitimate,
useful finding on its own, not a failure of the tooling — it means the
honest next step is either (a) accept none of these are ready to risk real
capital on and go looking for a different strategy shape, or (b) treat
this as a starting point and iterate on entry/exit logic with walk-forward
as the ongoing check, not skip straight to paper/live trading on the
strongest-looking single backtest.

Each run prints a full performance report (return, drawdown, Sharpe/Sortino,
profit factor, consecutive losses, etc.), saves a trade log CSV to `logs/`,
and persists everything to `data/trading.db` (SQLite) for later analysis.

It also prints a **prop-firm survival check** — whether the backtest's
equity curve would have breached a configured max trailing drawdown, and
whether it would have hit the profit target, using Topstep's $50K Combine
numbers as the default ($2,000 max loss / $3,000 target). This is the most
important number in the report: a strategy can be net profitable and still
have been disqualified from an evaluation along the way.

## Running the live paper-trading loop

This is fully operational today — no Tradovate API access needed. It polls
free yfinance data on an interval, re-runs your strategy on the
accumulated history, and routes any resulting signal through the risk
engine before filling it against a local in-memory paper broker.

```bash
# Single tick -- good for testing/debugging without waiting on a poll loop
uv run python run_paper.py --symbol MES --strategy ma_crossover --timeframe 1d --history-days 720 --once

# Continuous loop -- Ctrl+C or `kill <pid>` to stop cleanly
uv run python run_paper.py --symbol MES --strategy ma_crossover --timeframe 1d --history-days 720 --poll-seconds 3600
```

Important flags:
- `--history-days` must give your strategy's slowest indicator enough bars
  to warm up. The MA crossover's 30-period slow MA needs way more than 30
  *days* of calendar time once weekends/holidays thin it out — 720 is a
  safe default for daily bars. Too little history and the strategy will
  correctly report "no signal yet" indefinitely.
- `--poll-seconds` should roughly match your `--timeframe` — no benefit to
  polling every 60s on daily bars; you'd just be hammering Yahoo Finance
  for the same unchanged bar.

Every tick prints its status (flat / holding N / new signal) so you can
tell it's alive rather than silently doing nothing. Each run is recorded in
`data/trading.db` under `run_type='paper'`, separate from backtest runs, so
you can compare paper performance against what the backtest predicted.

Once your Tradovate API access is approved, swapping the data feed and
execution from yfinance/`PaperBrokerAdapter` to `TradovateAdapter`'s demo
environment is the next step — the strategy/risk code doesn't change.

## What's real vs. stubbed right now

| Component | Status |
|---|---|
| Data (yfinance) | Working — verified against live MES/ES data |
| Data (CSV loader) | Working — swap-in point for paid data later |
| Backtest engine | Working — built on `backtesting.py` |
| Risk engine | Working — 9 passing tests, hardcoded limits |
| SQLite trade DB | Working — schema + store verified |
| MA crossover strategy | Working baseline |
| Trend-following strategy | Working — EMA 20/50 + long-term MA filter + volume confirmation + ATR trailing stop + volatility-scaled size. Walk-forward shows it's currently over-filtered (many zero-trade windows) |
| Donchian breakout strategy | Working — 20-bar channel breakout + volume/ATR filters, mechanical entry logic |
| Mean-reversion strategy | Working — Bollinger Band + RSI entry, hard stop at 3 std dev, ADX gate to avoid fading confirmed trends |
| Regime-switching strategy | Working — ADX/ATR regime classifier routes to trend-following or mean-reversion logic, stands down in high volatility. Not yet outperforming its ungated sub-strategies |
| ORB strategy | Working — session detection fixed to anchor on 9:30am ET RTH open (was previously broken on 24h futures data, see git history / strategy docstring for what that bug looked like) |
| Regime detector | Working — `engine/strategy/regime.py`, ADX + ATR percentile based, rule-based (not ML) |
| Time-of-day filters | Working — `engine/strategy/time_filters.py`; dead-zone window wired into ORB. FOMC/CPI blackout list is a manually-maintained stub, not a live economic calendar |
| Tradovate broker adapter | Auth + order placement implemented, **not yet tested against a real demo account** — historical bars endpoint not wired up (use yfinance for backtesting) |
| IBKR broker adapter | Stub only — not implemented. Only build this out if you specifically need IBKR over Tradovate |
| Paper broker (local sim) | Working — fills instantly at last quote, tracks positions/balance in memory |
| Live paper-trading loop | Working — polls yfinance, runs strategy, checks risk engine, fills via paper broker, persists to DB. See `run_paper.py` |
| Walk-forward optimization | Working for daily-bar strategies (`walk_forward.py`) — 10-fold validation across ~7 years of MES data. Blocked for `orb` until real intraday data is available (yfinance's ~60-day intraday cap isn't enough bars for a meaningful fold count) |
| AI research agent | Not built yet — see "Next steps" |

## Next steps (in the order I'd tackle them)

1. **None of the 4 strategies has a walk-forward-validated edge yet** (see
   the walk-forward section above). Before doing anything else, decide
   whether to iterate on entry/exit logic for one of these (using
   `walk_forward.py` as the ongoing check on every change) or look for a
   different strategy shape entirely. Don't skip this and go straight to
   paper/live trading on whichever strategy's single-backtest table looked
   best — that's exactly the trap walk-forward exists to catch.

2. **Get Tradovate API access.** A regular Tradovate login is not enough —
   API access (the CID/SEC credentials in `.env.example`) is a separate
   request/approval process through Tradovate (check Settings → API Access
   in the Tradovate dashboard, or their developer portal at
   api.tradovate.com). This can take some back-and-forth, which is why the
   local paper-trading loop above doesn't depend on it. Once approved, fill
   in `.env` (copy `.env.example`) and test `TradovateAdapter.connect()`
   and `place_order()` against the demo environment before anything else.

3. **Get real intraday data.** yfinance's intraday bars are capped at
   roughly the last 60 days regardless of how far back you ask — fine for
   prototyping, not for serious backtesting, and specifically what's
   currently blocking `orb`'s walk-forward validation. Databento
   (pay-as-you-go, has a free tier for small pulls) is a reasonable next
   step for CME minute/tick data. Drop
   exported CSVs into `data/historical/` and point `CSVDataSource` at them
   — no other code needs to change.

4. **Add the AI research agent** — give it read access to `data/trading.db`
   and let it do the analysis job described in the original design doc
   ("find conditions where my strategy loses money," "test excluding the
   first 15 minutes," etc.). This should read the trade database and
   propose experiments; it should never call `place_order()` directly.

5. **Only after a strategy has a walk-forward-validated edge**: pick a prop
   firm and pay for an evaluation. Topstep is a reasonable default to
   research first (clear sim-to-funded structure), but check current
   terms/reputation before paying — trailing drawdown mechanics and payout
   consistency change firm-to-firm and over time.

## Equities: does switching markets change the picture?

Every strategy class works against equities (SPY, QQQ, individual tickers)
with zero code changes to the strategy logic itself — `YFinanceDataSource`
already resolves any ticker not in its futures symbol map, and the position
sizing functions (`volatility_scaled_size()`, `clenow_position_size()`) take
a `dollars_per_point` parameter that's just `1.0` for a stock (1 share
moving $1 = $1 P&L, no futures contract multiplier). Six `_spy` variants
exist in `engine/backtest/param_grids.py` reusing the exact same strategy
classes:

```bash
uv run python main.py --strategy ma_crossover_spy
uv run python walk_forward.py --strategy ma_crossover_spy
uv run python walk_forward.py --strategy trend_following_spy
```

### Results

| Strategy | Window | Folds | Aggregate OOS return | Folds profitable |
|---|---|---|---|---|
| `ma_crossover_spy` | daily, ~10yr | 14 | **+33.1%** | **57.1%** |
| `mean_reversion_spy` (ADX-gated) | daily, ~10yr | 14 | -4.5% | 7.1% |
| `trend_following_spy` | 1h, ~2yr (yfinance cap) | 21 | -3.8% | 19.0% |
| `donchian_breakout_spy` (Clenow rules) | 1h, ~2yr | 21 | -4.2% | 33.3% |
| `regime_switching_spy` | 1h, ~2yr | 21 | -1.6% | 23.8% |

**`ma_crossover_spy` is the strongest result anywhere in this project** —
14 out-of-sample folds spanning 2016-2026 (including the COVID crash and
recovery), +33.1% compounded, better-than-coin-flip fold win rate. Worth
being precise about what this does and doesn't show:

- **The parameters still don't converge** (`fast_period`/`slow_period`
  bounce across the whole grid fold to fold, same pattern as every other
  result in this project) — this is a meaningfully weaker signal than a
  result where the winning parameters stabilize. It suggests the return is
  driven substantially by SPY's long-term upward drift (buy-and-hold SPY
  over this window is also strongly positive) rather than a robust,
  parameter-stable timing edge. A crossover strategy tends to "work" in a
  persistently trending instrument mostly by staying long most of the time
  — which is a real, usable property, but it's closer to "a taxed, choppier
  version of buy-and-hold" than "a discovered edge."
- **The 1h equity strategies (trend_following, donchian_breakout,
  regime_switching) show the same weak/negative results as their futures
  counterparts.** Switching markets didn't fix what walk-forward has
  consistently found: these specific entry/exit rules don't have a
  demonstrated edge, independent of which instrument they're tested on.
- **A real practical friction surfaced testing equities**: `backtesting.py`
  logged "insufficient margin" warnings running `trend_following` against
  SPY, because share-based position sizing at SPY's ~$770 price ties up
  much more capital per position (~$18K of a $50K account for a single
  trade) than the equivalent futures contract exposure. This is a genuine
  capital-efficiency difference between the two markets, not a bug — worth
  weighing if position count/diversification matters to you.
- **SPY/QQQ also don't have the futures PDT exemption** — accounts under
  $25K face FINRA's Pattern Day Trader restrictions (max 3 day-trades per 5
  rolling business days) if trading stocks. This was the "advantage of
  futures" flagged back at the start of this project, and it fully applies
  here: a small equities account can't day-trade freely the way a futures
  account can.

**Bottom line**: switching to equities didn't produce "always profitable" —
nothing could — but it did surface the single most promising result in
this project so far (`ma_crossover_spy`), while confirming (via the
intraday strategies) that market choice alone doesn't fix a strategy
without a demonstrated edge.

## Risk engine — the non-negotiable part

`engine/risk/engine.py` hardcodes:
- Max daily loss
- Max risk per trade
- Max trades per day
- Max consecutive losses (kill switch)
- Max position size
- Max trailing drawdown (kill switch, prop-firm style)
- Profit target (auto-stop, prop-firm style)

These are dataclass fields, not runtime config — deliberately. To change a
limit you edit `RiskLimits` in code and restart the process. No strategy,
script, or agent has a code path to raise its own risk ceiling while
running. If you build the AI research agent in step 6 above, do not give it
write access to this file's logic — only read access to results.

## Known gaps / honest caveats

- **yfinance data quality**: continuous futures contracts splice contract
  rolls together; there can be small gaps/jumps at roll dates. Fine for
  prototyping, not for final validation.
- **Commission/slippage model** is a simplified flat percentage
  (`engine/backtest/runner.py`), not a precise per-contract commission +
  liquidity-aware slippage model. Good enough to avoid a "beautiful fake
  strategy," not perfectly realistic.
- **Tradovate historical bars endpoint** is not implemented — only order
  placement/account endpoints are. Backtesting uses yfinance/CSV; live
  paper-trading would need the market-data websocket wired up (step 5).
