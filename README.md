# VRP Strategy

A systematic short-volatility strategy on SPX/VIX that sells the variance risk premium (VRP),
sized by an HMM-detected market regime and a rolling percentile rank of the premium, scaled to
a target volatility. Built end-to-end in Jupyter notebooks: data pipeline, signal construction,
regime detection, position sizing, backtest, and three notebooks stress-testing the result
(permutation testing, an alternative sizing scheme, and transaction costs).

## The idea

Implied volatility (priced into VIX) tends to run higher than the volatility that actually
shows up afterward. That gap — the variance risk premium — is compensation option sellers earn
for taking on tail risk. It's usually small and positive, and goes sharply negative in a crash.
The strategy's job is to collect the premium on ordinary days and get small or flat before it
turns into a large loss.

## Pipeline

| Notebook | What it does |
|---|---|
| [`01_data_pipeline.ipynb`](notebooks/01_data_pipeline.ipynb) | Merges SPX, VIX, VIX3M into one dataset |
| [`02_signal_construction.ipynb`](notebooks/02_signal_construction.ipynb) | Builds VRP, term structure slope, realized vol |
| [`03_regime_hmm.ipynb`](notebooks/03_regime_hmm.ipynb) | 3-state Gaussian HMM regime classifier (calm / stressed / crisis), trained out-of-sample |
| [`04_position_sizing.ipynb`](notebooks/04_position_sizing.ipynb) | Position size = regime multiplier × rolling VRP percentile rank |
| [`05_backtest.ipynb`](notebooks/05_backtest.ipynb) | P&L proxy, vol-targeting overlay, performance metrics vs. SPX buy-and-hold |
| [`06_permutation_testing.ipynb`](notebooks/06_permutation_testing.ipynb) | Signal-shuffle permutation test: is the timing signal adding real skill? |
| [`07_position_sizing_comparison.ipynb`](notebooks/07_position_sizing_comparison.ipynb) | Regime-based sizing vs. a rolling half-Kelly fraction |
| [`08_transaction_costs.ipynb`](notebooks/08_transaction_costs.ipynb) | Turnover-based cost model, decomposed into signal vs. leverage-driven turnover |
| [`09_regime_multiplier_sweep.ipynb`](notebooks/09_regime_multiplier_sweep.ipynb) | Grid sweep on the regime multipliers, checked for stability across sub-periods |

`src/vrp_strategy/` holds the P&L and vol-targeting logic shared across notebooks 05–09, so
the math is defined once and reused rather than copy-pasted per notebook.

## Method, briefly

- **Signal**: `vrp = (VIX/100)^2 - realized_vol_30d^2` (implied minus realized variance), plus
  term structure slope `(VIX3M - VIX) / VIX`.
- **Regime**: a 3-state Gaussian HMM fit on `[vrp_vol, slope, realized_vol_30d]`, trained on
  data before 2016 and applied out-of-sample after that.
- **Sizing**: `position = regime_multiplier[state] × vrp_percentile_rank_252d`, clipped to
  [0, 1]. Regime multiplier: `calm → 1.0, stressed → 0.25, crisis → 0.0`.
- **Vol targeting**: the resulting return stream is scaled to a 10% annualized volatility
  target using trailing realized vol, capped at 100x leverage.
- **P&L**: there's no real options data, so P&L is proxied as
  `position(t-1) × (implied_daily_variance(t-1) - realized_daily_variance(t))` — roughly what
  a short variance swap earns, not a tradeable instrument as-is.

## Headline result (out-of-sample, 2016–2026, before costs)

| | Strategy | SPX buy & hold |
|---|---|---|
| Annualized return | 31.9% | 12.7% |
| Annualized vol | 10.4% | 18.0% |
| Sharpe | 3.08 | 0.70 |
| Max drawdown | -19.6% | -36.1% |

## What the stress-testing notebooks found

- **Permutation test**: shuffling the position-sizing signal across dates 2,000 times rejects
  the null hypothesis that the timing adds no skill (p = 0.004) — but the null distribution's
  mean Sharpe is 1.75, not 0, since even random short-vol exposure inherits VRP's structurally
  positive average over this sample. A meaningful chunk of the headline Sharpe is a base rate,
  not timing skill.
- **Transaction costs**: turnover splits into ~15x/year from the timing signal itself (cheap)
  and ~71x/year from the daily vol-target releveraging (expensive). At just 3bps on total
  turnover, annualized return goes from +31.9% to **-7.6%**. The strategy's cost sensitivity is
  almost entirely a leverage-cycling problem, not a signal-timing problem.
- **Parameter sweep**: the current regime multipliers sit at the 75th–99th percentile of a
  132-point grid across four different sub-periods — never the single best choice in any one
  window, but consistently good across all of them, which is a reasonable signature of a
  robust choice rather than one fit to a single historical path.

## Known limitations

- The P&L proxy isn't real options or variance-swap P&L — no skew, gamma, or roll cost.
- The regime HMM is trained on one decade (2006–2015) that includes 2008; "crisis" as the
  model understands it leans heavily on one event.
- The sizing parameters (regime multipliers, vol target, leverage cap) were originally chosen
  by observing backtest performance, which invites overfitting on a ~10-year window — the
  permutation test and parameter sweep notebooks exist specifically to pressure-test that risk.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Notebooks are numbered and meant to run in order — each one reads the previous notebook's
output from `data/processed/`.
