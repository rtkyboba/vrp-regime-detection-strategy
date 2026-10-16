# 06 - Combining the regime signal and the RNN signal

Code: `src/vrp_strategy/combiner.py`, `scripts/combine_signals.py`, `tests/test_combiner.py`.
Outputs: `results/combine/summary.csv`, `results/combine/equity.png`, `results/combine/ic_weights_hmm3_core.csv`.

## 1. Question
Can two weak signals be combined, weighted by their signal strength, into a position-sizing rule that
beats simply holding the instrument? Inputs: the causal walk-forward regime `stress_score`
(report 02) and the frozen walk-forward RNN forecast (report 03). Instrument: real SVXY
(2012-01 on, -1x until Feb 2018 then -0.5x), exposure in [0, 1], no leverage, 2 bps on exposure
change, excess of T-bills.

## 2. Method
Both signals are mapped to "how much short-vol am I allowed", a in [0, 1]: regime `1 - stress`,
RNN `(s + 1) / 2`. Rules, fixed before running: `mean`, `product` (AND-gate), `veto` (RNN
sizes, regime flag can only cut to zero), and `ic` (weights proportional to each component's
trailing 504-day rank IC against the held-to-maturity payoff, using only payoffs that have
matured by t; equal weights if no IC is positive). Two regime sources (`hmm3_core`,
`jump3_lam20`), plus `rnn_only` and `ridge_only`: **12 variants = 12 trials**.

**The control that matters:** every variant is compared with a *constant* exposure equal to its own
average exposure. A rule that simply holds less than 100% looks better on drawdown, but a
constant exposure does the same thing; only the difference is timing skill. Paired block-bootstrap
test (`stats_tests.bootstrap_sharpe_diff`) on the 2016+ test period.

Measurement note, stated plainly: my first run scored information content against the payoff in
vol points, which scales with the VIX level and mechanically rewards high-VIX days (it gave a
significantly negative IC for the regime signal). I switched the IC target to the scale-free
`1 - RV^2 / VIX^2` (the regime agent's definition) after seeing that output. This changes a
diagnostic only; the trading results were identical in both runs.

## 3. Results (2016+, net 2 bps)
| variant | avg exposure | Sharpe | constant same size | diff vs constant [95% CI] | p | IC vs 21d payoff [95% CI] |
|---|---|---|---|---|---|---|
| hmm3_core, regime only | 0.68 | 0.29 | 0.37 | -0.08 [-0.31, 0.25] | 0.54 | 0.01 [-0.13, 0.15] |
| hmm3_core, mean | 0.72 | 0.32 | 0.37 | -0.05 [-0.17, 0.15] | 0.52 | 0.01 [-0.13, 0.14] |
| hmm3_core, product | 0.52 | 0.29 | 0.37 | -0.08 [-0.31, 0.29] | 0.56 | 0.01 [-0.12, 0.14] |
| hmm3_core, veto | 0.52 | 0.29 | 0.37 | -0.08 [-0.31, 0.29] | 0.57 | 0.02 [-0.12, 0.15] |
| hmm3_core, IC-weighted | 0.77 | 0.28 | 0.37 | -0.08 [-0.23, 0.14] | 0.34 | 0.04 [-0.11, 0.17] |
| jump3_lam20, regime only | 0.93 | 0.46 | 0.37 | +0.10 [-0.11, 0.32] | 0.36 | 0.07 [-0.06, 0.19] |
| jump3_lam20, mean | 0.84 | 0.41 | 0.37 | +0.04 [-0.09, 0.17] | 0.53 | -0.07 [-0.19, 0.06] |
| jump3_lam20, product | 0.73 | 0.45 | 0.37 | +0.09 [-0.13, 0.32] | 0.42 | -0.07 [-0.19, 0.06] |
| jump3_lam20, veto | 0.73 | 0.39 | 0.37 | +0.02 [-0.16, 0.27] | 0.83 | -0.07 [-0.19, 0.06] |
| jump3_lam20, IC-weighted | 0.86 | 0.43 | 0.37 | +0.06 [-0.13, 0.28] | 0.56 | -0.14 [-0.27, -0.02] |
| RNN only | 0.75 | 0.33 | 0.37 | -0.04 [-0.12, 0.03] | 0.31 | -0.08 [-0.20, 0.05] |
| ridge only | 0.66 | 0.40 | 0.37 | +0.03 [-0.15, 0.16] | 0.72 | 0.09 [-0.05, 0.19] |

Always-short SVXY: Sharpe 0.37 in 2016+ (0.72 excluding February 2018). Max drawdowns of every
variant are -69% to -93%: none of the blends protects against the tail.

## 4. Interpretation
- **No combination beats a constant exposure of the same size.** All 12 differences are within
  [-0.08, +0.10] with p between 0.31 and 0.87. The blends do not add skill either.
- **No signal ranks the forward payoff.** 11 of 12 information coefficients have confidence
  intervals that include zero; the one that does not (`jump3_lam20|ic`, -0.14) has the wrong sign.
- **Why combining cannot help here:** combining reduces noise only when the components carry
  independent signal. Reports 02 and 03 showed neither component forecasts the short-variance
  payoff on 2016+ (the regime score predicts forward *realised vol* well, but so does raw VIX,
  and that does not translate into the payoff after the VIX has already priced it in).
  Averaging two zero-information signals gives a zero-information signal with lower exposure.
- The apparent gains for `jump3_lam20` (+0.10) are inside the noise and rest on the February 2018
  event (excluding it, every variant is at 0.66-0.78 vs 0.72 for always-short).
- IC-weighting did what it should mechanically (tilting toward whichever component had positive
  trailing IC) but with no stable winner there is nothing to tilt toward.

## 5. Caveats
- 12 variants is a small family but the family is not independent (all share the same inputs).
  Counting against the project-wide ~500 trials, nothing here would survive deflation anyway.
- One regime source's tail behaviour (Feb 2018) drives most of the sign differences.
- SVXY is a -1x then -0.5x VIX-futures ETF; a variance swap has different P&L (report 01).

## 6. What an interviewer might push on
- **"You combined HMM and RNN and it did not help. Why?"** Because the combination rule has no
  edge to amplify: each input has IC ~ 0 against the target, and I tested the combination against
  a constant exposure of the same average size, which isolates timing from de-leveraging.
- **"Why compare with a constant of the same exposure?"** A strategy that is 70% invested has a
  smaller drawdown than a 100% invested one by construction. Sharpe is scale-free, so the
  same-size constant has the baseline Sharpe; the paired test then asks only whether the timing
  added anything.
- **"Did you change the evaluation after seeing results?"** Once: the IC target (see Method). It
  affected the diagnostic only and is documented.
- **"Why weight by trailing IC?"** It is the simplest causal "signal strength" weight, and uses only
  matured payoffs (unit-tested). It is noisy when ICs are near zero, which is the finding.
