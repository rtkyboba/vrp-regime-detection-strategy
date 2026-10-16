# 03 - Does a GRU/LSTM add anything to a short-variance signal?

**Verdict in one line: no.** A small, carefully regularised GRU ensemble, trained walk-forward with
purging and an embargo, is statistically indistinguishable from "always short variance" and from
simple baselines. On the 2016+ test its 21-day Spearman IC is -0.08 (95% CI -0.23 to +0.07), its
probabilities are worse calibrated than the base rate (Brier skill -0.11), and its SVXY P&L has the
same Sharpe as constant exposure (0.33 vs 0.34, paired difference CI -0.08 to +0.05). Ridge did
nominally better on most economic measures, but none of its edge over always-short is
distinguishable from zero either (see "Caveats").

Reproduce everything with `python scripts/rnn_signal.py dev|select|final|report`
(about 40 min of CPU, mostly `dev`). Tests: `pytest tests/test_rnn_signal.py` (22 tests).
All outputs are in `results/rnn/`.

## 1. Question

Given only information available at the close of day *t*, can a recurrent network forecast whether
a short-variance position will be profitable well enough to (a) rank-forecast the payoff and
(b) improve a real, tradable short-vol P&L (SVXY) after 2 bps of cost, relative to
HAR-RV, ridge, gradient boosting, the legacy VRP-rank rule and just staying short?

## 2. Method

**Data and features** (`rnn_signal.build_features`, 20 causal daily features; all rolling windows /
diffs, forward-fill only). log VIX; VIX3M/VIX and VIX6M/VIX3M log ratios; log VVIX and VVIX/VIX;
SKEW; realised vol (zero-mean, annualised, log) over 5/10/21/63 days; two VRP measures
(log VIX - log RV21, and VIX^2 - RV21 in variance units); 1/5/21-day SPX log returns; drawdown from the
63-day high; 1- and 5-day change in log VIX; VIX relative to its 126-day mean; log |return|
(HAR "daily" term). Features start 2008-01 (VIX6M's first date). VIX9D (from 2011-01) was *not* used:
it would leave under 5 years of pre-2016 training history. SPY is redundant with SPX and is only
used as a buy-and-hold benchmark.

**Targets.**
(a) 21-day held-to-maturity short-variance payoff: enter at *t* at strike VIX_t^2 x 21/252, pay the
sum of squared SPX log returns over *t+1..t+21*. Regression target = payoff / strike = 1 - RV/IV
(training target winsorised to [-3, 1]; evaluation uses the unclipped value). Classification target
= sign. Base rate of a positive payoff: 82-84%.
(b) 5-day forward SVXY return sign (labelled from 2011-10 only).

**Walk-forward with purging and embargo** (`purged_split`, tested). Refit every 63 trading days,
expanding window, first prediction 2011-05-31. For a refit at row *t0* a training sample *i* has label
window (i, i+h]; it is admitted only if `i + h + embargo < t0` (embargo = h: 21 days for the payoff,
5 for SVXY). Early stopping uses the last 252 admissible samples as a validation tail, separated from
the inner-train samples by the same h + embargo gap. Standardisation, target scaling, ridge penalty and
early stopping use training rows only.

**Models.** RNN: one GRU or LSTM layer, 16-64 units, dropout on the last hidden state, Adam with weight
decay, joint Huber (payoff) + BCE (sign) loss, max 40 epochs, patience 6, best-validation epoch kept,
5-seed ensemble (mean of member probabilities / forecasts; a seed is a pure function of member and
refit row). Baselines on identical dates and targets: HAR-RV style OLS (RV 1d/5d/21d/63d + log VIX);
ridge on [x_t, 5d mean, 21d mean] of all features (alpha chosen on the validation tail);
`HistGradientBoosting` (classifier and regressor) on the same flattened features; the legacy rule
(trailing-252d percentile rank of VRP as the position); always short.

**Signal in [-1, 1].** Classifier heads: 2P - 1. Regression heads: tanh(forecast / training std of the
clipped target). Legacy: 2 rank - 1. Primary signal = classifier head for RNN and HistGB, regression
head for ridge and HAR; the RNN regression head and HistGB regression head are reported as separate
rows.

**Economic evaluation.** SVXY exposure = (signal + 1) / 2 (so exposure = P for classifier heads and
recovers the legacy position for the rank rule), decided at close *t*, earns the SVXY return of *t+1*,
rest in cash at the 13-week bill rate; 2 bps charged per unit of exposure traded; Sharpe is on
excess-of-risk-free returns; max drawdown is on total net return. Real `svxy` returns (-1x until
Feb-2018, -0.5x after). Two sensitivity maps are also run: long/flat `clip(s, 0, 1)` and `1[s > 0]`.
Test window 2016-01-04 to 2026-05-22 (2,612 days; signals end 2026-05-22 because SPX/VIX data do).

**Statistics.** 95% moving-block bootstrap (block 42 days = 2x the longest horizon, 1000 draws).
Differences (IC, Sharpe) are paired on the same resampled blocks. With overlapping 21-day labels
there are only about 120 effectively independent test observations.

**Model selection** used only data before 2016: configurations were scored on a copy of the dataset
truncated at 2016-01-01 whose labels reaching past the cut were set to NaN
(`rnn_signal.truncate`). The truncated run reproduces the full run's pre-2016 predictions exactly
(tested for ridge; verified for the RNN to 6e-8).

## 3. Results

### 3.1 Pre-2016 walk-forward (selection set, 1,135 dates, 2011-05 to 2015-11)

Criterion: Spearman IC of the classifier signal vs the 21-day scaled payoff.

| RNN config (cell, hidden, seq, dropout, wd) | dev IC (cls) | dev IC (reg head) | seed-IC mean +/- sd |
|---|---|---|---|
| **gru, 16, 20, 0.2, 1e-2 (frozen)** | **0.117** | 0.026 | 0.121 +/- 0.034 |
| gru, 64, 20, 0.4, 1e-2 | 0.112 | 0.057 | 0.110 +/- 0.053 |
| gru, 32, 20, 0.4, 1e-2 | 0.086 | 0.029 | 0.088 +/- 0.020 |
| gru, 16, 20, 0.2, 1e-3 | 0.085 | 0.009 | 0.080 +/- 0.041 |
| gru, 16, 40, 0.2, 1e-3 | 0.079 | -0.007 | 0.074 +/- 0.046 |
| gru, 32, 20 / 32, 40, 0.2, 1e-3 | 0.057 / 0.054 | 0.010 / 0.021 | |
| lstm, 64, 40, 0.4, 1e-2 | 0.035 | -0.012 | 0.040 +/- 0.051 |
| lstm, 16/32, 20/40, 0.2, 1e-3 (4 configs) | -0.008 to -0.054 | -0.030 to 0.058 | about 0 +/- 0.1 |
| HistGB depth 3 / depth 2 | 0.088 / 0.033 | -0.126 / -0.176 | |
| Ridge (reg) / HAR (reg) | n/a | -0.176 / -0.116 | |

GRUs beat LSTMs and higher weight decay helped; every dev IC is small and a dev IC standard error of
about 0.14 (about 50 independent 21-day blocks) means the ranking is mostly noise. The winning dev IC
of 0.117 did not survive: see 3.2. Ridge and HAR have *negative* dev IC (they extrapolate the
2008-2011 relation between VIX level and payoff into a calm 2012-2015) and positive test IC, i.e. the
linear relationship is itself unstable across regimes.

### 3.2 Forecast quality, 21-day payoff, 2016+ test (2,591 labelled dates)

Always-short hit rate (base rate of a positive payoff): 0.838.

| Model | Spearman IC [95% CI] | Hit rate [95% CI] | % days signal < 0 |
|---|---|---|---|
| GRU ensemble, cls head (frozen) | **-0.080** [-0.232, 0.074] | 0.838 [0.783, 0.891] | 0% |
| GRU ensemble, reg head | -0.058 [-0.213, 0.077] | 0.838 [0.783, 0.891] | 0% |
| Ridge | 0.085 [-0.067, 0.225] | 0.814 [0.759, 0.867] | 5.9% |
| HAR-RV OLS | 0.030 [-0.116, 0.161] | 0.836 [0.781, 0.889] | 0.2% |
| HistGB (cls) | -0.109 [-0.242, 0.031] | 0.811 [0.753, 0.864] | 4.0% |
| HistGB (reg) | -0.034 [-0.148, 0.089] | 0.766 [0.711, 0.822] | 10.1% |
| Legacy VRP-rank | 0.010 [-0.106, 0.130] | 0.491 [0.431, 0.552] | 48.4% |
| Always short | n/a | 0.838 [0.783, 0.891] | 0% |

No model has an IC interval that excludes zero. The RNN's hit rate equals the always-short hit rate
*because it never turns bearish* (signal < 0 on 0% of test days; 10% of dev days): it is a
position-scaling device around "short", not a timing signal. Paired IC differences (`ic_differences_test.csv`):
RNN minus ridge -0.17 [-0.33, +0.01], RNN minus HAR -0.11 [-0.32, +0.11], RNN minus HistGB +0.03
[-0.16, +0.18]. The RNN is not better than any baseline; its point estimate is worse than ridge and the
difference only just straddles zero.

**Calibration** (`calibration.png`, `calibration_rnn_test.csv`). Brier skill vs the base rate: RNN
-0.111, HistGB -0.224 (negative = worse than predicting 83.8% every day). Mean predicted P across test
deciles runs from 0.61 to 0.92 while the realised frequency is flat to inverted (0.87, 0.86, 0.85,
0.83, 0.81, 0.80, 0.89, 0.87, 0.88, 0.74): the most confident decile is the least reliable (0.74
realised).

**Seed stability.** Member-level IC on the test set: -0.027, -0.131, -0.087, -0.061, -0.030 (sd 0.04);
mean pairwise correlation between members' signals 0.68; member net Sharpes 0.31 to 0.37 (ensemble
0.33). Seeds agree on "no skill"; the seed noise is smaller than the CI half-width but is a visible
fraction of any claimed effect.

### 3.3 Economics on real SVXY, 2016+, exposure = (s+1)/2, 2 bps, excess of risk-free

| Strategy | Sharpe gross | Sharpe net [95% CI] | Max DD | Worst day | Avg exposure | Turnover /yr |
|---|---|---|---|---|---|---|
| GRU ensemble (cls) | 0.336 | 0.334 [-0.25, 1.22] | -84.8% | -63.8% | 0.78 | 2.5 |
| GRU ensemble (reg) | 0.353 | 0.353 [-0.25, 1.23] | -81.1% | -58.4% | 0.73 | 1.7 |
| Ridge | 0.402 | 0.399 [-0.16, 1.16] | -67.6% | -39.8% | 0.69 | 4.8 |
| HAR-RV OLS | 0.338 | 0.334 [-0.27, 1.20] | -77.2% | -53.8% | 0.67 | 6.4 |
| HistGB (cls) | 0.262 | 0.259 [-0.32, 1.21] | -92.8% | -78.8% | 0.84 | 6.8 |
| HistGB (reg) | 0.233 | 0.229 [-0.39, 1.19] | -86.3% | -64.8% | 0.68 | 7.2 |
| Legacy VRP-rank | 0.059 | 0.049 [-0.49, 0.96] | -94.5% | -83.0% | 0.51 | 20.2 |
| **Always short (constant 100% SVXY)** | 0.344 | **0.344** [-0.25, 1.23] | -95.2% | -83.0% | 1.00 | 0 |
| Buy and hold SPY (reference) | 0.757 | 0.757 | -33.7% | -10.9% | 1.00 | 0 |

Constant exposure at each model's own average exposure has Sharpe 0.344 and max drawdowns of -86%
(at 0.78, the RNN's average) down to -80% (at 0.67), so the RNN's -84.8% drawdown is what simply holding
about 0.78 of SVXY gives; the drawdown reduction vs always-short is just deleveraging.
Costs are immaterial at these turnovers (gross to net: 0.002-0.01), except for the legacy rule.

Paired net-Sharpe differences (block bootstrap; `sharpe_differences_test.csv`):

| a vs b | Sharpe a - b [95% CI] |
|---|---|
| RNN vs always short | -0.009 [-0.077, 0.049] |
| RNN vs ridge | -0.065 [-0.175, 0.107] |
| RNN vs HAR | 0.000 [-0.079, 0.084] |
| RNN vs HistGB | +0.075 [-0.064, 0.162] |
| RNN vs legacy rule | +0.286 [0.046, 0.480] |
| Ridge vs always short | +0.056 [-0.141, 0.169] |
| Legacy vs always short | -0.295 [-0.486, -0.079] |

The only distinguishable gaps are against the legacy rule, which is *worse* than doing nothing. Under
the other exposure maps (`sharpe_differences_other_maps_test.csv`): long/flat, no model differs from
always-short (ridge +0.10 [-0.49, 0.57], RNN -0.04 [-0.21, 0.10]); 1[s>0]: the RNN is identical to
always-short (never bearish), ridge reaches net Sharpe 0.59 (+0.25 vs always-short, CI [-0.14, 0.58],
max DD -54.5%), HistGB is significantly *worse* (-0.12 [-0.28, -0.01]) and the legacy rule is worse
(-0.44 [-0.74, -0.19]).

**Held-to-maturity payoff view** (variance-swap economics without SVXY roll mechanics,
`variance_payoff_economics_test.csv`, payoff in vol-point^2 per entry): mean payoff of always-short
5.96; exposure-weighted 5.18 (RNN), 5.34 (ridge), 4.10 (HAR), 5.05 (HistGB), 2.29 (legacy). The
"timing covariance" cov(exposure, payoff), the part of the payoff attributable to *when* the
model is short, has a CI containing zero for every model: RNN 0.52 [-0.30, 1.82], ridge 1.24
[-0.31, 3.90], legacy -0.77 [-4.11, 1.26]. The worst exposure-weighted single-entry payoff is -455
for the RNN vs -680 for always short, which is about the 0.78 average exposure applied to the
same crash, not avoidance of it.

### 3.4 5-day SVXY return sign (second target, same frozen architecture, test 2016+)

Always-long hit rate 0.615.

| Model | IC [95% CI] | Hit rate | Net Sharpe, exposure from signal |
|---|---|---|---|
| GRU cls | -0.022 [-0.097, 0.046] | 0.613 | 0.335 |
| GRU reg | 0.077 [0.013, 0.162] | 0.600 | 0.331 |
| Ridge | 0.055 [-0.008, 0.128] | 0.588 | 0.290 |
| HAR | 0.009 [-0.049, 0.081] | 0.595 | 0.248 |
| HistGB cls / reg | 0.073 [-0.014, 0.158] / 0.079 [0.003, 0.163] | 0.573 / 0.562 | 0.314 / 0.339 |
| Legacy | 0.010 [-0.056, 0.078] | 0.473 | 0.049 |
| Always long | n/a | 0.615 | 0.344 |

Two rows (GRU reg head, HistGB reg) have IC intervals that just exclude zero. Out of about 14 IC
tests (7 signals x 2 targets) one expects roughly one such result by chance at 5%, the effect (IC 0.08)
explains under 1% of return variance, the sign hit rates are *below* always-long, and none of it reaches the P&L (every net
Sharpe is at or below always-long 0.344). I do not count it as a finding.

### 3.5 Behaviour around the stress episodes (`events.png`, `events.csv`)

Strategy = exposure (s+1)/2 in SVXY, 2 bps; always-short equals SVXY.

| Episode | Signal at last close before window (RNN / ridge / HAR / HistGB / legacy) | Mean RNN exposure | RNN return | Ridge return | SVXY |
|---|---|---|---|---|---|
| 2018-02 (Jan 2 - Mar 29) | 0.32 / 0.58 / 0.43 / 0.99 / -0.57 | 0.76 | -76.7% | -58.2% | -90.8% |
| 2020-03 (Feb 3 - Apr 30) | 0.22 / -0.31 / 0.23 / 0.63 / 0.88 | 0.66 | -35.3% | -22.3% | -46.5% |
| 2022 (full year) | 0.85 / 0.74 / 0.45 / 0.94 / -0.99 | 0.84 | -5.5% | -7.3% | -4.7% |

The RNN signal does fall during the events (to about 0.1 in the mid-March-2020 trough, see the
figure), but it falls *with* the move, not ahead of it; it stays 50-75% long into both crashes
(minimum exposure over the windows 0.66 in 2018, 0.53 in 2020). It delivers only the loss reduction you
get from running less leverage. Ridge was already negative at the end of January 2020 (signal -0.31,
exposure down to 0.12 in the window), which probably accounts for much of its better numbers; in
Feb-2018 it was still 44-70% exposed.
The legacy rule was bearish before 2018 and bullish before COVID, and was not protective in either.

## 4. Honest verdict

1. The RNN adds nothing measurable over ridge, HAR, gradient boosting or always-short. Its point IC
   is negative, its probabilities are miscalibrated, its Sharpe equals constant exposure, and every CI
   comfortably contains zero.
2. "Ridge or HAR is better than the RNN" is also not established: the IC and Sharpe differences have
   CIs that straddle zero (RNN - ridge IC -0.17 [-0.33, 0.01]).
3. The honest summary of what any of these signals do on this data: they scale a short-vol position
   between 0.5x and 1x. That reduces the drawdown by about as much as holding a constant 0.7x would.
4. What drives short-vol P&L over 2016-2026 is a handful of tail events (Feb 2018, Mar 2020), each of
   which the 21-day forecast was long into. No model here predicts them; a test set containing two such
   events cannot establish that one does.
5. The only thing in the whole study that looks like an edge is ridge's behaviour in March 2020 and its
   1[s>0] Sharpe of 0.59, and a Sharpe difference with CI [-0.14, 0.58] after seeing about 32 Sharpe
   numbers is not evidence.

## 5. Caveats

- **Single path, few independent events.** About 120 independent 21-day outcomes, dominated by
  2018-02 and 2020-03. SVXY's drawdown is -95% over the window; CIs on Sharpe span -0.25 to 1.2 for
  *every* strategy including always-short.
- **SVXY mechanics.** Real SVXY changed from -1x to -0.5x in Feb 2018; the pre-2018 "-1x" regime had
  a catastrophic day (-83%) that the 21-day payoff target does not see (the payoff is held-to-maturity
  and is a different object from a daily-rebalanced futures ETN). The payoff target and the SVXY P&L
  therefore measure related but different things; I report both.
- **Strike approximation.** VIX^2 is a model-free proxy for the 21-day variance strike, and SPX
  close-to-close squared returns omit overnight/intraday composition; the 21-day window is 21 trading
  days vs 30 calendar days for VIX.
- **Dev/selection weakness.** Roughly 50 independent dev observations; a dev IC difference of 0.03 is
  noise. The frozen config is the dev maximum (0.117) and its test IC is -0.08: textbook selection
  decay. Ridge and HAR flip from negative dev IC to positive test IC; the data are non-stationary.
- **Sample for RNN.** At the first refit there are 758 labelled samples (about 3 years, of which
  2008-2009 is the only stress period), growing to about 4,500. GRU units trained on a few hundred
  nearly independent monthly outcomes are not going to learn a regime-dependent payoff.
- **Training sets are expanding**, so test-period models do see post-2015 data (as a live model
  would); only the architecture/hyperparameters were frozen from pre-2016.
- **Winsorising** the training target at [-3, 1] gives the net less to fit in the tail; it is a
  choice made a priori (not tuned) and is not applied at evaluation.
- Linear baselines use regression heads while RNN/HistGB use classifier heads for the primary
  signal; the RNN and HistGB regression heads are in the tables for a like-for-like comparison.
- The ridge/HAR signal scale (tanh of forecast / training std) is a convention, not calibrated.
- Dates: the signal file ends 2026-05-22 because `signals.csv`, SPX and VIX end there; VIX6M and VVIX
  extend further but cannot be used without SPX.

## 6. Trial count (docs/EVAL_PROTOCOL.md rule 3)

- **Dev (pre-2016 only):** 12 RNN configurations (2 cells x 2 hidden x 2 seq len at wd 1e-3, plus
  4 variants with higher weight decay/dropout/width), 2 HistGB configurations, ridge, HAR = **16
  configurations**, each also scored on its regression head. One additional unlogged dev-only timing
  run of the default GRU (hidden 16, wd 1e-3, started at a later refit) = 17 dev evaluations of the
  RNN-or-baseline family.
- **Test (2016+, computed once, after freezing):** 8 signal variants x 3 exposure maps on the 21d target + 8 on the 5d target
  (linear, long/flat, sign) = 32 Sharpe numbers seen; 14 IC estimates; no changes were made after
  seeing them. Trial-count deflation therefore applies to anything one wants to claim from them: with
  32 looks and the best net Sharpe of 0.59 (ridge, 1[s>0]) the deflated value is not distinguishable from
  the always-short Sharpe of 0.34.
- Fixed a priori, not tuned: refit step 63, validation tail 252, early-stopping patience 6, lr 2e-3,
  batch 128, 40 max epochs, 5 seeds, embargo = horizon, block length 42.
- No peeking variants: no model or hyperparameter was changed after seeing 2016+ results. (The
  report stage was re-run twice only to add diagnostics and fix a plot/dev-mask.)

## 7. What an interviewer might push on

1. **"Your network never says 'don't be short'. Isn't that just a failure of the loss?"** Partly. A
   classifier trained on an 83% positive class will output P > 0.5 nearly everywhere unless it truly finds
   discriminating information, and in 2008-2015 (what it trained on) it did not. That is the finding,
   not a bug: the sign map shows the RNN is identical to always-short.
2. **"Did you tune on the test set?"** No: selection used a truncated dataset (labels reaching past
   2016-01-01 NaN'd), the frozen config is reported with the dev max and its test decay, and the
   truncated run reproduces the full run pre-2016 exactly.
3. **"Why not a bigger model / more features / more epochs?"** The sample is ~120 independent monthly
   outcomes with two crashes; capacity is not the constraint. Larger GRUs (64 units) had no better dev IC.
   I could add VIX9D or SPY option-implied features, but the signal-to-noise problem is the target, not
   the input set.
4. **"Why the embargo of 21 days and a 42-day block?"** Labels overlap for 21 days, so training samples
   within 21 days of the prediction block share outcomes with it; the embargo doubles that gap. Bootstrap
   block of 2x horizon preserves the overlap-induced autocorrelation of the test statistic.
5. **"Multiple testing?"** 16 dev configs and 32 test Sharpes are listed above. The 5-day regression-head
   ICs whose CI barely excludes zero are 2 of 14 and have no P&L.
6. **"Is the 'always short' benchmark fair?"** In a window that includes only two tail events and one
   -0.5x leverage change, always-short Sharpe (0.34) is mostly luck; that is why I compare to it and
   to constant exposure at matched average exposure, and why I report drawdown and worst day.
7. **"Why is ridge better on some economics?"** Apparently one episode (March 2020, signal negative
   before the crash, exposure down to 0.12) and a different map (sign). Its IC CI [-0.07, 0.23] includes zero,
   its dev IC was -0.18, and the Sharpe difference vs always-short is [-0.14, 0.58].
8. **"Held-to-maturity payoff isn't tradable with SVXY."** Agreed, which is why both targets and P&L
   bases are shown; the tranche Sharpe in `variance_payoff_economics_test.csv` is an approximation
   (overlapping entries), not a P&L.
9. **"Is the 63-day refit stale?"** A faster refit uses more data labelled with less embargo room; I did
   not test it (it would add a trial and cannot fix a lack of signal).
