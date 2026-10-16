# 04 - Statistical validation: is the Sharpe real?

Reproduce: `python scripts/validate_sharpe.py` (about 30 s), tests: `pytest tests/test_stats_tests.py` (29 tests, about 17 s).
Toolkit: `src/vrp_strategy/stats_tests.py`. Outputs: `results/validation/`.
Entry point for other strategies: `from vrp_strategy.stats_tests import validate, validate_table`.

## 1. Question

The project's headline is a 2016-2026 Sharpe of 3.08 (Viterbi regime labels, proxy P&L, vol-targeted to 10%).
`docs/EVAL_PROTOCOL.md` already shows it is inflated by non-causal labels and by a P&L proxy that omits mark-to-market.
This report asks the narrower statistical questions a reviewer would ask next:

1. Given sampling noise, fat tails, autocorrelation and about 150 variants tried, is the Sharpe distinguishable from zero?
2. Does the regime logic add anything over a constant short position?
3. Does the "random-timing" evidence (notebook 06) hold up once the null is repaired?
4. Which of these tests would have caught the problem, and which could not?

## 2. Methods (all implemented in `stats_tests.py`, formulas in the docstrings)

| Tool | What it does | Reference |
|---|---|---|
| `sharpe_se` mertens / hac | SE of the Sharpe estimate allowing skew and kurtosis (Mertens), and additionally autocorrelation (delta-method influence function with a Newey-West long-run variance) | Mertens 2002, Opdyke 2007, Ledoit-Wolf 2008 |
| `lo_adjusted_sharpe` | Annualised Sharpe corrected for serial correlation, eta(q) = q / sqrt(q + 2 sum (q-k) rho_k) | Lo 2002 |
| `probabilistic_sharpe_ratio` | P(true SR > benchmark) | Bailey, Lopez de Prado 2012 |
| `expected_max_sharpe`, `deflated_sharpe_ratio` | PSR against the hurdle E[max of N null Sharpes] | Bailey, Lopez de Prado 2014 |
| `min_track_record_length` | Observations needed for significance at the observed SR, skew, kurtosis | Bailey, Lopez de Prado 2012 |
| `bootstrap_sharpe_ci`, `bootstrap_sharpe_diff` | Stationary block bootstrap (Politis-Romano), automatic block length (Politis-White / Patton-Politis-White); paired version for SR_a - SR_b | Politis, Romano 1994 |
| `probability_of_backtest_overfitting` | CSCV PBO over a matrix of candidate return streams (16 blocks, all 12,870 splits) | Bailey, Borwein, Lopez de Prado, Zhu 2017 |
| `reality_check` | White RC and Hansen SPA (lower, consistent, upper p-values), same stationary-bootstrap resamples for all rules | White 2000, Hansen 2005 |
| `timing_null_test` | Strategy Sharpe vs Sharpe of randomly timed copies of its own position: `circular_shift` (keeps autocorrelation and exposure), `block_permute` (keeps exposure level and regime persistence), `iid_shuffle` (the old, flawed null, kept for comparison) | this report |

What the unit tests establish (`tests/test_stats_tests.py`):

* Mertens SE equals the HAC SE at zero lags, reduces to Lo's (1 + SR^2/2)/T for normal data, and matches the empirical standard deviation of the Sharpe across simulated negatively skewed samples within 12% (the naive normal formula is more than 15% too small).
* Lo adjustment matches the analytic AR(1) value within 5%.
* PSR false-positive rate at the 95% level is 3-7%.
* `expected_max_sharpe` matches Monte Carlo maxima of N normals (N = 10, 100, 1000).
* DSR with N = 100 trials: naive PSR flags the best-of-100 noise strategy as significant more than 80% of the time, the DSR less than 9% (both with sample cross-trial variance and with the default variance).
* Bootstrap 90% interval coverage on iid and AR(1, 0.2) data lies in [0.82, 0.97].
* PBO is 0.5 on average for pure-noise candidates (a single dataset has a standard deviation of about 0.19 because all splits reuse the same data), below 0.05 when one candidate is truly better, above 0.6 when candidates flip sign between sample halves.
* Reality check / SPA reject at 5% in fewer than 12% of null simulations and detect a planted edge; shift and block-permutation nulls have size below 13% and power on a planted timing signal.

## 3. Trial count

Counted from `git log` and notebooks (`results/validation/trial_count.csv`):

| Source | Variants |
|---|---|
| Regime-multiplier grid, 12 x 11 (notebook 09; contains the hand-tuned 0.5 -> 0.25 change of commit 8e4a981) | 132 |
| HMM specification revisions (initial features, train/test split, "trained on vrp_vol instead") | 3 |
| Vol-target cap 150 -> 100 | 2 |
| Return-definition revision (unit exposure -> % capital at 10% vol) | 2 |
| Half-Kelly comparison (notebook 07) | 1 |
| Audit variants (causal hard, causal soft, no gate, constant, SVXY) | 5 |
| **Logged total** | **145** |

Used for deflation: **N = 150 (central)**, 132 (lower bound, the grid alone), 500 (upper, assumes two un-logged manual tries per logged variant; commit 8e4a981 literally says the multipliers were chosen by trying numbers). The DSR table below is also given for N up to 10,000. Caveat: notebook edits that were never committed are invisible to git, so 150 is a floor on honesty, not a ceiling. Cross-trial variance: the default is the null sampling variance 1/(T-1) (conservative for the highly correlated grid); the sensitivity table also uses the actual dispersion of the 132 grid Sharpes (SD 0.45 annualised, mean pairwise correlation 0.87).

## 4. Results (2016-01 to 2026-05, vol-targeted proxy P&L unless stated; SVXY rows are excess of T-bill)

### 4.1 Headline validation, N = 150

| Strategy | Sharpe | Lo-adj. | 95% block-boot CI | SE (Mertens / HAC) | DSR | Null-best hurdle | MinTRL (yrs) | Skew | Exc. kurt. |
|---|---|---|---|---|---|---|---|---|---|
| Original (Viterbi) | 3.08 | 2.20 | 1.65 to 5.26 | 0.71 / 0.78 | 0.999 | 0.85 | 1.4 | -10.7 | 202 |
| Causal, hard argmax | 2.74 | 2.00 | 1.60 to 4.49 | 0.66 / 0.73 | 0.998 | 0.85 | 1.5 | -10.5 | 191 |
| Causal, soft multiplier | 2.81 | 2.05 | 1.68 to 4.50 | 0.67 / 0.74 | 0.998 | 0.85 | 1.5 | -10.7 | 196 |
| Constant short, no signal | 2.89 | 1.69 | 1.10 to 5.63 | 0.56 / 0.78 | 1.000 | 0.85 | 1.0 | -7.3 | 84 |
| Causal positions x real SVXY | 0.02 | 0.02 | -0.55 to 0.70 | 0.31 / 0.33 | 0.005 | 0.83 | 9,653 | -5.6 | 86 |
| SVXY buy and hold | 0.37 | 0.35 | -0.28 to 1.19 | 0.34 / 0.38 | 0.091 | 0.81 | 24 | -8.0 | 179 |

The DSR of the proxy strategies stays above 0.97 even at N = 10,000 (`dsr_sensitivity.csv`). For the real-SVXY series it is below 0.1 at any N of 132 or more, and even a single trial gives a PSR of only 0.52 (strategy) / 0.86 (buy and hold).

### 4.2 Does the regime logic beat constant short? (paired block bootstrap of SR difference)

| Strategy vs benchmark | SR difference | 95% CI | p |
|---|---|---|---|
| Original vs constant short | +0.19 | -1.36 to 1.60 | 0.80 |
| Causal hard vs constant short | -0.15 | -1.68 to 1.19 | 0.84 |
| Causal soft vs constant short | -0.08 | -1.65 to 1.28 | 0.92 |
| Causal positions x SVXY vs SVXY buy and hold | -0.33 | -0.74 to 0.03 | 0.10 |

### 4.3 Timing-skill nulls (1000 draws each; strategy percentile in the null, one-sided p)

| Strategy (actual SR) | iid shuffle (old, flawed) | circular shift | block permute 21d | block permute 63d |
|---|---|---|---|---|
| Original (3.08) | null mean 1.75; pct 99.8%; p 0.003 | null mean 2.19; pct 90.8%; p 0.093 | null mean 2.04; pct 95.2%; p 0.049 | null mean 2.13; pct 92.6%; p 0.075 |
| Causal soft (2.81) | null mean 1.75; pct 99.3%; p 0.008 | null mean 2.19; pct 83.5%; p 0.166 | null mean 2.04; pct 91.4%; p 0.087 | null mean 2.13; pct 86.9%; p 0.132 |
| Causal x real SVXY (0.02) | null mean 0.29; pct 11%; p 0.89 | null mean 0.31; pct 14.7%; p 0.85 | null mean 0.29; pct 12.6%; p 0.87 | null mean 0.30; pct 13.2%; p 0.87 |

Reference: constant short = 2.89; a flat position at the strategy's mean exposure scores 2.28 (lower only because the 100x leverage cap binds on 52% of days at that smaller exposure, versus 9% at exposure 1.0).

### 4.4 Search over the 132-point multiplier grid (`grid_snooping.csv`)

| | Viterbi labels | Causal hard labels |
|---|---|---|
| Best Sharpe on grid (stressed 0.2, crisis 0.0) | 3.13 | 2.77 |
| Median Sharpe on grid | 2.04 | 1.96 |
| PBO (CSCV, 16 blocks) | 0.23 | 0.34 |
| Mean IS / OOS Sharpe of the IS winner | 3.45 / 2.75 | 3.09 / 2.32 |
| White RC p vs zero / SPA p vs zero | 0.001 / 0.001 | 0.001 / 0.001 |
| RC p / SPA p of best rule vs constant short | 1.00 / 1.00 | 1.00 / 1.00 |
| Best rule's mean return minus constant short | -11.3 pp/yr | -13.7 pp/yr |

### 4.5 Sub-period Sharpe (four equal blocks, 95% CI, from `subperiod_sharpe.csv`)

Original: 1.34 (CI -0.16 to 7.2), 2.06 (0.54 to 5.0), 6.30 (4.2 to 8.5), 4.76 (2.9 to 7.5). Causal x SVXY: 0.23, -0.35, 0.08, -0.13 (every CI includes zero).

## 5. Interpretation (plain language)

1. **The statistics do not reject the proxy Sharpe; the P&L definition is what is wrong.** On the proxy series, a Sharpe of 2.8 to 3.1 is about 4 standard errors from zero, survives deflation for any plausible N (DSR above 0.97 up to N = 10,000) and needs only about 1.5 years of data. Selection bias among the 150 tried variants is therefore not what inflated the number. The problem is documented in the protocol: the proxy omits mark-to-market, so a constant short position scores 2.89 on it. A significance test on a mis-measured P&L stream returns significance. This is the main lesson for the other agents: `validate()` PASS is necessary, never sufficient.
2. **Serial correlation and fat tails shrink but do not remove the proxy Sharpe.** The proxy returns have positive autocorrelation at lags 1 to 12 (rho_1 about 0.1), skew near -10 and excess kurtosis near 200. Lo's correction cuts 3.08 to 2.20 (and lower, 1.65, if 40 lags are used); the bootstrap CI is 1.65 to 5.26. A Sharpe quoted to two decimals has an uncertainty band of plus or minus 1.8.
3. **The regime logic adds nothing detectable.** Versus constant short the paired Sharpe difference is -0.15 to +0.19 with p between 0.8 and 0.9. Nothing in the 132-point grid beats constant short on mean return (RC and SPA p of 1.00; the best rule is 11 pp/yr worse). The grid itself is not overfit in the rank sense (PBO 0.23 to 0.34, because all 132 candidates are 0.87-correlated variants of the same short position and all are good), but that "robustness" is just "short variance works on this proxy".
4. **The original random-timing test was not a test of timing skill.** The shuffled null has a mean of 1.75 (reproducing the notebook's 1.75) because it inherits the premium, and it also turns the position into day-to-day noise, which changes the vol-target leverage path, so the null is a different strategy rather than an unskilled version of this one. Under null positions that keep the position's autocorrelation and exposure, the null mean rises to 2.0 to 2.2 and the strategy's percentile falls from 99.8% to 91% (circular shift) or 95% (21-day blocks). For the causal version the percentile is 83% to 91% and p is 0.09 to 0.17: not significant. The "p = 0.003" evidence for timing skill was an artefact. The original Viterbi strategy sits just at the 5% boundary under one null and fails under the others, and that strategy uses look-ahead labels.
5. **On a real traded instrument there is no edge.** Causal positions applied to SVXY earn a Sharpe of 0.02 (CI -0.55 to 0.70, DSR 0.005), sit at the 11th to 15th percentile of their own timing nulls (the timing makes them slightly worse than random, not better), and are statistically indistinguishable from SVXY buy and hold (difference -0.33, p = 0.10). SVXY buy and hold itself (0.37) would need about 24 years of data to be significant at 5%.
6. **Sub-period instability.** The proxy Sharpe runs from 1.3 to 6.3 across four equal blocks; the proxy returns of 2021-2026 dominate. A stable-edge story would not show a 5x range.

## 6. Caveats

* Statistical significance is conditional on the return stream being the true tradable P&L. Everything in 4.1 for the proxy series answers only "is the proxy mean different from zero?".
* DSR is an approximation: it treats trials as exchangeable with a common variance, assumes Gaussian-like tails in the expected-maximum formula, and takes N as given. N = 150 is our best count from git; uncommitted exploration is unknowable.
* Percentile bootstrap intervals for a Sharpe are mildly anti-conservative at small T and with extreme skew (here the effective sample is much smaller than 2,485 days because a handful of crash days drive the kurtosis). The tests show coverage of about 0.82 to 0.97 at a nominal 0.90 on well-behaved data; with excess kurtosis near 200, real coverage is probably worse.
* Lo's adjustment depends on the lag truncation (2.20 at 8 lags, 1.65 at 40); it is reported at the Newey-West lag, with the caveat shown.
* The reality check tests the mean return differential, not a Sharpe. PBO only measures overfitting within the supplied candidate matrix.
* Nulls: circular shift also moves the position across the vol-target warm-up and across the Feb 2018 event, and the null Sharpe depends on the 100x cap (flat exposure of 0.45 scores 2.28 while exposure 1.0 scores 2.89). None of the nulls adjusts for the fact that the regime model was fitted on pre-2016 data only for the headline, but selected on the full OOS period for the multiplier.
* SVXY returns are the traded ETN/ETF (-1x until Feb 2018, -0.5x after); the series mixes two leverage regimes. Excess return uses the 13-week bill.
* Proxy-strategy Sharpes are not excess of the risk-free rate (a variance swap is unfunded; this matches the 3.08 headline), but SVXY rows are.

## 7. What an interviewer might push on

* *"Your DSR says 0.999. So the strategy is fine?"* No. DSR corrects for selection, not for a wrong P&L. The decisive evidence is the real-SVXY row (0.02) and the constant-short comparison.
* *"How did you choose N = 150?"* From the git log and notebooks: 132 grid points plus 13 logged variants, rounded up. The DSR is shown from N = 1 to 10,000; the conclusion on the proxy does not move and the conclusion on SVXY does not move either, which is why the exact N does not matter here.
* *"Why not just use the cross-trial variance of the grid?"* Used as a sensitivity; it is lower-bound-ish because the 132 candidates are one idea (average correlation 0.87), which would understate the number of independent hypotheses explored across HMM, cap and return-definition changes.
* *"Why is your null centred at 2.2, not at zero?"* Because a short-variance book is structurally long the premium; the right question for timing skill is "better than randomly timed versions of the same exposure", so zero is the wrong reference.
* *"Percentile 95% under one null, 83% under another. Which one is right?"* They answer slightly different questions (block permutation keeps regime durations but reshuffles when each is held; circular shift preserves the exact sequence). Neither supports skill with the causal strategy; I report both and the spread.
* *"Is the bootstrap valid with 200 excess kurtosis?"* It is the standard remedy but not a guarantee; I would want a studentised bootstrap or a subsampling check, and ideally a longer sample.
* *"Why does Lo's adjustment shrink the Sharpe by a third but your HAC SE barely moves?"* They target different things: Lo changes the point estimate of the annual Sharpe given persistence in the 252-day aggregation; the HAC SE measures the precision of the daily mean/vol ratio with a tapered kernel. Both are shown; the lag choice is a stated sensitivity.
* *"Did you test the tests?"* Yes: size, coverage and power simulations for each (section 2).
