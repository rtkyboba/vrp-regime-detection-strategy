# 02 - Regime detection under strict causality

(Written up from the regime-detection agent's hand-back; all numbers are from `scripts/regime_comparison.py`, outputs in `results/regimes/`.)

## 1. Question
The original 3-state HMM was trained once on 2006-2015, decoded with Viterbi, and its states and features were chosen while watching 2016+. (1) Which causal regime method is usable, and is any distinguishable from the others or from noise? (2) Was 3 states defensible using pre-2016 information only? (3) Does gating a short-vol instrument on regime help (de-risk) or hurt (buy the spike)? Both directions are tested.

## 2. Method
- **Interface.** `fit(F_train)`, then `filter(F)` returns (T,K) probabilities whose row t uses rows <= t only. Columns are sorted by volatility so EM label switching cannot corrupt a walk-forward concatenation. `stress_score` is P(top-vol state) or the method's native [0,1] score. The hard flag is score > 0.5 (0.8 for GJR-GARCH and the VIX percentile; the CUSUM alarm level).
- **Walk-forward.** Expanding window, refit every 63 trading days, min_train 1260. Block [i, i+63) comes from a model fitted on rows < i and filtered over rows <= t. Common evaluation window 2012-01-03 to 2026-05-22 (3,617 days). The VIX9D family starts in 2013 and is compared with its own baselines.
- **34 variants:** HMM on original features n=2..5 (plus the original spec fit once on 2006-15 and causally filtered); HMM on core features (log VIX, log RV21, slope) n=2..5; 3-state core HMM with VVIX, SKEW, VIX6M, VIX9D or all added; GMM with no temporal structure (K=2,3,4 original, K=3 core); Markov-switching level models (MarkovRegression on log VIX k=2,3; MarkovAutoregression(1) k=2; MarkovRegression on log RV k=2); statistical jump model (K=3 with lambda 5, 20, 60; K=2 lambda 20); rules (VIX above rolling 80th/90th percentile, VIX percentile rank, VIX/VIX3M > 1, VIX9D/VIX > 1); CUSUM on dlogVIX and on level; GJR-GARCH.
- All hyper-parameters were fixed before any result was computed. Nothing was changed after seeing 2016+.
- **Targets.** Forward 21d realised vol, and the held-to-maturity short-variance payoff `1 - RV_fwd^2 / VIX_t^2`. Block bootstrap: block 21, 1000 resamples.
- **Economic tests.** SVXY: exposure set at close t earns the t+1 return, remainder earns T-bills, cost 2 bps x |change in exposure|, excess Sharpe. Variants: derisk_soft e = 1-s, derisk_hard, spike_soft e = s, spike_hard; baseline buy-and-hold. Rolling variance-swap book (1/21 notional cohorts, accrual P&L only, no MTM). White-style joint reality check of Sharpe minus buy-and-hold over 124 variants.

## 3. Results

### 3.1 How many states (pre-2016 information only)
| features | n | params | BIC | AIC | OOS LL/day 2009-15 | paired gain vs n-1 | GMM OOS LL | OOS LL 2016+ |
|---|---|---|---|---|---|---|---|---|
| orig | 2 | 21 | 13202 | 13081 | -2.39 +/- 0.10 | | -2.83 | -2.95 |
| orig | 3 | 35 | 11093 | 10891 | -2.22 +/- 0.13 | 0.17 (t=3.4) | -2.74 | -2.55 |
| orig | 4 | 51 | 9877 | 9583 | -2.04 +/- 0.13 | 0.18 (t=2.7) | -2.80 | -2.22 |
| orig | 5 | 69 | 8476 | 8078 | -1.95 +/- 0.12 | 0.09 (t=1.5) | -2.92 | -2.13 |
| orig | 6 | 89 | 8145 | 7632 | -1.82 +/- 0.13 | 0.13 (t=1.7) | -2.97 | |
| core | 2 | 21 | 12414 | 12293 | -2.56 +/- 0.09 | | -2.98 | -2.82 |
| core | 3 | 35 | 10862 | 10660 | -2.34 +/- 0.10 | 0.22 (t=3.2) | -3.10 | -2.47 |
| core | 4 | 51 | 9388 | 9094 | -2.33 +/- 0.12 | 0.01 (t=0.1) | -3.18 | -2.36 |
| core | 5 | 69 | 8516 | 8118 | -2.24 +/- 0.13 | 0.09 (t=1.5) | -3.26 | -2.20 |
| core | 6 | 89 | 8374 | 7861 | -2.14 +/- 0.12 | 0.11 (t=1.5) | -3.31 | |

- BIC and AIC fall monotonically through n=6 on both feature sets: no elbow. The features are 30-day rolling statistics (autocorrelated), so the HMM keeps adding states to track the path.
- A rule fixed in advance (add a state while the paired OOS gain has t >= 2) gives **n=4 on the original features and n=3 on core features**. So "3" is defensible only on core; the original features say 4. **The original 3 was not a selection result.**
- Persistence buys likelihood: HMM -2.22 vs GMM -2.74 at n=3.
- After 2016 the 4-vs-3 gain is large (t=6.7 orig, 2.4 core): that is non-stationarity, not cleaner regimes. Moving from 3 to 5 states changes the de-risk Sharpe from 0.36 to 0.47 (orig) and 0.39 to 0.45 (core), all at or below buy-and-hold.

### 3.2 Persistence, flicker, refit stability (2012-26, hard flag)
| method | stress % | mean run (days) | switches/yr | % episodes <= 3d | refit agree % |
|---|---|---|---|---|---|
| hmm3_orig | 11.3 | 27 | 2.0 | 33 | 98.7 |
| hmm3_core | 22.4 | 81 | 1.3 | 0 | 96.5 |
| hmm5_orig | 5.1 | 17 | 1.5 | 18 | 97.0 |
| hmm3_core_all | 27.6 | 29 | 4.9 | 29 | 85.5 |
| gmm3_orig | 2.9 | 6 | 2.2 | 81 | 99.8 |
| msar1_2_logvix | 19.5 | 3 | 31.5 | 69 | 99.7 |
| jump3_lam20 | 2.8 | 25 | 0.6 | 0 | 99.4 |
| rule_vix_vix3m_inv | 6.9 | 3 | 12.0 | 83 | 100 |
| rule_vix_q80 | 17.8 | 6 | 13.9 | 63 | 100 |
| cusum_diff | 3.0 | 3 | 6.0 | 88 | 100 |
| gjr_garch | 11.9 | 8 | 7.5 | 44 | 99.6 |

- HMMs are persistent: 1-3 flag switches/yr, top-state expected duration 60-200 days, refit agreement 91-99%. Rules and CUSUM are fast and noisy: 12-35 switches/yr. The GMM has no memory: 70-80% of its stress episodes last <= 3 days.
- Zero label flips across all refits for every method (the vol-sorting works).
- `hmm3_orig`'s calm and stressed states have close rv_30 means (0.14 vs 0.16), so only its top state is stable. `hmm3_core` flags March 2020 to early 2023 as one continuous stress regime.

### 3.3 Do regimes separate the future?
- Every method predicts forward RV (stress minus calm +4 to +20 vol points, t 2.0-6.0; e.g. `hmm3_orig` 22.6% vs 13.1%, t=4.5).
- **No method beats raw VIX** (Spearman 0.625 with forward RV). Best: `jump3_lam5` (0.58), `msreg3_logvix` / `msreg2_logvix` (0.57), `jump2_lam20` (0.57). VIX-percentile rule 0.51. HMMs 0.41-0.47. Richer feature sets are worse (`hmm3_core_all` 0.18, `hmm3_core_vix6m` 0.27).
- **The short-variance payoff is not significantly lower in stress.** Slow regimes show a higher payoff (hmm3_orig 0.36 vs 0.18, t=1.7; hmm3_core_all 0.42 vs 0.12, t=2.3). Fast flags show a lower one (-0.17 to -0.58) but |t| <= 1.2. Only 1 of 34 variants has |t| > 2.
- Next-day SVXY by regime: no |t| > 1.5.
- Only ~15 stress episodes per method, so bootstrap t-values are optimistic.

### 3.4 Early warning (dates fixed from market history before looking at any model)
- Mar 2020 and 2022: every method flags ahead of the peak (5-16 days in 2020).
- Aug 2015: slow HMMs flag only at the peak or not at all; fast rules and CUSUM lead by 1-2 days, VIX9D by 9 days.
- **Feb 2018** (SVXY -32% on 2/5, -83% on 2/6): never flagged by hmm3/4/5 (orig and core), the core HMMs with extra features, jump3_lam20/60, jump2, msreg2_logrv, gmm4. Flagged by the 2/5 close (0-7 days ahead) by VIX/VIX3M, VIX quantile and percentile rules, CUSUM, GJR, GMM-3, msreg2/3, MS-AR, jump3_lam5, hmm2, hmm3_core_all. Soft-gate return trigger-to-peak+10: buy-and-hold -89%, fast rules ~0%, slow HMMs -89%.
- The cost of lateness shows in 2020: slow HMMs stay out for the entire rebound (buy-and-hold +19% in the 40 days after the peak, gate 0%).

### 3.5 Economics on real SVXY (net, 2012-26)
Buy-and-hold Sharpe 0.50 (CI -0.05 to 1.12); 0.84 pre-2016; 0.34 in 2016+; max drawdown -95%; worst day -83%; **0.74 excluding Feb 2018**.

| method | derisk_soft net Sharpe | 95% CI | ex-Feb18 | 2016+ | spike_soft |
|---|---|---|---|---|---|
| rule_vix_vix3m_inv | 0.75 | [0.26, 1.22] | 0.76 | 0.71 | -0.18 |
| msar1_2_logvix | 0.71 | [0.22, 1.21] | 0.73 | 0.66 | 0.09 |
| cusum_diff | 0.70 | [0.23, 1.17] | 0.73 | 0.59 | -0.05 |
| rule_vix_q80 | 0.69 | [0.19, 1.19] | 0.69 | 0.67 | -0.01 |
| gmm3_orig | 0.63 | [0.13, 1.15] | 0.72 | 0.53 | -0.13 |
| jump3_lam20 | 0.61 | [0.08, 1.18] | 0.82 | 0.47 | -0.22 |
| hmm3_orig_static | 0.43 | | 0.54 | 0.43 | 0.04 |
| hmm4_core | 0.40 | [-0.15, 1.01] | 0.65 | 0.18 | 0.35 |
| hmm3_core | 0.39 | [-0.17, 1.02] | 0.65 | 0.29 | 0.35 |
| hmm3_orig | 0.36 | [-0.17, 0.98] | 0.60 | 0.16 | 0.51 |
| hmm3_core_vvix | 0.30 | [-0.27, 0.97] | 0.54 | 0.12 | 0.59 |

- **The de-risk gain is the Feb 2018 event.** Fast rules sit at 0.70-0.75 with or without it, while buy-and-hold rises from 0.50 to 0.74. Excluding Feb 2018 the best variant adds only +0.08 (t < 1). Pre-2016, fast rules equal buy-and-hold (0.84 vs 0.84).
- **Reality check over 124 variants:** max t = 1.45 (jump3_lam20 soft), 95% critical value of the maximum 3.41, **adjusted p = 0.78**. Excluding Feb 2018: max t = 0.96 vs critical 3.22.
- For spike variants vs buy-and-hold, t reaches -3.1 (VIX/VIX3M spike, adjusted two-sided p = 0.11).
- **Look-ahead reference, 2016+:** original Viterbi 0.60, the same model causal 0.36 (hard) / 0.35 (soft), buy-and-hold 0.34. Argmax agreement is 96%.

**Direction (neither established):**
| Evidence | De-risk | Buy the spike |
|---|---|---|
| SVXY, fast flags | 0.6-0.75 vs 0.50, but only via Feb 2018 | ~0 to -0.2 |
| SVXY, slow regimes | 0.30-0.47, below buy-and-hold (persistence misses 2018 and the rebounds) | 0.2-0.6 at 5-30% exposure, alpha t <= 1.7 |
| Variance payoff, slow regimes | payoff higher in stress, so de-risking gives up premium (swap book 0.5-1.1 vs constant 1.28) | swap book 1.5-3.5; 2016+ 1.3-4.0 |
| Variance payoff, fast flags | de-risk book 1.5-1.6 vs 1.28 | 0-0.5 (pre-2016 4.4-4.6, 2016+ negative: sign flips) |

The swap-book numbers are accrual-only (overlapping cohorts, no MTM or margin, skew -6 to -25). The top swap variant (hmm3_core_all spike, Sharpe 3.5 vs constant 1.28) has t = 1.74 and adjusted p = 0.32: it shows direction, not size.

### 3.6 Persistence vs mixture
The HMM has far better likelihood and 10-20x fewer switches than the mixture, but detects Feb 2018 later and exits later. The mixture's de-risk Sharpe is higher (0.63 vs 0.36). That difference was not separately tested.

### 3.7 Feature sets
Adding VVIX, SKEW, VIX6M or VIX9D to the core HMM improved nothing: forward-RV Spearman falls (0.43 to 0.42, 0.35, 0.27, and 0.18 for all three), de-risk Sharpe is 0.30-0.45 vs 0.39, and refit stability is worst for core_all (85% agreement). The VIX9D family (from 2013) scores 0.11 vs 0.38 for buy-and-hold on the same window; the raw VIX9D > VIX rule flickers 35 times a year (0.48).

## 4. Best method, and is it distinguishable from noise?
- Forecasting forward vol: nothing beats raw VIX. The jump model (lambda 5-20) and Markov-switching level models lead; HMMs are mid-pack.
- Stability: jump3_lam20 and the core HMMs.
- Early warning of a Feb-2018-type event: rules, CUSUM, GJR and MS-AR; HMMs do not.
- Economic value: the VIX/VIX3M inversion rule is nominally best (0.75 vs 0.50) but rests on one event and has p = 0.78 after correcting for 124 variants. **No method is distinguishable from buy-and-hold or from the others.**
- As a risk monitor, simple rules are at least as good as every model tested on latency, flicker and Sharpe. The HMM is weakest on latency.

## 5. Caveats
- One decisive event; SVXY leverage changed from -1x to -0.5x in Feb 2018.
- ~15 independent stress episodes per method, so bootstrap standard errors are optimistic and the 2011-15 sub-period columns are unreliable.
- Event dates are known ex post.
- Hard thresholds are not comparable across score types.
- The variance-swap book is accrual-only; 2 bps is not a real swap cost.
- statsmodels Markov-switching fits are not bit-reproducible (up to 8 restarts, degenerate fits with P(stay) < 0.8 rejected), so msreg*/msar* rows may shift slightly on rerun.
- GJR-GARCH hit alpha = 0 at the boundary; one specification only.
- Data end 2026-05-22. The first stress calibration sees only the 2008 crisis and 2011.
- Two technical evaluation fixes were made (static-from-2016 model excluded from the common reality-check window; NaN handling in the swap reality check). No method was changed.

## 6. Trial count
328 = 34 methods + 22 order-selection fits + 136 SVXY variants + 136 swap variants. The reality check uses 124 (the VIX9D family and the 2016-start static model are excluded).

## 7. What an interviewer might push on
- **"Nothing works. Why believe your null?"** Power: one decisive event and a Sharpe-difference standard error of 0.08-0.25 cannot detect gains below about 0.3-0.5. The statement is "not established", not "proven zero".
- **"The fast rules beat buy-and-hold."** Excluding Feb 2018 the gain is +0.02. That is insurance priced in hindsight: valid as a tail hedge, not as alpha.
- **"Why is the HMM late in 2018?"** Stay probability 0.98-0.99, and a 3-day jump from 13 to 37 is one outlying observation.
- **"Why 3 states?"** It was not selected. BIC keeps falling, out-of-sample log-likelihood says 4 on the original features and 3 on core, and more states do not help P&L.
- **"Why not Viterbi?"** Tomorrow's position cannot see tomorrow's data. Look-ahead is worth about 0.25 Sharpe on real SVXY.
- **"The swap book says buy the spike at 3.5."** Accrual-only, overlapping cohorts, no MTM, adjusted p = 0.32; SVXY gives 0.3-0.6 for the same exposure.
- **"Is the 0.8 threshold tuned?"** No, it is a fixed convention, and the soft variant is threshold-free.
- **"Does regime beat VIX?"** No (0.625 vs at most 0.58). The case for regimes is stability and communicability.
- **"Was the walk-forward honest?"** Refits use rows strictly before each block. Tests corrupt future rows for all 13 model classes plus the driver, and a negative control confirms the smoothed posterior does move.
