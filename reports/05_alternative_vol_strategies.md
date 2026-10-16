# 05 - Alternative ways to monetise volatility (tradable data only)

Reproduce: `python scripts/alt_strategies.py` (about 1 minute). Code: `src/vrp_strategy/alt_strategies.py`.
Tests: `pytest tests/test_alt_strategies.py` (no look-ahead for every strategy, cost accounting, lag, bootstrap).
Every number below is in `results/alt/` (`summary_*.csv`, `tables.md` has the full per-variant tables for every window,
`<variant>.csv` has date, exposure, ret for each of the 37 variants, `equity.png`).

## 1. Question

The audit (`docs/EVAL_PROTOCOL.md`) showed the original VRP strategy's Sharpe 3.08 was a proxy-P&L artefact: its positions on
real SVXY returns give about 0.1, versus SVXY buy-and-hold 0.4 and SPX 0.8 (2016+). So: **is there any way to make money from
volatility, or to raise Sharpe, that survives realistic P&L, costs and an honest out-of-sample test?**
The yardstick is not "positive Sharpe" (SPY itself has 0.86 since late 2011) but "better than buy-and-hold SPY, net of costs".

## 2. Method

**Data (tradable only).** SPY adjusted close (total return), SVXY (-1x short-term VIX futures until Feb 2018, -0.5x after),
VXX (real series only from 2018-01-25), T-bill from ^IRX. Signals: VIX, VIX3M, VIX9D, VVIX, SKEW. For long-vol before 2018
the series `lv` is real VXX where it exists and `-1 x SVXY` before (exact while SVXY was -1x: post-Mar-2018 the regression of VXX on
SVXY has slope -1.9 and correlation -0.98). Everything is evaluated up to 2026-05-22, the last VIX3M observation.

**Engine (one function costs everything).** Weights are set at the close of `t`, earn the return of `t+1` (explicit one-day lag, unit-tested).
Uninvested capital earns the T-bill rate, borrowing above 100% pays T-bill + 0.5%/yr. Cost is `bps x sum|change in held weights|`
at 0, **2 (headline)** and 5 bps. Constant-mix hedge portfolios charge cost on the trade back from drifted weights. Sharpe is excess of
the T-bill. The bootstrap is a circular block bootstrap (21-day blocks, 2000 draws, same resampled days for strategy and benchmark, so the
CI is for the **Sharpe difference vs SPY** on identical dates). Benchmarks are SPY and SVXY buy-and-hold on the same dates.

**Windows.** `all` = strategy's own history; `pre2016` (the development window); `test2016+` (out of sample); `from2011-10`
(common window where every SVXY sleeve is live and SPY is the comparable benchmark).

**Parameters are fixed a priori, not searched.** Rationale per family (details in docstrings):

| # | Idea | Literature | Fixed parameters and why |
|---|---|---|---|
| 1 | Vol-managed SPY, w = c/sigma | Moreira & Muir (2017); critics: Cederburg et al. (2020) | c = expanding historical mean of sigma (no full-sample c), so average exposure about 1; cap 1.5; monthly (their frequency) or daily; sigma = 21d realised or VIX; stress overlay = half size when VIX/VIX3M >= 1 |
| 2 | Term-structure carry switch | Simon & Campasano (2014), Eraker & Wu (2017) | VIX/VIX3M < 1 = contango (1.0 is structural); ramp 0 at 1.0 to full at 0.90 (about the pre-2016 median); 9D variant uses VIX9D/VIX 1.05 to 0.95; hysteresis on 0.95/1.00; off-state cash or SPY |
| 3 | Buy the spike vs de-risk in stress | Whaley (2009), vol mean reversion | spike = VIX max over 10d >= 1.4x its 60d median, curve back in contango and VIX 10% off peak; hold 15d; de-risk = flat when VIX 252d rank > 0.8 |
| 4 | VVIX / SKEW crash filter | tail-risk premia literature | flat when 252d percentile rank > 0.8 (rank because both gauges drifted up over the sample) |
| 5 | SPY + small long-vol; put-write | CBOE PUT index | hedge 5% (sensitivity 2, 10); put-write: monthly ATM, Black-Scholes at 0.93 x VIX (APPROXIMATION, sensitivity 0.85 / 1.00) |
| 6 | Risk-parity blend | inverse-vol | 126d vol, floor 4%, cap 60%, monthly; **sleeves chosen on pre-2016 only**: sleeve must have positive pre-2016 Sharpe and a 2-asset blend with SPY must beat SPY pre-2016 |

**Trials: 37 variants** (7 vol-managed, 10 term-structure, 4 spike/de-risk, 6 crash filter, 7 hedge/put-write, 3 blends), each run at 3 cost
levels. All are reported in `results/alt/summary_metrics.csv`. The headline column below is the pre-declared family representative,
not the winner of the sweep. For reference, the expected best Sharpe of 37 independent zero-skill trials over the 2016+ window is 0.67
(Bailey-Lopez de Prado SR0; conservative since trials are correlated). SPY itself scores 0.76 there, so a lone "best variant" proves little.

## 3. Results

Net of 2 bps, excess Sharpe. SPY and SVXY buy-and-hold over the common window **2011-10-06 to 2026-05-22**: SPY Sharpe 0.860,
max DD -33.7%, worst day -10.9%; SVXY Sharpe 0.520, max DD **-95.2%**, worst day **-83.0%**. 2016+: SPY 0.757, SVXY 0.344.
`dSPY` = Sharpe difference vs SPY on the same dates with 95% block-bootstrap CI.

### 3.1 Headline table (family representatives and the best alternatives), window from 2011-10

| Variant | Sharpe | Max DD | Worst day | Turnover/yr | dSPY [95% CI] | Pre-2016 Sharpe | 2016+ Sharpe |
|---|---|---|---|---|---|---|---|
| BH SPY | 0.860 | -33.7% | -10.9% | 0 | - | 1.207 (2011-15) | 0.757 |
| BH SVXY | 0.520 | -95.2% | -83.0% | 0 | - | 0.903 | 0.344 |
| vm_implied_m (VIX, monthly) | 0.751 | -32.8% | -11.5% | 2.2 | -0.11 [-0.27, +0.03] | 0.512 (2005-15) | 0.626 |
| vm_realised_d (21d RV, daily) | 0.956 | -23.1% | -6.3% | 4.0 | +0.10 [-0.12, +0.31] | 0.598 (2005-15) | 0.883 |
| ts_bin_cash (SVXY if VIX/VIX3M<1) | 0.757 | -49.4% | -26.5% | 12.5 | -0.10 [-0.58, +0.37] | 0.877 | 0.710 |
| ts_bin_spy (else SPY) | 0.795 | -49.9% | -26.5% | 25.0 | -0.07 [-0.50, +0.35] | 0.928 | 0.738 |
| spike_buy | 0.468 | -55.5% | -26.5% | 8.2 | -0.39 [-0.99, +0.17] | 1.145 | **0.093** |
| derisk_highvol | 0.633 | -55.7% | -26.4% | 14.6 | -0.23 [-0.78, +0.32] | 0.617 | 0.671 |
| ts_vvix (VVIX filter on ts) | 0.581 | -52.1% | -26.5% | 26.1 | -0.28 [-0.83, +0.26] | 0.984 | **0.350** |
| ts_skew (SKEW filter on ts) | 0.840 | -39.2% | -19.2% | 32.0 | -0.02 [-0.54, +0.52] | 1.113 | 0.699 |
| spy_lv5 (95% SPY + 5% long-vol) | 0.847 | -26.4% | -8.5% | 0.9 | -0.01 [-0.10, +0.07] | 1.226 | 0.744 |
| putwrite (BS model, approx.) | 0.963 | -25.9% | -7.8% | 12.1 | +0.10 [-0.09, +0.34] | 1.599 | 0.772 |
| blend_rp_survivors (from 2012-04) | 0.830 | -24.1% | -7.6% | 1.0 | +0.02 [-0.24, +0.30] | 1.477 | 0.592 |

(The vol-managed rows use their own history in `tables.md`; on their full 2005-2026 history SPY has Sharpe 0.554 and max DD -55.2%:
vm_realised_d 0.737, dSPY +0.19 [-0.00, +0.37], max DD -35.8%, worst day -6.4%; vm_implied_m 0.569, dSPY +0.02 [-0.15, +0.19].)

### 3.2 Strategy by strategy

**1. Vol-managed equity.** The only family where the sign of the Sharpe difference is stable across periods, for the daily-rebalanced
variants: realised-vol daily +0.22 pre-2016, +0.13 in 2016+ (window 2005-26: +0.19, CI [-0.00, +0.37], one-sided p=0.03), with a much smaller
drawdown (2016+: -23% vs -34%) and worst day (-6.3% vs -10.9%). The monthly versions (Moreira-Muir's own frequency) are **worse**:
the weight set at the February 2020 rebalance (1.2x implied, 1.35x realised) was still on when SPY fell 9.6% on 2020-03-12, giving worst days of -11.5% / -12.9%
(larger than SPY's -10.9%), and 2016+ dSPY is -0.13 (implied) / -0.05 (realised).
The stress overlay (half size when VIX/VIX3M >= 1, tested only on the monthly versions) helps them (+0.07 to +0.08 in Sharpe, worst day about -5.7% to -6.5%) because it is the
part that reacts daily. Costs are immaterial (daily realised: 0.743 at 0 bps, 0.728 at 5 bps).
Caveat: 7 variants in this family, CI includes zero, and the post-publication literature (Cederburg et al.) finds the benefit is not robust out of sample.

**2. Term-structure carry switching.** It does what it should for the instrument: SVXY drawdown -95% to -49%, worst day -83% to -26%, and Sharpe 0.52 to 0.76
(dSVXY +0.24, CI [-0.17, +0.64]). It does **not** beat SPY (dSPY -0.10, CI wide). Two further findings.
(a) Almost all of the gain over SVXY buy-and-hold is **one event**: the signal flipped to backwardation at the 2018-02-02 close (VIX/VIX3M = 1.016) and so
missed the -32% / -83% days. Dropping 2018-01-25 to 2018-03-31 from both, ts_bin_cash scores 0.802 and SVXY buy-and-hold 0.775.
(b) After the leverage change the edge is small: -1x era Sharpe 1.03 vs 0.67, -0.5x era (2018-03+) 0.46 vs 0.37 (diff +0.09, CI [-0.32, +0.57]).
The ramp, the VIX9D condition and hysteresis all **lowered** Sharpe. Hysteresis did cut turnover (12.5 to 8.8 per year) but exits late: Sharpe 0.757 to 0.559.
The 9D variants triple turnover (36/yr) for a lower Sharpe. Switching into SPY instead of cash adds +0.04 and no drawdown benefit.

**3. Buy-the-spike versus de-risk.** Buying the post-spike normalisation **does not work**: Sharpe 0.47 overall, 0.09 in 2016+, only 16% of days invested.
Mean 15-day SVXY excess return after a trigger is 1.5% versus 1.7% on all other days: no edge, though the trigger reduces the worst case.
The conditional table in `tables.md` shows why: SVXY next-day excess return when VIX > 35 is about +500% annualised on just 5 pre-2016 days but -250% annualised (Sharpe -1.3) on 61 days in 2016+.
Rebounds are sharp, but so are the second legs, and the sample of true spikes is tiny.
The opposite logic (flat when VIX rank > 0.8) gives 0.63 vs 0.52 for always-long SVXY overall, but the sign **flips between periods**:
pre-2016 it costs 0.29 of Sharpe (0.62 vs 0.90, the "sharp rebounds hurt de-riskers" effect from the audit), 2016+ it adds 0.33 (0.67 vs 0.34).
So the audit's suggestion is supported only in the early sample; neither direction is reliable.

**4. VVIX / SKEW filter.** Not a reliable crash filter. On top of the term-structure rule, the VVIX filter *lowers* Sharpe (0.76 to 0.58; 2016+ 0.71 to 0.35).
The SKEW filter lifts it (0.84, max DD -39%) but SKEW alone on always-long SVXY misses the Feb 2018 collapse completely (max DD -94.6%, worst day -83%), so the one favourable
result in a 6-variant family is more likely noise than a signal. "Either gauge" cuts exposure to 55% of days and lowers drawdown (-40%), but its 2016+ Sharpe (0.52 on SVXY, 0.47 on the term-structure rule) is below SPY's 0.76.

**5. SPY plus long-vol, and put-write.**
*Long-vol hedge.* Long-vol correlates -0.77 with SPY and -0.81 with the short-vol sleeve (-0.92 pre-2016). It returned +17.7% on average on SPY's 20 worst days
(positive on 19 of 20), but it bleeds -44%/yr on average (roll cost in contango). A 5% sleeve cuts max DD from -33.7% to -26.4%, vol 16.8% to 13.5%, worst day -10.9% to -8.5%,
yet Sharpe is unchanged (0.847 vs 0.860, CI [-0.10, +0.07]); 10% is worse (0.78). Real-VXX-only 2018+: 0.656 vs 0.639. So it is **insurance with no Sharpe benefit**: diversification benefit is about zero
in Sharpe terms and positive in drawdown terms. Scaling SPY down to hold some cash leaves Sharpe unchanged by construction, so the hedge's value is the shape of the tail, not the Sharpe.
*Put-write.* Cash-secured monthly ATM puts priced off VIX reproduce the known profile (vol about 11-13% vs 17-19%, max DD -36% vs -55% in 2004-26, correlation with SPY 0.91).
Sharpe +0.10 vs SPY on 2011-10+ (CI [-0.09, +0.34]); +0.22 over 2004-2015 (CI [+0.02, +0.45]), +0.02 in 2016+. **This is the weakest-evidence result in the report**: it is a model, and it moves one-for-one with the
assumed ATM-IV/VIX ratio (0.85 gives dSPY -0.07, 1.00 gives +0.26 with CI excluding 0). It tells you the sign of the variance risk premium is positive, not that you can harvest it.

**6. Blend.** The pre-2016 selection kept `spike_buy` (pre-2016 Sharpe 1.14) and `putwrite`. `spike_buy` then collapsed in 2016+ (0.09), so the blend
(inverse-vol: SPY 28%, spike_buy 30%, putwrite 43% on average) has Sharpe 0.83 vs SPY 0.81 from 2012-04 (+0.02, CI [-0.24, +0.30]); 1.48 vs 1.03 pre-2016 but **0.59 vs 0.76 out of sample**
(CI [-0.44, +0.12]). Max DD -24% and vol 12% are lower than SPY's, but you get that from holding less equity risk. An equal-weight blend of the same sleeves gives the same picture (0.85 / 0.62 in test)
and the blend of all six candidates is no better (0.76 overall, 0.57 test). Risk-parity blending of weak sleeves does not manufacture alpha.

## 4. Honest verdict

**Nothing here beats buy-and-hold SPY net of costs with statistical confidence.** Every 95% CI on the Sharpe difference vs SPY contains zero (the narrowest favourable one,
daily realised vol-managed SPY at +0.19 over 2005-26, touches zero at -0.00), and with 37 trials the best out-of-sample Sharpe in the table (0.88 to 0.92 for vm_realised_d and the
most generous put-write) is not far above the 0.67 that chance alone would produce.

| Idea | Verdict |
|---|---|
| Vol-managed SPY (daily, realised vol, cap 1.5) | **Weak positive.** Same sign pre- and post-2016, roughly +0.1 to +0.2 Sharpe and a third off the drawdown and worst day; borderline significance; costs irrelevant. The only candidate worth carrying forward, with the monthly caveat (monthly rebalancing with 1.5x leverage is worse than SPY in the tail). |
| Term-structure switch on SVXY | **Useful risk control, not alpha vs SPY.** Fixes SVXY's tail (-95% to -49% DD) but lags SPY, and the improvement over SVXY buy-and-hold rests on one early exit in Feb 2018. |
| Hysteresis / ramps / VIX9D | **Do not help.** Lower Sharpe; 9D triples turnover. |
| Buy-the-spike | **No.** No excess 15d return after trigger; Sharpe 0.09 out of sample. |
| De-risk in high vol | **Inconclusive.** Hurts pre-2016, helps 2016+; never significant. |
| VVIX filter | **No** (cuts Sharpe). SKEW filter: one lucky variant; fails alone in Feb 2018. |
| Long-vol tail hedge | **Insurance, not return.** Drawdown and tail improve, Sharpe flat; costs -44%/yr in carry. Only worth it if the goal is a lower worst day. |
| Put-write | **Positive only under the assumed pricing**; not evidence. |
| Risk-parity blend | **No.** Lower vol, equal Sharpe; the pre-2016 selected sleeve failed out of sample. |

## 5. Caveats

- **SVXY leverage change (Feb 2018).** SVXY is -1x until 2018-02-27 and -0.5x after. A "100% SVXY" exposure therefore has half the risk in the second half; the era table splits both.
  The early era (Sharpes around 1.0 for the SVXY sleeves) is the 2011-2017 low-vol grind, which flatters every short-vol sleeve.
- **ETF fees and tracking.** SVXY/VXX returns include expense ratios (0.95%/0.89%) and futures roll costs, which is right for a tradable test; `lv` before 2018 is an inverse of SVXY (about VXX minus fee differences) and is an approximation.
  SVXY's 2018-02-05/06 prints (-32%, -83%) reflect an after-hours NAV reset not available to a stop order.
- **Short history.** SVXY starts 2011-10: 14.6 years with essentially three stress events (2015, Feb 2018, Mar 2020) plus Aug 2024; the pre-2016 development window for any SVXY sleeve is 4.2 years.
  Sharpe standard errors are about 0.26 on the full 14.6 years and about 0.5 on the 4.2-year development window, so differences below roughly 0.3 are not detectable.
  2016+ is not a pure test either: the ideas tested were designed with knowledge of 2018 and 2020.
- **Costs.** 2 bps on weight changes is realistic for SPY and SVXY ETF trading, not for options; the put-write charges 2 bps of notional per roll, below real option spreads (about 3 to 5 bps of notional per side). Slippage on days SVXY gaps is not modelled.
- **Put-write is model P&L** (Black-Scholes, VIX-derived vol, q = 0, no early exercise, no margin costs), so treat it as illustration. Financing for leverage at T-bill + 0.5% is assumed.
- **Single path.** Everything is one historical path; the block bootstrap quantifies sampling noise, not regime risk.
- **Data ends 2026-05-22 for VIX/VIX3M**; VIX9D/VVIX/SKEW run slightly longer but are truncated to the same end.
- The blend's 2-asset selection rule and candidate list were fixed before looking at results; the sleeves' weights are causal but the choice of survivors used only 4.2 years.

## 6. What an interviewer might push on

1. *"Your Sharpe differences all have CIs spanning zero. Why report them as positive?"* I do not claim significance; I report sign stability across the pre/post split and the CI, and label vol-managed equity "weak positive".
2. *"37 trials."* Yes, all are listed; the family representative was named up front; SR0 for 37 trials over the test window is 0.67 and the best candidates sit only slightly above it.
3. *"The term-structure result depends on one day."* Correct, and the report says so (0.80 vs 0.78 ex-Feb-2018); that is the main reason it is not presented as alpha.
4. *"Why does monthly vol-managing lose to daily?"* Leverage of 1.2x to 1.35x set in a calm February 2020 was held through the first crash week; the daily version cut it as realised vol rose. Daily weights track 21d realised vol, which is slow-moving, so turnover stays about 4 per year.
5. *"Is the put-write number real?"* No - it is a Black-Scholes mark. Its Sharpe edge swings from -0.07 to +0.26 over a plausible IV/VIX range, so it only illustrates the sign of the variance risk premium.
6. *"Hedge cost vs benefit?"* Long-vol loses about 44%/yr; a 5% sleeve costs about 2.2%/yr and removes about 7 points of max drawdown. At equal Sharpe it is a risk-appetite decision, not an improvement.
7. *"Why not tune thresholds?"* They are structural (1.0 for the curve, rank 0.8 from convention, 0.90 = pre-2016 median), not fitted; sweeping 7 or 10 values per rule would have produced a prettier backtest and a worthless number.
8. *"Look-ahead?"* Unit tests corrupt all data after a cut date and require identical weights before it, for every strategy and for the put-write and risk-parity weights; the engine lags weights by one day and charges costs on the held-weight changes.
