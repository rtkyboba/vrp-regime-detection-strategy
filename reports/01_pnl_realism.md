# 01 - P&L realism: what is the original strategy actually worth?

Code: `src/vrp_strategy/pnl.py`, `tests/test_pnl.py` (21 tests, offline), `scripts/pnl_realism.py`
(one command reproduces everything; log in `results/pnl/run_log.txt`, tables in `results/pnl/*.csv`).
Out-of-sample = 2016-01-01 .. 2026-05-22 (`data.SPLIT`). All positions are the original
strategy's *causal* ones (HMM fitted pre-2016, forward-filtered, VRP rank x soft regime multiplier,
logic copied from `scripts/audit_sharpe.py`).

## 1. Question

The audit found the headline Sharpe 3.08 is inflated, mainly because the proxy
`iv_daily(t-1) - rv_daily(t)` ignores the mark-to-market (MTM) change in the variance strike.
Build P&L models that do not have that flaw, validate them against real SVXY/VXX, and answer:
**what is a defensible Sharpe for this strategy?**

## 2. Method

**Held-to-maturity (HTM) variance swap.** Enter at the close of `t` at strike `(VIX_t/100)^2`;
pay realised variance, `252/21 * sum r^2` over closes `t+1..t+21` (SPX close-to-close log returns,
zero mean). Per-entry payoff in vol points per unit vega: `(K^2 - RV^2)/(2K)`.
*Return convention:* payoff / `VEGA_CAPITAL`, where capital is 48 vol points per unit vega notional
(about 2.5x the mean VIX). This is a calibration constant, not a margin model: it was set from the
2011-11..2015 regression slope of SVXY on the MTM swap return (slope 0.83 at capital 40, so ~48 gives
slope 1.0), making "1x" comparable to an unlevered -1x VIX-futures ETF. It only fixes the scale of
returns (drawdown, leverage cap); Sharpe is scale-free.
*Attribution:* the cohort entered at `t` is booked at its maturity `t+21`, sized by the position
known at `t` (`cohort_series`: overlapping daily entries, each 1/21 of the book; `monthly_nonoverlap`:
every 21st day, averaged over all 21 phase offsets). Tests show that corrupting data after a date
never changes earlier bookings.
*Annualisation trap:* the overlapping daily ladder series is autocorrelated, so
`sqrt(252)*mean/std` overstates the Sharpe by about `sqrt(21)=4.6`. We use Newey-West (21 lags) on
the ladder and the non-overlapping monthly series instead (naive ladder numbers are shown only to
make the point).

**Daily MTM of a rolling short variance swap** (`mtm_decomposition`). Hold a 21-day swap struck
at close `t-1`; at close `t` it has 20 days left and is revalued off the curve. Fair expected
variance for `n` remaining days comes from linear interpolation of total variance across the
available VIX9D (9d), VIX (30d), VIX3M (91d), VIX6M (182d) nodes, anchored at 0. P&L per day:

    carry (v_{t-1} = iv_daily)  -  realised (r_t^2)  +  MTM (F_{t-1}(21) - v_{t-1} - F_t(20))

so the legacy proxy is **exactly carry minus realised** (unit-tested). Summing a single cohort's daily
MTM over its life telescopes exactly to the HTM payoff (unit-tested). Vega notional is held fixed
over each day (divide by `2*VIX_{t-1}`) and the result divided by capital.
Caveat found on the way: for a 21-day tenor only VIX9D and VIX enter the interpolation, and
removing VIX9D changes nothing measurable (table 2a, "single VIX node"); VIX3M/6M matter only for
longer tenors (`h=63` is supported but I did not use it).

**Generic `backtest`** (`pnl.backtest`): the exposure that earns day `t` is the one decided at the
close of `t-1` (explicit one-day lag); cost = bps x |change in effective exposure| (entry from flat
included); `funding='funded'` gives `exposure*(r - rf)`, `'collateral'` treats unit returns as
already excess; leverage capped; metrics: Sharpe, Sortino (proper downside deviation), max DD with
ruin handling (equity floored at 0), worst day, skew, excess kurtosis, turnover (x/yr), % invested.
`VolTarget` supports a slow EWMA vol estimate and a relative rebalance band.

## 3. Results

### 3a. Held-to-maturity 1-month variance swap (the cleanest P&L)

| Entries | n | mean strike vol | mean realised vol | mean payoff (vol pts) | win rate | worst payoff (vol pts) | skew | ex. kurtosis |
|---|---|---|---|---|---|---|---|---|
| 2006-2026 | 9143 | 19.5 | 15.4 | +2.7 | 86% | -252 | -12.5 | 237 |
| 2016-2026 | 2591 | 18.6 | 14.9 | +1.7 | 84% | -252 (entry 2020-02-14) | -10.6 | 140 |
| 2006-2015 | 6552 | 19.8 | 15.6 | +3.1 | 87% | -98 | -6.1 | 58 |

The variance risk premium is real (strike beats realised ~84-86% of the time; in 2016+ the strike
vol averages 3.7 points above realised vol, though the variance payoff averages only +1.7 vol points
per unit vega once the crash months are included) but one entry lost 252 vol points = 5.3x the stated capital. A constant-notional
position that survives Feb-2020 needs leverage <= 0.19 of that capital. Positions *cannot* be sized
off the average payoff.

Sharpe of the HTM strategy, entries >= 2016 (annualised from 12 cohorts/yr):

| Position | monthly non-overlap (mean of 21 phases; min..max) | ladder, Newey-West | ladder, naive (wrong) | 1 / 2 / 5 bp entry cost (monthly) |
|---|---|---|---|---|
| soft (original, causal) | **0.51** (-0.10 .. 1.21) | 0.37 | 1.21 | 0.51 / 0.51 / 0.51 |
| hard argmax | 0.51 (-0.10 .. 1.22) | 0.35 | 1.10 | |
| rank only, no regime gate | 0.32 (0.01 .. 0.94) | 0.31 | 1.15 | |
| constant short | **0.46** (0.11 .. 0.82) | 0.44 | 1.66 | 0.46 / 0.46 / 0.46 |

Only 124 monthly cohorts exist: the iid standard error of an annualised Sharpe is about 0.31 (Lo 2002;
too small for this fat-tailed payoff), so 0.51 is roughly -0.1 to 1.1, and the spread across phase
offsets (-0.10 to 1.21 for the same strategy) shows it. Bootstrap CI for constant short on the daily
MTM series below: -0.24 to 1.36.

### 3b. Validation of the MTM swap model against SVXY (daily, -1x-equivalent)

SVXY was -1x until 2018-02-27 and -0.5x after; I double post-change returns for like-for-like
(raw numbers also in the log). Slope calibrated on 2011-11..2015 only.

| Model | corr (2011-15) | corr (2016-26) | slope (OOS) | OOS R^2 | OOS tracking error (pre-2016 slope) |
|---|---|---|---|---|---|
| **MTM swap (carry+realised+MTM)** | 0.81 | 0.77 | 0.71 | 0.59 | 54% ann |
| MTM, single VIX node (no term structure) | 0.81 | 0.77 | 0.72 | 0.59 | 54% |
| **Legacy proxy (carry - realised)** | **0.21** | 0.37 | 1.21 | 0.14 | 72% |

Against -VXX (2018+): MTM 0.82, legacy 0.39. Correlation by year for the MTM model is 0.79-0.90 in every
year except 2018 (0.47, the Feb event); legacy is 0.03-0.57, near 0.1-0.2 in 2011-13 and 2025-26.
On SVXY's 20 worst days (mean -27.9%) the MTM swap, scaled by the pre-2016 slope, shows -36.0%; the
legacy proxy shows -9.1%, i.e. it captures about a third of the tail.

Annual P&L decomposition of the constant-maturity short swap, OOS mean (vol points per unit vega per
year): carry +111, realised -77, **MTM -15**, total **+19**, legacy (carry+realised) **+34**. The legacy
proxy overstates the average edge by ~1.8x and removes the loss on every crisis year (2008 total
-85.5 vs legacy -37.6; 2018 -34.8 vs +10.3; 2020 -42.2 vs +5.4).

Unlevered path comparison (non-excess):

| Window | Series | Ann. return | Ann. vol | Sharpe | Max DD | Worst day |
|---|---|---|---|---|---|---|
| 2011-11..2018-02-02 | SVXY | 64% | 62% | 1.04 | -68% | -26% |
| | MTM swap (slope-scaled) | 64% | 49% | 1.30 | -62% | -37% |
| | legacy (slope-scaled) | 81% | 13% | **6.3** | -17% | -9% |
| 2011-11..2026 | SVXY as traded | 29% | 55% | 0.53 | -95% | -83% |
| | MTM swap (slope-scaled) | 46% | 75% | 0.61 | -99.9% | -93% |
| | legacy (slope-scaled) | 76% | 22% | **3.4** | -77% | -33% |

Events (see `results/pnl/event_*.csv`):

* **2018-02-05/06.** SVXY -32.0% then -83.0% (-91% over 2018-02-01..09). The MTM swap (slope-scaled)
  shows -76% on the 5th and **+13.5% on the 6th** (VIX fell 37.3 -> 30.0); the legacy proxy shows
  -12% and +1%. The two-day loss is roughly right (SVXY -88%, MTM -73%, legacy -11%) but the *split* is wrong.
  Plausible cause (not verified against intraday data): VIX is the 4:15pm SPX-option print, while the ETF was marked off VIX futures that kept rising after
  the cash close, so SVXY lost most of its value on the 6th. Day-level correlation around that event
  is not meaningful for any VIX-index-based model.
* **2020-02-24..03-18.** VIX 15.6 -> 82.7. SVXY (-1x-equivalent) compounds to -87% (as traded at -0.5x:
  -61%). The MTM swap compounds to **-99.7%** (-93% on 03-16 alone, slope-scaled); legacy -73%. The swap model is
  harsher than the ETF here (OOS slope 0.71 vs 0.99 pre-2016) because VIX futures mean-revert and
  move less than the spot VIX the swap reprices. Treat the swap as the cleaner but more punitive
  benchmark; SVXY is what a retail account can actually hold.

**Where each proxy breaks.**
*Legacy:* (i) no MTM, so it is blind to the vol-of-vol that is the whole risk (corr 0.21 pre-2016,
Sharpe 3-6 for a constant short, tail capture 1/3); (ii) on cohorts, the +34 vs +19 edge gap. 
*MTM swap:* (i) a swap model, not an ETF - it overstates crisis losses vs SVXY and the 2018-02-05/06
timing is wrong by a day; (ii) assumes VIX is the swap strike (VIX^2 includes a jump/variance-premium
convexity wedge and a 30-day vs 21-trading-day mismatch); (iii) mid prices, no bid-offer; (iv) daily
re-striking of the whole book is not what a dealer swap book does; (v) 2008 data pre-dates VIX9D and
uses a two-node curve (irrelevant for results here, which are 2016+). 
*HTM:* (i) only 12 independent observations a year and a -252 vol-point tail point; (ii) says nothing
about drawdown *within* the month; (iii) the ladder series hides mark-to-market risk entirely.
*SVXY:* a real instrument but its short history has two rebalances (-1x to -0.5x), a roll-yield
return that is not the same thing as the variance premium, and it is the only dataset with
financing/ETF-structure noise.

### 3c. Vol-targeting: how much turnover does dampening remove?

Notebook 08 reported "vol-target scalar turnover 70.9 x/yr". The same quantity (sum of |change in
leverage| per year, OOS 2016+) reproduces at 78.6 x/yr with the engine on the original settings
(strategy-vol, 126d rolling, daily, cap 100), with a mean leverage of 83x. The leverage was pinned at
the cap most days; the "71x" only looks small relative to total exposure turnover of
**1,291 x/yr**. On a real instrument with a 3x cap, same table:

| Vol-target spec | legacy proxy: scalar turnover (x/yr) | mean lev | real SVXY, 3x cap: scalar turnover | mean lev | SVXY total exposure turnover, soft position |
|---|---|---|---|---|---|
| V0 original (strategy-vol, 126d, daily) | 78.6 | 83 | 1.64 | 0.72 | 8.28 |
| V1 unit-vol, 126d rolling, daily | 120 | 49 | 0.49 | 0.26 | 4.01 |
| V2 EWMA half-life 63d, daily | 70 | 37 | 0.39 | 0.24 | 3.68 |
| **V3 EWMA 63d + 25% band** (default "dampened") | 40.5 | 34 | **0.11** | 0.23 | 3.51 |
| V4 EWMA 126d + 25% band | 22 | 28 | 0.04 | 0.21 | 3.11 |

Dampening (V3) removes **66% of scalar turnover vs the same estimator re-levered daily (V1)** and 48%
vs the original V0 on the proxy; on SVXY it removes 78% vs V1 and 93% vs V0. It does not move total
turnover much (4.0 -> 3.5 x/yr): in the real setting the **signal itself** turns over 15 x/yr unscaled
(soft position, no vol target), which dominates. Costs are therefore not what kills the strategy.
Vol-targeting also did not help Sharpe (table below, C and D vs B): scaling by trailing vol lags vol
spikes, so leverage is still high on the day a short-vol position gets hit.

### 3d. The original strategy through the honest models (OOS 2016-2026, soft causal position)

Sharpe is of excess returns (`data.excess` logic: funded SVXY earns `r - rf`; swap on cash collateral
is excess by construction). "Net" columns charge bps of *change in effective exposure*.

| P&L model | Sharpe gross | net 1 bp | net 2 bp | net 5 bp | max DD | worst day | skew | turnover x/yr |
|---|---|---|---|---|---|---|---|---|
| A legacy proxy, vol-tgt 10%, cap 100 (as published, causal) | **2.72** | n/a | n/a | n/a | -19.6% | -16.1% | -10.6 | 1,291 |
| B real SVXY, unlevered | **0.02** | 0.01 | 0.01 | -0.01 | -69.8% | -31.5% | -5.6 | 15.1 |
| C real SVXY, vol-tgt 10%, dampened, cap 3x | -0.08 | -0.08 | -0.09 | -0.11 | -21.4% | -8.3% | -7.5 | 3.5 |
| D real SVXY, vol-tgt 10%, daily re-lever, cap 3x | -0.14 | -0.15 | -0.16 | -0.17 | -25.1% | -7.1% | -5.0 | 4.0 |
| E MTM swap, 1x unlevered | **0.03** | 0.03 | 0.02 | 0.01 | -89.9% | -75.8% | -13.6 | 15.1 |
| F MTM swap, vol-tgt 10%, dampened, cap 3x | -0.19 | -0.19 | -0.19 | -0.20 | -38.7% | -25.8% | -26.4 | 2.7 |
| G legacy carry-realised on swap scale, 1x | 3.26 | 3.25 | 3.23 | 3.18 | -15.4% | -11.6% | -9.3 | 15.1 |
| H legacy on swap scale, vol-tgt 10%, cap 3x | 2.18 | 2.17 | 2.15 | 2.11 | -19.0% | -15.0% | -17.8 | 10.4 |
| HTM variance swap, monthly non-overlap (mean of 21 phases) | 0.51 | 0.51 | 0.51 | 0.51 | n/a (see 3a) | n/a | n/a | 12 entries/yr |
| HTM ladder, Newey-West | 0.37 | 0.37 | 0.37 | 0.37 | | | | |

Rows G and H are the point of the exercise: the *same* legacy P&L on a proper capital scale and a 3x
cap still gives 2.2-3.3, so the inflation is the proxy (no MTM), not leverage, costs, or the regime
logic. With the MTM term (E) the same positions earn 0.03.
The 3x cap never binds for B, C, D, E, F (a 10% vol target on a 25-80% vol instrument means mean
leverage 0.2-0.3x; peak 3x only in quiet spells); it binds on A, which is why A's 100x cap looked
harmless: the proxy's unit vol was ~0.1% a year.

Baselines and alternatives (same period, excess):

| | Sharpe | Max DD | Worst day |
|---|---|---|---|
| Constant short, SVXY (= SVXY buy and hold) | 0.34 | -95.4% | -83.0% |
| Constant short, MTM swap 1x | 0.48 (CI -0.24 .. 1.36) | -99.9% (ruin in 03/2020) | -93.9% |
| Constant short, MTM swap vol-tgt dampened 3x | 0.20 | -44.5% | -26.2% |
| Constant short, HTM monthly / ladder NW | 0.46 / 0.44 | | |
| Constant short, legacy proxy (reference) | 2.90 (A) / 3.00 (G) | | |
| SPY buy and hold | **0.76** | -33.8% | -10.9% |
| Soft causal position, SVXY / MTM swap / HTM monthly | 0.02 / 0.03 / 0.51 | | |
| Hard argmax, SVXY / MTM swap / HTM monthly | 0.02 / 0.02 / 0.51 | | |
| Rank only (no regime gate), SVXY / MTM swap / HTM monthly | 0.06 / 0.29 / 0.32 | | |

Block-bootstrap 95% CI (21-day blocks) on the soft strategy: SVXY unlevered 0.02 (-0.54 .. 0.78);
SVXY dampened vol-tgt -0.08 (-0.67 .. 0.71); MTM swap 0.03 (-0.47 .. 0.80). Sub-periods (SVXY
unlevered): 2016-19 0.18, 2020-22 -0.63, 2023-26 0.04; MTM swap 0.06, -0.52, 0.33.

Why HTM (0.5) and daily MTM (0.0) disagree for the gated position, **post hoc and labelled as such**:
the HTM ladder only lets the signal act on 1/21 of the book per day. Feeding the daily models a
position equal to the trailing 21-day mean of the soft position lifts the Sharpe to 0.14 (SVXY) and
0.29 (MTM swap) and cuts turnover from 15 to 3 x/yr. So a large part of the gap is intra-month
signal whipsaw (de-risking after a spike, re-risking before the next one), not a different
estimate of the premium.

## 4. What is a defensible Sharpe?

* **Not 3.08, 2.7 or 2.9.** Those numbers describe a proxy with no MTM, a leverage cap of 100x and a
  smoothed unit series. A constant always-short position gets 2.9 on it.
* On the realistic models, the original causal strategy scores **0.0 to 0.5** before costs
  (SVXY 0.02, MTM swap 0.03, 21d-slowed signal 0.14-0.29, HTM ladder 0.37 NW, HTM monthly 0.51), and
  every 95% CI I computed includes zero. The point estimate I would defend is **about 0.2, with a
  plausible range of 0 to 0.5, i.e. not statistically distinguishable from zero**.
* It does **not beat** the baselines: constant short on the same instruments earns 0.34 (SVXY),
  0.44-0.48 (swap), and SPY buy and hold earns 0.76 with a -34% drawdown against -70% to -90%.
  The regime/rank signal subtracts about 0.3 of Sharpe from SVXY when traded daily and adds ~0.05
  to the HTM ladder.
* Transaction costs are not the issue (1/2/5 bps cost 0.005/0.011/0.028 Sharpe on unlevered SVXY
  because the soft position turns over 15 x/yr); the issue is that after a correct MTM there is little
  alpha, and the tails (skew -5 to -14, worst day -31% to -76% unlevered) are large.
* The honest summary of the *premium itself* (not the strategy) is the constant HTM short:
  ~0.45 Sharpe with a -252 vol-point worst month, against 0.34 for SVXY buy and hold: a real but
  modest premium that pays in ~85% of months and crashes in a few.

## 5. Caveats

* 10 years of OOS data and effectively two stress events (2018-02, 2020-03) that dominate every
  statistic; the CIs above are wide and the block bootstrap with 21-day blocks understates tail
  clustering.
* VIX is not the exact swap strike; no bid-offer on swaps (I have not modelled it; an assumed
  0.3-0.5 vol point per entry, which I have not verified against dealer quotes, would be roughly a
  fifth to a third of the +1.7 vol-point 2016+ mean payoff); bps cost on SVXY ignores borrow and
  market impact.
* The capital convention is a calibration choice. `perf_stats` reports ruin (equity floored at
  zero); the unlevered 1x swap books a total loss in 03/2020, which the real ETF did not suffer
  (-61% at -0.5x). Sharpe is unaffected; drawdown/leverage numbers depend on it.
* SVXY is -1x then -0.5x; rows labelled "as traded" use the actual returns. The VXX file is only 2018+
  (first two values stale).
* VIX9D/VIX3M/VIX6M add nothing to the 21-day MTM (tested: single-node version tracks SVXY
  identically). I report that rather than claiming the term structure matters.
* The signals file ends 2026-05-22, while SVXY/VXX extend to 2026-10; evaluation stops at the signal end.
* Vol-target default (EWMA 63d, 25% band) was chosen before looking at results, but I report every
  alternative I ran. The MA21 diagnostic and the SVXY/MTM disagreement analysis are post hoc.

## 6. Trial count

113 evaluations (position x P&L model x vol-target/cost configuration), 47 distinct
(position, model) pairs: 4 positions x 8 P&L models x {0,1,2,5} bp (costs only for 2 positions),
5 vol-target specs x 2 instruments x 2 positions for turnover, 3 validation proxies, 4 HTM positions,
3 baselines/diagnostics. No parameter was tuned on 2016+. The one scale constant (capital = 48 vol
points) was calibrated on 2011-2015 data (exploratory regressions along the way briefly used the full
sample; the final value is the pre-2016 one). The deflated-Sharpe correction for the best row would be
irrelevant here since the best realistic row is below 0.6 with CI crossing zero.

## 7. What an interviewer might push on

* **"Your swap model says -99.7% in March 2020 and SVXY says -61%. Which is right?"** Neither is
  "right": SVXY was at -0.5x and holds futures that mean-revert; the swap model is an unlevered
  fixed-vega-notional position priced off spot VIX. Which is why I calibrate the *scale* on pre-2016
  data, and report SVXY as the tradable truth and the swap as a benchmark for the premium.
* **"The ladder Sharpe printed 1.2-1.7 and your NW number is 0.4. Why?"** Overlapping 21-day payoffs;
  the daily series' autocorrelation makes sqrt(252) scaling overstate by ~sqrt(21). Test included.
* **"VIX is not a variance swap strike."** Agreed; it is within a few percent for 1-month, includes
  tail strikes and a 30 vs 21-trading-day mismatch; I did not correct it. It biases the premium by
  a roughly constant offset, not the shape of the tail.
* **"Why does vol-targeting make Sharpe worse?"** It levers up after calm periods, which is exactly
  when short-vol is about to be hit; a trailing estimate cannot see a vol spike. Skew and kurtosis
  (F: skew -26, kurtosis 936) are worse after targeting.
* **"You changed the capital from 20 to 48 after seeing results."** I changed the scale from a
  first guess after a regression on SVXY; the final value is from pre-2016 data and only affects
  drawdown and leverage-cap numbers, not Sharpe.
* **"The HTM ladder gives 0.5 and the daily models 0.0 for the same signal. Isn't that
  cherry-picking?"** Reported both; the diagnostic shows it is signal speed. Honest range is 0 to 0.5.
* **"Is 0.2 a number you believe?"** No: it is a midpoint of estimates with CI crossing zero. What I
  believe is that the original 3.08 was 95%+ proxy error and the signal adds little over constant short.
* **"Why trust 1 to 5 bps?"** SVXY is liquid; 5 bps is generous at small size. Swap spreads are far
  larger and not modelled; for a swap-based implementation costs, not bps, would be the story.
