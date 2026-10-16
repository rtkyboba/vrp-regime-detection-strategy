# VRP Strategy: a short-volatility backtest, and an audit of its Sharpe ratio

A systematic short-volatility strategy on SPX/VIX that sells the variance risk premium (VRP),
sized by an HMM regime classifier and a rolling percentile rank of the premium. The first version
reported an out-of-sample **Sharpe of 3.08**. This repository is the original build **plus a
rigorous audit showing that number was inflated**, what the defensible number is (about 0.2,
statistically indistinguishable from zero), and a set of causal, walk-forward experiments
(regime-detection methods, an RNN signal, alternative volatility strategies) judged by the same
honest standard.

## Headline: where the 3.08 went

| Variant (2016 to 2026, vol-targeted to 10% unless noted) | Sharpe |
|---|---|
| Original: HMM decoded with Viterbi over the whole test block | 3.08 |
| Causal: forward-filtered regime probabilities only | 2.74 |
| **Constant short variance, no signal at all** (same legacy P&L proxy) | **2.89** |
| Same causal positions on **real SVXY returns**, unlevered | **0.02** (95% CI -0.55 to 0.70) |
| Held-to-maturity 21-day variance swap, regime strategy | about 0.4 to 0.5 (constant short 0.44 to 0.48) |
| SVXY buy and hold / SPY buy and hold | 0.41 / 0.79 |

1. **Look-ahead.** `model.predict()` over the test sequence is Viterbi decoding, so the regime label
   on day *t* depended on data after *t*. Causal filtering (`regimes.filtered_proba`) costs about 0.3 Sharpe.
2. **The P&L proxy.** `iv_daily(t-1) - rv_daily(t)` collects implied variance against a single day's
   squared return and omits the mark-to-market change in the variance strike, which is where
   short-vol tail losses live. It correlates only 0.21 with real SVXY daily returns, and a position
   that is *always short* already scores 2.9 on it. The regime logic added nothing; the proxy did the work.
3. **Leverage.** Vol-targeting up to 100x turned a smooth-looking series into a 10%-vol "Sharpe 3" stream.
4. **Costs.** 3 bps on exposure turnover turned +32% into -8% (notebook 08), and realistic costs
   are not the main problem once the P&L is honest.
5. **Selection.** The statistical tests show the original Sharpe is not a selection artefact (the
   deflated Sharpe stays above 0.97 even at 10,000 trials): the *measurement* was wrong, not just
   the search. The notebook-06 permutation null was also flawed (an iid shuffle has mean Sharpe 1.75);
   under block and circular-shift nulls the original is not significant at 5%.

**Defensible Sharpe for the strategy: about 0.2, plausibly 0 to 0.5. Every 95% interval includes zero,
and it does not beat constant exposure or SPY buy and hold.**

## What the follow-up research found

All of it follows [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md): strictly causal signals, walk-forward
refits, nothing tuned on the 2016+ test period, every trial counted, costs and excess-of-risk-free returns,
tested against the same-size constant exposure.

| Report | Question | Verdict |
|---|---|---|
| [01 P&L realism](reports/01_pnl_realism.md) | What is the honest P&L of short variance? | Held-to-maturity swap and a term-structure mark-to-market model (0.77 correlation with SVXY vs 0.37 for the legacy proxy). Defensible Sharpe about 0.2. |
| [02 Regime detection](reports/02_regime_detection.md) | 34 causal methods: HMM n=2..5, GMM, Markov-switching, jump model, rules, CUSUM, GJR-GARCH. How many states? | The original "3 states" was not a selection result (OOS log-likelihood says 3 or 4 depending on features). No method beats raw VIX at forecasting vol, and none beats buy and hold after a 124-variant reality check (adjusted p = 0.78). |
| [03 RNN signal](reports/03_rnn_signal.md) | Does a walk-forward GRU/LSTM forecast whether to be short vol? | No. IC -0.08 (95% CI -0.23 to 0.07), indistinguishable from ridge, HAR and always-short. |
| [04 Statistical validation](reports/04_statistical_validation.md) | Is the Sharpe real? Deflated Sharpe, block bootstrap, PBO, reality check, better nulls. | The proxy Sharpe passes every sampling test; the problem was the P&L definition. Regime logic adds nothing over constant short (p = 0.8 to 0.9). |
| [05 Alternative vol strategies](reports/05_alternative_vol_strategies.md) | Vol-managed SPY, term-structure carry, buy-the-spike, VVIX/SKEW filters, tail hedges. | Nothing beats SPY with statistical confidence. Daily vol-managed SPY was the only family with a stable sign (+0.19 Sharpe, CI -0.00 to +0.37). |
| [06 Combined signal](reports/06_combined_signal.md) | Do HMM and RNN signals blended by signal strength beat a constant exposure of the same size? | No: all 12 variants within -0.08 to +0.10 Sharpe of the constant (p 0.31 to 0.87). |

The honest conclusion is a null result, reached carefully: this volatility-timing signal set, on
this data, has no demonstrable edge once the P&L is realistic and the evaluation is causal.

## Layout

```
src/vrp_strategy/   data.py (loaders, split), regimes.py (causal HMM filter), regime_models.py (34-method
                    walk-forward comparison), pnl.py (honest P&L + generic backtester), rnn_signal.py,
                    stats_tests.py (DSR, PSR, bootstrap, PBO, reality check), alt_strategies.py,
                    combiner.py, backtest.py / metrics.py (original helpers)
scripts/            one command reproduces each report's numbers (audit_sharpe.py, pnl_realism.py, ...)
tests/              pytest suite; every model has a "corrupt the future, past output must not change" test
reports/            the six reports above, each with method, tables, caveats, trial count
notebooks/          the original 9-notebook build (01-09), kept as the starting point
docs/               EVAL_PROTOCOL.md: the rules every experiment follows
data/raw/           SPX, VIX, VIX3M plus SVXY, VXX, VIX9D, VIX6M, VVIX, SKEW, SPY, IRX
```

## The original build (notebooks 01 to 09)

| Notebook | What it does |
|---|---|
| `01_data_pipeline` | Merges SPX, VIX, VIX3M |
| `02_signal_construction` | VRP, term-structure slope, realised vol |
| `03_regime_hmm` | 3-state Gaussian HMM (calm / stressed / crisis), trained on 2006-2015 |
| `04_position_sizing` | Position = regime multiplier x rolling VRP percentile rank |
| `05_backtest` | Legacy variance-swap proxy P&L, vol-targeting, metrics (**headline now superseded, see above**) |
| `06_permutation_testing` | Signal-shuffle test (**null flawed, see report 04**) |
| `07_position_sizing_comparison` | Regime sizing vs rolling half-Kelly |
| `08_transaction_costs` | Turnover-based costs; found the leverage-cycling problem |
| `09_regime_multiplier_sweep` | Parameter sweep with sub-period and walk-forward checks |

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pytest -q                 # 130+ tests
.venv/Scripts/python scripts/audit_sharpe.py      # reproduces 3.08 and the decomposition
```

Each `scripts/*.py` regenerates the tables in its matching report under `results/`.

## Limitations

- No real options or variance-swap quotes: P&L comes from SVXY (a real traded instrument, but
  -1x then -0.5x leverage with its own roll mechanics) and from a variance-swap model priced off
  VIX. The variance-swap capital convention is a calibration constant, not a margin model.
- One decisive stress event (February 2018) dominates several comparisons, and each method sees
  only about 15 independent stress episodes, so the null results are statements of *not
  established*, not proofs of zero (see each report's power discussion).
- Data are daily closes from yfinance; SVXY history starts October 2011.
