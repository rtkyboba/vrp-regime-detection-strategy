# Evaluation protocol (all experiments must follow this)

This project's original headline (Sharpe 3.08, 2016-2026) did not survive a first audit.
The rules below exist so no new number can be inflated the same way.

## What the audit found so far (reproduce with `scripts/audit_sharpe.py`)

| Variant (2016-2026, vol-targeted to 10%, before costs) | Sharpe |
|---|---|
| Original: Viterbi `predict()` over the whole test sequence | 3.08 |
| Causal: forward-filtered regime posterior, hard argmax | 2.74 |
| Causal, soft expected multiplier | 2.81 |
| No regime gate (VRP rank only) | 1.73 |
| **Constant short variance, no signal at all** | **2.89** |
| Same causal positions applied to **real SVXY returns**, unlevered | **0.05 to 0.17** |
| SVXY buy and hold / SPX buy and hold | 0.41 / 0.79 |

Takeaways every agent must internalise:
1. `model.predict()` on a test block is **non-causal** (Viterbi uses future observations).
   Use `vrp_strategy.regimes.filtered_proba` (forward algorithm) instead.
2. The proxy P&L `iv_daily(t-1) - rv_daily(t)` has only ~0.21 daily correlation with SVXY.
   It omits the **mark-to-market** change in the variance strike (VIX^2 moves), which is where
   short-vol tail losses live. A constant short position already scores ~2.9 on it, so
   the proxy itself is the largest source of inflation, not the regime logic.
3. The 100x leverage cap + trailing vol-target scales an artificially smooth series
   into a 10%-vol "Sharpe 3" stream. The 3bps cost test already showed it is untradeable.

## Rules

1. **Causality.** A feature, regime, forecast or position dated `t` may use data through the
   close of `t` only. It earns the return of `t+1` (lag positions by one day explicitly).
   Every model that is *fitted* must be fitted on data strictly before the period it is
   evaluated on (expanding or rolling walk-forward refit; document refit frequency).
   Add a unit test that corrupting future data does not change past outputs.
2. **No tuning on the test period.** `vrp_strategy.data.SPLIT = 2016-01-01`. Hyperparameters
   (number of states, features, lookbacks, multipliers, network size, thresholds) must be
   chosen on pre-2016 data or inside a walk-forward loop. If you look at 2016+ results and
   then change something, that variant is "peeked" and must be labelled so.
3. **Count your trials.** Keep a list of *every* variant you evaluated (including failures).
   Report the count; the statistical-validation module deflates the best Sharpe by it.
4. **Honest P&L, in order of preference:**
   a. Real traded instrument returns: `data.asset_returns("svxy")` (2011-10 on; -1x VIX
      short-term futures until Feb 2018, -0.5x after) and `vxx` (2018 on), `spy`.
   b. Held-to-maturity variance swap P&L: enter at `t` at strike `VIX_t^2`, pay realised
      variance over the next 21 trading days. No MTM approximation needed; the only
      error is that VIX is a model-free approximation to the strike.
   c. The legacy `iv_daily - rv_daily` proxy: allowed **only** as a labelled comparison.
5. **Report net of costs and in excess of risk-free.** Use `data.excess()`. Always give a
   gross and a net number (state bps assumed, charge on the change in *exposure*).
   Vol-target leverage must be capped at something a broker would allow (<= 3x of the
   instrument) unless the instrument is a swap, in which case say so.
6. **Report full metrics:** Sharpe (excess), Sortino, max drawdown, worst day, skew,
   excess kurtosis, turnover, % days invested, plus a 95% block-bootstrap CI on Sharpe once
   `stats_tests` exists. A Sharpe without its drawdown and worst day is not a result.
7. **Baselines to beat in every comparison:** buy-and-hold of the same instrument,
   constant-exposure, and the simplest non-ML rule (e.g. VIX/VIX3M > 1 -> flat).
8. **File ownership.** Only edit the files your task owns (listed in your brief). Do not
   edit `requirements.txt`, `README.md`, `data/`, existing notebooks or other agents' modules.
   Tell the coordinator what extra dependencies you used.

## Layout

- `src/vrp_strategy/<module>.py` - reusable code with docstrings explaining the *why*.
- `tests/test_<module>.py` - pytest, runnable offline from the repo root.
- `scripts/<name>.py` - one command reproduces every number in your report.
- `reports/<NN>_<name>.md` - findings: question, method, table of results, caveats, trial
  count, and a short "what an interviewer might push on" section.
- `results/<name>/*.csv|png` - generated outputs (small).

Run Python with `C:\Users\ryanr\vrp_strategy\.venv\Scripts\python.exe` (the system and
conda envs on this machine have a broken NumPy BLAS). From a worktree, set
`PYTHONPATH=<your worktree>\src`.
