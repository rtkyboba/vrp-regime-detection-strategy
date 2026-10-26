"""Combine a regime signal and a forecast signal into one position-sizing signal.

Both inputs are mapped onto a common "how much short-vol am I allowed" scale a in [0, 1]:

* regime:  a_h = 1 - stress_score        (calm -> 1, crisis -> 0)
* RNN/ridge forecast s in [-1, 1]:  a_r = (s + 1) / 2

Combination rules (all causal, all fixed before looking at results):

* ``mean``     a = w a_h + (1-w) a_r              diversifies two noisy estimates
* ``product``  a = a_h * a_r                      both must agree to size up (AND-gate)
* ``veto``     a = a_r * (1 - 1[stress_flag])     forecast sizes, regime can only veto
* ``ic``       a = sum_i w_i a_i with w_i proportional to the component's trailing rank-IC with
               the held-to-maturity variance payoff, using only payoffs that have *matured*
               by t (entry date <= t - h). If no component has positive trailing IC the
               weights fall back to equal. This is the "weight by demonstrated signal
               strength" rule.

The point of the module is the evaluation harness around it (scripts/combine_signals.py):
a combination only counts if it beats a constant exposure of the same average size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

H = 21


def calm_score(stress_score: pd.Series) -> pd.Series:
    return (1.0 - stress_score).clip(0.0, 1.0)


def forecast_strength(signal: pd.Series) -> pd.Series:
    """Map a [-1, 1] forecast to [0, 1]."""
    return ((signal + 1.0) / 2.0).clip(0.0, 1.0)


def combine_mean(a: pd.Series, b: pd.Series, w: float = 0.5) -> pd.Series:
    return (w * a + (1.0 - w) * b).clip(0.0, 1.0)


def combine_product(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a * b).clip(0.0, 1.0)


def combine_veto(strength: pd.Series, stress_flag: pd.Series) -> pd.Series:
    return (strength * (1.0 - stress_flag.astype(float))).clip(0.0, 1.0)


def trailing_ic_weights(components: dict, payoff: pd.Series, h: int = H, window: int = 504,
                        min_pairs: int = 126) -> pd.DataFrame:
    """Causal IC-proportional weights.

    components: name -> Series (signal dated t). payoff: held-to-maturity payoff by ENTRY
    date (value at entry date i is only known at index[i + h]). At date t only entries
    i <= t - h are used, the most recent ``window`` of them. Weights are max(IC, 0)
    normalised; equal weights if every IC <= 0 or fewer than ``min_pairs`` matured pairs.
    """
    names = list(components)
    idx = components[names[0]].index
    sig = pd.DataFrame({n: components[n].reindex(idx) for n in names})
    pay = payoff.reindex(idx)
    k = len(names)
    out = np.full((len(idx), k), 1.0 / k)
    sv, pv = sig.to_numpy(), pay.to_numpy()
    for t in range(len(idx)):
        end = t - h + 1                      # exclusive; last matured entry index is t - h
        if end < min_pairs:
            continue
        lo = max(0, end - window)
        y = pv[lo:end]
        ics = np.zeros(k)
        for j in range(k):
            x = sv[lo:end, j]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= min_pairs:
                ic = spearmanr(x[ok], y[ok])[0]
                ics[j] = 0.0 if np.isnan(ic) else ic
        pos = np.clip(ics, 0.0, None)
        if pos.sum() > 0:
            out[t] = pos / pos.sum()
    return pd.DataFrame(out, index=idx, columns=names)


def combine_ic(components: dict, payoff: pd.Series, h: int = H, window: int = 504) -> tuple:
    """Return (combined signal in [0,1], weights DataFrame)."""
    w = trailing_ic_weights(components, payoff, h, window)
    comb = sum(w[n] * components[n].reindex(w.index) for n in components)
    return comb.clip(0.0, 1.0), w


def block_bootstrap_ic(x: pd.Series, y: pd.Series, block: int = 21, n_boot: int = 1000,
                       seed: int = 0) -> dict:
    """Spearman IC with a moving-block bootstrap 95% CI (targets overlap by h days, so the
    plain iid standard error is far too small)."""
    d = pd.concat([x, y], axis=1).dropna()
    xv, yv = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    n = len(d)
    ic = float(spearmanr(xv, yv)[0])
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), nb)
        ii = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        stats[b] = spearmanr(xv[ii], yv[ii])[0]
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return {"ic": ic, "ci_low": float(lo), "ci_high": float(hi), "n": n}
