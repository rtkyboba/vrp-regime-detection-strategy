"""Causal regime-detection models behind one interface, plus a walk-forward driver.

Why this module exists
----------------------
The original notebook used a 3-state Gaussian HMM trained once on 2006-2015 and
decoded with Viterbi over the test block (non-causal).  Here every method is
forced through the same contract so they can be compared fairly:

    model = factory()          # fresh, unfitted
    model.fit(F_train)         # F_train: DataFrame of causal features, rows <= t0
    P = model.filter(F)        # (T, K) probabilities; row t uses F.iloc[:t+1] ONLY
    s = model.stress_score(P)  # in [0, 1]; P(highest-vol regime) or equivalent

Conventions that matter
-----------------------
* Columns of P are sorted by increasing volatility (mean of `sort_col` under each
  state, computed from the *fitted* model).  Without this, EM re-labels states at
  every refit (label switching) and a walk-forward concatenation is meaningless.
* `filter` is the forward (filtering) recursion or an equivalent online rule; it
  never smooths.  tests/test_regime_models.py corrupts the future and checks the
  past is unchanged for every method.
* `walk_forward_regimes` refits on an expanding window every `refit_every` days.
  Probabilities for block [i, i+refit_every) come from a model fitted on rows < i
  and filtered over rows <= t, so the fit never sees the evaluated day.
* Rule/threshold/CUSUM models have nothing to fit (or fit trivially); they go
  through the same driver so the comparison is like-for-like.

A feature dated t uses data through the close of t.  A position that trades on
the regime must still be lagged one day by the caller.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from vrp_strategy.data import load_raw
from vrp_strategy.regimes import forward_filter

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

# name -> (columns, sort_col).  sort_col orders states by volatility.
FEATURE_SETS = {
    "orig": (["vrp_vol", "slope", "rv_30"], "rv_30"),
    "core": (["log_vix", "log_rv21", "slope"], "log_vix"),
    "core_vvix": (["log_vix", "log_rv21", "slope", "log_vvix"], "log_vix"),
    "core_skew": (["log_vix", "log_rv21", "slope", "skew"], "log_vix"),
    "core_vix6m": (["log_vix", "log_rv21", "slope", "slope6"], "log_vix"),
    "core_vix9d": (["log_vix", "log_rv21", "slope", "ts9"], "log_vix"),
    "core_all": (["log_vix", "log_rv21", "slope", "log_vvix", "skew", "slope6"], "log_vix"),
}


def features_from_closes(spx, vix, v3, aux=None, ffill_limit: int = 5) -> pd.DataFrame:
    """Pure function: closes -> feature table (also used by the unit tests on
    synthetic data).  `aux` maps 'vvix','skew','vix6m','vix9d' to Series."""
    idx = spx.index.intersection(vix.index).intersection(v3.index)
    F = pd.DataFrame({"spx": spx.reindex(idx), "vix": vix.reindex(idx), "vix3m": v3.reindex(idx)})
    lr = np.log(F["spx"]).diff()
    F["ret"] = lr * 100.0
    F["log_vix"] = np.log(F["vix"])
    F["rv_30"] = lr.rolling(30).std() * np.sqrt(252)
    F["log_rv21"] = np.log(np.sqrt((lr ** 2).rolling(21).mean() * 252) + 1e-4)
    F["vrp_vol"] = F["vix"] / 100.0 - F["rv_30"]
    F["slope"] = (F["vix3m"] - F["vix"]) / F["vix"]
    aux = aux or {}
    for col in ["vvix", "skew", "vix6m", "vix9d"]:
        if col in aux:
            x = aux[col]
            F[col] = x.reindex(F.index.union(x.index)).ffill(limit=ffill_limit).reindex(F.index)
        else:
            F[col] = np.nan
    F["log_vvix"] = np.log(F["vvix"])
    F["slope6"] = (F["vix6m"] - F["vix3m"]) / F["vix3m"]
    F["ts9"] = np.log(F["vix9d"] / F["vix"])
    return F


def build_features(ffill_limit: int = 5) -> pd.DataFrame:
    """Daily causal feature table on the SPX/VIX/VIX3M calendar (2006-07 on).

    Columns: spx, vix, vix3m, ret (log return, %), log_vix, rv_30 (annualised,
    30d, as in the original notebook), log_rv21, vrp_vol, slope
    ((vix3m-vix)/vix as originally defined), log_vvix, skew, slope6
    ((vix6m-vix3m)/vix3m), ts9 (log(vix9d/vix)), vix9d, vix6m.

    Every column at t is a function of closes <= t.  Auxiliary series (vvix,
    skew, vix6m, vix9d) are forward-filled for at most `ffill_limit` days (a
    stale value is causal; a back-filled one would not be) and are NaN before
    their first date: vvix 2007-01, vix6m 2008-01, vix9d 2011-01.
    """
    aux = {n: load_raw(n) for n in ("vvix", "skew", "vix6m", "vix9d")}
    return features_from_closes(load_raw("spx"), load_raw("vix"), load_raw("vix3m"), aux, ffill_limit)


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------

class RegimeModel:
    """Contract described in the module docstring."""

    name = "base"
    n_states = 2
    hard_threshold = 0.5  # stress_score > this  => "stress" flag

    def __init__(self, cols=None, sort_col=None):
        self.cols = list(cols) if cols is not None else None
        self.sort_col = sort_col

    # -- helpers --
    def _X(self, F: pd.DataFrame) -> np.ndarray:
        cols = self.cols if self.cols is not None else list(F.columns)
        return F[cols].to_numpy(dtype=float)

    def _sort_idx(self) -> int:
        cols = self.cols
        return cols.index(self.sort_col) if self.sort_col in cols else 0

    # -- API --
    def fit(self, F: pd.DataFrame):  # pragma: no cover - abstract
        raise NotImplementedError

    def filter(self, F: pd.DataFrame, return_ll: bool = False):  # pragma: no cover
        raise NotImplementedError

    def stress_score(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P)[:, -1]

    def describe(self) -> dict:
        """Per-refit diagnostics used for refit-stability tables."""
        return {}


# --------------------------------------------------------------------------
# Gaussian HMM and Gaussian mixture
# --------------------------------------------------------------------------

class HMMRegime(RegimeModel):
    """Gaussian HMM (hmmlearn) with causal forward filtering and vol-sorted states.

    Best of `n_init` EM restarts by training log-likelihood (EM on a 3-5 state
    full-covariance model has many local optima).  Features are standardised
    with the training window only.
    """

    def __init__(self, n_states=3, cols=None, sort_col=None, n_init=3, n_iter=300,
                 seed=42, covariance_type="full", name=None):
        super().__init__(cols, sort_col)
        self.n_states, self.n_init, self.n_iter = n_states, n_init, n_iter
        self.seed, self.covariance_type = seed, covariance_type
        self.name = name or f"hmm{n_states}"

    def fit(self, F):
        from hmmlearn import hmm
        X = self._X(F)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        best, best_ll = None, -np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in range(self.n_init):
                m = hmm.GaussianHMM(self.n_states, covariance_type=self.covariance_type,
                                    n_iter=self.n_iter, tol=1e-3, random_state=self.seed + k)
                try:
                    m.fit(Xs)
                    ll = m.score(Xs)
                except Exception:
                    continue
                if np.isfinite(ll) and ll > best_ll and np.isfinite(m.transmat_).all():
                    best, best_ll = m, ll
        if best is None:
            raise RuntimeError("all HMM restarts failed")
        self.model_, self.train_ll_, self.n_train_ = best, best_ll, len(Xs)
        means_raw = best.means_[:, self._sort_idx()] * self.sd_[self._sort_idx()] + self.mu_[self._sort_idx()]
        self.order_ = np.argsort(means_raw)
        return self

    @property
    def n_params(self):
        d, K = len(self.mu_), self.n_states
        cov = K * d * (d + 1) // 2 if self.covariance_type == "full" else K * d
        return (K - 1) + K * (K - 1) + K * d + cov

    def bic_aic(self):
        k, n = self.n_params, self.n_train_
        return {"bic": -2 * self.train_ll_ + k * np.log(n), "aic": -2 * self.train_ll_ + 2 * k}

    def filter(self, F, return_ll=False):
        Xs = (self._X(F) - self.mu_) / self.sd_
        m = self.model_
        logb = m._compute_log_likelihood(Xs)
        P, ll = forward_filter(logb, m.transmat_, m.startprob_)
        P = P[:, self.order_]
        return (P, ll) if return_ll else P

    def describe(self):
        m, o = self.model_, self.order_
        A = m.transmat_[np.ix_(o, o)]
        j = self._sort_idx()
        means = m.means_[o, j] * self.sd_[j] + self.mu_[j]
        d = {f"mean_{i}": means[i] for i in range(self.n_states)}
        d.update({f"pstay_{i}": A[i, i] for i in range(self.n_states)})
        return d


class GMMRegime(RegimeModel):
    """Gaussian mixture with NO temporal structure: P(state | x_t) only.  The
    comparison with the HMM shows what the transition matrix (persistence) buys."""

    def __init__(self, n_states=3, cols=None, sort_col=None, n_init=5, seed=42, name=None):
        super().__init__(cols, sort_col)
        self.n_states, self.n_init, self.seed = n_states, n_init, seed
        self.name = name or f"gmm{n_states}"

    def fit(self, F):
        from sklearn.mixture import GaussianMixture
        X = self._X(F)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        self.model_ = GaussianMixture(self.n_states, covariance_type="full", n_init=self.n_init,
                                      reg_covar=1e-4, random_state=self.seed).fit(Xs)
        self.train_ll_ = float(self.model_.score(Xs) * len(Xs))
        self._bic, self._aic = float(self.model_.bic(Xs)), float(self.model_.aic(Xs))
        self.n_train_ = len(Xs)
        j = self._sort_idx()
        self.order_ = np.argsort(self.model_.means_[:, j] * self.sd_[j] + self.mu_[j])
        return self

    def bic_aic(self):
        return {"bic": self._bic, "aic": self._aic}

    def filter(self, F, return_ll=False):
        Xs = (self._X(F) - self.mu_) / self.sd_
        P = self.model_.predict_proba(Xs)[:, self.order_]
        if return_ll:
            return P, self.model_.score_samples(Xs)
        return P

    def describe(self):
        j = self._sort_idx()
        means = self.model_.means_[self.order_, j] * self.sd_[j] + self.mu_[j]
        return {f"mean_{i}": means[i] for i in range(self.n_states)}


# --------------------------------------------------------------------------
# Markov-switching regression / AR (statsmodels)
# --------------------------------------------------------------------------

class MarkovSwitchingRegime(RegimeModel):
    """Markov-switching mean/variance model for ONE series (default log VIX).

    ar_order=0 -> MarkovRegression (regime-dependent level and variance);
    ar_order>=1 -> MarkovAutoregression with a common AR coefficient, which
    separates "VIX is high" from "VIX is persistent".  Filtering = applying the
    fitted parameters to the longer series with statsmodels' `filter`, whose
    filtered_marginal_probabilities at t use endog[:t+1] only.
    """

    def __init__(self, k=2, endog_col="log_vix", ar_order=0, seed=0, name=None):
        super().__init__([endog_col], endog_col)
        self.n_states, self.endog_col, self.ar_order, self.seed = k, endog_col, ar_order, seed
        self.name = name or (f"msreg{k}_{endog_col}" if ar_order == 0 else f"msar{ar_order}_{k}_{endog_col}")

    def _mod(self, y):
        import statsmodels.api as sm
        if self.ar_order == 0:
            return sm.tsa.MarkovRegression(y, self.n_states, trend="c", switching_variance=True)
        return sm.tsa.MarkovAutoregression(y, self.n_states, order=self.ar_order,
                                           switching_ar=False, switching_variance=True)

    def fit(self, F):
        y = F[self.endog_col].astype(float)
        mod = self._mod(y.to_numpy())
        # EM from random starts is needed: the default start sticks in poor local
        # optima for the level model (log-lik differs by >400), and statsmodels'
        # random-start search is not reproducible even with a fixed seed, and
        # sometimes returns a degenerate optimum (a regime with P(stay)~0 that
        # flips every day to soak up outliers).  So: restart up to 8 times, drop
        # degenerate fits (any P(stay) < 0.8), stop when the two best admissible
        # log-likelihoods agree to 1e-2, and keep the best.
        cands = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attempt in range(8):
                np.random.seed(self.seed + attempt)
                try:
                    cand = mod.fit(disp=False, maxiter=200, search_reps=5, search_iter=10)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    continue
                llf = float(cand.llf)
                if not np.isfinite(llf):
                    continue
                A = np.asarray(mod.regime_transition_matrix(np.asarray(cand.params)))
                A = A[:, :, 0] if A.ndim == 3 else A
                cands.append((float(np.diag(A).min()) > 0.8, llf, cand))
                good = sorted([c[1] for c in cands if c[0]], reverse=True)
                if len(good) >= 2 and good[0] - good[1] < 1e-2:
                    break
            if not cands:
                res = mod.fit(disp=False, maxiter=200)
            else:
                pool = [c for c in cands if c[0]] or cands
                res = max(pool, key=lambda c: c[1])[2]
        self.params_ = np.asarray(res.params, dtype=float)
        self.names_ = list(mod.param_names)
        consts = np.array([self.params_[self.names_.index(f"const[{i}]")] for i in range(self.n_states)])
        self.order_ = np.argsort(consts)
        self.consts_ = consts[self.order_]
        A = np.asarray(mod.regime_transition_matrix(self.params_))
        A = A[:, :, 0] if A.ndim == 3 else A   # [to, from]
        self.pstay_ = np.diag(A)[self.order_]
        self.train_ll_, self.n_train_ = float(res.llf), len(y)
        self.n_params = len(self.params_)
        return self

    def filter(self, F, return_ll=False):
        y = F[self.endog_col].astype(float).to_numpy()
        mod = self._mod(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mod.filter(self.params_)
        pr = np.asarray(res.filtered_marginal_probabilities, dtype=float)
        lag = len(y) - len(pr)
        if lag > 0:  # AR(p) loses p initial observations
            pr = np.vstack([np.full((lag, self.n_states), 1.0 / self.n_states), pr])
        P = pr[:, self.order_]
        if return_ll:
            ll = np.asarray(getattr(res, 'llf_obs', np.full(len(y) - lag, np.nan)), dtype=float)
            if lag > 0:
                ll = np.concatenate([np.full(lag, np.nan), ll])
            return P, ll
        return P

    def describe(self):
        d = {f"mean_{i}": c for i, c in enumerate(self.consts_)}
        d.update({f"pstay_{i}": float(self.pstay_[i]) for i in range(self.n_states)})
        return d


# --------------------------------------------------------------------------
# Statistical jump model
# --------------------------------------------------------------------------

class JumpModelRegime(RegimeModel):
    """Statistical jump model (Nystrup et al.): K centroids in feature space and
    a state path minimising  sum_t ||x_t - mu_{s_t}||^2 + lam * 1[s_t != s_{t-1}].

    Fit: k-means++ start, alternate (i) dynamic-programming path given centroids
    and (ii) centroid update, best of `n_init`.  Online use: the forward DP value
    V_t(k) = d_t(k) + min_j(V_{t-1}(j) + lam*1[j!=k]) is a function of x_{<=t}
    only, so s_t = argmin_k V_t(k) is causal (unlike the full-sample path, which
    is a Viterbi-style smoother).  Pseudo-probabilities: softmax(-0.5*(V - min V)),
    i.e. the isotropic-Gaussian-likelihood reading of the costs.
    """

    def __init__(self, n_states=3, lam=10.0, cols=None, sort_col=None, n_init=5, seed=42, name=None):
        super().__init__(cols, sort_col)
        self.n_states, self.lam, self.n_init, self.seed = n_states, lam, n_init, seed
        self.name = name or f"jump{n_states}_lam{lam:g}"

    @staticmethod
    def _dist(X, C):
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)

    def _dp_path(self, D):
        T, K = D.shape
        V = np.empty((T, K))
        back = np.zeros((T, K), dtype=int)
        V[0] = D[0]
        for t in range(1, T):
            stay = V[t - 1]
            j = int(np.argmin(stay))
            alt = stay[j] + self.lam
            take_stay = stay <= alt
            V[t] = D[t] + np.where(take_stay, stay, alt)
            back[t] = np.where(take_stay, np.arange(K), j)
        s = np.empty(T, dtype=int)
        s[-1] = int(np.argmin(V[-1]))
        for t in range(T - 1, 0, -1):
            s[t - 1] = back[t, s[t]]
        return s

    def fit(self, F):
        X = self._X(F)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        rng = np.random.default_rng(self.seed)
        best, best_obj = None, np.inf
        for _ in range(self.n_init):
            C = [Xs[rng.integers(len(Xs))]]
            for _k in range(1, self.n_states):  # k-means++
                d2 = self._dist(Xs, np.array(C)).min(1)
                C.append(Xs[rng.choice(len(Xs), p=d2 / d2.sum())])
            C = np.array(C)
            prev = None
            for _it in range(50):
                s = self._dp_path(self._dist(Xs, C))
                for k in range(self.n_states):
                    if (s == k).any():
                        C[k] = Xs[s == k].mean(0)
                if prev is not None and (s == prev).all():
                    break
                prev = s
            D = self._dist(Xs, C)
            obj = D[np.arange(len(Xs)), s].sum() + self.lam * (np.diff(s) != 0).sum()
            if obj < best_obj:
                best, best_obj = (C.copy(), s.copy()), obj
        C, s = best
        j = self._sort_idx()
        self.order_ = np.argsort(C[:, j] * self.sd_[j] + self.mu_[j])
        self.C_ = C
        self.train_path_ = s
        return self

    def filter(self, F, return_ll=False):
        Xs = (self._X(F) - self.mu_) / self.sd_
        D = self._dist(Xs, self.C_)
        T, K = D.shape
        V = np.empty((T, K))
        V[0] = D[0]
        for t in range(1, T):
            stay = V[t - 1]
            V[t] = D[t] + np.minimum(stay, stay.min() + self.lam)
        Z = V - V.min(1, keepdims=True)
        P = np.exp(-0.5 * Z)
        P /= P.sum(1, keepdims=True)
        P = P[:, self.order_]
        if return_ll:
            return P, np.full(T, np.nan)
        return P

    def describe(self):
        j = self._sort_idx()
        means = self.C_[self.order_, j] * self.sd_[j] + self.mu_[j]
        return {f"mean_{i}": means[i] for i in range(self.n_states)}


# --------------------------------------------------------------------------
# Simple rules
# --------------------------------------------------------------------------

class RuleRegime(RegimeModel):
    """Non-ML baselines.  `kind`:

    vix_q      stress = VIX_t > q-quantile of the *previous* `window` closes
    vix_pct    stress_score = trailing `window` percentile rank of VIX_t
               (continuous; hard flag at 0.8)
    inversion  stress = VIX_t / VIX3M_t > 1  (term-structure backwardation)
    inv9d      stress = VIX9D_t / VIX_t > 1  (needs vix9d; starts 2011-01)
    The constants (q, window, 1.0) are conventional, not fitted.
    """

    def __init__(self, kind, q=0.8, window=252, name=None):
        super().__init__(None, None)
        self.kind, self.q, self.window = kind, q, window
        self.n_states = 2
        self.name = name or {"vix_q": f"rule_vix_q{int(q * 100)}", "vix_pct": "rule_vix_pctile",
                             "inversion": "rule_vix_vix3m_inv", "inv9d": "rule_vix9d_inv"}[kind]
        self.hard_threshold = 0.8 if kind == "vix_pct" else 0.5

    def fit(self, F):
        return self

    def filter(self, F, return_ll=False):
        vix = F["vix"].astype(float)
        if self.kind == "vix_q":
            thr = vix.shift(1).rolling(self.window).quantile(self.q)
            s = (vix > thr).astype(float).where(thr.notna(), 0.0)
        elif self.kind == "vix_pct":
            s = vix.rolling(self.window).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
            s = s.fillna(0.0)
        elif self.kind == "inversion":
            s = (vix / F["vix3m"] > 1.0).astype(float)
        elif self.kind == "inv9d":
            s = (F["vix9d"] / vix > 1.0).astype(float)
        else:
            raise ValueError(self.kind)
        s = s.to_numpy(dtype=float)
        P = np.column_stack([1 - s, s])
        return (P, np.full(len(s), np.nan)) if return_ll else P


class CusumRegime(RegimeModel):
    """One-sided CUSUM change-point detector on log VIX.

    mode 'diff':  S_t = max(0, S_{t-1} + z_t - k), z_t = dlogVIX_t / trailing-63d
                  std of dlogVIX (shifted one day).  Detects *rising runs*.
    mode 'level': z_t = (logVIX_t - EWMA_{t-1}) / EW-std_{t-1}  (half-life 126),
                  S as above.  Detects *shifts of level* relative to a slow baseline.
    stress_score = min(1, S_t / h); hard flag is S_t >= h (threshold 0.999).
    k, h are textbook defaults (0.5, 4), not tuned.
    """

    def __init__(self, mode="diff", k=0.5, h=4.0, name=None):
        super().__init__(None, None)
        self.mode, self.k, self.h = mode, k, h
        self.n_states = 2
        self.name = name or f"cusum_{mode}"
        self.hard_threshold = 0.999

    def fit(self, F):
        return self

    def filter(self, F, return_ll=False):
        x = F["log_vix"].astype(float)
        if self.mode == "diff":
            d = x.diff()
            z = d / d.rolling(63).std().shift(1)
        else:
            m = x.ewm(halflife=126, adjust=False).mean().shift(1)
            sd = (x - m).ewm(halflife=126, adjust=False).std().shift(1)
            z = (x - m) / sd
        z = z.fillna(0.0).to_numpy()
        S = np.zeros(len(z))
        s = 0.0
        for t in range(len(z)):
            s = max(0.0, s + z[t] - self.k)
            S[t] = s
        sc = np.clip(S / self.h, 0, 1)
        P = np.column_stack([1 - sc, sc])
        return (P, np.full(len(sc), np.nan)) if return_ll else P


# --------------------------------------------------------------------------
# GJR-GARCH conditional-vol state
# --------------------------------------------------------------------------

class GJRGarchRegime(RegimeModel):
    """GJR-GARCH(1,1) on SPX returns (Gaussian QMLE, scipy SLSQP).

    h_{t+1} = w + (a + g*1[r_t<0]) r_t^2 + b h_t is known at the close of t, so
    sigma_{t+1|t} is a causal forecast.  stress_score = empirical CDF of that
    forecast within the *training-window* fitted path (so 0.8 means "higher than
    80% of conditional vols seen in training"); hard flag at 0.8.
    """

    def __init__(self, name="gjr_garch"):
        super().__init__(["ret"], "ret")
        self.n_states = 2
        self.name = name
        self.hard_threshold = 0.8

    @staticmethod
    def _h(theta, r, h0):
        w, a, g, b = theta
        u = w + (a + g * (r < 0)) * r ** 2  # contribution to h_{t+1} from r_t
        h_next = lfilter([1.0], [1.0, -b], u, zi=[b * h0])[0]  # h_{t+1}, t=0..T-1
        return h_next

    def _nll(self, theta, r, h0):
        h_next = self._h(theta, r, h0)
        h_cur = np.concatenate([[h0], h_next[:-1]])  # h_t for r_t
        if (h_cur <= 0).any() or not np.isfinite(h_cur).all():
            return 1e10
        return 0.5 * float(np.sum(np.log(h_cur) + r ** 2 / h_cur))

    def fit(self, F):
        r = F["ret"].astype(float).to_numpy()
        r = r[np.isfinite(r)]
        self.h0_ = float(np.var(r))
        x0 = np.array([0.02, 0.03, 0.10, 0.88])
        cons = [{"type": "ineq", "fun": lambda t: 0.999 - (t[1] + 0.5 * t[2] + t[3])}]
        res = minimize(self._nll, x0, args=(r, self.h0_), method="SLSQP",
                       bounds=[(1e-6, 1.0), (0.0, 0.5), (0.0, 0.8), (0.0, 0.999)],
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-8})
        self.theta_ = res.x
        hn = self._h(self.theta_, r, self.h0_)
        self.train_sorted_ = np.sort(hn)
        return self

    def forecast_vol(self, F):
        r = F["ret"].astype(float).to_numpy()
        return np.sqrt(252.0 * self._h(self.theta_, r, self.h0_)) / 100.0

    def filter(self, F, return_ll=False):
        r = F["ret"].astype(float).to_numpy()
        hn = self._h(self.theta_, r, self.h0_)
        s = np.searchsorted(self.train_sorted_, hn, side="right") / len(self.train_sorted_)
        s = np.clip(s, 0.0, 1.0)
        P = np.column_stack([1 - s, s])
        return (P, np.full(len(s), np.nan)) if return_ll else P

    def describe(self):
        w, a, g, b = self.theta_
        return {"alpha": a, "gamma": g, "beta": b, "persistence": a + 0.5 * g + b}


# --------------------------------------------------------------------------
# Walk-forward driver
# --------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    probs: pd.DataFrame            # p0..p{K-1}, vol-sorted
    stress: pd.Series              # stress_score in [0,1]
    ll: pd.Series                  # one-step predictive log-density (NaN if n/a)
    refits: pd.DataFrame           # one row per refit: date, n_train, diagnostics
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)  # agreement old vs new model
    failures: list = field(default_factory=list)


def walk_forward_regimes(factory, features: pd.DataFrame, refit_every: int = 63,
                         min_train: int = 1260, end: int | None = None,
                         stability: bool = True) -> WalkForwardResult:
    """Expanding-window walk-forward.

    For each block start i = min_train, min_train+refit_every, ...:
      model = factory().fit(features.iloc[:i])              # rows strictly before block
      P = model.filter(features.iloc[:i+refit_every])       # filter over prefix only
      keep rows i .. i+refit_every-1

    Because the model is fitted on rows < i and `filter` is causal, row t of the
    output is a function of data <= t and parameters estimated on data < i <= t.
    State labels are sorted by volatility inside each model, so concatenating
    blocks is meaningful; `refits` records the sorted state means so you can
    verify that (refit stability).  If a fit fails the previous model is kept
    (logged in `failures`).
    """
    F = features
    T = len(F) if end is None else min(end, len(F))
    blocks, lls, rows, stab, fails = [], [], [], [], []
    prev, K = None, None
    for i in range(min_train, T, refit_every):
        j = min(i + refit_every, T)
        try:
            m = factory().fit(F.iloc[:i])
        except Exception as e:  # keep going; the failure is reported
            fails.append((F.index[i], repr(e)))
            if prev is None:
                continue
            m = prev
        P, ll = m.filter(F.iloc[:j], return_ll=True)
        K = P.shape[1]
        blocks.append(pd.DataFrame(P[i:j], index=F.index[i:j], columns=[f"p{k}" for k in range(K)]))
        lls.append(pd.Series(ll[i:j], index=F.index[i:j]))
        d = {"date": F.index[i], "n_train": i}
        d.update(m.describe())
        rows.append(d)
        if stability and prev is not None and prev is not m:
            Pold = prev.filter(F.iloc[:j])[i:j]
            s_new, s_old = m.stress_score(P[i:j]), prev.stress_score(Pold)
            stab.append({"date": F.index[i],
                         "flag_agree": float(((s_new > m.hard_threshold) == (s_old > prev.hard_threshold)).mean()),
                         "mean_abs_diff": float(np.abs(s_new - s_old).mean())})
        prev = m
    if not blocks:
        raise RuntimeError("no successful fit")
    probs = pd.concat(blocks)
    stress = pd.Series(prev.stress_score(probs.to_numpy()), index=probs.index, name="stress_score")
    return WalkForwardResult(probs, stress, pd.concat(lls), pd.DataFrame(rows),
                             pd.DataFrame(stab), fails)
