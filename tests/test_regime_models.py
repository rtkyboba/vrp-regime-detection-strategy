"""Tests for vrp_strategy.regime_models: causality of EVERY method, label ordering,
and the walk-forward driver.  Synthetic data only, so they run offline and fast."""
import numpy as np
import pandas as pd
import pytest

from vrp_strategy import regime_models as rm
from vrp_strategy.regimes import filtered_proba

COLS, SORT = rm.FEATURE_SETS["core"]
ORIG, ORIG_SORT = rm.FEATURE_SETS["orig"]


def make_frame(n=1400, seed=0):
    """Synthetic SPX/VIX/VIX3M with a persistent 3-regime latent vol."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.985, 0.012, 0.003], [0.03, 0.95, 0.02], [0.05, 0.10, 0.85]])
    vol = np.array([0.10, 0.20, 0.45])  # annualised
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = rng.choice(3, p=A[s[t - 1]])
    sig = vol[s]
    ret = rng.normal(0, 1, n) * sig / np.sqrt(252)
    idx = pd.bdate_range("2008-01-01", periods=n)
    spx = pd.Series(1000 * np.exp(np.cumsum(ret)), index=idx)
    vix = pd.Series(100 * sig * np.exp(rng.normal(0, 0.08, n)) * 1.15, index=idx)
    vix3m = pd.Series(100 * (0.7 * sig + 0.3 * 0.2) * np.exp(rng.normal(0, 0.05, n)) * 1.1, index=idx)
    F = rm.features_from_closes(spx, vix, vix3m)
    return F.dropna(subset=["ret", "rv_30", "log_rv21", "vrp_vol"])


@pytest.fixture(scope="module")
def F():
    return make_frame()


def factories():
    return {
        "hmm2": lambda: rm.HMMRegime(2, COLS, SORT, n_init=1, n_iter=50),
        "hmm3": lambda: rm.HMMRegime(3, COLS, SORT, n_init=1, n_iter=50),
        "hmm3_orig": lambda: rm.HMMRegime(3, ORIG, ORIG_SORT, n_init=1, n_iter=50),
        "gmm3": lambda: rm.GMMRegime(3, COLS, SORT, n_init=1),
        "msreg2": lambda: rm.MarkovSwitchingRegime(2),
        "msar2": lambda: rm.MarkovSwitchingRegime(2, ar_order=1),
        "jump3": lambda: rm.JumpModelRegime(3, 10.0, COLS, SORT, n_init=2),
        "rule_q80": lambda: rm.RuleRegime("vix_q", q=0.8, window=100),
        "rule_pct": lambda: rm.RuleRegime("vix_pct", window=100),
        "rule_inv": lambda: rm.RuleRegime("inversion"),
        "cusum_diff": lambda: rm.CusumRegime("diff"),
        "cusum_level": lambda: rm.CusumRegime("level"),
        "gjr": lambda: rm.GJRGarchRegime(),
    }


FACTORY_NAMES = list(factories())


@pytest.mark.parametrize("name", FACTORY_NAMES)
def test_filter_is_causal(F, name):
    """Row t of filter() must not change when data after t is corrupted or removed."""
    m = factories()[name]().fit(F.iloc[:800])
    full = m.filter(F)
    assert full.shape[0] == len(F)
    assert np.isfinite(full).all()
    assert np.allclose(full.sum(axis=1), 1.0, atol=1e-6)
    for t in (850, 1000, 1200):
        # (a) truncation
        assert np.allclose(m.filter(F.iloc[: t + 1])[t], full[t], atol=1e-9)
        # (b) corruption of the future
        G = F.copy()
        num = G.select_dtypes("number").columns
        G.iloc[t + 1:, [G.columns.get_loc(c) for c in num]] *= 3.7
        assert np.allclose(m.filter(G)[t], full[t], atol=1e-9)
        assert np.allclose(m.filter(G)[: t + 1], full[: t + 1], atol=1e-9)


@pytest.mark.parametrize("name", ["hmm3", "gmm3", "msreg2", "jump3", "rule_q80", "cusum_diff", "gjr"])
def test_walk_forward_is_causal(F, name):
    """Corrupting rows >= c leaves walk-forward output for rows < c unchanged:
    refits before c never see the corrupted rows, and filtering is causal."""
    kw = dict(refit_every=100, min_train=500)
    base = rm.walk_forward_regimes(factories()[name], F, **kw)
    c = 1000
    G = F.copy()
    num = [G.columns.get_loc(x) for x in G.select_dtypes("number").columns]
    G.iloc[c:, num] *= 2.9
    alt = rm.walk_forward_regimes(factories()[name], G, **kw)
    d0 = F.index[c - 1]
    a = base.probs.loc[:d0].to_numpy()
    b = alt.probs.loc[:d0].to_numpy()
    assert len(a) > 300
    # atol 1e-4: statsmodels' random-start EM is not bit-reproducible (~1e-7 in params);
    # real leakage from rows x2.9 would move probabilities by O(1).
    assert np.allclose(a, b, atol=1e-4)


def test_stress_score_in_unit_interval(F):
    for name in FACTORY_NAMES:
        m = factories()[name]().fit(F.iloc[:800])
        s = m.stress_score(m.filter(F))
        assert (s >= 0).all() and (s <= 1 + 1e-12).all(), name


@pytest.mark.parametrize("name", ["hmm2", "hmm3", "gmm3", "msreg2", "jump3"])
def test_states_sorted_by_volatility(F, name):
    """Column k of P is the k-th lowest-volatility state: the P-weighted mean of
    the sort column rises with k, independent of EM's arbitrary labelling."""
    for seed in (1, 2, 3):
        fac = factories()[name]()
        if hasattr(fac, "seed"):
            fac.seed = seed * 17
        fac.fit(F.iloc[:900])
        P = fac.filter(F.iloc[:900])
        x = F.iloc[:900][SORT].to_numpy()
        means = [(P[:, k] * x).sum() / max(P[:, k].sum(), 1e-9) for k in range(P.shape[1]) if P[:, k].sum() > 5]
        # extreme states must be ordered; interior P-weighted means can be non-monotone
        # because soft posteriors overlap, so the strict check is on fitted parameters:
        if len(means) >= 2:
            assert means[0] < means[-1], (name, seed, means)
        d = fac.describe()
        ms = [d[f"mean_{i}"] for i in range(fac.n_states)]
        assert ms == sorted(ms)


def test_label_stability_across_refits(F):
    """Across all walk-forward refits the sorted state means stay ordered (no label flips)."""
    res = rm.walk_forward_regimes(lambda: rm.HMMRegime(3, COLS, SORT, n_init=1, n_iter=50),
                                  F, refit_every=100, min_train=500)
    mean_cols = [c for c in res.refits.columns if c.startswith("mean_")]
    M = res.refits[mean_cols].to_numpy()
    assert (np.diff(M, axis=1) > 0).all()
    assert len(res.refits) >= 5


def test_hmm_filter_matches_reference_filter(F):
    """Our forward recursion equals the one in vrp_strategy.regimes."""
    m = rm.HMMRegime(3, COLS, SORT, n_init=1, n_iter=50).fit(F.iloc[:800])
    Xs = (m._X(F) - m.mu_) / m.sd_
    ref = filtered_proba(m.model_, Xs)[:, m.order_]
    assert np.allclose(m.filter(F), ref, atol=1e-8)


def test_hmm_ll_increments_sum_to_model_score(F):
    m = rm.HMMRegime(2, COLS, SORT, n_init=1, n_iter=50).fit(F.iloc[:800])
    Xs = (m._X(F.iloc[:800]) - m.mu_) / m.sd_
    P, ll = m.filter(F.iloc[:800], return_ll=True)
    assert np.isclose(ll.sum(), m.model_.score(Xs), rtol=1e-6)


def test_walk_forward_block_alignment(F):
    res = rm.walk_forward_regimes(factories()["rule_inv"], F, refit_every=100, min_train=500)
    assert res.probs.index[0] == F.index[500]
    assert res.probs.index[-1] == F.index[-1]
    assert res.probs.index.is_unique
    assert ((res.stress == 0) | (res.stress == 1)).all()


def test_jump_model_penalty_reduces_switching(F):
    a = rm.JumpModelRegime(3, 1.0, COLS, SORT, n_init=2).fit(F.iloc[:900])
    b = rm.JumpModelRegime(3, 60.0, COLS, SORT, n_init=2).fit(F.iloc[:900])
    sw = lambda m: int((np.diff(m.filter(F.iloc[:900]).argmax(1)) != 0).sum())
    assert sw(b) <= sw(a)


def test_garch_forecast_vol_positive(F):
    m = rm.GJRGarchRegime().fit(F.iloc[:800])
    v = m.forecast_vol(F)
    assert (v > 0).all() and np.isfinite(v).all()


def test_causality_test_has_teeth(F):
    """Negative control: hmmlearn's smoothed posterior (forward-backward) DOES change
    when the future changes, so the corrupt-the-future checks above can fail."""
    m = rm.HMMRegime(3, COLS, SORT, n_init=1, n_iter=50).fit(F.iloc[:800])
    Xs = (m._X(F) - m.mu_) / m.sd_
    sm_full = m.model_.predict_proba(Xs)
    X2 = Xs.copy()
    X2[1000:] += 8.0
    sm_alt = m.model_.predict_proba(X2)
    assert not np.allclose(sm_full[990], sm_alt[990], atol=1e-6)


def test_static_walk_forward_never_refits(F):
    res = rm.walk_forward_regimes(factories()["hmm3"], F, refit_every=10 ** 6, min_train=800)
    assert len(res.refits) == 1
    assert res.probs.index[0] == F.index[800]
