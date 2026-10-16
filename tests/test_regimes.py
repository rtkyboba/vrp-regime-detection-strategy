import numpy as np
import pytest
from hmmlearn import hmm

from vrp_strategy.regimes import filtered_proba


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (300, 2))
    b = rng.normal(3, 1.5, (200, 2))
    X = np.vstack([a, b, a[:100]])
    m = hmm.GaussianHMM(2, covariance_type="full", n_iter=200, random_state=0).fit(X)
    return m, X


def test_rows_are_probabilities(fitted):
    m, X = fitted
    p = filtered_proba(m, X)
    assert p.shape == (len(X), 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_filter_is_causal(fitted):
    """Row t must not change when later observations are removed/altered."""
    m, X = fitted
    full = filtered_proba(m, X)
    for t in (10, 150, 400):
        prefix = filtered_proba(m, X[: t + 1])
        assert np.allclose(prefix[t], full[t])
        X2 = X.copy()
        X2[t + 1 :] += 100.0  # corrupt the future
        assert np.allclose(filtered_proba(m, X2)[t], full[t])


def test_smoothed_posterior_is_not_causal(fitted):
    """Sanity check on the test itself: hmmlearn's predict_proba (forward-
    backward) DOES change when the future changes, so the test above has teeth."""
    m, X = fitted
    t = 295  # near the regime switch
    X2 = X.copy()
    X2[t + 1 :] = X[t + 1 :] + 100.0
    assert not np.allclose(m.predict_proba(X)[t], m.predict_proba(X2)[t])
