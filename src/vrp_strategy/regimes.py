"""Regime-detection helpers.

The original notebook labelled the out-of-sample period with `model.predict`,
which is Viterbi decoding over the *entire* test sequence: the label for day t
depends on observations after t. A tradable signal only knows data up to t, so
here we provide the forward (filtering) recursion, whose posterior at t uses
observations <= t only.
"""
import numpy as np
from scipy.special import logsumexp


def filtered_proba(model, X):
    """Causal state posteriors P(state_t | x_1..x_t) from a fitted hmmlearn
    GaussianHMM, via the forward algorithm. Row t uses X[:t+1] only."""
    X = np.asarray(X, dtype=float)
    log_b = model._compute_log_likelihood(X)
    log_a = np.log(np.clip(model.transmat_, 1e-300, None))
    log_pi = np.log(np.clip(model.startprob_, 1e-300, None))
    T, K = log_b.shape
    out = np.empty((T, K))
    alpha = log_pi + log_b[0]
    alpha -= logsumexp(alpha)
    out[0] = np.exp(alpha)
    for t in range(1, T):
        alpha = log_b[t] + logsumexp(alpha[:, None] + log_a, axis=0)
        alpha -= logsumexp(alpha)
        out[t] = np.exp(alpha)
    return out


def state_order_by_feature(model, scaler, feature_idx):
    """State indices sorted ascending by the mean of one raw feature
    (e.g. realised vol) -> [calm, stressed, crisis] labelling that depends on
    the training fit only."""
    means = scaler.inverse_transform(model.means_)
    return np.argsort(means[:, feature_idx])
