"""Regime-detection helpers.

The original notebook labelled the out-of-sample period with `model.predict`,
which is Viterbi decoding over the *entire* test sequence: the label for day t
depends on observations after t. A tradable signal only knows data up to t, so
here we provide the forward (filtering) recursion, whose posterior at t uses
observations <= t only.
"""
import numpy as np


def forward_filter(log_b, transmat, startprob):
    """Normalised forward recursion on emission log-densities ``log_b`` (T, K).

    Returns ``(filtered, ll)``: filtered[t] = P(state_t | x_1..x_t) and
    ll[t] = log p(x_t | x_<t), the one-step predictive log-density (sums to the model's
    log-likelihood). Row t depends on rows <= t only.
    """
    T, K = log_b.shape
    m = log_b.max(axis=1)
    b = np.exp(log_b - m[:, None])
    alpha = np.empty((T, K))
    ll = np.empty(T)
    pred = np.asarray(startprob, dtype=float)
    for t in range(T):
        u = pred * b[t]
        c = max(u.sum(), 1e-300)
        alpha[t] = u / c
        ll[t] = np.log(c) + m[t]
        pred = alpha[t] @ transmat
    return alpha, ll


def filtered_proba(model, X):
    """Causal state posteriors P(state_t | x_1..x_t) from a fitted hmmlearn
    GaussianHMM. Row t uses X[:t+1] only."""
    X = np.asarray(X, dtype=float)
    return forward_filter(model._compute_log_likelihood(X), model.transmat_, model.startprob_)[0]


def state_order_by_feature(model, scaler, feature_idx):
    """State indices sorted ascending by the mean of one raw feature
    (e.g. realised vol) -> [calm, stressed, crisis] labelling that depends on
    the training fit only."""
    means = scaler.inverse_transform(model.means_)
    return np.argsort(means[:, feature_idx])
