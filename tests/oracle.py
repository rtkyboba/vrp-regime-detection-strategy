"""Independent reference implementation used only to cross-check production code.

A log-space forward filter (logsumexp), written differently from
``vrp_strategy.regimes.forward_filter`` (probability space with scaling) so that a
bug in one is unlikely to be replicated in the other.
"""
import numpy as np
from scipy.special import logsumexp


def reference_filter(log_b, transmat, startprob):
    log_a = np.log(np.clip(transmat, 1e-300, None))
    log_pi = np.log(np.clip(startprob, 1e-300, None))
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
