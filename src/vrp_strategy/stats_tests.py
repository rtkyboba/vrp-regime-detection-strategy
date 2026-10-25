"""Statistical validation toolkit: "is this Sharpe real?"

Every function takes DAILY (or other fixed-frequency) *excess* returns and works
in per-period units internally; user-facing Sharpe arguments and outputs are
annualised with sqrt(periods_per_year) unless a docstring says otherwise.

Contents
--------
1. Point estimates and standard errors
   - ``sharpe_ratio``            plain annualised Sharpe
   - ``lo_adjusted_sharpe``      Lo (2002) serial-correlation-adjusted annualisation
   - ``sharpe_se``               Mertens/Opdyke (skew+kurtosis) and HAC standard errors
2. Selection-bias aware inference
   - ``probabilistic_sharpe_ratio``   PSR  (Bailey & Lopez de Prado 2012)
   - ``expected_max_sharpe``          E[max of N null Sharpes]
   - ``deflated_sharpe_ratio``        DSR  (Bailey & Lopez de Prado 2014)
   - ``min_track_record_length``      MinTRL (Bailey & Lopez de Prado 2012)
3. Resampling
   - ``stationary_bootstrap_indices``, ``optimal_block_length``
   - ``bootstrap_sharpe_ci``          block-bootstrap CI for a Sharpe
   - ``bootstrap_sharpe_diff``        paired block-bootstrap CI / p-value for SR_a - SR_b
4. Data-snooping across many candidate rules
   - ``probability_of_backtest_overfitting``  CSCV PBO (Bailey et al. 2017)
   - ``reality_check``                White (2000) RC and Hansen (2005) SPA p-values
5. Better nulls for timing signals
   - ``timing_null_test``  circular-shift / block-permutation / (flawed) iid shuffle
6. One-call entry point
   - ``validate`` / ``validate_table``

Design notes
------------
* No dependency beyond numpy / pandas / scipy.
* All randomness takes an explicit ``seed`` so results are reproducible.
* Statistical tests answer "is the sample mean distinguishable from noise, given
  how many things were tried?".  They CANNOT detect a wrong P&L definition (see
  reports/04_statistical_validation.md: the legacy proxy P&L passes every test
  here and is still not a tradable return stream).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329

__all__ = [
    "sharpe_ratio", "lo_adjusted_sharpe", "sharpe_se",
    "probabilistic_sharpe_ratio", "expected_max_sharpe", "deflated_sharpe_ratio",
    "min_track_record_length",
    "stationary_bootstrap_indices", "optimal_block_length",
    "bootstrap_sharpe_ci", "bootstrap_sharpe_diff",
    "probability_of_backtest_overfitting", "reality_check",
    "timing_null_test", "null_positions",
    "validate", "validate_table",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _arr(x) -> np.ndarray:
    """1-D float array with NaNs dropped."""
    a = np.asarray(x, dtype=float).ravel()
    return a[~np.isnan(a)]


def _align(a, b):
    """Align two return inputs. Series are inner-joined on index and NaN-dropped
    jointly (paired comparison); arrays must have equal length."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        return j.iloc[:, 0].to_numpy(float), j.iloc[:, 1].to_numpy(float)
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if len(a) != len(b):
        raise ValueError("paired comparison needs equal-length inputs")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def _moments(r: np.ndarray):
    """(mean, std[ddof=1], skewness, RAW kurtosis) of a 1-D array."""
    mu = r.mean()
    sd = r.std(ddof=1)
    z = (r - mu) / r.std(ddof=0)
    return mu, sd, float((z ** 3).mean()), float((z ** 4).mean())


def _per_period(sr_annual: float, ppy: float) -> float:
    return sr_annual / np.sqrt(ppy)


def _autocorr(r: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelations rho_1..rho_nlags (biased /n estimator)."""
    n = len(r)
    x = r - r.mean()
    c0 = (x * x).sum()
    if c0 == 0:
        return np.zeros(nlags)
    return np.array([(x[k:] * x[:n - k]).sum() / c0 for k in range(1, nlags + 1)])


def _nw_lag(n: int) -> int:
    """Newey-West (1994) rule-of-thumb bandwidth floor(4 (n/100)^(2/9))."""
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


# --------------------------------------------------------------------------- #
# 1. point estimates and standard errors
# --------------------------------------------------------------------------- #
def sharpe_ratio(returns, periods_per_year: float = 252) -> float:
    """Annualised Sharpe of *excess* returns:  SR = mean / std(ddof=1) * sqrt(ppy).

    Pass excess returns (use ``data.excess``) for funded instruments; for
    self-financing P&L such as swaps the raw return is already an excess return.
    """
    r = _arr(returns)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def lo_adjusted_sharpe(returns, periods_per_year: float = 252, max_lag: int | None = None) -> float:
    """Sharpe annualised with Lo's (2002) serial-correlation correction.

    With a per-period Sharpe SR_1 and autocorrelations rho_k, the q-period Sharpe is

        SR_q = eta(q) * SR_1,   eta(q) = q / sqrt( q + 2 * sum_{k=1}^{q-1} (q-k) rho_k )

    The naive sqrt(q) scaling is the special case rho_k = 0.  Positive autocorrelation
    (smoothed / stale-priced returns, e.g. a vol-targeted series) *inflates* the naive
    annualised Sharpe, and eta(q) < sqrt(q) removes that inflation.  We truncate the
    sum at ``max_lag`` (default: Newey-West bandwidth) because high-lag sample
    autocorrelations are pure noise.

    Reference: Lo, A. W. (2002), "The Statistics of Sharpe Ratios", Financial
    Analysts Journal 58(4), 36-52.
    """
    r = _arr(returns)
    q = float(periods_per_year)
    L = max_lag if max_lag is not None else _nw_lag(len(r))
    L = int(min(L, q - 1, len(r) - 2))
    rho = _autocorr(r, L)
    k = np.arange(1, L + 1)
    rad = q + 2.0 * ((q - k) * rho).sum()
    rad = max(rad, 1e-12)
    sr1 = r.mean() / r.std(ddof=1)
    return float(sr1 * q / np.sqrt(rad))


def sharpe_se(returns, method: str = "mertens", periods_per_year: float | None = None,
              max_lag: int | None = None) -> float:
    """Standard error of the Sharpe estimator.

    ``method="mertens"`` (iid, non-normal; Mertens 2002, Opdyke 2007):

        Var(SR_hat) = (1 - g3 * SR + (g4 - 1)/4 * SR^2) / (T - 1)

    where SR is the per-period Sharpe, g3 the skewness, g4 the (raw, not excess)
    kurtosis.  Normal returns (g3 = 0, g4 = 3) give Lo's (1 + SR^2/2)/T.  Negative
    skew and fat tails (short-vol!) raise the variance.

    ``method="hac"``: delta-method / GMM standard error that also allows serial
    correlation.  The influence function of SR = mu/sigma is

        psi_t = z_t - (SR/2)(z_t^2 - 1),   z_t = (r_t - mu)/sigma

    and Var(SR_hat) = LRV(psi) / T with a Bartlett-kernel (Newey-West) long-run
    variance.  With zero lags this equals the Mertens formula (tested).  See also
    Ledoit & Wolf (2008).

    Returns the SE of the *per-period* Sharpe, or of the annualised Sharpe
    (x sqrt(periods_per_year)) if ``periods_per_year`` is given.
    """
    r = _arr(returns)
    T = len(r)
    if method == "mertens":
        mu, sd, g3, g4 = _moments(r)
        sr = mu / sd
        var = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) / (T - 1)
    elif method == "hac":
        sd0 = r.std(ddof=0)
        z = (r - r.mean()) / sd0
        sr = r.mean() / sd0
        psi = z - 0.5 * sr * (z ** 2 - 1.0)
        L = _nw_lag(T) if max_lag is None else int(max_lag)
        psi = psi - psi.mean()
        lrv = (psi * psi).mean()
        for j in range(1, L + 1):
            w = 1.0 - j / (L + 1.0)
            lrv += 2.0 * w * (psi[j:] * psi[:-j]).mean()
        var = max(lrv, 1e-12) / T
    else:
        raise ValueError("method must be 'mertens' or 'hac'")
    se = float(np.sqrt(max(var, 0.0)))
    return se * np.sqrt(periods_per_year) if periods_per_year else se


# --------------------------------------------------------------------------- #
# 2. selection-bias aware inference
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0, periods_per_year: float = 252,
                               se_method: str = "mertens") -> float:
    """PSR: probability that the TRUE Sharpe exceeds ``sr_benchmark`` (annualised).

        PSR = Phi( (SR_hat - SR*) / SE(SR_hat) )

    with SE from ``sharpe_se`` (default: skew/kurtosis-aware).  It is a one-sided
    p-value in disguise (1 - PSR is the p-value against SR <= SR*); it does *not*
    account for how many strategies were tried (that is the DSR's job).

    Reference: Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient
    Frontier", Journal of Risk 15(2).
    """
    r = _arr(returns)
    sr = r.mean() / r.std(ddof=1)
    se = sharpe_se(r, se_method)
    return float(stats.norm.cdf((sr - _per_period(sr_benchmark, periods_per_year)) / se))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe estimates whose true
    value is zero (the selection-bias hurdle).

        E[max_N] ~= sigma_SR * [ (1 - gamma) Phi^{-1}(1 - 1/N) + gamma Phi^{-1}(1 - 1/(N e)) ]

    gamma = Euler-Mascheroni constant, sigma_SR = std of the Sharpe estimates across
    trials (same units as the result).  This is the Gumbel approximation to the
    expected maximum of N standard normals scaled by sigma_SR.

    Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", Journal
    of Portfolio Management 40(5), eq. (6).
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0
    return float(sr_std * ((1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n)
                           + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n * np.e))))


def deflated_sharpe_ratio(returns, n_trials: int, trial_sharpes=None, sr_var: float | None = None,
                          periods_per_year: float = 252, se_method: str = "mertens",
                          return_details: bool = False):
    """DSR: PSR evaluated against the selection-bias hurdle SR_0 = E[max of N null Sharpes].

        DSR = Phi( (SR_hat - SR_0) / SE(SR_hat) ),
        SR_0 = expected_max_sharpe(N, sqrt(V))

    V is the variance of the (per-period) Sharpe estimates *across the N trials*.
    Priority: ``trial_sharpes`` (annualised Sharpes of all trials; V = their sample
    variance) > ``sr_var`` (per-period variance) > default V = 1/(T-1), the sampling
    variance of a Sharpe estimate when the true Sharpe is zero.  The default is the
    right null for N *independent* trials of a worthless family; correlated trials
    have smaller V but also fewer effective degrees of freedom, so state N honestly
    and report a sensitivity range rather than hunting for a single "effective N".

    DSR >= 0.95 is the usual bar.  Reference: Bailey & Lopez de Prado (2014).
    """
    r = _arr(returns)
    T = len(r)
    if trial_sharpes is not None:
        ts = _arr(trial_sharpes) / np.sqrt(periods_per_year)
        v = float(ts.var(ddof=1))
    elif sr_var is not None:
        v = float(sr_var)
    else:
        v = 1.0 / (T - 1)
    sr0 = expected_max_sharpe(n_trials, np.sqrt(v))
    sr = r.mean() / r.std(ddof=1)
    se = sharpe_se(r, se_method)
    dsr = float(stats.norm.cdf((sr - sr0) / se))
    if return_details:
        return {"dsr": dsr, "sr0_annual": sr0 * np.sqrt(periods_per_year),
                "sr_annual": sr * np.sqrt(periods_per_year), "se_annual": se * np.sqrt(periods_per_year),
                "trial_sr_std_annual": np.sqrt(v * periods_per_year)}
    return dsr


def min_track_record_length(returns, sr_benchmark: float = 0.0, alpha: float = 0.05,
                            periods_per_year: float = 252, se_method: str = "mertens") -> dict:
    """Minimum Track Record Length: observations needed before SR_hat is significantly
    above ``sr_benchmark`` at one-sided level ``alpha``, *holding the observed Sharpe,
    skew and kurtosis fixed*.

        MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) * (z_{1-alpha} / (SR - SR*))^2

    (per-period SR).  Returns ``{"n_obs": ..., "years": ...}``; ``inf`` if
    SR_hat <= SR*.  Reference: Bailey & Lopez de Prado (2012).
    """
    r = _arr(returns)
    mu, sd, g3, g4 = _moments(r)
    sr = mu / sd
    srb = _per_period(sr_benchmark, periods_per_year)
    if sr <= srb:
        return {"n_obs": np.inf, "years": np.inf}
    if se_method == "mertens":
        factor = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    else:  # HAC: back out the variance factor from the SE (SE^2 * T)
        factor = sharpe_se(r, "hac") ** 2 * len(r)
    n = 1.0 + factor * (stats.norm.ppf(1 - alpha) / (sr - srb)) ** 2
    return {"n_obs": float(n), "years": float(n / periods_per_year)}


# --------------------------------------------------------------------------- #
# 3. resampling
# --------------------------------------------------------------------------- #
def optimal_block_length(x, max_block: float | None = None) -> float:
    """Automatic mean block length for the stationary bootstrap.

    Politis & White (2004) with the Patton, Politis & White (2009) correction:

        b_SB = ( 2 G^2 / D_SB )^(1/3) n^(1/3),  G = sum_k lambda(k/M) |k| gamma_k,
        D_SB = 2 g(0)^2,  g(0) = sum_k lambda(k/M) gamma_k

    lambda is the flat-top trapezoid lag window; M = 2 m_hat where m_hat is the first
    lag after which K_n successive autocorrelations are insignificant.  White noise
    gives b ~ 1 (plain iid bootstrap); persistent series give longer blocks.
    Result is clipped to [1, min(3 sqrt(n), n/3)].
    """
    x = _arr(x)
    n = len(x)
    xc = x - x.mean()
    K = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    m_max = int(np.ceil(np.sqrt(n))) + K
    m_max = min(m_max, n - 2)
    acv = np.array([(xc[k:] * xc[:n - k]).sum() / n for k in range(m_max + 1)])
    if acv[0] == 0:
        return 1.0
    rho = acv / acv[0]
    thresh = 2.0 * np.sqrt(np.log10(n) / n)
    m_hat = max(m_max - K, 0)
    for m in range(0, m_max - K + 1):
        if np.all(np.abs(rho[m + 1:m + K + 1]) < thresh):
            m_hat = m
            break
    M = int(min(max(2 * m_hat, 1), m_max))
    k = np.arange(-M, M + 1)
    a = np.abs(k) / M
    lam = np.where(a <= 0.5, 1.0, 2.0 * (1.0 - a))
    gam = acv[np.abs(k)]
    G = (lam * np.abs(k) * gam).sum()
    g0 = (lam * gam).sum()
    if g0 <= 0:
        return 1.0
    b = (G ** 2 / g0 ** 2) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    cap = max_block if max_block is not None else min(3.0 * np.sqrt(n), n / 3.0)
    return float(np.clip(b, 1.0, cap))


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, rng) -> np.ndarray:
    """Index matrix (n_boot, n) of Politis-Romano (1994) stationary-bootstrap resamples.

    At each step, with probability p = 1/mean_block a new block starts at a uniform
    random location; otherwise the previous index advances by one (wrapping around
    the end of the sample).  Block lengths are Geometric(p), which makes the
    resampled series strictly stationary.  Fully vectorised.
    """
    p = 1.0 / max(mean_block, 1.0)
    new = rng.random((n_boot, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    t = np.arange(n)
    last_new = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    start_at = np.take_along_axis(starts, last_new, axis=1)
    return (start_at + t - last_new) % n


def _sr_rows(x: np.ndarray, ppy: float) -> np.ndarray:
    sd = x.std(axis=1, ddof=1)
    return x.mean(axis=1) / sd * np.sqrt(ppy)


def _block(r1, r2=None, mean_block=None) -> float:
    if mean_block is not None:
        return float(mean_block)
    cands = [optimal_block_length(r1), optimal_block_length(r1 ** 2)]
    if r2 is not None:
        cands += [optimal_block_length(r2), optimal_block_length(r2 ** 2)]
    return float(max(cands))


def bootstrap_sharpe_ci(returns, n_boot: int = 2000, alpha: float = 0.05, mean_block: float | None = None,
                        periods_per_year: float = 252, seed: int = 0) -> dict:
    """Stationary-bootstrap percentile confidence interval for the annualised Sharpe.

    Resamples blocks of consecutive days (random geometric length, mean from
    ``optimal_block_length`` on r and r^2 unless given), recomputes the Sharpe on
    each resample and returns the (alpha/2, 1-alpha/2) percentiles.  Block resampling
    preserves volatility clustering and autocorrelation that an iid bootstrap
    destroys.  Percentile intervals are slightly anti-conservative for small T and
    strongly skewed returns; coverage is checked in tests/test_stats_tests.py.

    Reference: Politis & Romano (1994), JASA 89(428); Ledoit & Wolf (2008).
    """
    r = _arr(returns)
    b = _block(r, mean_block=mean_block)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(len(r), n_boot, b, rng)
    srs = _sr_rows(r[idx], periods_per_year)
    lo, hi = np.percentile(srs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": sharpe_ratio(r, periods_per_year), "ci_low": float(lo), "ci_high": float(hi),
            "boot_se": float(srs.std(ddof=1)), "mean_block": b, "p_le_zero": float((srs <= 0).mean())}


def bootstrap_sharpe_diff(returns_a, returns_b, n_boot: int = 2000, alpha: float = 0.05,
                          mean_block: float | None = None, periods_per_year: float = 252,
                          seed: int = 0) -> dict:
    """Paired stationary-bootstrap inference on SR_a - SR_b.

    The SAME resampled dates are used for both series so their (often very high)
    correlation is retained - this is why the difference of two highly correlated
    Sharpes can be far less noisy than either alone.  Reports the percentile CI and
    a two-sided p-value for H0: SR_a = SR_b from the bootstrap distribution
    re-centred at zero: p = P(|d* - d_hat| >= |d_hat|).  Series inputs are
    inner-joined on their index.

    Reference: Ledoit & Wolf (2008), "Robust performance hypothesis testing with the
    Sharpe ratio", J. Empirical Finance 15 (we use the plain percentile version, not
    their studentised one).
    """
    a, b_ = _align(returns_a, returns_b)
    blk = _block(a, b_, mean_block)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(len(a), n_boot, blk, rng)
    d = _sr_rows(a[idx], periods_per_year) - _sr_rows(b_[idx], periods_per_year)
    obs = sharpe_ratio(a, periods_per_year) - sharpe_ratio(b_, periods_per_year)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = float((1 + (np.abs(d - obs) >= abs(obs)).sum()) / (n_boot + 1))
    return {"sharpe_a": sharpe_ratio(a, periods_per_year), "sharpe_b": sharpe_ratio(b_, periods_per_year),
            "diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_value": p,
            "mean_block": blk, "n_obs": len(a)}


# --------------------------------------------------------------------------- #
# 4. data-snooping across many candidate rules
# --------------------------------------------------------------------------- #
def probability_of_backtest_overfitting(returns_matrix, n_partitions: int = 16, max_combos: int | None = None,
                                        seed: int = 0) -> dict:
    """PBO by Combinatorially Symmetric Cross-Validation (CSCV).

    ``returns_matrix``: (T x N) return streams of N candidate configurations over
    the SAME dates (DataFrame or array).  Procedure:

    1. Split the T rows into S equal contiguous blocks (S even; leading remainder
       dropped).
    2. For each of the C(S, S/2) ways to choose S/2 blocks as in-sample (IS), the
       other half is out-of-sample (OOS).
    3. Find the configuration n* with the best IS Sharpe; find its relative rank
       omega = rank_OOS(n*) / (N + 1) among all configurations' OOS Sharpes.
    4. logit lambda = ln(omega / (1 - omega)).  PBO = P(lambda <= 0): the
       probability that the IS-optimal configuration is at or below the OOS median.

    Pure-noise candidates give PBO ~ 0.5; a genuinely superior candidate gives ~ 0.
    Also returns ``prob_loss`` (P(OOS Sharpe of the IS winner < 0)) and the OOS-on-IS
    regression ``degradation_slope`` (<1 means performance decays out of sample).
    Note PBO speaks only to the *selection procedure over the supplied candidates*;
    it is silent about anything tried but not included in the matrix.

    Reference: Bailey, Borwein, Lopez de Prado & Zhu (2017), "The probability of
    backtest overfitting", Journal of Computational Finance 20(4).
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("need a (T x N) matrix with N >= 2")
    M = M[~np.isnan(M).any(axis=1)]
    S = int(n_partitions)
    if S % 2:
        raise ValueError("n_partitions must be even")
    T, N = M.shape
    per = T // S
    M = M[T - per * S:]
    blocks = M.reshape(S, per, N)
    s1 = blocks.sum(axis=1)
    s2 = (blocks ** 2).sum(axis=1)
    tot1, tot2 = s1.sum(axis=0), s2.sum(axis=0)

    def sharpe_from(sum1, sum2, cnt):
        mean = sum1 / cnt
        var = (sum2 - cnt * mean ** 2) / (cnt - 1)
        return mean / np.sqrt(np.maximum(var, 1e-300))

    all_combos = list(combinations(range(S), S // 2))
    if max_combos is not None and len(all_combos) > max_combos:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(all_combos), size=max_combos, replace=False)
        all_combos = [all_combos[i] for i in pick]
    cnt = per * (S // 2)
    lam = np.empty(len(all_combos))
    is_best, oos_of_best = np.empty(len(all_combos)), np.empty(len(all_combos))
    for i, c in enumerate(all_combos):
        c = list(c)
        i1, i2 = s1[c].sum(axis=0), s2[c].sum(axis=0)
        sr_is = sharpe_from(i1, i2, cnt)
        sr_oos = sharpe_from(tot1 - i1, tot2 - i2, cnt)
        nstar = int(np.argmax(sr_is))
        rank = 1 + (sr_oos < sr_oos[nstar]).sum() + 0.5 * ((sr_oos == sr_oos[nstar]).sum() - 1)
        omega = rank / (N + 1.0)
        lam[i] = np.log(omega / (1.0 - omega))
        is_best[i], oos_of_best[i] = sr_is[nstar], sr_oos[nstar]
    xc = is_best - is_best.mean()
    slope = float((xc * (oos_of_best - oos_of_best.mean())).sum() / (xc ** 2).sum()) if (xc ** 2).sum() > 0 else np.nan
    return {"pbo": float((lam <= 0).mean()), "prob_loss": float((oos_of_best < 0).mean()),
            "degradation_slope": slope, "mean_is_sharpe_pp": float(is_best.mean()),
            "mean_oos_sharpe_pp": float(oos_of_best.mean()), "n_combos": len(all_combos),
            "n_candidates": N, "logits": lam}


def reality_check(returns_matrix, benchmark=None, n_boot: int = 1000, mean_block: float | None = None,
                  seed: int = 0) -> dict:
    """White's Reality Check and Hansen's SPA for "is the best of N rules better than
    the benchmark?".

    d_{t,k} = r_{t,k} - benchmark_t (benchmark = 0 if None).  H0: E[d_k] <= 0 for all
    k.  With dbar_k the sample means:

    * RC  (White 2000):    V = max_k sqrt(T) dbar_k ;   V*_b = max_k sqrt(T)(dbar*_{b,k} - dbar_k)
    * SPA (Hansen 2005):   studentise by omega_k (bootstrap std of sqrt(T) dbar*_k):
                           T_SPA = max(max_k sqrt(T) dbar_k / omega_k, 0), and recentre the
                           bootstrap at mu_k instead of dbar_k:  Z*_{b,k} = sqrt(T)(dbar*_{b,k} - dbar_k + mu_k)/omega_k.
        mu^u = 0                      -> "upper" p (= studentised RC, most conservative)
        mu^c = dbar_k 1{t_k <= -sqrt(2 ln ln T)} -> "consistent" p (the one to quote)
        mu^l = min(dbar_k, 0)         -> "lower" p (most liberal)

    Poor rules dilute the RC's critical value (it is not invariant to adding
    useless candidates); the SPA's recentring fixes that.  Uses the same
    stationary-bootstrap resamples for all N rules, so cross-rule dependence is
    preserved.  The statistic is the *mean* differential, not a Sharpe (a Sharpe
    version would resample the ratio; the studentised SPA t-stat is its close cousin).

    Returns p-values, the best rule's column index/name, and its mean differential
    (per period and annualised x252).
    """
    cols = None
    if isinstance(returns_matrix, pd.DataFrame):
        cols = list(returns_matrix.columns)
        R = returns_matrix.to_numpy(float)
        if benchmark is not None and isinstance(benchmark, pd.Series):
            benchmark = benchmark.reindex(returns_matrix.index)
    else:
        R = np.asarray(returns_matrix, float)
    bm = np.zeros(len(R)) if benchmark is None else np.asarray(benchmark, float).ravel()
    D = R - bm[:, None]
    keep = ~np.isnan(D).any(axis=1)
    D = D[keep]
    T, N = D.shape
    dbar = D.mean(axis=0)
    blk = _block(D.mean(axis=1)) if mean_block is None else float(mean_block)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, n_boot, blk, rng)
    boot_means = np.empty((n_boot, N))
    for b in range(n_boot):
        boot_means[b] = D[idx[b]].mean(axis=0)
    dev = np.sqrt(T) * (boot_means - dbar)          # B x N, centred at dbar
    omega = np.maximum(dev.std(axis=0, ddof=1), 1e-12)
    tstat = np.sqrt(T) * dbar / omega

    v_rc = np.sqrt(T) * dbar.max()
    rc_p = float((1 + (dev.max(axis=1) >= v_rc).sum()) / (n_boot + 1))
    t_spa = max(tstat.max(), 0.0)
    thr = -np.sqrt(2.0 * np.log(np.log(T)))
    mus = {"upper": np.zeros(N), "consistent": np.where(tstat <= thr, dbar, 0.0), "lower": np.minimum(dbar, 0.0)}
    spa = {}
    for name, mu in mus.items():
        z = (dev + np.sqrt(T) * mu) / omega
        stat_b = np.maximum(z.max(axis=1), 0.0)
        spa[name] = float((1 + (stat_b >= t_spa).sum()) / (n_boot + 1))
    best_by_mean = int(np.argmax(dbar))
    return {"rc_pvalue": rc_p, "spa_pvalue": spa["consistent"], "spa_upper": spa["upper"], "spa_lower": spa["lower"],
            "best_rule": cols[best_by_mean] if cols else best_by_mean, "best_rule_idx": best_by_mean,
            "best_mean_diff_ann": float(dbar[best_by_mean] * 252), "best_t": float(tstat[best_by_mean]),
            "n_rules": N, "n_obs": T, "mean_block": blk}


# --------------------------------------------------------------------------- #
# 5. better nulls for a timing signal
# --------------------------------------------------------------------------- #
def null_positions(position: pd.Series, kind: str, rng, min_shift: int | None = None,
                   mean_block: float = 21.0) -> pd.Series:
    """One draw from a null distribution of position series (same index as input).

    ``iid_shuffle``     permute individual days (the FLAWED null of notebook 06: it
                        destroys the persistence of the position, so the null is
                        a strategy that flips exposure randomly every day).
    ``circular_shift``  roll the whole position series by a random lag >= ``min_shift``
                        (default max(21, n/20)).  Preserves the position's exact
                        marginal distribution, autocorrelation and turnover; only its
                        alignment with the return series is broken.
    ``block_permute``   cut the series into geometric(mean ``mean_block``) blocks and
                        shuffle the blocks.  Matches exposure level, marginal
                        distribution and local persistence (regime durations) while
                        randomising *when* each regime/size is held.
    """
    v = position.to_numpy(float)
    n = len(v)
    if kind == "iid_shuffle":
        out = rng.permutation(v)
    elif kind == "circular_shift":
        ms = max(21, n // 20) if min_shift is None else int(min_shift)
        k = int(rng.integers(ms, n - ms + 1))
        out = np.roll(v, k)
    elif kind == "block_permute":
        lens, tot = [], 0
        p = 1.0 / max(mean_block, 1.0)
        while tot < n:
            L = int(rng.geometric(p))
            L = min(L, n - tot)
            lens.append(L)
            tot += L
        edges = np.concatenate([[0], np.cumsum(lens)])
        blocks = [v[edges[i]:edges[i + 1]] for i in range(len(lens))]
        order = rng.permutation(len(blocks))
        out = np.concatenate([blocks[i] for i in order])
    else:
        raise ValueError("kind must be iid_shuffle | circular_shift | block_permute")
    return pd.Series(out, index=position.index)


def timing_null_test(position: pd.Series, score_fn, kind: str = "circular_shift", n_sims: int = 1000,
                     seed: int = 0, **kw) -> dict:
    """Is the strategy's Sharpe better than that of *randomly timed* versions of itself?

    ``score_fn(position: pd.Series) -> float`` must run the FULL pipeline (lagging,
    P&L, vol-targeting, costs) and return a Sharpe, so the null differs from the
    actual run only in the position series.  Returns the actual score, the null
    draws, the strategy's percentile in the null (fraction of nulls <= actual), the
    one-sided p-value (1 + #{null >= actual}) / (n_sims + 1), and
    ``constant_exposure_score``: the score of a flat position at the same mean
    exposure - the "no-skill, just ride the premium" reference.

    Why the original null is flawed: a short-variance book earns the variance risk
    premium on average, so shuffling positions across dates leaves a null centred
    far above zero (mean Sharpe 1.75 in notebook 06) - and, worse, it makes the null
    positions erratic day-to-day, which changes the vol-target leverage path, so the
    "null" is a different strategy, not an unskilled version of this one.  Shift and
    block-permutation nulls keep the position's dynamics and exposure intact.
    """
    rng = np.random.default_rng(seed)
    actual = float(score_fn(position))
    null = np.empty(n_sims)
    for i in range(n_sims):
        null[i] = score_fn(null_positions(position, kind, rng, **kw))
    null_f = null[~np.isnan(null)]
    const = float(score_fn(pd.Series(position.mean(), index=position.index)))
    return {"kind": kind, "actual": actual, "null": null_f, "null_mean": float(null_f.mean()),
            "null_std": float(null_f.std(ddof=1)), "null_p95": float(np.percentile(null_f, 95)),
            "percentile": float((null_f <= actual).mean()),
            "p_value": float((1 + (null_f >= actual).sum()) / (len(null_f) + 1)),
            "constant_exposure_score": const, "n_sims": len(null_f)}


# --------------------------------------------------------------------------- #
# 6. one-call entry point
# --------------------------------------------------------------------------- #
def validate(returns, n_trials: int = 1, benchmark=None, periods_per_year: float = 252,
             trial_sharpes=None, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
             mean_block: float | None = None) -> dict:
    """Run the full "is this Sharpe real?" battery on one return stream.

    Parameters
    ----------
    returns : daily EXCESS returns (Series/array).  Be honest about the P&L first.
    n_trials : number of distinct strategy variants evaluated on this data in the
        whole project, including failures (docs/EVAL_PROTOCOL.md rule 3).  Drives the DSR.
    benchmark : optional second return stream (e.g. constant-exposure or buy-and-hold);
        adds a paired block-bootstrap test of SR(returns) - SR(benchmark).
    trial_sharpes : optional annualised Sharpes of all trials (sets cross-trial variance).

    Returns a flat dict of floats.  Key fields: ``sharpe``; ``sharpe_lo`` (Lo-adjusted);
    ``se_mertens`` / ``se_hac`` (annualised); ``psr0``; ``dsr``; ``sr0_hurdle`` (the
    annualised Sharpe a null best-of-N would show); ``min_trl_years``;
    ``ci_low``/``ci_high`` (block bootstrap); ``verdict``.

    ``verdict`` is "PASS" only if DSR >= 1-alpha AND the bootstrap CI excludes 0;
    otherwise "FAIL" with the reason.  PASS is necessary, not sufficient: it says
    nothing about whether the P&L is tradable (costs, MTM, capacity).
    """
    r_series = returns
    r = _arr(returns)
    T = len(r)
    ppy = periods_per_year
    mu, sd, g3, g4 = _moments(r)
    ci = bootstrap_sharpe_ci(r, n_boot, alpha, mean_block, ppy, seed)
    det = deflated_sharpe_ratio(r, n_trials, trial_sharpes, None, ppy, return_details=True)
    mtrl = min_track_record_length(r, 0.0, alpha, ppy)
    mtrl_h = min_track_record_length(r, det["sr0_annual"], alpha, ppy)
    out = {
        "n_obs": T, "years": T / ppy, "n_trials": int(n_trials),
        "sharpe": sharpe_ratio(r, ppy), "sharpe_lo": lo_adjusted_sharpe(r, ppy),
        "rho1": float(_autocorr(r, 1)[0]), "skew": g3, "excess_kurtosis": g4 - 3.0,
        "se_mertens": sharpe_se(r, "mertens", ppy), "se_hac": sharpe_se(r, "hac", ppy),
        "psr0": probabilistic_sharpe_ratio(r, 0.0, ppy),
        "dsr": det["dsr"], "sr0_hurdle": det["sr0_annual"],
        "min_trl_years": mtrl["years"], "min_trl_years_vs_hurdle": mtrl_h["years"],
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "boot_se": ci["boot_se"],
        "mean_block": ci["mean_block"],
    }
    if benchmark is not None:
        d = bootstrap_sharpe_diff(r_series, benchmark, n_boot, alpha, mean_block, ppy, seed)
        out.update({"bench_sharpe": d["sharpe_b"], "sharpe_diff": d["diff"], "diff_ci_low": d["ci_low"],
                    "diff_ci_high": d["ci_high"], "diff_p_value": d["p_value"]})
    reasons = []
    if not out["dsr"] >= 1 - alpha:
        reasons.append(f"DSR {out['dsr']:.2f} < {1 - alpha:.2f}")
    if not out["ci_low"] > 0:
        reasons.append("bootstrap CI includes 0")
    out["verdict"] = "PASS" if not reasons else "FAIL: " + "; ".join(reasons)
    return out


def validate_table(strategies: dict, n_trials: int = 1, benchmark=None, **kw) -> pd.DataFrame:
    """``validate`` for several named return streams -> DataFrame (one row each).
    ``benchmark`` may be one series (applied to all) or a dict keyed like ``strategies``."""
    rows = {}
    for name, r in strategies.items():
        bm = benchmark.get(name) if isinstance(benchmark, dict) else benchmark
        rows[name] = validate(r, n_trials=n_trials, benchmark=bm, **kw)
    return pd.DataFrame(rows).T.infer_objects()
