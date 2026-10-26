"""Tests for vrp_strategy.stats_tests: analytic identities plus Monte-Carlo
size/coverage/power checks.  Monte-Carlo tolerances are set wide enough to be
stable with fixed seeds but tight enough to catch a wrong formula."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vrp_strategy import stats_tests as st

PPY = 252


def ar1(n, phi, mu, sd_uncond, rng):
    """AR(1) with given unconditional mean/sd."""
    e = rng.standard_normal(n + 200) * sd_uncond * np.sqrt(1 - phi ** 2)
    x = np.zeros(n + 200)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t]
    return mu + x[200:]


# ------------------------------------------------------------------ basics
def test_sharpe_known_value():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    assert st.sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))


def test_nans_are_dropped():
    r = np.array([0.01, np.nan, -0.01, 0.02])
    assert st.sharpe_ratio(r) == pytest.approx(st.sharpe_ratio(r[~np.isnan(r)]))


# ------------------------------------------------------------------ SE / Lo
def test_mertens_equals_hac_with_zero_lags():
    rng = np.random.default_rng(1)
    r = rng.standard_t(5, 3000) * 0.01 + 0.0006
    assert st.sharpe_se(r, "mertens") == pytest.approx(st.sharpe_se(r, "hac", max_lag=0), rel=0.01)


def test_mertens_reduces_to_normal_formula():
    rng = np.random.default_rng(2)
    r = rng.standard_normal(200_000) * 0.01 + 0.001
    sr = r.mean() / r.std(ddof=1)
    assert st.sharpe_se(r) == pytest.approx(np.sqrt((1 + sr ** 2 / 2) / len(r)), rel=0.03)


def test_se_matches_empirical_sampling_sd_for_skewed_returns():
    """Short-vol-like returns (negative skew, fat left tail): empirical SD of SR_hat
    across simulated samples must match the skew/kurtosis-aware SE, and the naive
    normal formula must be visibly too small."""
    rng = np.random.default_rng(3)
    T, reps = 750, 1500

    def draw():
        # 97% small gain, 3% large loss -> strong negative skew
        crash = rng.random(T) < 0.03
        return np.where(crash, -0.05 + rng.standard_normal(T) * 0.01, 0.004 + rng.standard_normal(T) * 0.003)

    srs = np.array([(lambda r: r.mean() / r.std(ddof=1))(draw()) for _ in range(reps)])
    big = draw()
    big = np.concatenate([draw() for _ in range(40)])  # for stable moments
    se_mert = st.sharpe_se(big) * np.sqrt(len(big) / T)  # rescale to sample length T
    se_naive = np.sqrt((1 + (big.mean() / big.std(ddof=1)) ** 2 / 2) / T)
    emp = srs.std(ddof=1)
    assert se_mert == pytest.approx(emp, rel=0.12)
    assert se_naive < 0.85 * emp


def test_hac_se_larger_than_iid_se_under_positive_autocorrelation():
    rng = np.random.default_rng(4)
    r = ar1(20_000, 0.3, 0.0005, 0.01, rng)
    assert st.sharpe_se(r, "hac") > 1.3 * st.sharpe_se(r, "mertens")


def test_lo_adjustment_matches_analytic_ar1_and_iid():
    rng = np.random.default_rng(5)
    phi, mu, sd = 0.2, 0.0006, 0.01
    r = ar1(200_000, phi, mu, sd, rng)
    q = 252
    k = np.arange(1, q)
    eta = q / np.sqrt(q + 2 * ((q - k) * phi ** k).sum())
    expected = mu / sd * eta
    assert st.lo_adjusted_sharpe(r, max_lag=30) == pytest.approx(expected, rel=0.05)
    assert st.lo_adjusted_sharpe(r, max_lag=30) < st.sharpe_ratio(r)  # positive rho inflates naive SR
    iid = rng.standard_normal(200_000) * sd + mu
    assert st.lo_adjusted_sharpe(iid) == pytest.approx(st.sharpe_ratio(iid), rel=0.05)


# ------------------------------------------------------------------ PSR / DSR / MinTRL
def test_psr_false_positive_rate_is_nominal():
    rng = np.random.default_rng(6)
    hits = np.mean([st.probabilistic_sharpe_ratio(rng.standard_normal(500) * 0.01) > 0.95 for _ in range(1500)])
    assert 0.03 < hits < 0.07


def test_expected_max_sharpe_matches_monte_carlo():
    rng = np.random.default_rng(7)
    for n in (10, 100, 1000):
        mc = rng.standard_normal((4000, n)).max(axis=1).mean()
        assert st.expected_max_sharpe(n, 1.0) == pytest.approx(mc, abs=0.12)
    assert st.expected_max_sharpe(1, 1.0) == 0.0
    assert st.expected_max_sharpe(500, 2.0) == pytest.approx(2 * st.expected_max_sharpe(500, 1.0))


def test_dsr_controls_false_positives_with_many_trials():
    """Pick the best of N=100 zero-edge strategies.  Naive PSR calls it 'significant'
    almost always; the DSR (with the true N) rejects about <= 5-8% of the time."""
    rng = np.random.default_rng(8)
    N, T, reps = 100, 252, 400
    dsr_hit = psr_hit = 0
    for _ in range(reps):
        X = rng.standard_normal((T, N)) * 0.01
        srs = X.mean(0) / X.std(0, ddof=1) * np.sqrt(PPY)
        best = X[:, int(np.argmax(srs))]
        psr_hit += st.probabilistic_sharpe_ratio(best) > 0.95
        dsr_hit += st.deflated_sharpe_ratio(best, N, trial_sharpes=srs) > 0.95
    assert psr_hit / reps > 0.80
    assert dsr_hit / reps < 0.09


def test_dsr_default_variance_also_controls_size():
    rng = np.random.default_rng(9)
    N, T, reps = 50, 500, 300
    hit = 0
    for _ in range(reps):
        X = rng.standard_normal((T, N)) * 0.01
        best = X[:, int(np.argmax(X.mean(0) / X.std(0, ddof=1)))]
        hit += st.deflated_sharpe_ratio(best, N) > 0.95
    assert hit / reps < 0.09


def test_dsr_has_power_and_decreases_in_n():
    rng = np.random.default_rng(10)
    r = rng.standard_normal(1000) * 0.01 + 0.0022  # SR ~ 3.5 annualised
    d1, d50, d5000 = (st.deflated_sharpe_ratio(r, n) for n in (1, 50, 5000))
    assert d1 >= d50 >= d5000
    assert d50 > 0.95
    assert d5000 < d50


def test_min_track_record_consistent_with_psr_and_normal_case():
    rng = np.random.default_rng(11)
    r = rng.standard_normal(5000) * 0.01 + 0.0008
    out = st.min_track_record_length(r, 0.0, alpha=0.05)
    mu, sd, g3, g4 = st._moments(r)
    sr = mu / sd
    factor = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    n = out["n_obs"]
    assert sr / np.sqrt(factor / (n - 1)) == pytest.approx(stats.norm.ppf(0.95), rel=1e-6)
    assert out["years"] == pytest.approx(n / 252)
    # normal-return closed form 1 + (1 + SR^2/2)(1.645/SR)^2
    assert n == pytest.approx(1 + (1 + sr ** 2 / 2) * (1.6449 / sr) ** 2, rel=0.06)
    assert np.isinf(st.min_track_record_length(-r)["n_obs"])


# ------------------------------------------------------------------ bootstrap
def test_block_length_selection():
    rng = np.random.default_rng(12)
    assert st.optimal_block_length(rng.standard_normal(3000)) < 5
    assert st.optimal_block_length(ar1(3000, 0.8, 0, 1, rng)) > 8


def test_stationary_bootstrap_indices_properties():
    rng = np.random.default_rng(13)
    idx = st.stationary_bootstrap_indices(500, 400, 10.0, rng)
    assert idx.min() >= 0 and idx.max() < 500
    runs = (np.diff(idx, axis=1) % 500 == 1)  # continuation (new blocks also hit it w.p. 1/n)
    mean_block = 1.0 / (1.0 - runs.mean())
    assert mean_block == pytest.approx(10.0, rel=0.1)


@pytest.mark.parametrize("phi", [0.0, 0.2])
def test_bootstrap_ci_coverage(phi):
    """Nominal 90% percentile interval should cover the true Sharpe ~90% of the time
    (iid and AR(1) data)."""
    rng = np.random.default_rng(14 + int(phi * 10))
    mu, sd, T = 0.0004, 0.01, 750
    true_sr = mu / sd * np.sqrt(PPY)
    reps, cover = 250, 0
    for i in range(reps):
        r = ar1(T, phi, mu, sd, rng)
        ci = st.bootstrap_sharpe_ci(r, n_boot=250, alpha=0.10, seed=i)
        cover += ci["ci_low"] <= true_sr <= ci["ci_high"]
    assert 0.82 <= cover / reps <= 0.97


def test_bootstrap_diff_paired():
    rng = np.random.default_rng(15)
    common = rng.standard_normal(1500) * 0.01
    a = pd.Series(common + 0.0004 + rng.standard_normal(1500) * 0.002)
    same = st.bootstrap_sharpe_diff(a, a, n_boot=300)
    assert same["diff"] == 0 and same["ci_low"] == 0 == same["ci_high"]
    b = pd.Series(common + rng.standard_normal(1500) * 0.002)  # same vol, zero mean
    out = st.bootstrap_sharpe_diff(a, b, n_boot=500, seed=1)
    assert out["diff"] > 0 and out["ci_low"] < out["diff"] < out["ci_high"]
    # pairing: CI for the diff is much narrower than the independent-bootstrap sum of widths
    ci_a = st.bootstrap_sharpe_ci(a, n_boot=500, seed=1)
    assert (out["ci_high"] - out["ci_low"]) < 0.8 * (ci_a["ci_high"] - ci_a["ci_low"])


def test_bootstrap_diff_size_under_equal_sharpes():
    rng = np.random.default_rng(16)
    rej, reps = 0, 150
    for i in range(reps):
        common = rng.standard_normal(750) * 0.01
        a = common + 0.0003 + rng.standard_normal(750) * 0.004
        b = common + 0.0003 + rng.standard_normal(750) * 0.004
        rej += st.bootstrap_sharpe_diff(a, b, n_boot=200, seed=i)["p_value"] < 0.05
    assert rej / reps < 0.11


# ------------------------------------------------------------------ PBO
def test_pbo_pure_noise_is_about_half():
    rng = np.random.default_rng(17)
    # PBO of a single dataset is very noisy (sd ~0.19: all CSCV splits reuse the same
    # data) but unbiased at 0.5, so average over many datasets.
    pbos = [st.probability_of_backtest_overfitting(rng.standard_normal((1600, 40)) * 0.01, 16,
                                                   max_combos=1000, seed=i)["pbo"] for i in range(50)]
    assert 0.42 < np.mean(pbos) < 0.58


def test_pbo_low_with_one_truly_better_candidate():
    rng = np.random.default_rng(18)
    X = rng.standard_normal((1600, 40)) * 0.01
    X[:, 7] += 0.0015
    out = st.probability_of_backtest_overfitting(X, 16)
    assert out["pbo"] < 0.05
    assert out["prob_loss"] < 0.05
    assert out["n_combos"] == 12870
    assert out["mean_oos_sharpe_pp"] > 0.03


def test_pbo_high_when_is_winner_is_anti_persistent():
    """Candidates whose performance flips sign between halves of the sample: the IS
    winner is the OOS loser, PBO should be large."""
    rng = np.random.default_rng(19)
    T, N = 1600, 30
    X = rng.standard_normal((T, N)) * 0.01
    sign = np.where(np.arange(T) < T // 2, 1.0, -1.0)[:, None]
    X += sign * rng.normal(0, 0.0015, N)[None, :]
    assert st.probability_of_backtest_overfitting(X, 16)["pbo"] > 0.6


def test_pbo_validation_and_subsampling():
    X = np.random.default_rng(0).standard_normal((320, 5))
    with pytest.raises(ValueError):
        st.probability_of_backtest_overfitting(X, 7)
    assert st.probability_of_backtest_overfitting(X, 16, max_combos=100)["n_combos"] == 100


# ------------------------------------------------------------------ RC / SPA
def test_reality_check_size_under_null():
    rng = np.random.default_rng(20)
    rc = spa = 0
    reps = 120
    for i in range(reps):
        X = rng.standard_normal((500, 40)) * 0.01
        out = st.reality_check(X, n_boot=200, seed=i)
        rc += out["rc_pvalue"] < 0.05
        spa += out["spa_pvalue"] < 0.05
        assert out["spa_lower"] <= out["spa_pvalue"] + 1e-12 <= out["spa_upper"] + 2e-12
    assert rc / reps < 0.12 and spa / reps < 0.12


def test_reality_check_power_and_benchmark_handling():
    rng = np.random.default_rng(21)
    X = rng.standard_normal((1000, 40)) * 0.01
    X[:, 3] += 0.0010
    out = st.reality_check(pd.DataFrame(X, columns=[f"r{k}" for k in range(40)]), n_boot=400)
    assert out["best_rule"] == "r3" and out["rc_pvalue"] < 0.05 and out["spa_pvalue"] < 0.05
    # a benchmark that already earns the edge removes the evidence
    out2 = st.reality_check(X, benchmark=X[:, 3], n_boot=400)
    assert out2["spa_pvalue"] > 0.2


# ------------------------------------------------------------------ nulls
def test_null_position_properties():
    rng = np.random.default_rng(22)
    pos = pd.Series(np.clip(np.cumsum(rng.standard_normal(1000)) * 0.1 + 0.5, 0, 1),
                    index=pd.bdate_range("2016-01-01", periods=1000))
    for kind in ("iid_shuffle", "circular_shift", "block_permute"):
        p = st.null_positions(pos, kind, rng)
        assert p.index.equals(pos.index)
        assert np.sort(p.values) == pytest.approx(np.sort(pos.values))  # same exposure distribution
    def ac1(x):
        x = np.asarray(x) - np.mean(x)
        return (x[1:] * x[:-1]).sum() / (x * x).sum()
    sh = st.null_positions(pos, "circular_shift", rng)
    assert ac1(sh) == pytest.approx(ac1(pos), abs=0.01)
    assert abs(ac1(st.null_positions(pos, "iid_shuffle", rng))) < 0.2
    bp = st.null_positions(pos, "block_permute", rng, mean_block=21)
    assert ac1(bp) > 0.8 * ac1(pos)
    assert not np.allclose(sh.values, pos.values)


def _toy_world(rng, n=750, informative=False):
    ret = rng.standard_normal(n) * 0.01 + 0.0003
    if informative:
        # position tomorrow tracks a persistent latent that truly predicts returns
        lat = ar1(n, 0.95, 0, 1, rng)
        ret = ret + 0.004 * np.tanh(lat)
        pos = np.clip(0.5 + 0.5 * np.tanh(lat), 0, 1)
    else:
        pos = np.clip(0.5 + 0.5 * np.tanh(ar1(n, 0.95, 0, 1, rng)), 0, 1)
    idx = pd.bdate_range("2016-01-01", periods=n)
    return pd.Series(pos, idx), pd.Series(ret, idx)


@pytest.mark.parametrize("kind", ["circular_shift", "block_permute"])
def test_timing_null_size_and_power(kind):
    rng = np.random.default_rng(23)
    rej, reps = 0, 60
    for i in range(reps):
        pos, ret = _toy_world(rng)
        score = lambda p: st.sharpe_ratio(p * ret)
        rej += st.timing_null_test(pos, score, kind, n_sims=99, seed=i)["p_value"] < 0.05
    assert rej / reps < 0.13
    pos, ret = _toy_world(rng, informative=True)
    out = st.timing_null_test(pos, lambda p: st.sharpe_ratio(p * ret), kind, n_sims=299, seed=1)
    assert out["p_value"] < 0.05 and out["percentile"] > 0.95
    assert out["constant_exposure_score"] < out["actual"]


# ------------------------------------------------------------------ validate
def test_validate_noise_fails_and_strong_passes():
    rng = np.random.default_rng(24)
    idx = pd.bdate_range("2016-01-01", periods=1500)
    noise = pd.Series(rng.standard_normal(1500) * 0.01, index=idx)
    strong = pd.Series(rng.standard_normal(1500) * 0.01 + 0.0015, index=idx)
    bad = st.validate(noise, n_trials=100, n_boot=300)
    good = st.validate(strong, n_trials=100, benchmark=noise, n_boot=300)
    assert bad["verdict"].startswith("FAIL") and good["verdict"] == "PASS"
    assert good["sharpe_diff"] > 0 and good["diff_p_value"] < 0.05
    for k in ("sharpe", "sharpe_lo", "se_mertens", "se_hac", "psr0", "dsr", "sr0_hurdle",
              "min_trl_years", "ci_low", "ci_high"):
        assert k in good
    tab = st.validate_table({"noise": noise, "strong": strong}, n_trials=10, n_boot=200)
    assert list(tab.index) == ["noise", "strong"] and tab.loc["strong", "sharpe"] > tab.loc["noise", "sharpe"]
