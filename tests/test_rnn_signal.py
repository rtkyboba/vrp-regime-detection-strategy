"""Causality, purge/embargo and determinism tests for vrp_strategy.rnn_signal (offline, synthetic data)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from vrp_strategy import rnn_signal as R


def synth_raw(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    spx = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    vix = 15 + 5 * np.abs(np.sin(np.arange(n) / 30)) + rng.normal(0, 0.5, n).cumsum() * 0.05
    return pd.DataFrame({"spx": spx, "vix": vix, "vix3m": vix + 1.0 + rng.normal(0, .2, n),
                         "vix6m": vix + 2.0 + rng.normal(0, .2, n), "vvix": 90 + rng.normal(0, 3, n).cumsum() * 0.1,
                         "skew": 125 + rng.normal(0, 2, n)}, index=idx)


def synth_ds(n=700, d=len(R.FEATURE_COLS), h=5, seed=1):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, d)).cumsum(0) * 0.05 + rng.normal(size=(n, d))
    y = 0.5 * np.tanh(F[:, 0]) + rng.normal(0, 0.5, n)
    y[-h:] = np.nan
    cls = np.where(np.isnan(y), np.nan, (y > 0).astype(float))
    return R.Dataset(pd.bdate_range("2010-01-04", periods=n), F, y, cls, h, list(R.FEATURE_COLS))


# ------------------------------------------------------------------ features / targets
def test_features_do_not_change_when_future_is_corrupted():
    raw = synth_raw()
    f0 = R.build_features(raw)
    cut = 300
    bad = raw.copy()
    bad.iloc[cut + 1:] = bad.iloc[cut + 1:] * np.random.default_rng(5).uniform(0.3, 3.0, size=(len(bad) - cut - 1, bad.shape[1]))
    f1 = R.build_features(bad)
    pd.testing.assert_frame_equal(f0.iloc[: cut + 1], f1.iloc[: cut + 1])
    assert not f0.iloc[cut + 1:].equals(f1.iloc[cut + 1:])


def test_variance_payoff_known_value_and_future_window():
    n, h, r = 80, 21, 0.01
    spx = pd.Series(100 * np.exp(r * np.arange(n)), index=pd.bdate_range("2020-01-01", periods=n))
    vix = pd.Series(20.0, index=spx.index)
    p = R.variance_payoff(vix, spx, h)
    strike = 0.2 ** 2 * h / 252
    assert p["payoff"].iloc[10] == pytest.approx(strike - h * r * r)
    assert p["scaled"].iloc[10] == pytest.approx(1 - h * r * r / strike)
    assert p["payoff"].iloc[-h:].isna().all() and p["payoff"].iloc[-h - 1] == p["payoff"].iloc[-h - 1]
    # payoff at t must not depend on returns at or before t
    spx2 = spx.copy()
    spx2.iloc[:10] *= 3.0          # changes returns r_1..r_10 only (rows 0..9 shifted, jump lands on r_10)
    p2 = R.variance_payoff(vix, spx2, h)
    assert p2["payoff"].iloc[10] == pytest.approx(p["payoff"].iloc[10])
    assert p2["payoff"].iloc[9] != pytest.approx(p["payoff"].iloc[9])   # t=9 does see the r_10 jump


def test_forward_return_window():
    px = pd.Series([100, 101, 102, 103, 104, 105, 106.0], index=pd.bdate_range("2020-01-01", periods=7))
    f = R.forward_return(px, 3)
    assert f.iloc[0] == pytest.approx(103 / 100 - 1)
    assert f.iloc[-3:].isna().all()


# ------------------------------------------------------------------ purge / embargo
@pytest.mark.parametrize("h,emb", [(21, 21), (5, 5), (21, 30)])
def test_purged_split_labels_never_overlap_prediction_block_or_embargo(h, emb):
    t0, V = 900, 252
    tr, va, full = R.purged_split(t0, h, emb, V, min_row=59)
    # label window of i is (i, i+h]; it must end strictly before t0 - emb
    assert full.max() + h + emb < t0
    assert full.max() + h + emb == t0 - 1       # tight: no needless slack
    assert (tr.max() + h + emb) < va.min()      # inner-train labels do not reach the validation block
    assert tr.min() == 59 and len(va) == V
    assert set(tr).isdisjoint(va)
    # Purged samples really are removed: nothing between te and t0
    assert not np.isin(np.arange(t0 - h - emb, t0), full).any()


def test_walk_forward_training_set_excludes_samples_whose_labels_touch_the_block():
    ds = synth_ds()
    seen = []
    orig = R.har_fit_predict

    def spy(ds_, tr, va, pred, t0):
        full = np.concatenate([tr, va])
        seen.append((t0, full.max(), pred.min()))
        return orig(ds_, tr, va, pred, t0)

    R.har_fit_predict = spy
    try:
        R.walk_forward(ds, "har", step=50, first_t0=300, val_len=80)
    finally:
        R.har_fit_predict = orig
    assert seen
    for t0, last, p0 in seen:
        assert last + ds.h + ds.h < t0 == p0


# ------------------------------------------------------------------ standardisation / windowing causality
def test_standardise_uses_only_rows_up_to_cutoff():
    F = np.random.default_rng(0).normal(size=(200, 4))
    Z0 = R._standardise(F, 99)
    F2 = F.copy()
    F2[100:] += 50.0
    Z1 = R._standardise(F2, 99)
    assert np.allclose(Z0[:100], Z1[:100])


def test_flat_features_are_trailing():
    F = np.random.default_rng(0).normal(size=(100, 3))
    X0 = R._flat_features(F)
    F2 = F.copy()
    F2[60:] = 99.0
    X1 = R._flat_features(F2)
    assert np.allclose(X0[:60], X1[:60])


def test_windows_end_at_the_sample_row():
    Z = np.arange(40, dtype=float).reshape(20, 2)
    w = R._windows(Z, np.array([9, 15]), 4).numpy()
    assert w.shape == (2, 4, 2)
    assert np.allclose(w[0, -1], Z[9]) and np.allclose(w[0, 0], Z[6])


def corrupt_future(ds, c):
    """Return a copy with every feature row >= c and every label that reaches past c-1 scrambled."""
    rng = np.random.default_rng(9)
    F = ds.F.copy()
    F[c:] = rng.normal(size=F[c:].shape) * 10
    y = ds.y_reg.copy()
    y[c - ds.h:] = rng.normal(size=len(y[c - ds.h:])) * 3
    cls = np.where(np.isnan(y), np.nan, (y > 0).astype(float))
    return dataclasses.replace(ds, F=F, y_reg=y, y_cls=cls)


@pytest.mark.parametrize("kind", ["har", "ridge", "hgb", "rnn"])
def test_walk_forward_prediction_is_causal(kind):
    ds = synth_ds()
    c = 520
    ds2 = corrupt_future(ds, c)
    cfg = R.RNNConfig("gru", 4, 10, 0.1, 1e-3, max_epochs=3, patience=2)
    kw = dict(step=40, first_t0=300, val_len=80, seeds=(0, 1), cfg=cfg, hgb_kw=dict(max_iter=10))
    a = R.walk_forward(ds, kind, **kw)
    b = R.walk_forward(ds2, kind, **kw)
    m = np.asarray(a.dates < ds.index[c])
    assert m.sum() > 100
    np.testing.assert_allclose(a.reg[:, m], b.reg[:, m], atol=1e-6)
    if a.prob is not None:
        np.testing.assert_allclose(a.prob[:, m], b.prob[:, m], atol=1e-6)
    # ...and the later predictions DO change (the test is not vacuous)
    assert not np.allclose(a.reg[:, ~m], b.reg[:, ~m])


def test_truncated_run_reproduces_full_run_predictions():
    ds = synth_ds()
    cut = 560
    short = R.truncate(ds, str(ds.index[cut].date()))
    a = R.walk_forward(ds, "ridge", step=40, first_t0=300, val_len=80)
    b = R.walk_forward(short, "ridge", step=40, first_t0=300, val_len=80)
    common = b.dates
    ia = a.dates.get_indexer(common)
    np.testing.assert_allclose(a.reg[:, ia], b.reg, atol=1e-8)


# ------------------------------------------------------------------ determinism
def test_rnn_is_deterministic_and_seeds_differ():
    ds = synth_ds()
    cfg = R.RNNConfig("lstm", 4, 10, 0.1, 1e-3, max_epochs=4, patience=3)
    kw = dict(step=100, first_t0=350, val_len=80, cfg=cfg)
    a = R.walk_forward(ds, "rnn", seeds=(0, 1), **kw)
    b = R.walk_forward(ds, "rnn", seeds=(0, 1), **kw)
    np.testing.assert_array_equal(a.prob, b.prob)
    assert not np.allclose(a.prob[0], a.prob[1])
    c = R.walk_forward(ds, "rnn", seeds=(0,), **kw)
    np.testing.assert_array_equal(a.prob[0], c.prob[0])   # member 0 does not depend on ensemble size


# ------------------------------------------------------------------ signals / stats / economics
def test_signal_ranges():
    p = np.linspace(0, 1, 11)
    s = R.signal_from_prob(p)
    assert s.min() == -1 and s.max() == 1
    t = R.signal_from_reg(np.array([-100, 0, 100.0]), np.array([1.0, 1.0, 1.0]))
    assert (np.abs(t) <= 1).all() and t[1] == 0


def test_exposure_map():
    s = pd.Series([-1.0, 0.0, 0.5, 1.0, np.nan])
    assert R.exposure_from_signal(s, "linear").dropna().tolist() == [0.0, 0.5, 0.75, 1.0]
    assert R.exposure_from_signal(s, "longflat").dropna().tolist() == [0.0, 0.0, 0.5, 1.0]
    assert R.exposure_from_signal(s, "sign").dropna().tolist() == [0.0, 0.0, 1.0, 1.0]


def test_spearman_and_hit_rate():
    a = np.arange(50.0)
    assert R.spearman(a, a ** 3) == pytest.approx(1.0)
    assert R.spearman(a, -a) == pytest.approx(-1.0)
    assert R.hit_rate(np.array([1, -1, 1, 1.0]), np.array([2, 3, 1, -1.0])) == 0.5


def test_block_bootstrap_ci_covers_truth_for_noise_and_detects_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    est, lo, hi, _ = R.block_bootstrap(R.spearman, [x, rng.normal(size=600)], block=20, reps=300)
    assert lo < 0 < hi
    est, lo, hi, _ = R.block_bootstrap(R.spearman, [x, x + rng.normal(size=600)], block=20, reps=300)
    assert lo > 0.4


def test_exposure_pnl_is_lagged_and_charges_costs(monkeypatch):
    idx = pd.bdate_range("2020-01-01", periods=6)
    ret = pd.Series([0.0, 0.10, -0.50, 0.0, 0.0, 0.0], index=idx)
    monkeypatch.setattr(R.D, "rf_daily", lambda i: pd.Series(0.0, index=i))
    # exposure decided at close of day 1 (=1) must NOT earn day 1's return, only day 2's
    expo = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=idx)
    pnl = R.exposure_pnl(expo, ret, cost_bps=2.0)
    assert pnl.loc[idx[1], "gross_excess"] == 0.0
    assert pnl.loc[idx[2], "gross_excess"] == pytest.approx(-0.50)
    # entry cost on day 2 (position goes 0 -> 1) and exit cost on day 4
    assert pnl.loc[idx[2], "net_excess"] == pytest.approx(-0.50 - 2e-4)
    assert pnl.loc[idx[4], "net_excess"] == pytest.approx(-2e-4)


def test_calibration_and_brier():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(float)
    t = R.calibration_table(p, y, 10)
    assert np.abs(t.mean_pred - t.freq).max() < 0.08
    assert R.brier_skill(p, y) > 0.1
    assert R.brier_skill(np.full(5000, y.mean()), y) == pytest.approx(0.0, abs=1e-9)
