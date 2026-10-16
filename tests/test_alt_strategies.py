"""Tests for alt_strategies: causality (no look-ahead) and cost accounting."""
import numpy as np
import pandas as pd
import pytest

from vrp_strategy import alt_strategies as A

STRATS = {
    "vm_implied": lambda P: A.vol_managed(P, "implied"),
    "vm_realised_daily": lambda P: A.vol_managed(P, "realised", rebal=1),
    "vm_overlay": lambda P: A.vol_managed(P, "implied", overlay=True),
    "ts_bin": lambda P: A.ts_carry(P),
    "ts_ramp9d_spy": lambda P: A.ts_carry(P, "ramp", use_9d=True, off="spy"),
    "ts_schmitt": lambda P: A.ts_carry(P, hysteresis="schmitt"),
    "ts_deadband": lambda P: A.ts_carry(P, "ramp", hysteresis="deadband"),
    "spike_buy": lambda P: A.spike_buy(P),
    "derisk": lambda P: A.derisk_highvol(P),
    "highvol_only": lambda P: A.derisk_highvol(P, invert=True),
    "crash_vvix": lambda P: A.crash_filter(P, "ts", "vvix"),
    "crash_either": lambda P: A.crash_filter(P, "always", "either"),
    "hedge": lambda P: A.spy_plus_hedge(P, 0.05, "lv"),
}


@pytest.fixture(scope="module")
def panel():
    return A.build_panel().loc["2010-06-01":]  # shorter for speed; includes all signals


def _corrupt(P, cut, seed=0):
    """Replace every value strictly after ``cut`` with garbage."""
    rng = np.random.default_rng(seed)
    P2 = P.copy()
    after = P2.index > cut
    for c in P2.columns:
        P2.loc[after, c] = P2.loc[after, c] * rng.uniform(0.2, 5.0, after.sum()) + rng.normal(0, 0.5, after.sum())
    return P2


@pytest.mark.parametrize("name", list(STRATS))
def test_weights_do_not_use_future_data(panel, name):
    cut = panel.index[1500]
    W1 = STRATS[name](panel)
    W2 = STRATS[name](_corrupt(panel, cut))
    a, b = W1.loc[:cut], W2.loc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_put_write_does_not_use_future_data(panel):
    cut = panel.index[1500]
    # keep vix positive so the BS pricer on the corrupted tail stays finite
    P2 = _corrupt(panel, cut)
    P2.loc[P2.index > cut, "vix"] = P2.loc[P2.index > cut, "vix"].abs() + 5
    r1 = A.put_write(panel)
    r2 = A.put_write(P2)
    pd.testing.assert_series_equal(r1.loc[:cut, "ret"], r2.loc[:cut, "ret"], check_exact=False, rtol=1e-12, atol=1e-12)


def test_risk_parity_weights_causal_and_sum_to_one():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=600)
    S = pd.DataFrame(rng.normal(0, 0.01, (600, 3)), index=idx, columns=list("abc"))
    rf = pd.Series(0.0, index=idx)
    res, w = A.risk_parity_blend(S, rf)
    assert np.allclose(w.dropna().sum(axis=1), 1.0)
    S2 = S.copy()
    S2.iloc[400:] *= 10
    res2, w2 = A.risk_parity_blend(S2, rf)
    pd.testing.assert_frame_equal(w.iloc[:400], w2.iloc[:400])
    # returns up to and including day 399 use only weights/returns through 399
    pd.testing.assert_series_equal(res["ret"].loc[: idx[399]], res2["ret"].loc[: idx[399]])


def test_one_day_execution_lag():
    idx = pd.bdate_range("2020-01-01", periods=6)
    W = pd.DataFrame({"x": [0, 1, 1, 0, 0, 0.0]}, index=idx)
    R = pd.DataFrame({"x": [0.0, 0.10, 0.20, 0.30, 0.0, 0.0]}, index=idx)
    res = A.run_weights(W, R, pd.Series(0.0, index=idx), cost_bps=0.0)
    # weight set at close of day1 earns day2's 20% and day3's 30%, not day1's 10%
    assert res.loc[idx[1], "ret"] == pytest.approx(0.0)
    assert res.loc[idx[2], "ret"] == pytest.approx(0.20)
    assert res.loc[idx[3], "ret"] == pytest.approx(0.30)
    assert res.loc[idx[4], "ret"] == pytest.approx(0.0)


def test_cost_is_bps_times_change_in_weight():
    idx = pd.bdate_range("2020-01-01", periods=6)
    W = pd.DataFrame({"x": [0.0, 1.0, 1.0, 0.5, 0.5, 0.0]}, index=idx)
    R = pd.DataFrame({"x": [0.0, 0.01, 0.01, 0.01, 0.01, 0.01]}, index=idx)
    rf = pd.Series(0.0001, index=idx)
    g = A.run_weights(W, R, rf, cost_bps=0.0)
    n = A.run_weights(W, R, rf, cost_bps=5.0)
    # held weights: day1 0, day2 1, day3 1, day4 0.5, day5 0.5 -> |dw| = 0,1,0,.5,0 after first row
    expected_turn = pd.Series([0.0, 1.0, 0.0, 0.5, 0.0], index=idx[1:])
    pd.testing.assert_series_equal(n["turnover"], expected_turn, check_names=False)
    pd.testing.assert_series_equal(g["ret"] - n["ret"], expected_turn * 5e-4, check_names=False, atol=1e-15)
    # gross return: invested part earns the asset, remainder earns rf
    assert g.loc[idx[4], "ret"] == pytest.approx(0.5 * 0.01 + 0.5 * 0.0001)


def test_leverage_pays_financing_spread():
    idx = pd.bdate_range("2020-01-01", periods=4)
    W = pd.DataFrame({"x": [1.5] * 4}, index=idx)
    R = pd.DataFrame({"x": [0.0, 0.01, 0.01, 0.01]}, index=idx)
    rf = pd.Series(0.0002, index=idx)
    res = A.run_weights(W, R, rf, cost_bps=0.0, fin_spread=0.0252)
    assert res["ret"].iloc[-1] == pytest.approx(1.5 * 0.01 - 0.5 * (0.0002 + 0.0252 / 252))


def test_zero_weight_on_missing_return_ok_but_held_position_raises():
    idx = pd.bdate_range("2020-01-01", periods=4)
    R = pd.DataFrame({"x": [np.nan, 0.01, 0.01, 0.01]}, index=idx)
    rf = pd.Series(0.0, index=idx)
    A.run_weights(pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0]}, index=idx), R, rf, 0.0)  # row 0 return is never earned
    R2 = pd.DataFrame({"x": [0.01, 0.01, np.nan, 0.01]}, index=idx)
    with pytest.raises(ValueError):
        A.run_weights(pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0]}, index=idx), R2, rf, 0.0)


def test_sharpe_is_excess_of_riskfree():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0006, 0.01, 300), index=idx)
    rf = pd.Series(0.0002, index=idx)
    res = pd.DataFrame({"ret": r, "rf": rf, "exposure": 1.0, "turnover": 0.0})
    m = A.metrics(res)
    ex = r - rf
    assert m["sharpe"] == pytest.approx(ex.mean() / ex.std() * np.sqrt(252))


def test_hysteresis_reduces_flips():
    rng = np.random.default_rng(0)
    x = np.empty(3000)
    x[0] = 0.95
    for i in range(1, 3000):  # persistent series wobbling around the threshold
        x[i] = 0.95 + 0.9 * (x[i - 1] - 0.95) + 0.02 * rng.standard_normal()
    x = pd.Series(x)
    raw = (x < 1.0).astype(float)
    hyst = A.schmitt(x, 0.93, 1.0)
    assert hyst.diff().abs().sum() < 0.7 * raw.diff().abs().sum()


def test_deadband_never_delays_exit_to_zero():
    t = pd.Series([0.0, 0.5, 0.6, 0.55, 0.0, 0.1])
    out = A.deadband(t, 0.25)
    assert out.iloc[4] == 0.0
    assert out.iloc[2] == 0.5  # 0.1 move is inside the band


def test_bootstrap_diff():
    rng = np.random.default_rng(5)
    a = rng.normal(0.001, 0.01, 1500)
    same = A.block_bootstrap_sharpe(a, a, n_boot=200)
    assert same["diff"] == pytest.approx(0.0) and same["diff_lo"] == pytest.approx(0.0) and same["diff_hi"] == pytest.approx(0.0)
    b = a - 0.0015  # clearly worse series with the same noise
    d = A.block_bootstrap_sharpe(a, b, n_boot=400)
    assert d["diff_lo"] > 0 and d["diff"] > 0


def test_put_write_payoff_sanity():
    # BS put: at-the-money 21d, 20% vol ~ 0.4*S*sigma*sqrt(T) ~ 2.3% of S
    price = A._bs_put(100.0, 100.0, 21 / 252, 0.20, 0.0)
    assert 2.0 < price < 2.6
    assert A._bs_put(90.0, 100.0, 0.0, 0.2, 0.0) == 10.0
    assert -1.0 < A._bs_put_delta(100.0, 100.0, 21 / 252, 0.2, 0.0) < 0.0


def test_vol_managed_cap_and_inverse_vol(panel):
    W = A.vol_managed(panel, "implied", cap=1.5, rebal=1).dropna()
    assert W["spy"].max() <= 1.5 + 1e-12 and W["spy"].min() >= 0.0
    hi_vix = panel["vix"].loc[W.index].idxmax()
    lo_vix = panel["vix"].loc[W.index].idxmin()
    assert W.loc[hi_vix, "spy"] < W.loc[lo_vix, "spy"]
