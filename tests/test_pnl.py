"""Tests for vrp_strategy.pnl on synthetic data (no files needed)."""
import numpy as np
import pandas as pd
import pytest

from vrp_strategy import pnl


def _idx(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _returns(n, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, sigma, n), index=_idx(n))


# ---------------------------------------------------------------- realised variance / HTM
def test_forward_rv_matches_manual_and_has_no_lookahead():
    r = _returns(80)
    rv = pnl.forward_realized_variance(r, 21)
    t = 10
    manual = 252 / 21 * (r.iloc[t + 1:t + 22] ** 2).sum()
    assert rv.iloc[t] == pytest.approx(manual)
    assert rv.iloc[-21:].isna().all()           # not matured yet -> unknown
    # corrupt everything after t+21: value at t must not change
    r2 = r.copy()
    r2.iloc[t + 22:] = 0.5
    assert pnl.forward_realized_variance(r2, 21).iloc[t] == pytest.approx(rv.iloc[t])
    # today's own return is NOT part of the payoff of an entry made at the close of t
    r3 = r.copy()
    r3.iloc[t] = 0.9
    assert pnl.forward_realized_variance(r3, 21).iloc[t] == pytest.approx(rv.iloc[t])


def test_hold_to_maturity_zero_when_strike_equals_realised():
    n = 60
    sigma = 0.01
    idx = _idx(n)
    r = pd.Series(sigma * np.where(np.arange(n) % 2 == 0, 1, -1), index=idx)   # |r| constant
    vix = pd.Series(100 * sigma * np.sqrt(252), index=idx)
    h = pnl.hold_to_maturity(vix, r, 21)
    assert h["payoff_var"].dropna().abs().max() < 1e-12
    assert h["payoff_vega"].dropna().abs().max() < 1e-9
    assert h["maturity"].iloc[0] == idx[21]


def test_hold_to_maturity_sign_short_wins_when_realised_below_strike():
    idx = _idx(60)
    r = pd.Series(0.005, index=idx) * np.where(np.arange(60) % 2 == 0, 1, -1)
    vix = pd.Series(30.0, index=idx)             # strike far above realised (~7.9)
    h = pnl.hold_to_maturity(vix, r, 21)
    assert (h["payoff_var"].dropna() > 0).all() and (h["ret"].dropna() > 0).all()


# ---------------------------------------------------------------- cohort attribution
def test_cohort_series_is_dated_at_maturity_with_entry_position():
    n, h = 70, 21
    entry = pd.Series(np.arange(n, dtype=float), index=_idx(n))
    pos = pd.Series(np.where(np.arange(n) < 30, 1.0, 2.0), index=entry.index)
    c = pnl.cohort_series(entry, pos, h)
    for t in (0, 5, 29, 30, 40):
        assert c.iloc[t + h] == pytest.approx(pos.iloc[t] * entry.iloc[t] / h)
    assert c.iloc[:h].isna().all()               # nothing matures before h days


def test_cohort_series_no_lookahead_future_changes_do_not_touch_the_past():
    n, h = 90, 21
    r = _returns(n, seed=3)
    vix = pd.Series(18.0, index=r.index)
    pos = pd.Series(np.random.default_rng(1).uniform(0, 1, n), index=r.index)
    ent = pnl.hold_to_maturity(vix, r, h)["ret"]
    base = pnl.cohort_series(ent, pos, h)
    k = 50                                       # corrupt returns and positions strictly after day k
    r2, pos2 = r.copy(), pos.copy()
    r2.iloc[k + 1:] = 0.3
    pos2.iloc[k + 1:] = 7.0
    ent2 = pnl.hold_to_maturity(vix, r2, h)["ret"]
    new = pnl.cohort_series(ent2, pos2, h)
    # a payment dated d depends only on returns through d and positions through d-h
    pd.testing.assert_series_equal(base.iloc[:k + 1], new.iloc[:k + 1])


def test_monthly_nonoverlap_spacing_and_dates():
    n, h = 100, 21
    entry = pd.Series(np.arange(n, dtype=float), index=_idx(n))
    m = pnl.monthly_nonoverlap(entry, None, h, offset=3)
    assert list(m.values[:3]) == [3.0, 24.0, 45.0]
    assert m.index[0] == entry.index[3 + h]
    assert (np.diff(m.index.values).astype("timedelta64[D]").astype(int) >= 21).all()


def test_ladder_naive_annualisation_overstates_sharpe_vs_newey_west():
    # overlapping windows of an i.i.d. cohort payoff -> daily series is strongly autocorrelated
    rng = np.random.default_rng(0)
    n, h = 4000, 21
    idx = _idx(n)
    daily = pd.Series(rng.normal(0.05, 1.0, n), index=idx)
    ent = daily.rolling(h).sum().shift(-h)          # payoff of the entry at t = sum of next h days
    lad = pnl.cohort_series(ent, None, h).dropna()
    naive = lad.mean() / lad.std() * np.sqrt(252)
    nw = pnl.sharpe_nw(lad, 2 * h)
    assert naive > 3.0 * nw                         # true ratio is sqrt(21) ~ 4.6


# ---------------------------------------------------------------- term structure / MTM
def _curve(n, vols=(15, 18, 20, 22), seed=None):
    idx = _idx(n)
    base = np.array(vols, float)
    if seed is not None:
        rng = np.random.default_rng(seed)
        data = base * np.exp(rng.normal(0, 0.15, (n, 4)).cumsum(0) * 0.2)
    else:
        data = np.tile(base, (n, 1))
    return pd.DataFrame(data, index=idx, columns=["vix9d", "vix", "vix3m", "vix6m"])


def test_fair_variance_sum_flat_curve_and_30d_anchor():
    c = _curve(5, vols=(20, 20, 20, 20))
    F = pnl.fair_variance_sum(c, 21)
    v = (0.20 ** 2) / 252
    np.testing.assert_allclose(F[:, 5], 5 * v, rtol=1e-12)
    np.testing.assert_allclose(F[:, 21], 21 * v, rtol=1e-12)
    assert (F[:, 0] == 0).all()
    # n=21 maps to the 30d VIX node exactly even on a sloped curve
    c2 = _curve(3)
    F2 = pnl.fair_variance_sum(c2, 21)
    np.testing.assert_allclose(F2[:, 21], 21 * (c2["vix"].to_numpy() / 100) ** 2 / 252, rtol=1e-12)
    # monotone: more days -> more expected variance
    assert (np.diff(F2, axis=1) > 0).all()


def test_fair_variance_sum_drops_missing_nodes():
    c = _curve(3)
    c.loc[:, "vix9d"] = np.nan
    F = pnl.fair_variance_sum(c, 21)
    assert np.isfinite(F).all()
    np.testing.assert_allclose(F[:, 21], 21 * (c["vix"].to_numpy() / 100) ** 2 / 252, rtol=1e-12)


def test_mtm_telescopes_to_held_to_maturity_payoff():
    """Summing the daily MTM P&L of ONE cohort over its life must give exactly
    strike - realised variance (whatever the curve does in between)."""
    n, h = 70, 21
    c = _curve(n, seed=5)
    r = _returns(n, 0.012, seed=2)
    F = pnl.fair_variance_sum(c, h)
    t0 = 7
    total = 0.0
    for k in range(1, h + 1):
        t = t0 + k
        # cohort with h-k+1 days left at close t-1, h-k days left at close t
        total += F[t - 1, h - k + 1] - F[t, h - k] - r.iloc[t] ** 2
    expected = F[t0, h] - (r.iloc[t0 + 1:t0 + h + 1] ** 2).sum()
    assert total == pytest.approx(expected)


def test_mtm_legacy_is_carry_minus_realised_and_mtm_zero_on_static_flat_curve():
    n = 40
    c = _curve(n, vols=(20, 20, 20, 20))
    r = _returns(n, 0.01, seed=4)
    d = pnl.mtm_decomposition(c, r, 21)
    iv_daily = (c["vix"] / 100) ** 2 / 252
    legacy = iv_daily.shift(1) - r ** 2
    np.testing.assert_allclose(d["legacy"].dropna(), legacy.dropna())
    # flat static curve: v(21)=v(20) so F_{t-1}(21) - v - F_t(20) = 0
    assert d["mtm"].dropna().abs().max() < 1e-14
    np.testing.assert_allclose(d["total"].dropna(), d["legacy"].dropna())


def test_mtm_loses_on_a_vol_spike_while_legacy_does_not():
    n = 30
    c = _curve(n, vols=(15, 15, 15, 15))
    c.iloc[15:] = 40.0                              # VIX 15 -> 40 overnight
    r = pd.Series(0.0, index=c.index)
    d = pnl.mtm_decomposition(c, r, 21)
    spike_day = c.index[15]
    assert d.loc[spike_day, "mtm"] < -20 * ((0.40 ** 2 - 0.15 ** 2) / 252) * 0.9
    assert d.loc[spike_day, "legacy"] > 0           # proxy still shows a small profit
    assert d.loc[spike_day, "total"] < 0


# ---------------------------------------------------------------- backtest engine
def test_backtest_lag_position_set_at_t_earns_t_plus_1():
    idx = _idx(10)
    ret = pd.Series(0.0, index=idx)
    ret.iloc[5] = 0.10
    pos_same_day = pd.Series(0.0, index=idx)
    pos_same_day.iloc[5] = 1.0                      # decided at the close of day 5 -> must NOT earn day 5
    res = pnl.backtest(pos_same_day, ret)
    assert res["gross"].sum() == 0.0
    pos_prev = pd.Series(0.0, index=idx)
    pos_prev.iloc[4] = 1.0                          # decided at close of day 4 -> earns day 5
    res2 = pnl.backtest(pos_prev, ret)
    assert res2["gross"].iloc[5] == pytest.approx(0.10)
    assert res2["exposure"].iloc[5] == 1.0 and res2["exposure"].iloc[4] == 0.0


def test_backtest_cost_charged_on_exposure_change_only():
    idx = _idx(10)
    ret = pd.Series(0.0, index=idx)
    pos = pd.Series([0, 1, 1, 1, 0.5, 0.5, 0, 0, 0, 0], index=idx, dtype=float)
    res = pnl.backtest(pos, ret, cost_bps=10)
    # exposure changes: 0->1 (1.0), 1->0.5 (0.5), 0.5->0 (0.5)  => 2.0 traded
    assert res["cost"].sum() == pytest.approx(2.0 * 10 / 1e4)
    assert res["returns"].sum() == pytest.approx(-2.0 * 10 / 1e4)
    # holding a constant position costs nothing after the entry
    res_c = pnl.backtest(pd.Series(1.0, index=idx), ret, cost_bps=10)
    assert res_c["cost"].sum() == pytest.approx(1.0 * 10 / 1e4)
    assert res_c["cost"].iloc[2:].abs().sum() == 0.0


def test_backtest_excess_funding_modes():
    idx = _idx(8)
    ret = pd.Series(0.001, index=idx)
    rf = pd.Series(0.0002, index=idx)
    pos = pd.Series(2.0, index=idx)
    funded = pnl.backtest(pos, ret, rf=rf, funding="funded", max_leverage=5)
    coll = pnl.backtest(pos, ret, rf=rf, funding="collateral", max_leverage=5)
    assert funded["gross"].iloc[-1] == pytest.approx(2.0 * (0.001 - 0.0002))
    assert coll["gross"].iloc[-1] == pytest.approx(2.0 * 0.001)


def test_backtest_leverage_cap_and_nan_position_is_flat():
    idx = _idx(6)
    ret = pd.Series(0.01, index=idx)
    pos = pd.Series([10, np.nan, 10, 10, 10, 10.0], index=idx)
    res = pnl.backtest(pos, ret, max_leverage=3.0)
    assert res["exposure"].max() == 3.0
    assert res["exposure"].iloc[2] == 0.0           # NaN decided on day 1 -> flat on day 2


def test_vol_target_is_causal_and_band_reduces_turnover():
    n = 800
    r = _returns(n, 0.01, seed=7)
    r.iloc[300:400] *= 3                            # vol regime shift
    pos = pd.Series(1.0, index=r.index)
    vt_daily = pnl.VolTarget(0.10, window=63, band=0.0)
    vt_band = pnl.VolTarget(0.10, halflife=63, band=0.25)
    a = pnl.backtest(pos, r, vol_target=vt_daily, max_leverage=10)
    b = pnl.backtest(pos, r, vol_target=vt_band, max_leverage=10)
    assert b["stats"]["turnover_per_yr"] < 0.5 * a["stats"]["turnover_per_yr"]
    # causality: corrupting returns after day k leaves exposures through day k+1 unchanged
    k = 500
    r2 = r.copy()
    r2.iloc[k + 1:] = 0.2
    b2 = pnl.backtest(pos, r2, vol_target=vt_band, max_leverage=10)
    pd.testing.assert_series_equal(b["exposure"].iloc[:k + 2], b2["exposure"].iloc[:k + 2])
    # realised vol is near target after warm-up in a stationary stretch
    s = a["returns"].iloc[100:300]
    assert 0.05 < s.std() * np.sqrt(252) < 0.2


def test_perf_stats_ruin_and_turnover_units():
    idx = _idx(5)
    r = pd.Series([0.1, -1.5, 0.2, 0.2, 0.2], index=idx)
    s = pnl.perf_stats(r, pd.Series([1.0, 1, 1, 1, 1], index=idx))
    assert s["ruined"] and s["max_drawdown"] == pytest.approx(-1.0)
    assert s["pct_invested"] == 1.0
    assert s["turnover_per_yr"] == pytest.approx(1.0 / 5 * 252)   # one unit of entry traded over 5 days
