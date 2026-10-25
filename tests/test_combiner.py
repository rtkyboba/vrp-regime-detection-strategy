import numpy as np
import pandas as pd

from vrp_strategy.combiner import (
    calm_score,
    combine_mean,
    combine_product,
    combine_veto,
    forecast_strength,
    trailing_ic_weights,
)


def _series(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    return idx, rng


def test_maps_and_ranges():
    s = pd.Series([-1.0, 0.0, 1.0])
    assert forecast_strength(s).tolist() == [0.0, 0.5, 1.0]
    assert calm_score(pd.Series([0.0, 0.25, 1.0])).tolist() == [1.0, 0.75, 0.0]
    a, b = pd.Series([0.2, 1.0]), pd.Series([0.5, 0.0])
    assert combine_product(a, b).tolist() == [0.1, 0.0]
    assert combine_mean(a, b).tolist() == [0.35, 0.5]
    assert combine_veto(pd.Series([0.9, 0.9]), pd.Series([0, 1])).tolist() == [0.9, 0.0]


def test_ic_weights_prefer_informative_component_and_sum_to_one():
    idx, rng = _series()
    payoff = pd.Series(rng.normal(size=len(idx)), index=idx)
    good = payoff + rng.normal(scale=0.5, size=len(idx))        # informative
    noise = pd.Series(rng.normal(size=len(idx)), index=idx)      # uninformative
    w = trailing_ic_weights({"good": good, "noise": noise}, payoff)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert w["good"].iloc[-1] > 0.8


def test_ic_weights_only_use_matured_payoffs():
    """Corrupting payoffs that have not matured by t (entries after t-h) must not
    change the weights at t; this is the look-ahead guard for the IC rule."""
    idx, rng = _series()
    payoff = pd.Series(rng.normal(size=len(idx)), index=idx)
    comp = {"a": payoff + rng.normal(size=len(idx)), "b": pd.Series(rng.normal(size=len(idx)), index=idx)}
    base = trailing_ic_weights(comp, payoff, h=21)
    t = 600
    bad = payoff.copy()
    bad.iloc[t - 21 + 1:] = 1e6 * rng.normal(size=len(idx) - (t - 21 + 1))
    comp_bad = {k: v.copy() for k, v in comp.items()}
    out = trailing_ic_weights(comp_bad, bad, h=21)
    assert np.allclose(out.iloc[t].to_numpy(), base.iloc[t].to_numpy())
    # but weights do react once those payoffs have matured
    assert not np.allclose(out.iloc[t + 100].to_numpy(), base.iloc[t + 100].to_numpy())


def test_equal_weights_before_enough_history():
    idx, rng = _series(300)
    payoff = pd.Series(rng.normal(size=300), index=idx)
    w = trailing_ic_weights({"a": payoff, "b": payoff}, payoff)
    assert np.allclose(w.iloc[:100].to_numpy(), 0.5)
