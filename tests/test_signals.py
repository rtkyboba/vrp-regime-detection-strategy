import numpy as np
import pandas as pd
import pytest

from vrp_strategy.data import PROCESSED, RAW
from vrp_strategy.signals import build_master, build_signals, construct_signals


def _synthetic(n=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    spx = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    vix = pd.Series(18 + rng.normal(0, 1, n), index=idx)
    vix3m = vix + 1.0
    return spx, vix, vix3m


def test_construct_signals_formulas_and_warmup():
    spx, vix, vix3m = _synthetic()
    s = construct_signals(build_master(spx, vix, vix3m))
    assert len(s) == len(spx) - 30                       # 1 for the return, 29 more for the 30d window
    t = s.index[10]
    assert s.loc[t, "iv_daily"] == pytest.approx((vix[t] / 100) ** 2 / 252)
    assert s.loc[t, "vrp"] == pytest.approx((vix[t] / 100) ** 2 - s.loc[t, "rv_30"] ** 2)
    assert s.loc[t, "vrp_vol"] == pytest.approx(vix[t] / 100 - s.loc[t, "rv_30"])
    assert s.loc[t, "slope"] == pytest.approx((vix3m[t] - vix[t]) / vix[t])


def test_rv_is_causal():
    """Changing prices after t must not change any signal dated <= t."""
    spx, vix, vix3m = _synthetic()
    base = construct_signals(build_master(spx, vix, vix3m))
    t = base.index[100]
    spx2, vix2 = spx.copy(), vix.copy()
    spx2[spx2.index > t] *= 3.0
    vix2[vix2.index > t] += 40.0
    alt = construct_signals(build_master(spx2, vix2, vix2 + 1.0))
    assert np.allclose(base.loc[:t, ["rv_30", "log_ret"]].to_numpy(), alt.loc[:t, ["rv_30", "log_ret"]].to_numpy())


@pytest.mark.skipif(not (RAW / "spx.csv").exists() or not (PROCESSED / "signals.csv").exists(),
                    reason="raw/processed data not present (run scripts/fetch_data.py)")
def test_build_matches_notebook_output():
    """build_signals() reproduces the table notebook 02 wrote, on the same raw data."""
    built = build_signals()
    saved = pd.read_csv(PROCESSED / "signals.csv", index_col=0, parse_dates=True)
    common = built.index.intersection(saved.index)
    assert len(common) > 4900
    assert np.allclose(built.loc[common].to_numpy(), saved.loc[common].to_numpy(), rtol=1e-9, atol=1e-12)
