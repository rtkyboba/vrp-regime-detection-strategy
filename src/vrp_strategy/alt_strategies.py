"""Alternative ways to monetise volatility, evaluated on tradable data only.

Every strategy here is a function ``panel -> weights`` (a DataFrame of target
portfolio weights dated at the *close of t*).  A single engine, ``run_weights``,
turns weights into returns: the weight set at close ``t`` earns the return of
``t+1`` (explicit one-day lag), un-invested capital earns the T-bill rate,
borrowed capital (weight sum > 1) pays T-bill + a spread, and a proportional
cost is charged on the change in weights.  Nothing else touches returns, so
every strategy is costed identically.

Honest P&L comes from traded ETF returns (SPY, SVXY, VXX).  The only
model-based P&L is the put-write sleeve (Black-Scholes with VIX as the
implied-vol input), which is labelled as an approximation wherever it is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from vrp_strategy import data

# VIX3M's last observation; every strategy is evaluated up to here so that all
# signals (VIX, VIX3M, ...) are genuine observations rather than forward fills.
END = pd.Timestamp("2026-05-22")
SPLIT = pd.Timestamp(data.SPLIT)
ANN = 252

# --------------------------------------------------------------------------
# Data panel
# --------------------------------------------------------------------------


def build_panel(end: pd.Timestamp = END) -> pd.DataFrame:
    """Daily panel on the SPY trading calendar.

    Return columns (simple daily returns of tradable instruments):
      spy   SPY adjusted close (total return).
      svxy  SVXY ETF (-1x short-term VIX futures to Feb 2018, -0.5x after).
      vxx   VXX ETN, real series, 2018-01-25 on only.
      lv    "long-vol" series: real VXX where it exists, otherwise -1 x SVXY.
            Valid before Feb 2018 because SVXY then was the exact daily
            inverse of the index VXX is long (post-Mar-2018 regression of VXX
            on SVXY has slope -1.9 and correlation -0.98, so the inverse
            relation holds there too once the -0.5x is accounted for).
    Signal columns (levels): vix, vix3m, vix6m, vix9d, vvix, skew.
    ``rf`` is the daily T-bill return.  Signals are forward-filled by at most 3
    days to bridge exchange-calendar mismatches only.
    """
    spy_px = data.load_raw("spy")
    idx = spy_px.index
    P = pd.DataFrame(index=idx)
    P["spy"] = spy_px.pct_change()
    P["svxy"] = data.load_raw("svxy").pct_change().reindex(idx)
    P["vxx"] = data.load_raw("vxx").pct_change().reindex(idx)
    P["lv"] = P["vxx"].where(P["vxx"].notna(), -P["svxy"])
    for n in ["vix", "vix3m", "vix6m", "vix9d", "vvix", "skew"]:
        P[n] = data.load_raw(n).reindex(idx).ffill(limit=3)
    P["rf"] = data.rf_daily(idx)
    P = P.iloc[1:]  # first row has no return
    return P.loc[:end]


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


def run_weights(
    W: pd.DataFrame,
    R: pd.DataFrame,
    rf: pd.Series,
    cost_bps: float = 2.0,
    fin_spread: float = 0.005,
    drift: bool = False,
) -> pd.DataFrame:
    """Turn target weights (dated at close t) into net daily returns.

    ret_t = sum_i Wl_{t,i} r_{t,i} + cash_t * rf_t - cost_t, where
      Wl = W.shift(1)                           (decided t-1, earned t)
      cash = 1 - sum(Wl); if negative the borrowing pays rf + fin_spread
      cost_t = cost_bps/1e4 * sum_i |Wl_{t,i} - Wl_{t-1,i}|   (incl. entry)
    With ``drift=True`` the cost is instead charged on the trade needed to get
    from the *drifted* previous weights back to target (for constant-mix
    portfolios, whose weights change even when targets do not).

    Leading rows where any weight is NaN are dropped (strategy not yet live);
    later NaN weights are treated as 0 (flat).
    """
    W = W.copy()
    first = W.notna().all(axis=1).idxmax() if W.notna().all(axis=1).any() else None
    if first is None:
        raise ValueError("strategy never live")
    W = W.loc[first:].fillna(0.0)
    R = R.reindex(index=W.index, columns=W.columns)
    rf = rf.reindex(W.index)
    Wl = W.shift(1).fillna(0.0)
    if ((Wl.abs() > 0) & R.isna()).any().any():
        raise ValueError("position held in an asset with missing return")
    Rf = R.fillna(0.0)
    gross_assets = (Wl * Rf).sum(axis=1)
    cash = 1.0 - Wl.sum(axis=1)
    cash_ret = np.where(cash >= 0, cash * rf, cash * (rf + fin_spread / ANN))
    gross = gross_assets + cash_ret
    if drift:
        prev_ret = gross.shift(1).fillna(0.0)
        wd = Wl.shift(1).fillna(0.0).mul((1 + Rf.shift(1).fillna(0.0)).mul(1.0 / (1 + prev_ret), axis=0))
        turnover = (Wl - wd).abs().sum(axis=1)
    else:
        turnover = Wl.diff().abs().sum(axis=1)
        turnover.iloc[0] = Wl.iloc[0].abs().sum()
    cost = turnover * cost_bps / 1e4
    out = pd.DataFrame(
        {
            "exposure": Wl.sum(axis=1),
            "ret_gross": gross,
            "cost": cost,
            "ret": gross - cost,
            "rf": rf,
            "turnover": turnover,
        }
    )
    for c in W.columns:
        out["w_" + c] = Wl[c]
    return out.iloc[1:]  # first row is the (zero-return) entry day


def benchmark(P: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Buy-and-hold of one traded asset in ``run_weights`` format (no costs)."""
    r = P[asset].dropna()
    return pd.DataFrame(
        {"exposure": 1.0, "ret_gross": r, "cost": 0.0, "ret": r, "rf": P["rf"].reindex(r.index), "turnover": 0.0}
    )


# --------------------------------------------------------------------------
# Metrics and bootstrap
# --------------------------------------------------------------------------


def metrics(res: pd.DataFrame) -> dict:
    """Full metric set from an engine result (excess-of-rf Sharpe)."""
    r, rf = res["ret"], res["rf"]
    ex = r - rf
    n = len(r)
    vol = ex.std() * np.sqrt(ANN)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    dside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean() * ANN)
    yrs = n / ANN
    return {
        "sharpe": ex.mean() * ANN / vol if vol > 0 else np.nan,
        "sortino": ex.mean() * ANN / dside if dside > 0 else np.nan,
        "ann_excess_ret": ex.mean() * ANN,
        "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
        "ann_vol": vol,
        "max_dd": dd,
        "worst_day": r.min(),
        "skew": float(skew(r)),
        "ex_kurt": float(kurtosis(r)),
        "turnover_pa": res["turnover"].sum() / yrs,
        "pct_invested": float((res["exposure"].abs() > 1e-9).mean()),
        "n": n,
    }


def block_bootstrap_sharpe(a, b=None, block: int = 21, n_boot: int = 2000, seed: int = 0, chunk: int = 250):
    """Circular block bootstrap of annualised Sharpe of excess returns ``a``
    and, if ``b`` is given, of the Sharpe *difference* a - b (paired: the same
    resampled days are used for both series, preserving their correlation).

    Returns dict with point estimates, 95% percentile CIs and the bootstrap
    one-sided p-value P*(diff <= 0).  ``block`` = 21 days (one month) so that
    vol clustering and the serial dependence of vol-timing returns are kept.
    """
    a = np.asarray(a, dtype=float)
    T = len(a)
    if b is not None:
        b = np.asarray(b, dtype=float)
        assert len(b) == T
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    sa, sb = [], []
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        starts = rng.integers(0, T, size=(m, nb))
        idx = ((starts[:, :, None] + np.arange(block)) % T).reshape(m, -1)[:, :T]
        A = a[idx]
        sa.append(A.mean(1) / A.std(1, ddof=1) * np.sqrt(ANN))
        if b is not None:
            B = b[idx]
            sb.append(B.mean(1) / B.std(1, ddof=1) * np.sqrt(ANN))
    sa = np.concatenate(sa)
    out = {
        "sharpe": a.mean() / a.std(ddof=1) * np.sqrt(ANN),
        "sharpe_lo": np.percentile(sa, 2.5),
        "sharpe_hi": np.percentile(sa, 97.5),
    }
    if b is not None:
        sb = np.concatenate(sb)
        d = sa - sb
        out.update(
            sharpe_b=b.mean() / b.std(ddof=1) * np.sqrt(ANN),
            diff=out["sharpe"] - b.mean() / b.std(ddof=1) * np.sqrt(ANN),
            diff_lo=np.percentile(d, 2.5),
            diff_hi=np.percentile(d, 97.5),
            p_diff_le0=float((d <= 0).mean()),
        )
    return out


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """Annualised Sharpe one expects from the best of ``n_trials`` independent
    zero-skill strategies over ``n_obs`` daily observations (Bailey & Lopez de
    Prado 2014, SR0).  A crude yardstick: trials here are highly correlated, so
    the true hurdle is lower, but it shows how little a single 'best' Sharpe
    proves."""
    g = 0.5772156649
    se = 1.0 / np.sqrt(n_obs / ANN)  # annualised SE of a Sharpe estimate (iid)
    z = (1 - g) * norm.ppf(1 - 1.0 / n_trials) + g * norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(se * z)


# --------------------------------------------------------------------------
# Small causal helpers
# --------------------------------------------------------------------------


def ramp(x: pd.Series, zero_at: float, full_at: float) -> pd.Series:
    """Monotone linear map: 0 at ``zero_at``, 1 at ``full_at`` (either order)."""
    return ((x - zero_at) / (full_at - zero_at)).clip(0.0, 1.0)


def trailing_rank(x: pd.Series, window: int = 252) -> pd.Series:
    """Percentile rank of x_t within its last ``window`` values (incl. t)."""
    return x.rolling(window, min_periods=window // 2).apply(lambda v: (v <= v[-1]).mean(), raw=True)


def schmitt(x: pd.Series, enter_below: float, exit_above: float) -> pd.Series:
    """Hysteresis switch: go on when x < enter_below, off when x > exit_above,
    otherwise keep the previous state.  Fewer flips than a single threshold."""
    v = x.to_numpy(float)
    out = np.zeros(len(v))
    state = 0.0
    for i, xi in enumerate(v):
        if np.isnan(xi):
            state = 0.0
        elif state == 0.0 and xi < enter_below:
            state = 1.0
        elif state == 1.0 and xi > exit_above:
            state = 0.0
        out[i] = state
    return pd.Series(np.where(np.isnan(v), np.nan, out), index=x.index)


def deadband(target: pd.Series, thr: float) -> pd.Series:
    """Only trade when the target moves by >= thr from the current exposure,
    or when it hits the 0/1 bounds (so a stress exit is never delayed)."""
    t = target.to_numpy(float)
    out = np.full(len(t), np.nan)
    cur = 0.0
    for i, ti in enumerate(t):
        if np.isnan(ti):
            cur = 0.0
            continue
        if abs(ti - cur) >= thr or (ti in (0.0, 1.0) and ti != cur):
            cur = ti
        out[i] = cur
    return pd.Series(out, index=target.index)


def _mask(W: pd.DataFrame, ok: pd.Series) -> pd.DataFrame:
    return W.where(ok, np.nan)


# --------------------------------------------------------------------------
# 1. Vol-managed equity (Moreira & Muir 2017)
# --------------------------------------------------------------------------


def vol_managed(
    P: pd.DataFrame,
    kind: str = "implied",
    window: int = 21,
    cap: float = 1.5,
    rebal: int = 21,
    overlay: bool = False,
    overlay_scale: float = 0.5,
    min_hist: int = 252,
) -> pd.DataFrame:
    """SPY exposure inversely proportional to forecast vol, w = c / sigma_t.

    Moreira & Muir (2017, JF) scale monthly by 1/(previous-month realised
    variance) with c chosen so the managed and unmanaged vol agree.  Choosing
    c on the full sample is mild look-ahead, so here c is the *expanding-window*
    historical level of sigma (min 252 days): w_t = mean_hist(sigma) / sigma_t,
    i.e. exposure 1 when vol is at its historical average.  No free parameter
    except the cap (1.5, a retail-margin-friendly bound) and the rebalance
    interval (21 days = their monthly frequency; 1 = daily).

    kind='realised': sigma = trailing ``window``-day std of SPY returns
                     (expanding level = expanding std of SPY returns).
    kind='implied':  sigma = VIX/100 (expanding level = expanding mean of VIX).
    overlay: multiply by ``overlay_scale`` on days VIX/VIX3M >= 1
             (backwardation = stress), applied daily on top of the rebalance.
    """
    if kind == "realised":
        sigma = P["spy"].rolling(window).std() * np.sqrt(ANN)
        level = P["spy"].expanding(min_hist).std() * np.sqrt(ANN)
    elif kind == "implied":
        sigma = P["vix"] / 100.0
        level = sigma.expanding(min_hist).mean()
    else:
        raise ValueError(kind)
    raw = (level / sigma).clip(0.0, cap)
    if rebal > 1:
        keep = np.arange(len(raw)) % rebal == 0
        raw = raw.where(keep).ffill()
    ok = raw.notna()
    if overlay:
        stress = (P["vix"] / P["vix3m"]) >= 1.0
        raw = raw * np.where(stress, overlay_scale, 1.0)
        ok &= P["vix3m"].notna()
    return _mask(pd.DataFrame({"spy": raw}), ok)


# --------------------------------------------------------------------------
# 2. Term-structure carry switching
# --------------------------------------------------------------------------


def ts_carry(
    P: pd.DataFrame,
    mode: str = "binary",
    use_9d: bool = False,
    off: str = "cash",
    hysteresis: str | None = None,
) -> pd.DataFrame:
    """SVXY only when the VIX futures curve is in contango.

    Idea (Simon & Campasano 2014; Eraker & Wu 2017): the roll yield earned by
    short-term-futures shorts is the negative of the futures basis, and
    backwardation (VIX > VIX3M) is when the short side bleeds.  r1 = VIX/VIX3M
    proxies the curve slope; r2 = VIX9D/VIX flags front-end stress.
    mode 'binary': exposure 1 if r1 < 1 (and r2 < 1 if use_9d) else 0.
    mode 'ramp'  : monotone exposure, 0 at r1 >= 1.00, 1 at r1 <= 0.90 (0.90 is
                   about the pre-2016 median of r1); with use_9d also
                   min(., ramp(r2: 0 at >=1.05, 1 at <=0.95)).
    hysteresis 'schmitt': binary switch on r1 (on <0.95, off >1.00).
    hysteresis 'deadband': ramp exposure only traded when it moves >= 0.25.
    off 'cash' leaves 1-e in T-bills; off 'spy' puts 1-e in SPY.
    All thresholds are structural (1.0 = flat curve) or a pre-2016 median.
    """
    r1 = P["vix"] / P["vix3m"]
    r2 = P["vix9d"] / P["vix"]
    ok = r1.notna() & P["svxy"].notna()
    if use_9d:
        ok &= r2.notna()
    if hysteresis == "schmitt":
        e = schmitt(r1, 0.95, 1.00)
        if use_9d:
            e = e * (r2 < 1.05)
    elif mode == "binary":
        e = (r1 < 1.0).astype(float)
        if use_9d:
            e = e * (r2 < 1.0)
    elif mode == "ramp":
        e = ramp(r1, 1.0, 0.90)
        if use_9d:
            e = np.minimum(e, ramp(r2, 1.05, 0.95))
        if hysteresis == "deadband":
            e = deadband(e, 0.25)
    else:
        raise ValueError(mode)
    W = pd.DataFrame({"svxy": e})
    if off == "spy":
        W["spy"] = 1.0 - e
    return _mask(W, ok)


# --------------------------------------------------------------------------
# 3. Buy-the-spike vs de-risk-in-stress
# --------------------------------------------------------------------------


def spike_buy(
    P: pd.DataFrame, spike_ratio: float = 1.4, lookback: int = 10, retrace: float = 0.90, hold: int = 15
) -> pd.DataFrame:
    """Enter SVXY after a VIX spike once the curve has normalised.

    Rationale (Whaley 2009 "fear gauge"; vol mean reversion): spikes in VIX
    are transient, and the term-structure roll yield is largest right after.
    Spike: within the last ``lookback`` days VIX reached >= ``spike_ratio`` x
    its trailing 60-day median.  Normalised: VIX/VIX3M < 1 and VIX has fallen
    to <= ``retrace`` x that recent peak.  Hold ``hold`` days; leave early if
    VIX/VIX3M climbs back to >= 1 (curve re-inverts).  Otherwise cash.
    """
    vix = P["vix"]
    med = vix.rolling(60).median()
    peak = vix.rolling(lookback).max()
    spiked = (peak / med.shift(0)) >= spike_ratio
    r1 = vix / P["vix3m"]
    trigger = (spiked & (r1 < 1.0) & (vix <= retrace * peak)).to_numpy()
    r1v = r1.to_numpy()
    ok = r1.notna() & med.notna() & P["svxy"].notna()
    n = len(vix)
    e = np.zeros(n)
    left = 0
    for i in range(n):
        if left > 0:
            if r1v[i] >= 1.0:
                left = 0
            else:
                e[i] = 1.0
                left -= 1
                continue
        if left == 0 and trigger[i]:
            e[i] = 1.0
            left = hold - 1
    return _mask(pd.DataFrame({"svxy": e}, index=P.index), ok)


def derisk_highvol(P: pd.DataFrame, rank_thr: float = 0.8, invert: bool = False) -> pd.DataFrame:
    """Long SVXY except when VIX is in the top (1-rank_thr) of its trailing
    252-day range ('de-risk in stress').  invert=True is the opposite bet: long
    SVXY *only* in stress, and only once the curve is in contango."""
    rk = trailing_rank(P["vix"], 252)
    hi = rk > rank_thr
    if invert:
        e = (hi & (P["vix"] / P["vix3m"] < 1.0)).astype(float)
    else:
        e = (~hi).astype(float)
    ok = rk.notna() & P["svxy"].notna() & P["vix3m"].notna()
    return _mask(pd.DataFrame({"svxy": e}), ok)


# --------------------------------------------------------------------------
# 4. VVIX / SKEW crash-risk filter
# --------------------------------------------------------------------------


def crash_filter(
    P: pd.DataFrame, base: str = "always", which: str = "vvix", thr: float = 0.8, window: int = 252
) -> pd.DataFrame:
    """Switch short-vol exposure off when tail-risk gauges are elevated.

    VVIX (vol of VIX) high => VIX is fragile / upside vol explosions are
    priced; SKEW high => demand for crash protection (Bollerslev-Todorov style
    tail-risk premia).  Gauge is 'elevated' when its trailing-252-day
    percentile rank exceeds ``thr`` (a level-free, regime-adaptive rule: VVIX
    and SKEW both drifted up over the sample, so fixed levels would not be
    stationary).  base 'always' = long SVXY; base 'ts' = the binary
    term-structure rule.
    """
    gauges = {"vvix": ["vvix"], "skew": ["skew"], "either": ["vvix", "skew"]}[which]
    flag = pd.Series(False, index=P.index)
    ok = P["svxy"].notna()
    for g in gauges:
        rk = trailing_rank(P[g], window)
        flag |= rk > thr
        ok &= rk.notna()
    if base == "ts":
        e = (P["vix"] / P["vix3m"] < 1.0).astype(float)
        ok &= P["vix3m"].notna()
    else:
        e = pd.Series(1.0, index=P.index)
    e = e * (~flag)
    return _mask(pd.DataFrame({"svxy": e}), ok)


# --------------------------------------------------------------------------
# 5. SPY + long-vol hedge, put-write approximation
# --------------------------------------------------------------------------


def spy_plus_hedge(P: pd.DataFrame, h: float = 0.05, hedge: str = "lv") -> pd.DataFrame:
    """(1-h) SPY + h long-vol (constant mix, daily rebalanced).  hedge='lv'
    uses real VXX where it exists, -SVXY before 2018 (2011-10 on); hedge='vxx'
    uses real VXX only (2018-01-25 on)."""
    ok = P[hedge].notna()
    return _mask(pd.DataFrame({"spy": 1 - h, hedge: h}, index=P.index), ok)


def _bs_put(S, K, T, sigma, r):
    """Black-Scholes European put (q = 0; SPY adj-close is total return)."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_put_delta(S, K, T, sigma, r):
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def put_write(
    P: pd.DataFrame, iv_mult: float = 0.93, tenor: int = 21, cost_bps: float = 2.0, fin_spread: float = 0.005
) -> pd.DataFrame:
    """Cash-secured monthly at-the-money SPY put-write, Black-Scholes marked.

    APPROXIMATION - no option prices exist in the data.  Every ``tenor``
    trading days (positional, causal) sell one ATM put (K = S at the close)
    priced at sigma = iv_mult * VIX_t; mark it daily with the day's S and VIX
    until expiry, collateral (K) earns T-bill.  Mirrors the CBOE PUT index.
    iv_mult 0.93: ATM implied vol runs a few vol points under VIX (VIX is a
    strip of OTM-weighted options that embeds the skew); this is an
    *assumption*, sensitivity at 1.00 and 0.85 is reported.  Bid/ask is
    charged as ``cost_bps`` of notional per roll.  The result has the same
    columns as ``run_weights`` (exposure = |put delta| held during the day).
    """
    S = (1 + P["spy"]).cumprod().to_numpy()
    vix = P["vix"].to_numpy() / 100.0
    rf = P["rf"].to_numpy()
    n = len(S)
    ret = np.zeros(n)
    expo = np.zeros(n)
    cost = np.zeros(n)
    turn = np.zeros(n)
    K = V_prev = delta_prev = None
    k_open = None
    first_open = None
    for i in range(n):
        if k_open is not None:  # day i earns on the put sold at an earlier close
            age = i - k_open
            T = max(tenor - age, 0) / ANN
            V = _bs_put(S[i], K, T, iv_mult * vix[i], rf[i] * ANN) if age < tenor else max(K - S[i], 0.0)
            ret[i] = rf[i] - (V - V_prev) / K
            expo[i] = -delta_prev
            V_prev = V
            delta_prev = _bs_put_delta(S[i], K, T, iv_mult * vix[i], rf[i] * ANN)
            if age >= tenor:
                k_open = None
        if k_open is None and i % tenor == 0 and not np.isnan(vix[i]):  # roll at the close of i
            K = S[i]
            k_open = i
            first_open = i if first_open is None else first_open
            V_prev = _bs_put(S[i], K, tenor / ANN, iv_mult * vix[i], rf[i] * ANN)
            delta_prev = _bs_put_delta(S[i], K, tenor / ANN, iv_mult * vix[i], rf[i] * ANN)
            if i + 1 < n:
                cost[i + 1] = cost_bps / 1e4
                turn[i + 1] = 1.0
    out = pd.DataFrame(
        {"exposure": expo, "ret_gross": ret, "cost": cost, "ret": ret - cost, "rf": rf, "turnover": turn},
        index=P.index,
    )
    return out.iloc[first_open + 1 :]


# --------------------------------------------------------------------------
# 6. Risk-parity blend
# --------------------------------------------------------------------------


def risk_parity_blend(
    sleeves: pd.DataFrame,
    rf: pd.Series,
    vol_window: int = 126,
    rebal: int = 21,
    vol_floor: float = 0.04,
    cap: float = 0.6,
    cost_bps: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inverse-volatility ('naive risk parity') blend of sleeve *excess* returns.

    Weights at close t use each sleeve's trailing ``vol_window``-day vol
    through t, floored at ``vol_floor`` (a sleeve parked in cash has ~0 vol
    and would otherwise get all the weight) and capped at ``cap``; they are
    re-set every ``rebal`` days and earn from t+1.  Fully invested (weights sum
    to 1).  Blend turnover is charged ``cost_bps`` on top of the sleeve costs
    (already in the sleeve returns).  Returns (result in ``run_weights``
    format, weights DataFrame).
    """
    vol = sleeves.rolling(vol_window).std() * np.sqrt(ANN)
    inv = 1.0 / vol.clip(lower=vol_floor)
    w = inv.div(inv.sum(axis=1), axis=0)
    for _ in range(5):  # iterative cap
        w = w.clip(upper=cap)
        w = w.div(w.sum(axis=1), axis=0)
    ok = vol.notna().all(axis=1)
    keep = pd.Series(np.arange(len(w)) % rebal == 0, index=w.index) & ok
    w = w.where(keep).ffill().where(ok)
    wl = w.shift(1)
    valid = wl.notna().all(axis=1)
    wl, ex = wl[valid], sleeves[valid]
    port_ex = (wl * ex).sum(axis=1)
    turnover = wl.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * cost_bps / 1e4
    rfv = rf.reindex(wl.index)
    res = pd.DataFrame(
        {
            "exposure": wl.sum(axis=1),
            "ret_gross": port_ex + rfv,
            "cost": cost,
            "ret": port_ex + rfv - cost,
            "rf": rfv,
            "turnover": turnover,
        }
    )
    for c in wl.columns:
        res["w_" + c] = wl[c]
    return res, w
