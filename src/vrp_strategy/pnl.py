"""Honest P&L models for short variance, plus a generic position backtester.

Why this module exists
----------------------
The original project scored a short-variance strategy with the proxy
``iv_daily(t-1) - rv_daily(t)``: collect yesterday's implied daily variance, pay today's
squared return. That proxy is *carry minus realised*; it leaves out the mark-to-market
(MTM) change in the variance strike, which is where short-vol tail losses live (a VIX
jump from 15 to 40 costs a short-variance book roughly the whole month's premium in one
day, but the proxy only charges today's r^2). A constant short position therefore scored
Sharpe 2.9 on it, and the same positions on real SVXY scored ~0.1.

Three P&L models are provided, in decreasing order of realism-per-complexity:

1. ``hold_to_maturity``  - enter a 21-day variance swap at VIX_t^2, pay realised variance
   over t+1..t+21. No MTM approximation; only error is VIX != exact swap strike.
   ``cohort_series`` / ``monthly_nonoverlap`` attribute that payoff to its *maturity* date.
2. ``mtm_returns``       - daily MTM of a constant-maturity (rolling, re-struck daily)
   short variance swap using the VIX9D / VIX / VIX3M / VIX6M term structure for the
   remaining tenor. Decomposes into carry, realised and the MTM/roll-down term the
   legacy proxy omits (``legacy_proxy`` is exactly carry - realised).
3. Real SVXY / VXX returns (``data.asset_returns``) - ground truth, used to validate (2).

Capital convention (unit returns)
---------------------------------
Variance swaps are quoted in *vega notional* N (dollars per vol point). Payoff of a short
swap per unit vega, in vol points, is ``(K^2 - RV^2) / (2 K)`` with K = strike vol
(VIX level), RV = realised vol (both in vol points). We express returns as payoff divided
by ``VEGA_CAPITAL = 48`` vol points of capital per unit vega (about 2.5x the long-run
average VIX). It is a *calibration constant*, not a margin model: on 2011-11..2015 data only
(before the 2016 test split), the daily regression slope of SVXY (-1x) returns on the MTM
swap return at capital = 40 was 0.83 (a swap reprices the whole spot-VIX spike; VIX
futures, which SVXY holds, mean-revert and move less), so capital ~ 48 makes "1x" swap
exposure comparable in sensitivity to an unlevered -1x inverse VIX-futures ETF. It only sets
the *scale* of returns (drawdown, leverage cap); Sharpe is scale-free. A fixed-notional
held-to-maturity position can lose many times this capital (Feb 2020 entries lose ~5x),
which ``perf_stats`` reports as ruin (equity floored at zero). Returns are excess of cash
by construction when the collateral earns the risk-free rate.

Timing: VIX_t, the SPX close of t and the strike of an entry at t are all end-of-day t
information; payoff accrues over closes t+1 .. t+h. Everything is lagged explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data import load_raw
from .stats_tests import bootstrap_sharpe_ci

TRADING_DAYS = 252
H = 21                       # trading days in a 1-month variance swap
VEGA_CAPITAL = 48.0          # vol points of capital per unit vega notional (see docstring)
CAL_PER_TD = 30.0 / 21.0     # calendar days per trading day so that 21 td == the 30d VIX tenor
# calendar-day tenors of the CBOE term-structure nodes
NODE_DAYS = {"vix9d": 9.0, "vix": 30.0, "vix3m": 91.0, "vix6m": 182.0}


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------
def spx_log_returns() -> pd.Series:
    """Close-to-close SPX log returns from the raw file (independent of signals.csv)."""
    return np.log(load_raw("spx")).diff().dropna().rename("log_ret")


def load_curve() -> pd.DataFrame:
    """Daily implied vols in vol points (columns vix9d, vix, vix3m, vix6m), NaN before listing."""
    cols = {k: load_raw(k) for k in NODE_DAYS}
    return pd.DataFrame(cols).sort_index()


# --------------------------------------------------------------------------------------
# 1. Held-to-maturity variance swap
# --------------------------------------------------------------------------------------
def forward_realized_variance(log_ret: pd.Series, h: int = H) -> pd.Series:
    """Annualised realised variance over closes t+1..t+h, dated t (known only at t+h).

    Uses zero-mean squared close-to-close log returns, the variance-swap convention.
    NaN for the last h dates (not yet matured)."""
    ssq = (log_ret ** 2).rolling(h).sum()          # value at row t+h covers t+1..t+h
    return (TRADING_DAYS / h * ssq).shift(-h)


def hold_to_maturity(vix: pd.Series, log_ret: pd.Series, h: int = H,
                     capital: float = VEGA_CAPITAL) -> pd.DataFrame:
    """Per-entry-date payoff of a short h-day variance swap struck at VIX_t^2.

    Columns (indexed by ENTRY date t):
      strike_var   (VIX/100)^2, annualised variance
      rv_var       realised annualised variance over t+1..t+h  (NaN until matured)
      payoff_var   strike_var - rv_var   (short, per unit variance notional)
      payoff_vega  (K^2 - RV^2) / (2K) in vol points per unit vega notional
      ret          payoff_vega / capital   (the return-on-capital convention)
      maturity     date on which the payoff is known (index[t+h])
    """
    idx = vix.index.intersection(log_ret.index)
    vix, lr = vix.reindex(idx), log_ret.reindex(idx)
    k = vix
    rv = forward_realized_variance(lr, h)
    rv_vol = 100.0 * np.sqrt(rv)
    out = pd.DataFrame({
        "strike_var": (k / 100.0) ** 2,
        "rv_var": rv,
        "payoff_var": (k / 100.0) ** 2 - rv,
        "payoff_vega": (k ** 2 - rv_vol ** 2) / (2.0 * k),
    }, index=idx)
    out["ret"] = out["payoff_vega"] / capital
    out["maturity"] = pd.Series(idx, index=idx).shift(-h)
    return out


def cohort_series(entry_ret: pd.Series, position: Optional[pd.Series] = None, h: int = H,
                  entry_cost_bps: float = 0.0) -> pd.Series:
    """Overlapping daily-entry ladder, each cohort booked on its MATURITY date.

    A unit book is split into h cohorts of weight 1/h; the cohort entered at t (size
    ``position[t]``, decided with data through t) is paid at t+h, so row d of the output
    is ``position[d-h] * entry_ret[d-h] / h``. Nothing about the future is used to size a
    cohort, and nothing is booked before it is known. ``entry_cost_bps`` charges the cost of
    opening each cohort (on its own notional) at the entry date, but booked at maturity
    alongside the payoff so the series stays cash-flow consistent.

    WARNING: consecutive values share 20/21 of their underlying window, so the daily
    series is heavily autocorrelated. sqrt(252) * mean/std OVERSTATES the annual Sharpe by
    about sqrt(21); use ``sharpe_nw`` or ``monthly_nonoverlap`` instead.
    """
    pos = pd.Series(1.0, index=entry_ret.index) if position is None else position.reindex(entry_ret.index)
    cohort = pos * entry_ret - entry_cost_bps / 1e4 * pos.abs()
    return (cohort / h).shift(h).rename("cohort_ladder")


def monthly_nonoverlap(entry_ret: pd.Series, position: Optional[pd.Series] = None,
                       h: int = H, offset: int = 0, entry_cost_bps: float = 0.0) -> pd.Series:
    """Non-overlapping version: enter every h-th day starting at row ``offset``, one return
    per cohort indexed by its maturity date. Different offsets are different phase draws of
    the same strategy; average across them (``monthly_sharpe_over_offsets``)."""
    pos = pd.Series(1.0, index=entry_ret.index) if position is None else position.reindex(entry_ret.index)
    cohort = pos * entry_ret - entry_cost_bps / 1e4 * pos.abs()
    mat = pd.Series(entry_ret.index, index=entry_ret.index).shift(-h)
    sel = cohort.iloc[offset::h].dropna()
    out = pd.Series(sel.values, index=pd.DatetimeIndex(mat.loc[sel.index].values), name="monthly_cohort")
    return out[~out.index.isna()]


def monthly_sharpe_over_offsets(entry_ret: pd.Series, position: Optional[pd.Series] = None,
                                h: int = H, entry_cost_bps: float = 0.0,
                                start: Optional[str] = None) -> pd.Series:
    """Annualised Sharpe of the non-overlapping monthly series for every phase offset 0..h-1."""
    res = {}
    for off in range(h):
        s = monthly_nonoverlap(entry_ret, position, h, off, entry_cost_bps)
        if start is not None:
            s = s[s.index >= start]
        res[off] = s.mean() / s.std(ddof=1) * np.sqrt(TRADING_DAYS / h) if len(s) > 2 and s.std() > 0 else np.nan
    return pd.Series(res, name="monthly_sharpe")


# --------------------------------------------------------------------------------------
# 2. Mark-to-market constant-maturity short variance swap
# --------------------------------------------------------------------------------------
def fair_variance_sum(curve: pd.DataFrame, h: int = H, use_nodes=("vix9d", "vix", "vix3m", "vix6m")) -> np.ndarray:
    """Matrix F[t, n], n = 0..h: expected sum of the next n daily squared returns, as priced
    on date t, in units of daily variance (so F[t, n] = n * sigma_n^2 / 252).

    sigma_n comes from linear interpolation of *total variance* w(c) = sigma^2 * c across the
    available term-structure nodes (c in calendar days, node tenors in ``NODE_DAYS``),
    anchored at w(0) = 0 (flat forward variance below the shortest node, flat extrapolation
    of sigma beyond the longest). n trading days map to ``n * CAL_PER_TD`` calendar days so
    that n = 21 hits the 30-day VIX node exactly: F[t, 21] = 21 * (VIX/100)^2 / 252.
    Nodes missing on a date (e.g. VIX9D before 2011) are simply dropped for that date.
    """
    cols = [c for c in use_nodes if c in curve.columns]
    nodes = np.array([NODE_DAYS[c] for c in cols])
    vols = curve[cols].to_numpy(dtype=float) / 100.0
    ns = np.arange(h + 1)
    cal = ns * CAL_PER_TD
    F = np.full((len(curve), h + 1), np.nan)
    F[:, 0] = 0.0
    for i in range(len(curve)):
        ok = ~np.isnan(vols[i])
        if not ok.any():
            continue
        c_nodes = np.concatenate([[0.0], nodes[ok]])
        w_nodes = np.concatenate([[0.0], vols[i, ok] ** 2 * nodes[ok]])
        w = np.interp(cal, c_nodes, w_nodes)
        # beyond the last node keep sigma flat -> w grows linearly
        beyond = cal > c_nodes[-1]
        if beyond.any():
            w[beyond] = vols[i, ok][-1] ** 2 * cal[beyond]
        with np.errstate(invalid="ignore", divide="ignore"):
            var_n = np.where(cal > 0, w / np.where(cal > 0, cal, 1.0), 0.0)
        F[i, 1:] = ns[1:] * var_n[1:] / TRADING_DAYS
    return F


def mtm_decomposition(curve: pd.DataFrame, log_ret: pd.Series, h: int = H,
                      use_nodes=("vix9d", "vix", "vix3m", "vix6m")) -> pd.DataFrame:
    """Daily P&L of a short, constant-maturity (h-day, re-struck every day) variance swap.

    On day t we hold the swap struck/valued at close t-1 with h days left; at close t it has
    h-1 days left and is valued off the curve at t. All terms are in *daily variance units*
    (so ``carry - realised`` is exactly the legacy proxy iv_daily(t-1) - rv_daily(t)):

        carry     v_{t-1}(h)                         = (VIX_{t-1}/100)^2 / 252
        realised  -r_t^2
        mtm       F_{t-1}(h) - v_{t-1}(h) - F_t(h-1)  (revaluation of the remaining h-1
                  days of variance, incl. roll-down along the curve; F from ``fair_variance_sum``)
        total     carry + realised + mtm = F_{t-1}(h) - F_t(h-1) - r_t^2

    The legacy proxy stops after ``realised``.
    """
    idx = curve.index.intersection(log_ret.index)
    curve = curve.reindex(idx)
    r2 = (log_ret.reindex(idx) ** 2)
    F = fair_variance_sum(curve, h, use_nodes)
    f_h_prev = pd.Series(F[:, h], index=idx).shift(1)
    f_hm1 = pd.Series(F[:, h - 1], index=idx)
    carry = f_h_prev / h
    out = pd.DataFrame({
        "carry": carry,
        "realised": -r2,
        "mtm": f_h_prev - carry - f_hm1,
    })
    out["total"] = out["carry"] + out["realised"] + out["mtm"]
    out["legacy"] = out["carry"] + out["realised"]
    return out


def daily_var_to_return(var_pnl: pd.Series, vix: pd.Series, h: int = H,
                        capital: float = VEGA_CAPITAL) -> pd.Series:
    """Convert a daily-variance-unit P&L of a constant-maturity swap into a return on capital.

    Scale to annualised variance units (252/h), then vol points per unit vega: divide by the
    strike vol 2*VIX_{t-1} (vega notional fixed at the start of the day), then by capital."""
    ann_var_pts2 = var_pnl * (TRADING_DAYS / h) * 1e4
    return ann_var_pts2 / (2.0 * vix.shift(1)) / capital


def mtm_returns(curve: Optional[pd.DataFrame] = None, log_ret: Optional[pd.Series] = None,
                h: int = H, capital: float = VEGA_CAPITAL,
                use_nodes=("vix9d", "vix", "vix3m", "vix6m")) -> pd.DataFrame:
    """Return-on-capital streams: honest MTM ('mtm'), carry-only parts, and the legacy proxy
    expressed on the same scale ('legacy') so they are directly comparable."""
    curve = load_curve() if curve is None else curve
    log_ret = spx_log_returns() if log_ret is None else log_ret
    dec = mtm_decomposition(curve, log_ret, h, use_nodes)
    vix = curve["vix"].reindex(dec.index)
    res = pd.DataFrame({k: daily_var_to_return(dec[k], vix, h, capital) for k in dec.columns})
    # 'mtm' = honest total (carry + realised + revaluation); 'mtm_term' = the revaluation part alone
    return res.rename(columns={"mtm": "mtm_term", "total": "mtm"})


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def sharpe_nw(r: pd.Series, lags: int = 21, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe with a Newey-West (Bartlett) long-run variance - the right way to
    annualise a series with overlapping-window autocorrelation (e.g. ``cohort_series``)."""
    x = r.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < lags + 2:
        return np.nan
    d = x - x.mean()
    lrv = (d @ d) / n
    for L in range(1, lags + 1):
        lrv += 2 * (1 - L / (lags + 1)) * (d[L:] @ d[:-L]) / n
    return float(x.mean() / np.sqrt(lrv) * np.sqrt(periods)) if lrv > 0 else np.nan


def sharpe_bootstrap_ci(r: pd.Series, block: int = 21, n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05, periods: int = TRADING_DAYS) -> tuple[float, float]:
    """Stationary-bootstrap CI for the annualised Sharpe with mean block length ``block``
    (default ~1 month). Thin wrapper over ``stats_tests.bootstrap_sharpe_ci``."""
    x = r.dropna()
    if len(x) < 2 * block:
        return (np.nan, np.nan)
    ci = bootstrap_sharpe_ci(x, n_boot=n_boot, alpha=alpha, mean_block=block,
                             periods_per_year=periods, seed=seed)
    return float(ci["ci_low"]), float(ci["ci_high"])


def perf_stats(r: pd.Series, exposure: Optional[pd.Series] = None, periods: int = TRADING_DAYS) -> dict:
    """Full metric set for a daily (excess) return series.

    Equity is compounded and floored at zero (ruin is absorbing); ``max_drawdown`` is
    therefore >= -100% and ``ruined`` flags a wipe-out. Sortino uses the proper downside
    deviation sqrt(mean(min(r,0)^2)). ``turnover`` is the sum of |change in exposure| per
    year (x/yr); ``pct_invested`` is the share of days with non-zero exposure."""
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {}
    mu, sd = r.mean() * periods, r.std(ddof=1) * np.sqrt(periods)
    dd_dev = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)) * np.sqrt(periods)
    eq = np.empty(n)
    cur = 1.0
    for i, x in enumerate(r.to_numpy()):
        cur = max(cur * (1.0 + x), 0.0)
        eq[i] = cur
    eq = pd.Series(eq, index=r.index)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq.to_numpy()]))[1:]
    dd = eq.to_numpy() / peak - 1.0
    out = {
        "ann_return": mu, "ann_vol": sd,
        "sharpe": mu / sd if sd > 0 else np.nan,
        "sortino": mu / dd_dev if dd_dev > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "ruined": bool((eq.to_numpy() <= 0).any()),
        "worst_day": float(r.min()), "best_day": float(r.max()),
        "skew": float(r.skew()), "excess_kurtosis": float(r.kurt()),
        "n_obs": n,
    }
    if exposure is not None:
        e = exposure.reindex(r.index).fillna(0.0)
        out["turnover_per_yr"] = float(e.diff().abs().fillna(e.abs()).sum() / n * periods)
        out["pct_invested"] = float((e.abs() > 1e-9).mean())
        out["avg_abs_exposure"] = float(e.abs().mean())
        out["max_abs_exposure"] = float(e.abs().max())
    return out


# --------------------------------------------------------------------------------------
# 3. Generic backtester
# --------------------------------------------------------------------------------------
@dataclass
class VolTarget:
    """Volatility-targeting spec.

    target       annualised vol target.
    window       rolling-std window (days) if ``halflife`` is None, else
    halflife     EWMA half-life in days (slow, e.g. 63, to avoid daily re-levering).
    band         relative no-trade band on leverage: only re-lever when the desired
                 leverage differs from the current one by more than ``band`` (0 = daily).
    basis        'unit'     - vol of the instrument's unit returns (leverage independent of
                            the signal; recommended)
                 'strategy' - vol of lag(position) * unit returns, as in the original
                            notebook (feeds back: leverage explodes when the position is ~0).
    min_periods  warm-up days before any leverage is produced (exposure is 0 until then).
    """
    target: float = 0.10
    window: int = 126
    halflife: Optional[float] = None
    band: float = 0.0
    basis: str = "unit"
    min_periods: int = 63


def leverage_series(unit_returns: pd.Series, position: pd.Series, vt: VolTarget, max_leverage: float) -> pd.Series:
    """Leverage dated t from data through t (applied to the return of t+1 after the lag)."""
    base = unit_returns if vt.basis == "unit" else position.shift(1) * unit_returns
    if vt.halflife is not None:
        vol = base.ewm(halflife=vt.halflife, min_periods=vt.min_periods).std()
    else:
        vol = base.rolling(vt.window, min_periods=min(vt.window, vt.min_periods)).std()
    vol = vol * np.sqrt(TRADING_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        tgt = (vt.target / vol).where(vol > 0).clip(upper=max_leverage)
    if vt.band <= 0:
        return tgt
    out = np.full(len(tgt), np.nan)
    cur = np.nan
    for i, v in enumerate(tgt.to_numpy()):
        if np.isnan(v):
            out[i] = np.nan
            continue
        if np.isnan(cur) or abs(v / cur - 1.0) > vt.band:
            cur = v
        out[i] = cur
    return pd.Series(out, index=tgt.index)


def backtest(position: pd.Series, unit_returns: pd.Series, cost_bps: float = 0.0,
             vol_target: Optional[VolTarget] = None, max_leverage: float = 3.0,
             rf: Optional[pd.Series] = None, funding: str = "funded") -> dict:
    """Generic, causal position backtester.

    position       desired exposure dated t (computed from data through the close of t),
                   in units of the instrument (1 = 100% long unit_returns; may be fractional
                   or negative). NaN is treated as 0 (flat).
    unit_returns   return of the instrument over day t (close t-1 -> close t).
    cost_bps       one-way cost in bps of the traded exposure, charged on |change in
                   exposure| on the day the new exposure first earns return (initial entry
                   from flat is charged too).
    vol_target     optional ``VolTarget``; final exposure = position * leverage, with
                   |exposure| capped at ``max_leverage`` (cap applied to the leverage and to
                   the final exposure).
    rf             daily risk-free return. ``funding='funded'`` (ETF-like, cash-financed):
                   excess return = exposure * (unit - rf). ``'collateral'``: unit returns are
                   already excess-of-cash (e.g. swap on cash collateral), rf is not subtracted.

    Lag: exposure that earns day t's return is the one decided at the close of t-1.
    Returns dict(returns=net excess series, gross=..., cost=..., exposure=effective exposure
    series, decided=exposure as decided on the date, stats=perf_stats dict).
    """
    idx = unit_returns.index
    pos = position.reindex(idx).fillna(0.0).astype(float)
    ur = unit_returns.astype(float)
    if vol_target is not None:
        lev = leverage_series(ur, pos, vol_target, max_leverage)
        decided = (pos * lev).clip(-max_leverage, max_leverage).fillna(0.0)
    else:
        decided = pos.clip(-max_leverage, max_leverage)
    eff = decided.shift(1).fillna(0.0)                     # explicit one-day lag
    if rf is not None and funding == "funded":
        ret_unit = ur - rf.reindex(idx).ffill().fillna(0.0)
    else:
        ret_unit = ur
    gross = eff * ret_unit
    cost = cost_bps / 1e4 * eff.diff().abs().fillna(eff.abs())
    net = (gross - cost).where(ur.notna())
    valid = net.dropna()
    stats = perf_stats(valid, eff)
    stats["gross_sharpe"] = (gross.dropna().mean() / gross.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS)
                             if gross.dropna().std() > 0 else np.nan)
    stats["cost_drag_ann"] = float(cost.dropna().mean() * TRADING_DAYS)
    return {"returns": net, "gross": gross, "cost": cost, "exposure": eff,
            "decided": decided, "stats": stats}


# --------------------------------------------------------------------------------------
# Real-instrument helpers and validation
# --------------------------------------------------------------------------------------
def svxy_minus1x_equivalent(svxy_ret: pd.Series) -> pd.Series:
    """SVXY was -1x VIX short-term futures until the 2018-02-28 rebalance, -0.5x after.
    Doubling post-change returns gives an (approximate, ignores financing) -1x series."""
    out = svxy_ret.copy()
    out[out.index > "2018-02-27"] *= 2.0
    return out


def regression_stats(y: pd.Series, x: pd.Series) -> dict:
    """OLS y = a + b x on the overlap, plus correlation, R^2 and tracking error of y - b x
    (annualised). Hand-rolled (no BLAS dot) per the environment note in metrics.safe_corr."""
    j = pd.concat([y, x], axis=1, join="inner").dropna()
    yy, xx = j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy()
    xm, ym = xx - xx.mean(), yy - yy.mean()
    sxx, sxy, syy = (xm ** 2).sum(), (xm * ym).sum(), (ym ** 2).sum()
    beta = sxy / sxx
    corr = sxy / np.sqrt(sxx * syy)
    resid = ym - beta * xm
    return {"n": len(j), "corr": float(corr), "beta": float(beta), "r2": float(corr ** 2),
            "te_ann": float(resid.std(ddof=1) * np.sqrt(TRADING_DAYS))}
