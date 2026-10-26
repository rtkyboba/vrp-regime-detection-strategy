"""Compare regime-detection methods under strict causality.

    python scripts/regime_comparison.py                # everything (hours -> ~20-40 min)
    python scripts/regime_comparison.py --stage run    # only walk-forward the methods
    python scripts/regime_comparison.py --stage eval   # only evaluation (needs CSVs)
    python scripts/regime_comparison.py --stage select # only HMM order selection

Outputs (results/regimes/):
  <method>.csv            date, stress_score, p0.., ll, stress_flag, oos (date >= 2016-01-01)
  select_n.csv            BIC/AIC/OOS log-lik for n = 2..5 (pre-2016 only) and beyond
  methods_meta.csv        one row per method
  persistence.csv         duration / flicker / refit stability
  separation.csv          forward 21d RV and variance-swap payoff by regime, block-bootstrap t
  events.csv              early-warning table (2015-08, 2018-02, 2020-03, 2022)
  econ_svxy.csv           gated SVXY strategies (gross+net of 2bps, excess Sharpe, ...)
  econ_regime_cond.csv    next-day SVXY and variance payoff conditioned on regime
  econ_varswap.csv        gated held-to-maturity variance-swap book
  reality_check.csv       joint block-bootstrap max-t over every gated variant
  trial_ledger.csv        every variant evaluated (the trial count)

See reports/02_regime_detection.md for interpretation.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from vrp_strategy import regime_models as rm
from vrp_strategy.data import ROOT, SPLIT, asset_returns, rf_daily

OUT = ROOT / "results" / "regimes"
SPLIT_TS = pd.Timestamp(SPLIT)
EVAL_START = pd.Timestamp("2012-01-03")   # common window for the main comparison
CSV_START = pd.Timestamp("2011-01-01")
COST_BPS = 2.0
REFIT_EVERY = 63
BOOT_B = 1000
BLOCK = 21
SEED = 12345

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------


@dataclass
class Spec:
    name: str
    family: str
    cls: type
    kwargs: dict
    need: list            # columns that must be non-NaN
    min_train: int = 1260
    note: str = ""
    refit_every: int = REFIT_EVERY
    causal: bool = True

    def factory(self):
        return partial(self.cls, **self.kwargs)


def _fs(key):
    cols, sort = rm.FEATURE_SETS[key]
    return cols, sort


def build_specs() -> list[Spec]:
    S: list[Spec] = []
    # --- Gaussian HMMs on the ORIGINAL feature set, n = 2..5 --------------
    c, s = _fs("orig")
    for n in (2, 3, 4, 5):
        S.append(Spec(f"hmm{n}_orig", "hmm", rm.HMMRegime, dict(n_states=n, cols=c, sort_col=s, name=f"hmm{n}_orig"),
                      need=c, note="original features (vrp_vol, slope, rv_30), walk-forward filtered"))
    # original 3-state spec frozen: trained once on 2006-2015, filtered afterwards
    n_pre = int((rm.build_features().dropna(subset=["rv_30"]).index < SPLIT_TS).sum())
    S.append(Spec("hmm3_orig_static", "hmm", rm.HMMRegime, dict(n_states=3, cols=c, sort_col=s, name="hmm3_orig_static"),
                  need=c, min_train=n_pre, refit_every=10 ** 6,
                  note="original spec: fit ONCE on 2006-2015, causal filter afterwards (no refit)"))
    # --- HMMs on 'core' features (log VIX, log RV21, slope), n = 2..4 -------
    c, s = _fs("core")
    for n in (2, 3, 4, 5):
        S.append(Spec(f"hmm{n}_core", "hmm", rm.HMMRegime, dict(n_states=n, cols=c, sort_col=s, name=f"hmm{n}_core"),
                      need=c, note="core features (log VIX, log RV21, slope)"))
    # --- feature-set variants of the 3-state HMM ----------------------------
    for key, mt in (("core_vvix", 1000), ("core_skew", 1260), ("core_vix6m", 940),
                    ("core_all", 940), ("core_vix9d", 504)):
        c, s = _fs(key)
        S.append(Spec(f"hmm3_{key}", "hmm_features", rm.HMMRegime, dict(n_states=3, cols=c, sort_col=s, name=f"hmm3_{key}"),
                      need=c, min_train=mt, note=f"features {c}"))
    # --- Gaussian mixture (no temporal structure) ----------------------------
    c, s = _fs("orig")
    for k in (2, 3, 4):
        S.append(Spec(f"gmm{k}_orig", "gmm", rm.GMMRegime, dict(n_states=k, cols=c, sort_col=s, name=f"gmm{k}_orig"),
                      need=c, note="mixture, no persistence"))
    c, s = _fs("core")
    S.append(Spec("gmm3_core", "gmm", rm.GMMRegime, dict(n_states=3, cols=c, sort_col=s, name="gmm3_core"),
                  need=c, note="mixture, no persistence"))
    # --- Markov switching ------------------------------------------------------
    S.append(Spec("msreg2_logvix", "markov_switching", rm.MarkovSwitchingRegime,
                  dict(k=2, endog_col="log_vix", name="msreg2_logvix"), need=["log_vix"]))
    S.append(Spec("msreg3_logvix", "markov_switching", rm.MarkovSwitchingRegime,
                  dict(k=3, endog_col="log_vix", name="msreg3_logvix"), need=["log_vix"]))
    S.append(Spec("msar1_2_logvix", "markov_switching", rm.MarkovSwitchingRegime,
                  dict(k=2, endog_col="log_vix", ar_order=1, name="msar1_2_logvix"), need=["log_vix"]))
    S.append(Spec("msreg2_logrv", "markov_switching", rm.MarkovSwitchingRegime,
                  dict(k=2, endog_col="log_rv21", name="msreg2_logrv"), need=["log_rv21"]))
    # --- statistical jump model -------------------------------------------------
    c, s = _fs("core")
    for lam in (5.0, 20.0, 60.0):
        S.append(Spec(f"jump3_lam{lam:g}", "jump", rm.JumpModelRegime,
                      dict(n_states=3, lam=lam, cols=c, sort_col=s, name=f"jump3_lam{lam:g}"), need=c,
                      note="penalised clustering on core features"))
    S.append(Spec("jump2_lam20", "jump", rm.JumpModelRegime,
                  dict(n_states=2, lam=20.0, cols=c, sort_col=s, name="jump2_lam20"), need=c))
    # --- simple rules --------------------------------------------------------------
    S.append(Spec("rule_vix_q80", "rule", rm.RuleRegime, dict(kind="vix_q", q=0.8, window=252), need=["vix"], min_train=504))
    S.append(Spec("rule_vix_q90", "rule", rm.RuleRegime, dict(kind="vix_q", q=0.9, window=252, name="rule_vix_q90"),
                  need=["vix"], min_train=504))
    S.append(Spec("rule_vix_pctile", "rule", rm.RuleRegime, dict(kind="vix_pct", window=504), need=["vix"], min_train=504,
                  note="stress_score = trailing 504d percentile rank of VIX; hard flag at 0.8"))
    S.append(Spec("rule_vix_vix3m_inv", "rule", rm.RuleRegime, dict(kind="inversion"), need=["vix", "vix3m"], min_train=504))
    S.append(Spec("rule_vix9d_inv", "rule", rm.RuleRegime, dict(kind="inv9d"), need=["vix", "vix9d"], min_train=504,
                  note="VIX9D/VIX > 1; vix9d starts 2011-01"))
    # --- change-point / CUSUM -----------------------------------------------------
    S.append(Spec("cusum_diff", "changepoint", rm.CusumRegime, dict(mode="diff"), need=["log_vix"], min_train=504))
    S.append(Spec("cusum_level", "changepoint", rm.CusumRegime, dict(mode="level"), need=["log_vix"], min_train=504))
    # --- GJR-GARCH conditional-vol state ------------------------------------------
    S.append(Spec("gjr_garch", "garch", rm.GJRGarchRegime, dict(), need=["ret"],
                  note="stress = training-ECDF of sigma_{t+1|t}; hard flag at 0.8"))
    return S


def run_one(spec: Spec, force: bool = False):
    """Walk-forward one method; write results/regimes/<name>.csv (+ refit and
    stability logs under _refits/ and _stability/)."""
    t0 = time.time()
    F = rm.build_features().dropna(subset=spec.need)
    res = rm.walk_forward_regimes(spec.factory(), F, refit_every=spec.refit_every, min_train=spec.min_train)
    thr = spec.factory()().hard_threshold
    df = res.probs.copy()
    df.insert(0, "stress_score", res.stress)
    df["ll"] = res.ll
    df["stress_flag"] = (df["stress_score"] > thr).astype(int)
    df["oos"] = (df.index >= SPLIT_TS).astype(int)
    df = df[df.index >= CSV_START]
    df.index.name = "date"
    for d in ("_refits", "_stability"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    res.refits.to_csv(OUT / "_refits" / f"{spec.name}.csv", index=False)
    res.stability.to_csv(OUT / "_stability" / f"{spec.name}.csv", index=False)
    if res.failures:
        pd.DataFrame(res.failures, columns=["date", "error"]).to_csv(OUT / "_refits" / f"{spec.name}_failures.csv", index=False)
    df.to_csv(OUT / f"{spec.name}.csv", float_format="%.6g")
    return spec.name, "done", round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# Order selection for the Gaussian HMM (pre-2016 data only)
# ---------------------------------------------------------------------------

def _boot_idx(N, block, B, rng):
    nb = int(np.ceil(N / block))
    starts = rng.integers(0, N, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % N
    return idx.reshape(B, -1)[:, :N]


def _mean_se_block(x, block=BLOCK, B=BOOT_B, seed=SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    idx = _boot_idx(len(x), block, B, rng)
    return x.mean(), x[idx].mean(1).std()


def run_select(force=False):
    """Order selection for the Gaussian HMM / mixture using PRE-2016 data only:
    (a) BIC/AIC of a single fit on 2006-08..2015-12,
    (b) walk-forward one-step-ahead predictive log-likelihood (expanding window,
        refit every 63d, min_train 756) over 2009-08..2015-12.
    Also paired OOS log-lik gains n vs n-1 with block-bootstrap se."""
    path, ppath = OUT / "select_n.csv", OUT / "select_n_paired.csv"
    if path.exists() and ppath.exists() and not force:
        return pd.read_csv(path), pd.read_csv(ppath)
    F = rm.build_features().dropna(subset=["ret", "rv_30", "log_rv21", "vrp_vol"])
    pre = F[F.index < SPLIT_TS]
    rows, prow, lls = [], [], {}
    for key in ("orig", "core"):
        cols, sort = rm.FEATURE_SETS[key]
        for kind in ("hmm", "gmm"):
            for n in ((2, 3, 4, 5, 6) if kind == "hmm" else (1, 2, 3, 4, 5, 6)):
                if kind == "hmm":
                    mk = partial(rm.HMMRegime, n, cols, sort, n_init=5)
                else:
                    mk = partial(rm.GMMRegime, n, cols, sort, n_init=5)
                m = mk().fit(pre)
                ba = m.bic_aic()
                wf = rm.walk_forward_regimes(mk, pre, refit_every=REFIT_EVERY, min_train=756, stability=False)
                ll = wf.ll.dropna()
                lls[(key, kind, n)] = ll
                rows.append(dict(features=key, model=kind, n_states=n, n_params=getattr(m, "n_params", np.nan),
                                 bic_pre2016=ba["bic"], aic_pre2016=ba["aic"], train_ll_pre2016=m.train_ll_,
                                 oos_ll_per_day_pre2016=ll.mean(), oos_ll_se=_mean_se_block(ll.to_numpy())[1],
                                 oos_start=ll.index[0].date(), oos_days=len(ll)))
            for n in (3, 4, 5, 6):
                if (key, kind, n - 1) in lls:
                    d = (lls[(key, kind, n)] - lls[(key, kind, n - 1)]).dropna().to_numpy()
                    mu, se = _mean_se_block(d)
                    prow.append(dict(features=key, model=kind, comparison=f"{n} vs {n - 1}", mean_ll_gain_per_day=mu,
                                     se=se, t=mu / se, total_ll_gain=d.sum()))
    df, pf = pd.DataFrame(rows), pd.DataFrame(prow)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    pf.to_csv(ppath, index=False)
    return df, pf


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def load_world() -> pd.DataFrame:
    F = rm.build_features()
    lr = np.log(F["spx"]).diff()
    fwd_sum = (lr ** 2).rolling(21).sum().shift(-21)        # sum r^2 over t+1..t+21
    strike = (F["vix"] / 100.0) ** 2 * 21.0 / 252.0         # variance strike over 21d
    W = pd.DataFrame(index=F.index)
    W["lr"] = lr
    W["vix"] = F["vix"]
    W["fwd_rv"] = np.sqrt(fwd_sum * 252.0 / 21.0)           # annualised fwd 21d realised vol
    W["pnl_frac"] = 1.0 - fwd_sum / strike                  # short-variance payoff / strike
    W["pnl_pts"] = (strike - fwd_sum) * 1e4                 # short-variance payoff, var points (x1e4)
    W["svxy"] = asset_returns("svxy").reindex(F.index)
    W["rf"] = rf_daily(F.index)
    W["cur_rv"] = np.sqrt((lr ** 2).rolling(21).mean() * 252.0)
    return W


def load_method(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"{name}.csv", index_col=0, parse_dates=True)


def perf(r_total: pd.Series, r_ex: pd.Series, e: pd.Series | None = None) -> dict:
    r_total, r_ex = r_total.dropna(), r_ex.dropna()
    n = len(r_ex)
    sd = r_ex.std()
    sh = r_ex.mean() / sd * np.sqrt(252) if sd > 0 else np.nan
    dd = np.sqrt((np.minimum(r_ex, 0) ** 2).mean())
    cum = (1 + r_total).cumprod()
    out = dict(sharpe=sh, sortino=r_ex.mean() / dd * np.sqrt(252) if dd > 0 else np.nan,
               ann_ret_excess=r_ex.mean() * 252, ann_vol=sd * np.sqrt(252),
               max_dd=float((cum / cum.cummax() - 1).min()), worst_day=float(r_total.min()),
               skew=float(stats.skew(r_total)), ex_kurt=float(stats.kurtosis(r_total)), n_days=n)
    if e is not None:
        e = e.reindex(r_ex.index)
        out.update(avg_exposure=float(e.mean()), pct_days_invested=float((e > 0.01).mean()),
                   turnover_per_year=float(e.diff().abs().sum() / (n / 252)))
    return out


def sharpe_cols(R: np.ndarray) -> np.ndarray:
    sd = R.std(0, ddof=1)
    return np.where(sd > 0, R.mean(0) / np.where(sd > 0, sd, 1) * np.sqrt(252), np.nan)


def boot_sharpe_ci(r_ex: np.ndarray, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = _boot_idx(len(r_ex), BLOCK, BOOT_B, rng)
    R = r_ex[idx]
    sh = R.mean(1) / R.std(1, ddof=1) * np.sqrt(252)
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


def boot_diff(x: np.ndarray, flag: np.ndarray, seed=SEED):
    """mean(x | flag) - mean(x | ~flag) with circular block bootstrap se and t."""
    ok = np.isfinite(x)
    x, flag = x[ok], flag[ok].astype(bool)
    if flag.sum() < 5 or (~flag).sum() < 5:
        return dict(mean_stress=np.nan, mean_calm=np.nan, diff=np.nan, se=np.nan, t=np.nan, n_stress=int(flag.sum()))
    rng = np.random.default_rng(seed)
    idx = _boot_idx(len(x), BLOCK, BOOT_B, rng)
    xb, fb = x[idx], flag[idx].astype(float)
    ns, nc = fb.sum(1), (1 - fb).sum(1)
    good = (ns >= 3) & (nc >= 3)
    ms = (xb * fb).sum(1)[good] / ns[good]
    mc = (xb * (1 - fb)).sum(1)[good] / nc[good]
    d = x[flag].mean() - x[~flag].mean()
    se = (ms - mc).std()
    return dict(mean_stress=x[flag].mean(), mean_calm=x[~flag].mean(), diff=d, se=se, t=d / se if se > 0 else np.nan,
                n_stress=int(flag.sum()))


def runs(flag: np.ndarray):
    """Run-length encode a boolean array -> (values, lengths)."""
    if len(flag) == 0:
        return np.array([]), np.array([])
    ch = np.flatnonzero(np.diff(flag.astype(int)) != 0) + 1
    starts = np.r_[0, ch]
    lens = np.diff(np.r_[starts, len(flag)])
    return flag[starts], lens


# ---------------- strategies -----------------------------------------------

VARIANTS = ("derisk_soft", "derisk_hard", "spike_soft", "spike_hard")


def exposure(score: pd.Series, thr: float, variant: str) -> pd.Series:
    if variant == "derisk_soft":
        return 1.0 - score
    if variant == "derisk_hard":
        return (score <= thr).astype(float)
    if variant == "spike_soft":
        return score.copy()
    if variant == "spike_hard":
        return (score > thr).astype(float)
    raise ValueError(variant)


def svxy_strategy(e: pd.Series, W: pd.DataFrame, cost_bps=COST_BPS):
    """Exposure e_t decided at the close of t earns the SVXY return of t+1; the
    rest of equity earns the T-bill.  Cost = cost_bps * |change in exposure|."""
    pos = e.shift(1)
    d = pd.concat([pos.rename("pos"), W["svxy"], W["rf"]], axis=1).dropna()
    gross = d["pos"] * d["svxy"] + (1 - d["pos"]) * d["rf"]
    cost = (cost_bps * 1e-4) * d["pos"].diff().abs().fillna(0.0)
    net = gross - cost
    return gross, net, d["rf"], d["pos"]


def varswap_book(e: pd.Series, W: pd.DataFrame, cost_bps=COST_BPS):
    """Rolling book of held-to-maturity 21d variance swaps, one cohort per day.

    Cohort entered at close t (exposure e_t, notional 1/21, strike VIX_t^2) accrues
    on day d in (t, t+21]:  e_t/21 * [1/21 - 252*r_d^2/(21*VIX_t^2)]; summed over the
    21 days this is exactly e_t/21 * (1 - RV_fwd^2/VIX_t^2).  No MTM, no funding
    (swap), so 'excess' = total.  Cost = cost_bps * |change in average exposure|."""
    r2 = (W["lr"] ** 2).to_numpy()
    v2 = ((W["vix"] / 100.0) ** 2).to_numpy()
    ev = e.reindex(W.index).to_numpy()
    n = len(W)
    book = np.full(n, np.nan)
    for d in range(n):
        acc, ok = 0.0, False
        for k in range(1, 22):
            t = d - k
            if t < 0 or not np.isfinite(ev[t]):
                continue
            acc += ev[t] / 21.0 * (1.0 / 21.0 - 252.0 * r2[d] / (21.0 * v2[t]))
            ok = True
        if ok and np.isfinite(r2[d]):
            book[d] = acc
    Ebar = pd.Series(ev, index=W.index).shift(1).rolling(21, min_periods=1).mean()
    gross = pd.Series(book, index=W.index)
    net = gross - (cost_bps * 1e-4) * Ebar.diff().abs().fillna(0.0)
    return gross, net


def regression_alpha(r_ex: pd.Series, bench_ex: pd.Series):
    import statsmodels.api as sm
    d = pd.concat([r_ex, bench_ex], axis=1).dropna()
    X = sm.add_constant(d.iloc[:, 1].to_numpy())
    res = sm.OLS(d.iloc[:, 0].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    return float(res.params[0] * 252), float(res.tvalues[0]), float(res.params[1])


# ---------------------------------------------------------------------------
# Evaluation stage
# ---------------------------------------------------------------------------

EVENTS = {
    "2015-08": dict(trigger="2015-08-20", peak="2015-08-24", note="SPX -2.1% on 8/20; VIX close spike 8/24"),
    "2018-02": dict(trigger="2018-02-02", peak="2018-02-05", note="'Volmageddon': SVXY -32% on 2/5, -83% on 2/6"),
    "2020-03": dict(trigger="2020-02-24", peak="2020-03-16", note="COVID: SPX -3.4% on 2/24; VIX close peak 3/16"),
    "2022": dict(trigger="2022-01-05", peak="2022-03-07", note="rates bear market; VIX close high of H1 on 3/7"),
}


def pre_registered_event_dates():
    return EVENTS


def reality_check(df_ret: pd.DataFrame, bench: pd.Series, block: int, tag: str):
    """White-style joint block bootstrap over ALL variants: the max over variants of the
    centred t-stat of (Sharpe_variant - Sharpe_benchmark) gives the null distribution
    for 'best of N' (variants share the same resampled dates, so correlation is kept)."""
    keys = list(df_ret.columns)
    R = df_ret.to_numpy()
    b = bench.to_numpy()
    rng = np.random.default_rng(SEED + 1)
    idx = _boot_idx(len(R), block, BOOT_B, rng)
    shb = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)
    sh0, shb0 = sharpe_cols(R), float(shb(b))
    dobs = sh0 - shb0
    dboot = np.empty((BOOT_B, R.shape[1]))
    for i in range(BOOT_B):
        dboot[i] = sharpe_cols(R[idx[i]]) - shb(b[idx[i]])
    se = dboot.std(0)
    tobs = dobs / se
    z = (dboot - dobs) / se
    maxnull, maxabs = np.nanmax(z, axis=1), np.nanmax(np.abs(z), axis=1)
    rc = pd.DataFrame(dict(name=[k[0] for k in keys], variant=[k[1] for k in keys], net_sharpe=sh0,
                           bench_net_sharpe=shb0, d_sharpe_vs_bench=dobs, se=se, t=tobs,
                           p_single=2 * (1 - stats.norm.cdf(np.abs(tobs))),
                           p_rc_one_sided=[(maxnull >= t).mean() for t in tobs],
                           p_rc_two_sided=[(maxabs >= abs(t)).mean() for t in tobs])).sort_values("t", ascending=False)
    rc.to_csv(OUT / f"{tag}.csv", index=False)
    pd.DataFrame(dict(crit95_one_sided=[np.percentile(maxnull, 95)], crit95_two_sided=[np.percentile(maxabs, 95)],
                      n=[len(rc)], days=[len(R)], block=[block], window=[f"{df_ret.index[0].date()}..{df_ret.index[-1].date()}"])
                 ).to_csv(OUT / f"{tag}_crit.csv", index=False)
    return rc


def boot_corr_diff(a, b, y, seed=SEED):
    """Spearman(a,y) - Spearman(b,y) with circular block bootstrap (ranks fixed in-sample)."""
    ra, rb, ry = (stats.rankdata(v) for v in (a, b, y))

    def corr(u, v):
        u, v = u - u.mean(1, keepdims=True), v - v.mean(1, keepdims=True)
        return (u * v).sum(1) / np.sqrt((u ** 2).sum(1) * (v ** 2).sum(1))
    rng = np.random.default_rng(seed)
    idx = _boot_idx(len(y), BLOCK, 500, rng)
    d = corr(ra[idx], ry[idx]) - corr(rb[idx], ry[idx])
    obs = float(corr(ra[None], ry[None])[0] - corr(rb[None], ry[None])[0])
    sd = float(d.std())
    return obs, sd, obs / sd if sd > 0 else np.nan


def evaluate(specs, W, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(OUT / "methods_meta.csv", index_col=0)
    by_name = {s.name: s for s in specs}
    names = [s.name for s in specs if (OUT / f"{s.name}.csv").exists()]
    data = {n: load_method(n) for n in names}

    persist, sep, cond, events, econ, vsw = [], [], [], [], [], []
    ledger = []
    ret_store = {}     # (name, variant) -> net excess returns on the main window
    bh_ex_main = None
    vs_store, vs_base = {}, None
    scores = {}

    svx = W["svxy"]
    for name in names:
        df = data[name]
        thr = float(meta.loc[name, "hard_threshold"])
        fam = by_name[name].family
        is9d = "vix9d" in name or name.endswith("vix9d")
        start = max(EVAL_START, df.index[0]) if not is9d else df.index[0]
        win = df.index[(df.index >= start)]
        s = df.loc[win, "stress_score"]
        flag = (s > thr).to_numpy()
        if not is9d and not name.endswith("_static"):
            scores[name] = s

        # ---- persistence / flicker -----------------------------------------
        yrs = len(win) / 252.0
        vals, lens = runs(flag)
        st_lens, ca_lens = lens[vals == True], lens[vals == False]  # noqa: E712
        p = dict(name=name, family=fam, start=win[0].date(), days=len(win), pct_stress=flag.mean(),
                 mean_stress_run=st_lens.mean() if len(st_lens) else np.nan,
                 median_stress_run=np.median(st_lens) if len(st_lens) else np.nan,
                 mean_calm_run=ca_lens.mean() if len(ca_lens) else np.nan,
                 switches_per_year=(np.diff(flag.astype(int)) != 0).sum() / yrs,
                 stress_episodes_per_year=len(st_lens) / yrs,
                 pct_episodes_le3d=(st_lens <= 3).mean() if len(st_lens) else np.nan)
        pcols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
        if len(pcols) > 2:
            am = df.loc[win, pcols].to_numpy().argmax(1)
            p["argmax_switches_per_year"] = (np.diff(am) != 0).sum() / yrs
        for k in ("n_refits", "flag_agree_old_new", "mean_abs_stress_diff", "top_state_mean_cv", "label_flips",
                  "pstay_top_mean", "exp_duration_top_days", "pstay_low_mean"):
            if k in meta.columns:
                p[k] = meta.loc[name, k]
        persist.append(p)

        # ---- forward separation --------------------------------------------
        Wd = W.loc[win]
        for target in ("fwd_rv", "pnl_frac", "pnl_pts"):
            for label, mask in (("all", np.ones(len(win), bool)), ("pre2016", win < SPLIT_TS), ("2016+", win >= SPLIT_TS)):
                x = Wd[target].to_numpy()[mask]
                f = flag[mask]
                r = boot_diff(x, f)
                sep.append(dict(name=name, family=fam, target=target, period=label, **r))
        # rank correlation of the continuous score with forward targets
        for target in ("fwd_rv", "pnl_frac"):
            d = pd.concat([s, Wd[target]], axis=1).dropna()
            rho = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if d.iloc[:, 0].nunique() > 1 else np.nan
            sep.append(dict(name=name, family=fam, target=target, period="spearman_score", diff=rho))

        # ---- SVXY next-day conditional on regime (decided at close t) -------
        nxt = (svx - W["rf"]).shift(-1).loc[win]
        r = boot_diff(nxt.to_numpy() * 252, flag)
        cond.append(dict(name=name, family=fam, window="all", target="svxy_next_day_excess_ann", **r))
        for label, mask in (("pre2016", win < SPLIT_TS), ("2016+", win >= SPLIT_TS)):
            r = boot_diff(nxt.to_numpy()[mask] * 252, flag[mask])
            cond.append(dict(name=name, family=fam, window=label, target="svxy_next_day_excess_ann", **r))

        # ---- gated SVXY strategies ------------------------------------------
        for var in VARIANTS:
            e = exposure(s, thr, var)
            g, nt, rf, pos = svxy_strategy(e.reindex(W.index), W)
            g, nt, rf, pos = [z.loc[(z.index > win[0])] for z in (g, nt, rf, pos)]
            row = dict(name=name, family=fam, variant=var, start=g.index[0].date())
            for label, sl in (("all", slice(None)), ("pre2016", g.index < SPLIT_TS), ("2016+", g.index >= SPLIT_TS)):
                m = np.ones(len(g), bool) if label == "all" else sl
                pg = perf(g[m], g[m] - rf[m], pos[m])
                pn = perf(nt[m], nt[m] - rf[m], pos[m])
                for k in ("sharpe", "sortino", "max_dd", "worst_day", "skew", "ex_kurt", "ann_ret_excess", "ann_vol"):
                    row[f"{label}_net_{k}"] = pn[k]
                row[f"{label}_gross_sharpe"] = pg["sharpe"]
                if label == "all":
                    row.update(avg_exposure=pn["avg_exposure"], pct_days_invested=pn["pct_days_invested"],
                               turnover_per_year=pn["turnover_per_year"], n_days=pn["n_days"])
            exf = ~((nt.index >= "2018-02-01") & (nt.index < "2018-03-01"))
            row["all_net_sharpe_exFeb2018"] = perf(nt[exf], nt[exf] - rf[exf])["sharpe"]
            bh = (svx - W["rf"]).reindex(g.index)
            a, ta, beta = regression_alpha(nt - rf, bh)
            row.update(alpha_ann_vs_svxy=a, alpha_t=ta, beta_to_svxy=beta)
            if not is9d and not name.endswith("_static"):
                lo, hi = boot_sharpe_ci((nt - rf).to_numpy())
                row.update(net_sharpe_ci_lo=lo, net_sharpe_ci_hi=hi)
                ret_store[(name, var)] = (nt - rf)
            econ.append(row)
            ledger.append(dict(kind="strategy_variant", name=name, variant=var))

        # ---- gated variance-swap book --------------------------------------
        for var in VARIANTS:
            e = exposure(s, thr, var).reindex(W.index)
            g, nt = varswap_book(e, W)
            g, nt = g.loc[g.index > win[0]].dropna(), nt.loc[nt.index > win[0]].dropna()
            row = dict(name=name, family=fam, variant=var)
            for label, m in (("all", np.ones(len(g), bool)), ("pre2016", g.index < SPLIT_TS), ("2016+", g.index >= SPLIT_TS)):
                pg, pn = perf(g[m], g[m]), perf(nt[m], nt[m])
                row[f"{label}_gross_sharpe"] = pg["sharpe"]
                row[f"{label}_net_sharpe"] = pn["sharpe"]
                row[f"{label}_net_skew"] = pn["skew"]
            row.update(avg_exposure=float(e.loc[win].mean()))
            if not is9d and not name.endswith("_static"):
                vs_store[(name, var)] = nt
            vsw.append(row)
            ledger.append(dict(kind="varswap_variant", name=name, variant=var))

        # ---- event study -----------------------------------------------------
        e_soft = exposure(s, thr, "derisk_soft").reindex(W.index)
        pos_soft = e_soft.shift(1)
        for ev, spec in EVENTS.items():
            tr, pk = pd.Timestamp(spec["trigger"]), pd.Timestamp(spec["peak"])
            if tr not in df.index or pk not in df.index or df.index[0] > tr - pd.Timedelta(days=15):
                events.append(dict(name=name, event=ev, available=False))
                continue
            i_tr, i_pk = df.index.get_loc(tr), df.index.get_loc(pk)
            sc = df["stress_score"]
            fl = (sc > thr).astype(int)
            pre_win = df.index[max(0, i_tr - 10): i_pk + 1]
            hit = [d for d in pre_win if fl.loc[d] == 1]
            first = hit[0] if hit else None
            # contiguous flag run that is active at the peak day (to avoid stale earlier flags)
            lead = (i_pk - df.index.get_loc(first)) if first is not None else np.nan
            idxW = list(W.index)
            a, b = idxW.index(tr) + 1, idxW.index(pk) + 10
            crash_bh = (1 + svx.iloc[a:b + 1].fillna(0)).prod() - 1
            crash_gate = (1 + (pos_soft.iloc[a:b + 1] * svx.iloc[a:b + 1]
                               + (1 - pos_soft.iloc[a:b + 1]) * W["rf"].iloc[a:b + 1]).fillna(0)).prod() - 1
            c, d_ = idxW.index(pk) + 1, idxW.index(pk) + 40
            reb_bh = (1 + svx.iloc[c:d_ + 1].fillna(0)).prod() - 1
            reb_gate = (1 + (pos_soft.iloc[c:d_ + 1] * svx.iloc[c:d_ + 1]
                             + (1 - pos_soft.iloc[c:d_ + 1]) * W["rf"].iloc[c:d_ + 1]).fillna(0)).prod() - 1
            events.append(dict(name=name, event=ev, available=True,
                               score_day_before_trigger=sc.iloc[i_tr - 1], score_trigger=sc.iloc[i_tr],
                               score_day_before_peak=sc.iloc[i_pk - 1], score_peak=sc.iloc[i_pk],
                               first_flag=first.date() if first is not None else None,
                               lead_days_before_peak=lead,
                               flag_day_before_peak=int(fl.iloc[i_pk - 1]), flag_at_peak=int(fl.iloc[i_pk]),
                               flag_at_trigger=int(fl.iloc[i_tr]),
                               svxy_bh_trigger_to_peak10=crash_bh, svxy_soft_gate_trigger_to_peak10=crash_gate,
                               svxy_bh_rebound_peak_to_40d=reb_bh, svxy_soft_gate_rebound_40d=reb_gate,
                               avg_exposure_crash_window=float(pos_soft.iloc[a:b + 1].mean())))
    # extra: 2018-02-05 close -> SVXY return dated 2018-02-06
    pd.DataFrame(persist).to_csv(OUT / "persistence.csv", index=False)
    pd.DataFrame(sep).to_csv(OUT / "separation.csv", index=False)
    pd.DataFrame(cond).to_csv(OUT / "econ_regime_cond.csv", index=False)
    pd.DataFrame(events).to_csv(OUT / "events.csv", index=False)

    econ_df = pd.DataFrame(econ)
    # ---- baselines: buy & hold, constant exposures, VIX/VIX3M rule is a method already ----
    base_rows = []
    for label, start in (("main_2012", EVAL_START), ("vix9d_window", pd.Timestamp(persist_start_9d(persist)))):
        idx = W.index[W.index >= start]
        idx = idx[idx > idx[0]]
        for cname, cexp in (("buy_and_hold", 1.0), ("constant_0.5", 0.5)):
            e = pd.Series(cexp, index=W.index)
            g, nt, rf, pos = svxy_strategy(e, W)
            g, nt, rf, pos = [z.reindex(idx) for z in (g, nt, rf, pos)]
            row = dict(name=cname, family="baseline", variant=label, start=idx[0].date())
            for lab, m in (("all", np.ones(len(g), bool)), ("pre2016", g.index < SPLIT_TS), ("2016+", g.index >= SPLIT_TS)):
                pn = perf(nt[m], nt[m] - rf[m], pos[m])
                pg = perf(g[m], g[m] - rf[m], pos[m])
                for k in ("sharpe", "sortino", "max_dd", "worst_day", "skew", "ex_kurt", "ann_ret_excess", "ann_vol"):
                    row[f"{lab}_net_{k}"] = pn[k]
                row[f"{lab}_gross_sharpe"] = pg["sharpe"]
            exf = ~((nt.index >= "2018-02-01") & (nt.index < "2018-03-01"))
            row["all_net_sharpe_exFeb2018"] = perf(nt[exf], nt[exf] - rf[exf])["sharpe"]
            lo, hi = boot_sharpe_ci((nt - rf).to_numpy())
            row.update(net_sharpe_ci_lo=lo, net_sharpe_ci_hi=hi, avg_exposure=cexp, pct_days_invested=1.0)
            base_rows.append(row)
            if label == "main_2012" and cname == "buy_and_hold":
                bh_ex_main = (nt - rf)
        # variance swap constant book
        g, nt = varswap_book(pd.Series(1.0, index=W.index), W)
        g = g.loc[idx].dropna()
        nt = nt.loc[g.index]
        if label == "main_2012":
            vs_base = nt
        base_rows.append(dict(name="varswap_constant_1", family="baseline_varswap", variant=label,
                              all_gross_sharpe=perf(g, g)["sharpe"], all_net_sharpe=perf(nt, nt)["sharpe"],
                              **{"pre2016_gross_sharpe": perf(g[g.index < SPLIT_TS], g[g.index < SPLIT_TS])["sharpe"],
                                 "2016+_gross_sharpe": perf(g[g.index >= SPLIT_TS], g[g.index >= SPLIT_TS])["sharpe"]}))
    econ_df.to_csv(OUT / "econ_svxy.csv", index=False)
    pd.DataFrame(base_rows).to_csv(OUT / "econ_baselines.csv", index=False)
    pd.DataFrame(vsw).to_csv(OUT / "econ_varswap.csv", index=False)

    # ---- joint reality checks on the main window -----------------------------------
    df_ret = pd.concat({k: ret_store[k] for k in ret_store}, axis=1).loc[EVAL_START:].dropna()
    print("reality-check window", df_ret.index[0].date(), df_ret.index[-1].date(), len(df_ret), "x", df_ret.shape[1], flush=True)
    bench = bh_ex_main.reindex(df_ret.index)
    reality_check(df_ret, bench, BLOCK, "reality_check")
    exf = ~((df_ret.index >= "2018-02-01") & (df_ret.index < "2018-03-01"))
    reality_check(df_ret[exf], bench[exf], BLOCK, "reality_check_exFeb2018")
    df_vs = pd.concat({k: vs_store[k] for k in vs_store}, axis=1).loc[EVAL_START:].dropna()
    reality_check(df_vs, vs_base.reindex(df_vs.index), 63, "reality_check_varswap")

    # ---- Spearman skill vs simple benchmarks (VIX level / VIX percentile rule) ---------
    common = None
    for n_, sc_ in scores.items():
        common = sc_.index if common is None else common.intersection(sc_.index)
    tgt = W["fwd_rv"].reindex(common)
    vixr = W["vix"].reindex(common)
    pct = scores.get("rule_vix_pctile")
    skill = []
    ok = tgt.notna()
    for n_, sc_ in scores.items():
        a_ = sc_.reindex(common)[ok]
        rho = stats.spearmanr(a_, tgt[ok])[0]
        d, se, t = boot_corr_diff(a_.to_numpy(), pct.reindex(common)[ok].to_numpy(), tgt[ok].to_numpy())
        skill.append(dict(name=n_, spearman_fwd_rv=rho, diff_vs_vix_pctile=d, se=se, t=t))
    skill.append(dict(name="VIX_level_itself", spearman_fwd_rv=stats.spearmanr(vixr[ok], tgt[ok])[0]))
    pd.DataFrame(skill).to_csv(OUT / "rv_forecast_skill.csv", index=False)

    # ---- order selection: what happens beyond 2016 (main walk-forward, min_train 1260) -----
    rows = []
    for key in ("orig", "core"):
        for n in (2, 3, 4, 5):
            pth = OUT / f"hmm{n}_{key}.csv"
            if pth.exists():
                d = load_method(f"hmm{n}_{key}")["ll"].dropna()
                a_, b_ = d[d.index < SPLIT_TS], d[d.index >= SPLIT_TS]
                rows.append(dict(features=key, n_states=n, oos_ll_2011_2015=a_.mean(), se_pre=_mean_se_block(a_.to_numpy())[1],
                                 oos_ll_2016plus=b_.mean(), se_post=_mean_se_block(b_.to_numpy())[1]))
    gains = []
    for key in ("orig", "core"):
        for n in (3, 4, 5):
            p1, p0 = OUT / f"hmm{n}_{key}.csv", OUT / f"hmm{n - 1}_{key}.csv"
            if p1.exists() and p0.exists():
                dd = (load_method(f"hmm{n}_{key}")["ll"] - load_method(f"hmm{n - 1}_{key}")["ll"]).dropna()
                for lab, m in (("2011-15", dd.index < SPLIT_TS), ("2016+", dd.index >= SPLIT_TS)):
                    mu, se_ = _mean_se_block(dd[m].to_numpy())
                    gains.append(dict(features=key, comparison=f"{n} vs {n - 1}", period=lab, mean_ll_gain_per_day=mu, se=se_, t=mu / se_))
    pd.DataFrame(rows).to_csv(OUT / "select_n_beyond.csv", index=False)
    pd.DataFrame(gains).to_csv(OUT / "select_n_beyond_paired.csv", index=False)

    # ---- trial ledger --------------------------------------------------------------
    led = pd.DataFrame(ledger)
    return econ_df, None, led


def stage_reference(W):
    """NON-CAUSAL reference: the original recipe (fit once on 2006-2015, Viterbi
    `predict` over the whole 2016+ block) vs the same model filtered causally,
    evaluated on real SVXY.  Quantifies what look-ahead buys.  Not a candidate method."""
    F = rm.build_features().dropna(subset=["rv_30", "vrp_vol", "slope"])
    cols, sort = rm.FEATURE_SETS["orig"]
    tr, te = F[F.index < SPLIT_TS], F[F.index >= SPLIT_TS]
    m = rm.HMMRegime(3, cols, sort, n_init=1, seed=42).fit(tr)       # same seed as the notebook (first restart)
    Xs = (m._X(te) - m.mu_) / m.sd_
    vit = m.model_.predict(Xs)
    rank = np.argsort(m.order_)[vit]                                  # vol rank of each decoded state
    P_f = m.filter(F)[len(tr):]
    MULT = np.array([1.0, 0.25, 0.0])                                 # original notebook: calm/stressed/crisis
    rows = []
    for lab, e_arr in (("original_viterbi_NONCAUSAL", MULT[rank]),
                       ("same_model_filtered_hard_argmax", MULT[P_f.argmax(1)]),
                       ("same_model_filtered_soft", P_f @ MULT)):
        e = pd.Series(e_arr, te.index)
        g, nt, rf, pos = svxy_strategy(e.reindex(W.index), W)
        pn = perf(nt, nt - rf, pos)
        rows.append(dict(model=lab, variant="exposure=[1,.25,0][state]", net_sharpe=pn["sharpe"], max_dd=pn["max_dd"],
                         worst_day=pn["worst_day"], avg_exposure=pn["avg_exposure"], turnover=pn["turnover_per_year"]))
    # label agreement
    agree = float(((P_f.argmax(1)) == rank).mean())
    bh = svxy_strategy(pd.Series(1.0, index=W.index), W)
    mask = bh[1].index >= SPLIT_TS
    pb = perf(bh[1][mask], bh[1][mask] - bh[2][mask])
    rows.append(dict(model="svxy_buy_and_hold", variant="-", net_sharpe=pb["sharpe"], max_dd=pb["max_dd"],
                     worst_day=pb["worst_day"], avg_exposure=1.0, turnover=0.0))
    out = pd.DataFrame(rows)
    out["viterbi_vs_filter_argmax_agreement"] = agree
    out.to_csv(OUT / "ref_viterbi_vs_filtered.csv", index=False)
    return out


def stage_figs(W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 1. stress scores over time
    names = ["hmm3_orig", "hmm3_core", "jump3_lam20", "rule_vix_vix3m_inv", "gjr_garch"]
    names = [n for n in names if (OUT / f"{n}.csv").exists()]
    fig, ax = plt.subplots(len(names) + 1, 1, figsize=(13, 2.0 * (len(names) + 1)), sharex=True)
    ax[0].plot(W.index, W["vix"], lw=0.7, color="k")
    ax[0].set_ylabel("VIX")
    for d in ("2015-08-24", "2018-02-05", "2020-03-16", "2022-03-07"):
        for a_ in ax:
            a_.axvline(pd.Timestamp(d), color="r", lw=0.6, ls="--")
    for a_, n in zip(ax[1:], names):
        df = load_method(n)
        a_.fill_between(df.index, df["stress_score"], color="C3", alpha=0.5, lw=0)
        a_.set_ylim(0, 1)
        a_.set_ylabel(n, fontsize=7)
    ax[0].set_xlim(pd.Timestamp("2012-01-01"), W.index[-1])
    ax[-1].axvline(SPLIT_TS, color="b", lw=0.8)
    fig.suptitle("Causal stress scores (walk-forward). Dashed = Aug-15, Feb-18, Mar-20, Mar-22 spikes; blue = 2016 split")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stress_timeline.png", dpi=110)
    plt.close(fig)
    # 2. order selection
    sel = pd.read_csv(OUT / "select_n.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for key, c in (("orig", "C0"), ("core", "C1")):
        h = sel[(sel.features == key) & (sel.model == "hmm")]
        g = sel[(sel.features == key) & (sel.model == "gmm")]
        ax[0].plot(h.n_states, h.bic_pre2016, "o-", color=c, label=f"HMM {key}")
        ax[0].plot(g.n_states, g.bic_pre2016, "s--", color=c, alpha=0.6, label=f"GMM {key}")
        ax[1].errorbar(h.n_states, h.oos_ll_per_day_pre2016, h.oos_ll_se, fmt="o-", color=c, label=f"HMM {key}")
        ax[1].errorbar(g.n_states, g.oos_ll_per_day_pre2016, g.oos_ll_se, fmt="s--", color=c, alpha=0.6, label=f"GMM {key}")
    ax[0].set_title("BIC (fit on 2006-2015)")
    ax[1].set_title("Walk-forward OOS log-lik / day (2009-08..2015-12)")
    for a_ in ax[:2]:
        a_.set_xlabel("n states")
        a_.legend(fontsize=7)
    # beyond: 2016+ OOS ll from the main walk-forward runs
    rows = []
    for key in ("orig", "core"):
        for n in (2, 3, 4, 5):
            p = OUT / f"hmm{n}_{key}.csv"
            if p.exists():
                d = pd.read_csv(p, index_col=0, parse_dates=True)["ll"]
                rows.append((key, n, d[d.index < SPLIT_TS].mean(), d[d.index >= SPLIT_TS].mean()))
    for key, c in (("orig", "C0"), ("core", "C1")):
        r = [x for x in rows if x[0] == key]
        ax[2].plot([x[1] for x in r], [x[3] for x in r], "o-", color=c, label=f"HMM {key} 2016+")
        ax[2].plot([x[1] for x in r], [x[2] for x in r], "x:", color=c, label=f"HMM {key} 2011-15")
    ax[2].set_title("Main walk-forward (min_train 1260): OOS ll/day")
    ax[2].set_xlabel("n states")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_order_selection.png", dpi=110)
    plt.close(fig)
    # 3. net Sharpe of de-risk variants
    econ = pd.read_csv(OUT / "econ_svxy.csv")
    base = pd.read_csv(OUT / "econ_baselines.csv")
    bh = base[(base.name == "buy_and_hold") & (base.variant == "main_2012")].iloc[0]
    e = econ[(econ.variant == "derisk_soft") & econ.net_sharpe_ci_lo.notna()].sort_values("all_net_sharpe")
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(e) + 1.5))
    ax.errorbar(e.all_net_sharpe, range(len(e)), xerr=[e.all_net_sharpe - e.net_sharpe_ci_lo, e.net_sharpe_ci_hi - e.all_net_sharpe],
                fmt="o", ms=3, lw=0.8)
    ax.axvline(bh.all_net_sharpe, color="r", ls="--", label=f"SVXY buy&hold {bh.all_net_sharpe:.2f}")
    ax.set_yticks(range(len(e)))
    ax.set_yticklabels(e.name, fontsize=7)
    ax.set_xlabel("net excess Sharpe, soft de-risk gate (1 - stress), 2012-2026, 95% block-bootstrap CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_derisk_sharpe.png", dpi=110)
    plt.close(fig)


def persist_start_9d(persist):
    d = [p["start"] for p in persist if "vix9d" in p["name"]]
    return str(max(d)) if d else str(EVAL_START.date())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def stage_run(specs, force, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_refits").mkdir(exist_ok=True)
    todo = [s for s in specs if force or not (OUT / f"{s.name}.csv").exists()]
    print(f"running {len(todo)} / {len(specs)} methods with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, s, True): s for s in todo}
        for f, s in futs.items():
            try:
                name, status, secs = f.result()
                print(f"  {name}: {status} ({secs}s)", flush=True)
            except Exception as e:  # report and carry on
                print(f"  {s.name}: FAILED {e!r}", flush=True)
    # (re)build meta from per-method CSVs + any cached info
    rows = []
    for s in specs:
        p = OUT / f"{s.name}.csv"
        if p.exists():
            rows.append(s.name)
    return rows


def write_meta(specs):
    """methods_meta.csv; if infos are not cached we recompute them from the CSV + refit logs."""
    rows = []
    for s in specs:
        p = OUT / f"{s.name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        model = s.factory()()
        info = dict(name=s.name, family=s.family, min_train=s.min_train, refit_every=s.refit_every,
                    hard_threshold=model.hard_threshold, first_date=df.index[0].date(), n_rows=len(df), note=s.note)
        rp = OUT / "_refits" / f"{s.name}.csv"
        if rp.exists():
            rf = pd.read_csv(rp)
            info["n_refits"] = len(rf)
            mc = [c for c in rf.columns if c.startswith("mean_")]
            if mc:
                top = rf[mc[-1]]
                info["top_state_mean_cv"] = float(top.std() / abs(top.mean())) if abs(top.mean()) > 0 else np.nan
                info["label_flips"] = int((np.diff(rf[mc].to_numpy(), axis=1) <= 0).any(axis=1).sum())
            pc = [c for c in rf.columns if c.startswith("pstay_")]
            if pc:
                info["pstay_top_mean"] = float(rf[pc[-1]].mean())
                info["pstay_low_mean"] = float(rf[pc[0]].mean())
                info["exp_duration_top_days"] = float((1.0 / (1.0 - rf[pc[-1]].clip(upper=0.9999))).median())
        sp = OUT / "_stability" / f"{s.name}.csv"
        if sp.exists() and sp.stat().st_size > 5:
            st = pd.read_csv(sp)
            info["flag_agree_old_new"] = st["flag_agree"].mean()
            info["mean_abs_stress_diff"] = st["mean_abs_diff"].mean()
        rows.append(info)
    meta = pd.DataFrame(rows).set_index("name")
    meta.to_csv(OUT / "methods_meta.csv")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "select", "run", "eval"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    if a.stage in ("all", "select"):
        t = time.time()
        d1, d2 = run_select(a.force)
        print(d1.round(3).to_string(), flush=True)
        print(d2.round(3).to_string(), flush=True)
        print(f"select done {time.time() - t:.0f}s", flush=True)
    if a.stage in ("all", "run"):
        stage_run(specs, a.force, a.workers)
    if a.stage in ("all", "eval"):
        meta = write_meta(specs)
        W = load_world()
        print(stage_reference(W).round(3).to_string(), flush=True)
        econ, rc, led = evaluate(specs, W)
        stage_figs(W)
        sel = pd.read_csv(OUT / "select_n.csv")
        extra = pd.DataFrame([dict(kind="order_selection_fit", name=f"{r.model}{r.n_states}_{r.features}", variant="bic/aic/wf_ll")
                              for r in sel.itertuples()])
        meths = pd.DataFrame([dict(kind="regime_method", name=n, variant="walk_forward") for n in meta.index])
        ledger = pd.concat([meths, extra, led], ignore_index=True)
        ledger.to_csv(OUT / "trial_ledger.csv", index=False)
        print(ledger.groupby("kind").size())
        print("TOTAL trials:", len(ledger))


if __name__ == "__main__":
    main()
