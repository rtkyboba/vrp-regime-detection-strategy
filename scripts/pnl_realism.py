"""Honest P&L for short variance: validate the models, then re-score the original strategy.

    python scripts/pnl_realism.py

One command reproduces every number in reports/01_pnl_realism.md and writes CSV/PNG files
to results/pnl/. Run with the project venv, PYTHONPATH=<worktree>/src.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from vrp_strategy import pnl
from vrp_strategy.backtest import raw_strategy_returns, vol_target_scale
from vrp_strategy.data import ROOT, SPLIT, asset_returns, load_raw, load_signals, rf_daily
from vrp_strategy.metrics import performance_metrics
from vrp_strategy.regimes import filtered_proba, state_order_by_feature

OUT = ROOT / "results" / "pnl"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 70)
fmt = lambda df, n=3: df.round(n)

TRIALS = []          # every (position, P&L model, config, cost) we evaluate - see report


def trial(*key):
    TRIALS.append(key)


def save(df, name):
    df.to_csv(OUT / name)


# ======================================================================================
# 0. Data and the ORIGINAL strategy's causal positions (logic of scripts/audit_sharpe.py)
# ======================================================================================
sig = load_signals()
END = sig.index[-1]
lr = pnl.spx_log_returns()
curve = pnl.load_curve()
vix = curve["vix"]
svxy = asset_returns("svxy")
svxy1x = pnl.svxy_minus1x_equivalent(svxy)
spy = asset_returns("spy")

FEATURES = ["vrp_vol", "slope", "rv_30"]
MULT_BY_RANK = np.array([1.0, 0.25, 0.0])
train = sig[sig.index < SPLIT]
scaler = StandardScaler().fit(train[FEATURES])
model = hmm.GaussianHMM(3, covariance_type="full", n_iter=1000, random_state=42).fit(scaler.transform(train[FEATURES]))
order = list(state_order_by_feature(model, scaler, FEATURES.index("rv_30")))
mult_of_state = np.array([MULT_BY_RANK[order.index(s)] for s in range(3)])
# forward filter over the full history in one pass (fit is pre-2016 only; posterior at t uses X[:t+1])
p_all = filtered_proba(model, scaler.transform(sig[FEATURES]))
rank = sig["vrp"].rolling(252).rank(pct=True)
POS = {
    "soft (original, causal)": (rank * pd.Series(p_all @ mult_of_state, index=sig.index)).clip(0, 1),
    "hard argmax (causal)": (rank * pd.Series(mult_of_state[p_all.argmax(1)], index=sig.index)).clip(0, 1),
    "rank only (no regime gate)": rank.clip(0, 1),
    "constant short": pd.Series(1.0, index=sig.index),
}
POS_MAIN = POS["soft (original, causal)"]
print(f"Sample: OOS = {SPLIT} .. {END.date()}; positions: {len(POS)} variants; "
      f"mean soft position OOS = {POS_MAIN[POS_MAIN.index >= SPLIT].mean():.3f}")

# ======================================================================================
# 1. Held-to-maturity variance swap
# ======================================================================================
htm = pnl.hold_to_maturity(vix, lr)
htm_oos = htm[(htm.index >= SPLIT)].dropna(subset=["ret"])
rows = []
for label, sub in [("all entries 2006-2026", htm.dropna(subset=["ret"])), ("entries >= 2016", htm_oos),
                   ("entries 2006-2015", htm[htm.index < SPLIT].dropna(subset=["ret"]))]:
    p = sub["payoff_vega"]
    rows.append({"sample": label, "n_entries": len(sub), "avg_strike_vol": vix.reindex(sub.index).mean(),
                 "avg_realised_vol": (100 * np.sqrt(sub["rv_var"])).mean(),
                 "mean_payoff_volpts": p.mean(), "median_payoff": p.median(), "std_payoff": p.std(),
                 "win_rate": (p > 0).mean(), "worst_payoff_volpts": p.min(),
                 "skew": p.skew(), "ex_kurt": p.kurt(),
                 "ruin_free_leverage": 1.0 / (-sub["ret"].min()) if sub["ret"].min() < 0 else np.nan,
                 "pct_months_loss_gt_capital": (sub["ret"] < -1).mean()})
t_htm = pd.DataFrame(rows).set_index("sample")
print("\n[1] Held-to-maturity 1-month short variance swap (payoff in vol points per unit vega; capital = "
      f"{pnl.VEGA_CAPITAL:.0f} vol pts)"); print(fmt(t_htm.T))
save(t_htm, "htm_summary.csv")


def htm_sharpes(position, cost_bps=0.0, start=SPLIT, label=""):
    """Sharpe of the held-to-maturity strategy three ways. Entries before `start` are masked."""
    ent = htm["ret"].where(htm.index >= start)
    ent = ent.where(htm["maturity"].notna() & (htm["maturity"] <= END))
    off = pnl.monthly_sharpe_over_offsets(ent, position, entry_cost_bps=cost_bps)
    lad = pnl.cohort_series(ent, position, entry_cost_bps=cost_bps).dropna()
    n_c = len(pnl.monthly_nonoverlap(ent, position, offset=0, entry_cost_bps=cost_bps))
    s_m = off.mean() / np.sqrt(12)          # monthly Sharpe; Lo (2002) iid standard error, annualised
    se = np.sqrt(12 * (1 + 0.5 * s_m ** 2) / n_c)
    return {"monthly_sharpe_mean": off.mean(), "monthly_sharpe_min": off.min(), "monthly_sharpe_max": off.max(),
            "ladder_sharpe_NW": pnl.sharpe_nw(lad), "ladder_sharpe_naive": lad.mean() / lad.std() * np.sqrt(252),
            "approx_se_iid": se, "n_cohorts": n_c}


rows = {}
for name, pos in POS.items():
    p = pos.reindex(htm.index)
    for c in (0, 1, 2, 5):
        if name not in ("soft (original, causal)", "constant short") and c > 0:
            continue
        trial(name, "HTM", "monthly/ladder", c)
        rows[(name, c)] = htm_sharpes(p, c)
t_htm_s = pd.DataFrame(rows).T
t_htm_s.index.names = ["position", "cost_bps"]
print("\n[1b] HTM Sharpe, entries >= 2016 (monthly = non-overlapping, averaged over the 21 phase offsets; "
      "ladder = overlapping daily entries booked at maturity)"); print(t_htm_s.round(3))
save(t_htm_s, "htm_sharpe.csv")

# ======================================================================================
# 2. MTM constant-maturity swap vs real SVXY / VXX
# ======================================================================================
m = pnl.mtm_returns(curve, lr)                       # columns carry, realised, mtm (total), legacy
m_flat = pnl.mtm_returns(curve, lr, use_nodes=("vix",))   # no term structure (single VIX node)
PRE = (svxy1x.index >= "2011-11-01") & (svxy1x.index < SPLIT)   # calibration window (pre-2016 only)
post = lambda s: s[s.index >= SPLIT]
pre = lambda s: s[(s.index >= "2011-11-01") & (s.index < SPLIT)]

streams = {"MTM swap (carry+realised+MTM)": m["mtm"], "MTM swap, single VIX node (no term str.)": m_flat["mtm"],
           "legacy proxy (carry - realised)": m["legacy"]}
rows = []
beta_pre = {}
for k, s in streams.items():
    trial("validation", k)
    b0 = pnl.regression_stats(pre(svxy1x), pre(s))["beta"]
    beta_pre[k] = b0
    for lab, f in [("2011-11..2015 (calibration)", pre), ("2016..2026 (OOS)", post)]:
        y, x = f(svxy1x), f(s)
        st = pnl.regression_stats(y, x)
        j = pd.concat([y, x], axis=1, join="inner").dropna()
        te_oos = (j.iloc[:, 0] - b0 * j.iloc[:, 1]).std() * np.sqrt(252)
        rows.append({"model": k, "period": lab, "corr": st["corr"], "beta (SVXY on model)": st["beta"],
                     "R2": st["r2"], "TE_ann_with_pre2016_beta": te_oos, "n": st["n"]})
t_val = pd.DataFrame(rows).set_index(["model", "period"])
print("\n[2] Daily validation vs real SVXY (-1x-equivalent: post-2018-02-27 returns doubled)"); print(fmt(t_val))
save(t_val, "validation_vs_svxy.csv")
print("  calibration: pre-2016 beta of SVXY on MTM swap at capital", pnl.VEGA_CAPITAL, "=",
      round(beta_pre["MTM swap (carry+realised+MTM)"], 3))

# raw (not -1x equivalent) SVXY too, and -VXX
for lab, y in [("SVXY raw (as traded)", svxy), ("-VXX (2018-01..)", -asset_returns("vxx"))]:
    for k in ("MTM swap (carry+realised+MTM)", "legacy proxy (carry - realised)"):
        st = pnl.regression_stats(post(y) if lab.startswith("SVXY") else y, streams[k])
        print(f"  {lab:24s} vs {k:32s} corr={st['corr']:.3f} beta={st['beta']:.3f} n={st['n']}")

# where the MTM swap breaks: by year corr, tail days
yr = pd.concat([svxy1x, m["mtm"], m["legacy"]], axis=1, join="inner").dropna()
yr.columns = ["svxy1x", "mtm", "legacy"]
by_year = yr.groupby(yr.index.year).apply(lambda d: pd.Series({
    "corr_mtm": pnl.regression_stats(d.svxy1x, d.mtm)["corr"], "corr_legacy": pnl.regression_stats(d.svxy1x, d.legacy)["corr"]}))
print("\n  correlation with SVXY by year"); print(fmt(by_year, 2).T)
save(by_year, "corr_by_year.csv")
b_mtm, b_leg = beta_pre["MTM swap (carry+realised+MTM)"], beta_pre["legacy proxy (carry - realised)"]
worst = yr.nsmallest(20, "svxy1x")
tail = pd.Series({"SVXY -1x worst-20-day mean": worst.svxy1x.mean(), "MTM swap x pre2016 beta": b_mtm * worst.mtm.mean(),
                  "legacy x pre2016 beta": b_leg * worst.legacy.mean()})
print("\n  worst 20 SVXY days: mean daily return captured by each model (beta-scaled)"); print(fmt(tail))
save(tail.to_frame("worst20_mean"), "tail_capture.csv")

# drawdown / risk comparison at 1x
eq_rows = []
for lab, lo, hi in [("2011-11 .. 2018-02-02", "2011-11-01", "2018-02-02"), ("2011-11 .. end", "2011-11-01", None),
                    ("2018-03 .. end", "2018-03-01", None)]:
    for k, s in [("SVXY (as traded)", svxy), ("SVXY (-1x equivalent)", svxy1x), ("MTM swap 1x", m["mtm"]), ("MTM swap x pre2016 beta (SVXY-sized)", b_mtm * m["mtm"]),
                 ("legacy x pre2016 beta (SVXY-sized)", b_leg * m["legacy"])]:
        s = s[(s.index >= lo)]
        s = s[s.index <= (hi or END)]
        s = s.dropna()
        st = pnl.perf_stats(s)
        eq_rows.append({"window": lab, "series": k, "ann_return": st["ann_return"], "ann_vol": st["ann_vol"],
                        "sharpe": st["sharpe"], "max_dd": st["max_drawdown"], "worst_day": st["worst_day"],
                        "ruined": st["ruined"]})
t_dd = pd.DataFrame(eq_rows).set_index(["window", "series"])
print("\n[2b] Unlevered return/drawdown comparison (non-excess, 1x)"); print(fmt(t_dd))
save(t_dd, "drawdown_comparison.csv")

# events
ev = pd.DataFrame({"svxy_as_traded": svxy, "svxy_minus1x_equiv": svxy1x, "mtm_swap_1x": m["mtm"], "mtm_x_beta": b_mtm * m["mtm"],
                   "legacy_x_beta": b_leg * m["legacy"], "mtm_term_only_1x": m["mtm_term"]})
ev["vix"] = vix
ev["vix_chg"] = vix.diff()
for lab, a, b in [("2018-02", "2018-02-01", "2018-02-09"), ("2020-02/03", "2020-02-20", "2020-03-18")]:
    e = ev.loc[a:b]
    print(f"\n[2c] Event {lab}: daily returns (mtm_swap_1x in capital units; *_x_beta = pre-2016-beta-scaled, i.e. SVXY(-1x)-sized)")
    print(fmt(e, 3))
    cum = (1 + e.drop(columns=["vix", "vix_chg", "mtm_term_only_1x"])).prod() - 1
    print("  compounded over window:", cum.round(3).to_dict())
    save(e, f"event_{lab.replace('/', '_')}.csv")

# decomposition by year: where the legacy proxy leaves out the loss
dec = pnl.mtm_decomposition(curve, lr)
# express in vol points per unit vega per year for readability (sum of daily capital-free vega points)
vp = pd.DataFrame({k: (dec[k] * 252 / 21 * 1e4) / (2 * vix.shift(1)) for k in ["carry", "realised", "mtm", "total", "legacy"]})
dec_y = vp.groupby(vp.index.year).sum()
print("\n[2d] Annual P&L decomposition of the constant-maturity short swap (vol points per unit vega)"); print(fmt(dec_y, 1))
save(dec_y, "annual_decomposition_volpts.csv")
oos = vp[vp.index >= SPLIT]
print("  OOS mean annual vol-pts:", (oos.mean() * 252).round(2).to_dict())

# ======================================================================================
# 3. Vol-targeting: turnover of the original vs dampened
# ======================================================================================
proxy_var = (sig["iv_daily"].shift(1) - sig["rv_daily"])          # legacy, daily-variance units
rf = rf_daily(svxy.index)
cfgs = {
    "V0 original: strategy-vol, 126d rolling, daily re-lever, cap 100": (pnl.VolTarget(0.10, window=126, basis="strategy", min_periods=126), 100.0),
    "V1 unit-vol, 126d rolling, daily": (pnl.VolTarget(0.10, window=126, basis="unit", min_periods=126), 100.0),
    "V2 unit-vol, EWMA hl=63, daily": (pnl.VolTarget(0.10, halflife=63, basis="unit", min_periods=63), 100.0),
    "V3 unit-vol, EWMA hl=63, band 25%": (pnl.VolTarget(0.10, halflife=63, band=0.25, basis="unit", min_periods=63), 100.0),
    "V4 unit-vol, EWMA hl=126, band 25%": (pnl.VolTarget(0.10, halflife=126, band=0.25, basis="unit", min_periods=63), 100.0),
}
DAMP = pnl.VolTarget(0.10, halflife=63, band=0.25, basis="unit", min_periods=63)   # default "dampened" spec (chosen a priori)
oos_slice = lambda r: r[r.index >= SPLIT]
svx_idx = svxy[svxy.index <= END]


def lev_turnover(unit, pos, vt, cap):
    """Turnover of the leverage scalar alone (sum |d lev| per year, OOS) as in notebook 08, plus
    its mean level so the damping can be read in relative terms."""
    lev = pnl.leverage_series(unit, pos, vt, cap)
    lv = oos_slice(lev).dropna()
    return lv.diff().abs().sum() / (len(lv) / 252), lv.mean()


rows = []
for uname, unit, cap, instr_rf in [("legacy proxy (cap 100)", proxy_var, 100.0, None), ("real SVXY (cap 3x)", svx_idx, 3.0, rf)]:
    for pname in ("soft (original, causal)", "constant short"):
        pos = POS[pname]
        for cname, (vt, _) in cfgs.items():
            trial(pname, "vol-target turnover", uname, cname)
            lt, lm = lev_turnover(unit, pos, vt, cap)
            res = pnl.backtest(pos, unit, vol_target=vt, max_leverage=cap, rf=instr_rf,
                               funding="funded" if instr_rf is not None else "collateral")
            r, e = oos_slice(res["returns"]).dropna(), oos_slice(res["exposure"])
            st = pnl.perf_stats(r, e)
            rows.append({"instrument": uname, "position": pname, "config": cname, "scalar_turnover_x_yr": lt, "mean_scalar": lm,
                         "scalar_turnover_per_unit_lev": lt / lm, "total_exposure_turnover_x_yr": st["turnover_per_yr"],
                         "sharpe_gross": st["sharpe"]})
t_turn = pd.DataFrame(rows)
res0 = pnl.backtest(POS_MAIN, svx_idx, max_leverage=3.0, rf=rf)       # signal-only turnover, no vol target
sig_turn = pnl.perf_stats(oos_slice(res0["returns"]).dropna(), oos_slice(res0["exposure"]))["turnover_per_yr"]
print("\n[3] Vol-target turnover, OOS (x per year). scalar_turnover = sum|d leverage| (notebook-08 definition)")
print(fmt(t_turn, 2).to_string())
print(f"  signal-only turnover (no vol target, soft position): {sig_turn:.2f} x/yr")
save(t_turn, "voltarget_turnover.csv")

# reproduce the headline with the old tools for reference
d = sig.assign(position=POS_MAIN)
d = d[d.index >= SPLIT].dropna(subset=["position"])
orig_r = raw_strategy_returns(d["position"], d["iv_daily"], d["rv_daily"]).dropna()
sc, scalar = vol_target_scale(orig_r, 0.10, 126, 100.0)
legacy_headline = performance_metrics(sc.dropna())
print(f"\n  original tooling, causal soft positions (legacy proxy, 10% target, cap 100): Sharpe={legacy_headline['sharpe']:.3f}, "
      f"mean scalar={scalar.mean():.1f}, max={scalar.max():.1f}")

# ======================================================================================
# 4. The ORIGINAL strategy through the honest P&L models
# ======================================================================================
def run(pos, unit, cost, vt, cap, funding, rf_=None):
    res = pnl.backtest(pos, unit, cost_bps=cost, vol_target=vt, max_leverage=cap, rf=rf_, funding=funding)
    r = oos_slice(res["returns"]).dropna()
    st = pnl.perf_stats(r, oos_slice(res["exposure"]))
    st["gross_sharpe"] = res["stats"]["gross_sharpe"] if False else (oos_slice(res["gross"]).dropna().mean() / oos_slice(res["gross"]).dropna().std() * np.sqrt(252))
    return st, r


MODELS = {      # name: (unit returns, funding, rf, vol_target, cap)
    "A legacy proxy, as published (vol-tgt 10%, cap 100)": (proxy_var, "collateral", None, pnl.VolTarget(0.10, window=126, basis="strategy", min_periods=126), 100.0),
    "B real SVXY, unlevered": (svx_idx, "funded", rf, None, 3.0),
    "C real SVXY, vol-tgt 10% dampened, cap 3x": (svx_idx, "funded", rf, DAMP, 3.0),
    "D real SVXY, vol-tgt 10% daily re-lever, cap 3x": (svx_idx, "funded", rf, cfgs["V1 unit-vol, 126d rolling, daily"][0], 3.0),
    "E MTM swap (cm 1m), unlevered 1x": (m["mtm"], "collateral", None, None, 3.0),
    "F MTM swap, vol-tgt 10% dampened, cap 3x": (m["mtm"], "collateral", None, DAMP, 3.0),
    "G legacy carry-realised on swap scale, unlevered 1x": (m["legacy"], "collateral", None, None, 3.0),
    "H legacy carry-realised on swap scale, vol-tgt 10% dampened, cap 3x": (m["legacy"], "collateral", None, DAMP, 3.0),
}
rows = []
curves = {}
for pname, pos in POS.items():
    for mname, (unit, funding, rf_, vt, cap) in MODELS.items():
        costs = (0, 1, 2, 5) if (pname in ("soft (original, causal)", "constant short") and not mname.startswith("A")) else (0,)
        for c in costs:
            trial(pname, mname, c)
            st, r = run(pos, unit, c, vt, cap, funding, rf_)
            rows.append({"position": pname, "model": mname, "cost_bps": c, **{k: st[k] for k in (
                "sharpe", "gross_sharpe", "sortino", "ann_return", "ann_vol", "max_drawdown", "worst_day", "skew",
                "excess_kurtosis", "turnover_per_yr", "pct_invested", "avg_abs_exposure", "ruined")}})
            if pname.startswith("soft") and c == 0:
                curves[mname] = r
t_main = pd.DataFrame(rows)
save(t_main, "original_strategy_rescored.csv")
cols = ["sharpe", "gross_sharpe", "sortino", "ann_return", "ann_vol", "max_drawdown", "worst_day", "skew", "excess_kurtosis", "turnover_per_yr", "pct_invested"]
print("\n[4] Original strategy (causal) through each P&L model, OOS 2016+, excess of rf where funded")
for pname in POS:
    sub = t_main[t_main.position == pname].set_index(["model", "cost_bps"])[cols]
    print(f"\n  -- position: {pname}"); print(fmt(sub, 3).to_string())

# baselines
base = []
for lab, r in [("SVXY buy&hold (excess)", pnl.backtest(pd.Series(1.0, index=svx_idx.index), svx_idx, rf=rf, max_leverage=3.0)["returns"]),
               ("SPY buy&hold (excess)", pnl.backtest(pd.Series(1.0, index=spy.index), spy[spy.index <= END], rf=rf_daily(spy.index), max_leverage=3.0)["returns"]),
               ("-VXX buy&hold (short, excess)", None)]:
    if r is None:
        continue
    trial("baseline", lab)
    r = oos_slice(r).dropna()
    st = pnl.perf_stats(r)
    base.append({"baseline": lab, **{k: st[k] for k in ("sharpe", "sortino", "ann_return", "ann_vol", "max_drawdown", "worst_day", "skew", "excess_kurtosis")}})
t_base = pd.DataFrame(base).set_index("baseline")
print("\n  Baselines (OOS)"); print(fmt(t_base))
save(t_base, "baselines.csv")

# diagnostic: why does HTM-ladder (0.4-0.5) differ from daily-traded MTM (0.0) for the gated position?
# A cohort ladder only lets the signal act on 1/21 of the book per day, i.e. an exposure equal to the
# trailing-21-day mean of the position. Feed that slowed position to the daily MTM models.
pos_ma = POS_MAIN.rolling(21, min_periods=1).mean()
diag = []
for mname in ("B real SVXY, unlevered", "E MTM swap (cm 1m), unlevered 1x"):
    unit, funding, rf_, vt, cap = MODELS[mname]
    for lab, pos in [("soft daily position", POS_MAIN), ("soft position, 21d trailing mean (ladder-equivalent)", pos_ma)]:
        trial(lab, mname, "diagnostic")
        st, _ = run(pos, unit, 0, vt, cap, funding, rf_)
        diag.append({"model": mname, "position": lab, "sharpe": st["sharpe"], "turnover_x_yr": st["turnover_per_yr"], "max_dd": st["max_drawdown"]})
t_diag = pd.DataFrame(diag).set_index(["model", "position"])
print("\n  Diagnostic: slowing the signal to the ladder's effective speed"); print(fmt(t_diag))
save(t_diag, "diagnostic_ladder_equivalent.csv")

# bootstrap CIs on the headline realistic numbers
ci_rows = []
for mname in ("B real SVXY, unlevered", "C real SVXY, vol-tgt 10% dampened, cap 3x", "E MTM swap (cm 1m), unlevered 1x"):
    st, r = run(POS_MAIN, *[MODELS[mname][i] for i in (0,)], 0, MODELS[mname][3], MODELS[mname][4], MODELS[mname][1], MODELS[mname][2])
    lo, hi = pnl.sharpe_bootstrap_ci(r)
    ci_rows.append({"model": mname, "sharpe": st["sharpe"], "ci95_lo": lo, "ci95_hi": hi, "nw_sharpe": pnl.sharpe_nw(r, 10)})
st, r = run(POS["constant short"], *[MODELS["E MTM swap (cm 1m), unlevered 1x"][i] for i in (0,)], 0, None, 3.0, "collateral", None)
lo, hi = pnl.sharpe_bootstrap_ci(r)
ci_rows.append({"model": "E constant short (swap MTM)", "sharpe": st["sharpe"], "ci95_lo": lo, "ci95_hi": hi, "nw_sharpe": pnl.sharpe_nw(r, 10)})
t_ci = pd.DataFrame(ci_rows).set_index("model")
print("\n  Block-bootstrap 95% CI on Sharpe (21d blocks)"); print(fmt(t_ci))
save(t_ci, "sharpe_ci.csv")

# subperiod split for the headline realistic row
sp = []
for mname in ("B real SVXY, unlevered", "E MTM swap (cm 1m), unlevered 1x", "C real SVXY, vol-tgt 10% dampened, cap 3x"):
    r = curves[mname]
    for lab, a, b in [("2016-2019", "2016-01-01", "2019-12-31"), ("2020-2022", "2020-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-12-31")]:
        s = r[(r.index >= a) & (r.index <= b)]
        sp.append({"model": mname, "period": lab, "sharpe": pnl.perf_stats(s)["sharpe"], "max_dd": pnl.perf_stats(s)["max_drawdown"]})
t_sp = pd.DataFrame(sp).set_index(["model", "period"])
print("\n  Sub-period Sharpe (soft position)"); print(fmt(t_sp))
save(t_sp, "subperiods.csv")

# ======================================================================================
# plots
# ======================================================================================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
w = slice("2011-11-01", None)
for lab, s in [("SVXY (as traded)", svxy), ("MTM swap x beta (SVXY-sized)", b_mtm * m["mtm"]), ("legacy carry-realised x beta", b_leg * m["legacy"])]:
    s = s[w]
    s = s[s.index <= END]
    ax[0].plot((1 + s.fillna(0)).cumprod().clip(lower=1e-3), label=lab)
ax[0].set_yscale("log"); ax[0].set_title("Unlevered short-vol equity (log), 2011-2026"); ax[0].legend(fontsize=8)
for lab, r in [("A legacy as published", curves["A legacy proxy, as published (vol-tgt 10%, cap 100)"]),
               ("C SVXY dampened vol-tgt", curves["C real SVXY, vol-tgt 10% dampened, cap 3x"]),
               ("F MTM swap dampened vol-tgt", curves["F MTM swap, vol-tgt 10% dampened, cap 3x"])]:
    ax[1].plot((1 + r).cumprod(), label=lab)
ax[1].set_yscale("log"); ax[1].set_title("Original causal positions, OOS 2016+"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "pnl_realism.png", dpi=110); plt.close()

# ======================================================================================
print(f"\nTRIAL COUNT: {len(TRIALS)} evaluations (position x P&L model x config x cost), "
      f"{len(set(t[:2] for t in TRIALS))} distinct (position, model) pairs; no parameter was tuned on 2016+.")
pd.Series({"n_trials": len(TRIALS)}).to_csv(OUT / "trial_count.csv")
