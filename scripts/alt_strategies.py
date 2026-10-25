"""Evaluate the alternative volatility strategies; reproduces every number in
reports/05_alternative_vol_strategies.md.

    python scripts/alt_strategies.py

Outputs (results/alt/): one <name>.csv per variant (date, exposure, ret, ...),
summary_*.csv, tables.md (markdown tables used in the report), equity.png.

Protocol (docs/EVAL_PROTOCOL.md): signals at close t earn t+1; excess-of-rf
Sharpe; 2 bps on the change in weights (0 and 5 also shown); parameters fixed
a priori (see the docstrings in vrp_strategy.alt_strategies); every variant
evaluated is registered in VARIANTS and counted as a trial; the blend's sleeve
selection uses pre-2016 data only and 2016+ is reported as the test.
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from vrp_strategy import alt_strategies as A
from vrp_strategy.data import ROOT
from vrp_strategy.metrics import safe_corr

OUT = ROOT / "results" / "alt"
OUT.mkdir(parents=True, exist_ok=True)
COSTS = (0.0, 2.0, 5.0)
HEAD_COST = 2.0
SPLIT = A.SPLIT
C11 = pd.Timestamp("2011-10-06")  # first day every SVXY sleeve can earn a return

P = A.build_panel()
R = P[["spy", "svxy", "lv", "vxx"]]

# ---------------------------------------------------------------- registry
VARIANTS = []  # every variant evaluated = one trial


def V(name, fam, desc, fn, kind="w", headline=False, drift=False):
    VARIANTS.append(dict(name=name, fam=fam, desc=desc, fn=fn, kind=kind, headline=headline, drift=drift))


# 1. vol-managed equity
V("vm_implied_m", "1 vol-managed", "SPY x (hist mean VIX / VIX), cap 1.5, monthly", lambda P: A.vol_managed(P, "implied"), headline=True)
V("vm_realised_m", "1 vol-managed", "SPY x (hist vol / 21d realised vol), cap 1.5, monthly", lambda P: A.vol_managed(P, "realised"))
V("vm_implied_d", "1 vol-managed", "as vm_implied_m, rebalanced daily", lambda P: A.vol_managed(P, "implied", rebal=1))
V("vm_realised_d", "1 vol-managed", "as vm_realised_m, rebalanced daily", lambda P: A.vol_managed(P, "realised", rebal=1))
V("vm_implied_m_cap1", "1 vol-managed", "vm_implied_m, no leverage (cap 1.0)", lambda P: A.vol_managed(P, "implied", cap=1.0))
V("vm_implied_stress", "1 vol-managed", "vm_implied_m x0.5 when VIX/VIX3M>=1", lambda P: A.vol_managed(P, "implied", overlay=True))
V("vm_realised_stress", "1 vol-managed", "vm_realised_m x0.5 when VIX/VIX3M>=1", lambda P: A.vol_managed(P, "realised", overlay=True))
# 2. term-structure carry switching
V("ts_bin_cash", "2 term-structure", "SVXY if VIX/VIX3M<1 else cash", lambda P: A.ts_carry(P), headline=True)
V("ts_bin9d_cash", "2 term-structure", "SVXY if VIX/VIX3M<1 and VIX9D/VIX<1 else cash", lambda P: A.ts_carry(P, use_9d=True))
V("ts_ramp_cash", "2 term-structure", "SVXY exposure ramp 0@1.0 -> 1@0.9 of VIX/VIX3M, else cash", lambda P: A.ts_carry(P, "ramp"))
V("ts_ramp9d_cash", "2 term-structure", "ramp, also min with VIX9D/VIX ramp", lambda P: A.ts_carry(P, "ramp", use_9d=True))
V("ts_schmitt_cash", "2 term-structure", "hysteresis: on <0.95, off >1.00", lambda P: A.ts_carry(P, hysteresis="schmitt"))
V("ts_deadband_cash", "2 term-structure", "ramp with 0.25 trade deadband", lambda P: A.ts_carry(P, "ramp", hysteresis="deadband"))
V("ts_bin_spy", "2 term-structure", "SVXY if VIX/VIX3M<1 else SPY", lambda P: A.ts_carry(P, off="spy"), headline=True)
V("ts_ramp_spy", "2 term-structure", "ramp: e*SVXY + (1-e)*SPY", lambda P: A.ts_carry(P, "ramp", off="spy"))
V("ts_schmitt_spy", "2 term-structure", "hysteresis, off-state SPY", lambda P: A.ts_carry(P, hysteresis="schmitt", off="spy"))
V("ts_deadband_spy", "2 term-structure", "ramp with deadband, off-state SPY", lambda P: A.ts_carry(P, "ramp", hysteresis="deadband", off="spy"))
# 3. buy-the-spike vs de-risk
V("spike_buy", "3 spike", "SVXY 15d after VIX spike once VIX/VIX3M<1 and VIX retraced 10%", lambda P: A.spike_buy(P), headline=True)
V("spike_buy_h21", "3 spike", "as spike_buy, hold 21d", lambda P: A.spike_buy(P, hold=21))
V("derisk_highvol", "3 spike", "SVXY always, flat when VIX 252d rank > 0.8", lambda P: A.derisk_highvol(P), headline=True)
V("highvol_only", "3 spike", "SVXY only when VIX rank > 0.8 and VIX/VIX3M<1", lambda P: A.derisk_highvol(P, invert=True))
# 4. VVIX / SKEW filter
V("svxy_vvix", "4 crash filter", "SVXY always, flat when VVIX 252d rank > 0.8", lambda P: A.crash_filter(P, "always", "vvix"))
V("svxy_skew", "4 crash filter", "SVXY always, flat when SKEW rank > 0.8", lambda P: A.crash_filter(P, "always", "skew"))
V("svxy_either", "4 crash filter", "SVXY always, flat when VVIX or SKEW rank > 0.8", lambda P: A.crash_filter(P, "always", "either"))
V("ts_vvix", "4 crash filter", "ts_bin_cash and VVIX rank <= 0.8", lambda P: A.crash_filter(P, "ts", "vvix"), headline=True)
V("ts_skew", "4 crash filter", "ts_bin_cash and SKEW rank <= 0.8", lambda P: A.crash_filter(P, "ts", "skew"))
V("ts_either", "4 crash filter", "ts_bin_cash and both gauges calm", lambda P: A.crash_filter(P, "ts", "either"))
# 5. hedges
V("spy_lv5", "5 hedge", "95% SPY + 5% long vol (VXX, -SVXY before 2018), daily mix", lambda P: A.spy_plus_hedge(P, 0.05, "lv"), headline=True, drift=True)
V("spy_lv2", "5 hedge", "98% SPY + 2% long vol", lambda P: A.spy_plus_hedge(P, 0.02, "lv"), drift=True)
V("spy_lv10", "5 hedge", "90% SPY + 10% long vol", lambda P: A.spy_plus_hedge(P, 0.10, "lv"), drift=True)
V("spy_vxx5", "5 hedge", "95% SPY + 5% real VXX (2018-01-25 on only)", lambda P: A.spy_plus_hedge(P, 0.05, "vxx"), drift=True)
V("putwrite", "5 hedge", "cash-secured monthly ATM put-write, BS at 0.93 x VIX (approximation)", lambda P, c: A.put_write(P, 0.93, cost_bps=c), kind="pw", headline=True)
V("putwrite_iv100", "5 hedge", "put-write priced at 1.00 x VIX", lambda P, c: A.put_write(P, 1.00, cost_bps=c), kind="pw")
V("putwrite_iv085", "5 hedge", "put-write priced at 0.85 x VIX", lambda P, c: A.put_write(P, 0.85, cost_bps=c), kind="pw")

# ---------------------------------------------------------------- run
W_CACHE, RES = {}, {}
for v in VARIANTS:
    for c in COSTS:
        if v["kind"] == "pw":
            RES[(v["name"], c)] = v["fn"](P, c)
        else:
            if v["name"] not in W_CACHE:
                W_CACHE[v["name"]] = v["fn"](P)
            RES[(v["name"], c)] = A.run_weights(W_CACHE[v["name"]], R, P["rf"], c, drift=v["drift"])
SPY = A.benchmark(P, "spy")
SVXY = A.benchmark(P, "svxy")
BENCH = {"SPY": SPY, "SVXY": SVXY}

WINDOWS = {
    "all": (None, None),
    "pre2016": (None, SPLIT - pd.Timedelta(days=1)),
    "test2016+": (SPLIT, None),
    "from2011-10": (C11, None),
}


def sl(res, w):
    a, b = WINDOWS[w]
    return res.loc[a:b] if (a is not None or b is not None) else res


def ex(res):
    return (res["ret"] - res["rf"]).to_numpy()


def fmt_md(df, floatfmt=3):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, row in df.iterrows():
        cells = [str(i)]
        for x in row:
            if isinstance(x, (float, np.floating)):
                cells.append("" if np.isnan(x) else f"{x:.{floatfmt}f}")
            else:
                cells.append(str(x))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


md = []  # collected markdown

# ---- headline metrics at 2 bps over windows
rows = []
for v in VARIANTS:
    res = RES[(v["name"], HEAD_COST)]
    for w in WINDOWS:
        r = sl(res, w)
        if len(r) < 100:
            continue
        m = A.metrics(r)
        spy_w = SPY.loc[r.index[0] : r.index[-1]]
        bs = A.block_bootstrap_sharpe(ex(r), ex(spy_w), seed=1)
        rows.append(dict(variant=v["name"], fam=v["fam"], window=w, start=r.index[0].date(), **m,
                         spy_sharpe=bs["sharpe_b"], diff_vs_spy=bs["diff"], diff_lo=bs["diff_lo"], diff_hi=bs["diff_hi"],
                         p_le0=bs["p_diff_le0"], sh_lo=bs["sharpe_lo"], sh_hi=bs["sharpe_hi"]))
for nm, b in BENCH.items():
    for w in WINDOWS:
        r = sl(b, w)
        if w == "all" and nm == "SVXY":
            pass
        m = A.metrics(r)
        rows.append(dict(variant="BH_" + nm, fam="benchmark", window=w, start=r.index[0].date(), **m))
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "summary_metrics.csv", index=False)

# ---- cost sensitivity
crow = []
for v in VARIANTS:
    d = {"variant": v["name"]}
    for c in COSTS:
        res = RES[(v["name"], c)]
        d[f"sharpe_{int(c)}bp_all"] = A.metrics(res)["sharpe"]
        d[f"sharpe_{int(c)}bp_test"] = A.metrics(sl(res, "test2016+"))["sharpe"] if len(sl(res, "test2016+")) > 100 else np.nan
    d["turnover_pa"] = A.metrics(RES[(v["name"], HEAD_COST)])["turnover_pa"]
    crow.append(d)
cost_tbl = pd.DataFrame(crow).set_index("variant")
cost_tbl.to_csv(OUT / "summary_costs.csv")

# ---- print compact tables
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)
for w in ["all", "pre2016", "test2016+", "from2011-10"]:
    t = summ[summ.window == w].set_index("variant")
    cols = ["start", "sharpe", "sh_lo", "sh_hi", "sortino", "cagr", "ann_vol", "max_dd", "worst_day", "skew", "ex_kurt", "turnover_pa", "pct_invested", "spy_sharpe", "diff_vs_spy", "diff_lo", "diff_hi", "p_le0"]
    t = t[cols]
    print(f"\n=== window {w} (net 2bps) ===")
    print(t.round(3))
    t.index.name = "variant"
    md.append(f"### Window: {w} (net of 2 bps)\n\n" + fmt_md(t.drop(columns=["sh_lo", "sh_hi", "skew", "ex_kurt"]).assign(start=t["start"].astype(str)), 3) + "\n")
print("\n=== cost sensitivity ===")
print(cost_tbl.round(3))
md.append("### Cost sensitivity (excess Sharpe)\n\n" + fmt_md(cost_tbl, 3) + "\n")

# ---- SVXY conditional diagnostics (strategy 3: does de-risking in stress hurt?)
sv_ex = (P["svxy"] - P["rf"])
fwd = sv_ex.shift(-1)  # next-day SVXY excess return given the state at close t
state = pd.DataFrame({
    "vix": P["vix"], "r1": P["vix"] / P["vix3m"], "rank": A.trailing_rank(P["vix"], 252), "fwd": fwd,
}).dropna()
state = state.loc[C11:]


def bucket_table(col, bins, labels):
    s = state.copy()
    s["b"] = pd.cut(s[col], bins, labels=labels)
    out = {}
    for per, sub in [("pre2016", s[s.index < SPLIT]), ("2016+", s[s.index >= SPLIT]), ("all", s)]:
        g = sub.groupby("b", observed=False)["fwd"]
        out[(per, "days")] = g.size()
        out[(per, "ann_mean")] = g.mean() * 252
        out[(per, "ann_vol")] = g.std() * np.sqrt(252)
        out[(per, "sharpe")] = g.mean() / g.std() * np.sqrt(252)
    t = pd.DataFrame(out)
    t.columns = [f"{a}:{b}" for a, b in t.columns]
    return t


bt_vix = bucket_table("vix", [0, 15, 20, 25, 35, 200], ["<15", "15-20", "20-25", "25-35", ">35"])
bt_r1 = bucket_table("r1", [0, 0.85, 0.90, 0.95, 1.0, 5], ["<0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00", ">=1.00"])
bt_rk = bucket_table("rank", [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0], ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
for nm, t in [("VIX level", bt_vix), ("VIX/VIX3M", bt_r1), ("VIX 252d rank", bt_rk)]:
    t.index.name = nm
    print(f"\n=== SVXY next-day excess return by {nm} ===")
    print(t.round(2))
    t.to_csv(OUT / f"cond_svxy_{nm.replace('/', '_').replace(' ', '_')}.csv")
    md.append(f"### SVXY next-day excess return by {nm}\n\n" + fmt_md(t[[c for c in t.columns if c.startswith(("pre2016", "2016+")) and ("days" in c or "ann_mean" in c or "sharpe" in c)]], 2) + "\n")

# post-spike forward returns: SVXY 15d excess after trigger days, vs unconditional
vix = P["vix"]
med = vix.rolling(60).median()
peak = vix.rolling(10).max()
trig = ((peak / med >= 1.4) & (vix / P["vix3m"] < 1.0) & (vix <= 0.9 * peak))
cum15 = (1 + sv_ex).rolling(15).apply(np.prod, raw=True).shift(-15) - 1
spk = pd.DataFrame({"trig": trig, "f15": cum15}).dropna().loc[C11:]
spk_tbl = pd.DataFrame({
    "n_days": [int(spk.trig.sum()), int((~spk.trig).sum())],
    "mean_15d_excess": [spk[spk.trig].f15.mean(), spk[~spk.trig].f15.mean()],
    "median_15d": [spk[spk.trig].f15.median(), spk[~spk.trig].f15.median()],
    "worst_15d": [spk[spk.trig].f15.min(), spk[~spk.trig].f15.min()],
}, index=["spike-normalised days", "all other days"])
spk_tbl.index.name = "state"
print("\n=== SVXY forward 15d excess return after spike trigger ===")
print(spk_tbl.round(4))
md.append("### SVXY forward 15-day excess return (overlapping) after the spike trigger\n\n" + fmt_md(spk_tbl, 4) + "\n")

# ---- hedge / diversification analysis
lv = P["lv"].loc[C11:]
spy_c = P["spy"].loc[C11:]
sleeve = RES[("ts_bin_cash", HEAD_COST)]
sleeve_ex = (sleeve["ret"] - sleeve["rf"]).loc[C11:]
lv_ex = (lv - P["rf"].loc[C11:])
spy_ex = (spy_c - P["rf"].loc[C11:])
pw_res = RES[("putwrite", HEAD_COST)]
pw_ex = (pw_res["ret"] - pw_res["rf"])
j = pd.concat([spy_ex.rename("SPY"), lv_ex.rename("long-vol"), sleeve_ex.rename("ts_bin_cash sleeve"), pw_ex.rename("putwrite")], axis=1).dropna()
cm = pd.DataFrame({a: {b: safe_corr(j[a], j[b]) for b in j.columns} for a in j.columns})
cm.index.name = "daily excess-return correlation"
print("\n=== correlations (2011-10 on) ===")
print(cm.round(2))
md.append("### Correlations of daily excess returns (from 2011-10)\n\n" + fmt_md(cm, 2) + "\n")
j_pre, j_post = j[j.index < SPLIT], j[j.index >= SPLIT]
cm2 = pd.DataFrame({"pre2016": [safe_corr(x["SPY"], x["long-vol"]) for x in (j_pre,)] + [safe_corr(x["ts_bin_cash sleeve"], x["long-vol"]) for x in (j_pre,)] + [safe_corr(x["ts_bin_cash sleeve"], x["SPY"]) for x in (j_pre,)] + [safe_corr(x["putwrite"], x["ts_bin_cash sleeve"]) for x in (j_pre,)],
                    "2016+": [safe_corr(x["SPY"], x["long-vol"]) for x in (j_post,)] + [safe_corr(x["ts_bin_cash sleeve"], x["long-vol"]) for x in (j_post,)] + [safe_corr(x["ts_bin_cash sleeve"], x["SPY"]) for x in (j_post,)] + [safe_corr(x["putwrite"], x["ts_bin_cash sleeve"]) for x in (j_post,)]},
                   index=["SPY vs long-vol", "short-vol sleeve vs long-vol", "short-vol sleeve vs SPY", "putwrite vs short-vol sleeve"])
cm2.index.name = "pair"
md.append("### Subperiod correlations\n\n" + fmt_md(cm2, 2) + "\n")
print(cm2.round(2))

# long-vol payoff on SPY's worst days
worst = spy_c.nsmallest(20).index
lv_on_worst = pd.DataFrame({"SPY": spy_c[worst], "long-vol": lv[worst], "5% hedge P&L": 0.05 * lv[worst]}).sort_index()
print("\nlong-vol on SPY's 20 worst days: mean", lv[worst].mean().round(4), "hit-rate", (lv[worst] > 0).mean())
md.append(f"Long-vol return on SPY's 20 worst days (2011-10 on): mean {lv[worst].mean():.3f}, positive on {(lv[worst] > 0).mean():.0%} of them; "
          f"on all days the long-vol series averages {lv.mean() * 252:.1%} per year (carry drag).\n")

# diversification benefit of the hedge: sharpe and DD vs spy on the same dates
hedge_rows = []
for nm in ["spy_lv2", "spy_lv5", "spy_lv10", "spy_vxx5", "putwrite", "putwrite_iv100", "putwrite_iv085"]:
    for w in ["from2011-10", "pre2016", "test2016+"]:
        r = sl(RES[(nm, HEAD_COST)], w)
        if nm == "spy_vxx5" and w == "pre2016":
            continue
        s = SPY.loc[r.index[0] : r.index[-1]]
        mm, ms = A.metrics(r), A.metrics(s)
        bs = A.block_bootstrap_sharpe(ex(r), ex(s), seed=2)
        hedge_rows.append(dict(variant=nm, window=w, sharpe=mm["sharpe"], spy_sharpe=ms["sharpe"], diff=bs["diff"], lo=bs["diff_lo"], hi=bs["diff_hi"],
                               max_dd=mm["max_dd"], spy_max_dd=ms["max_dd"], ann_vol=mm["ann_vol"], spy_vol=ms["ann_vol"], worst_day=mm["worst_day"], spy_worst=ms["worst_day"]))
hedge_tbl = pd.DataFrame(hedge_rows).set_index("variant")
hedge_tbl.to_csv(OUT / "summary_hedges.csv")
print("\n=== hedges ===")
print(hedge_tbl.round(3))
md.append("### Hedge sleeves vs SPY (net 2 bps)\n\n" + fmt_md(hedge_tbl, 3) + "\n")

# ---- 6. blend, sleeve selection on pre-2016 only
CANDIDATES = ["vm_implied_m", "ts_bin_cash", "spike_buy", "derisk_highvol", "ts_vvix", "putwrite"]
spy_ex_full = (P["spy"] - P["rf"]).loc[C11:]


def sleeve_ex_series(name):
    r = RES[(name, HEAD_COST)]
    return (r["ret"] - r["rf"])


sel_rows, survivors = [], []
for nm in CANDIDATES:
    pair = pd.concat([spy_ex_full.rename("spy"), sleeve_ex_series(nm).rename(nm)], axis=1).loc[C11:].dropna()
    res, _ = A.risk_parity_blend(pair, P["rf"], cost_bps=HEAD_COST)
    pre = res[res.index < SPLIT]
    post = res[res.index >= SPLIT]
    s_pre = A.metrics(pre)["sharpe"]
    spy_pre = A.metrics(SPY.loc[pre.index[0] : pre.index[-1]])["sharpe"]
    alone_pre = A.metrics(RES[(nm, HEAD_COST)].loc[C11 : SPLIT - pd.Timedelta(days=1)])["sharpe"]
    ok = (s_pre > spy_pre) and (alone_pre > 0)
    if ok:
        survivors.append(nm)
    s_post = A.metrics(post)["sharpe"]
    spy_post = A.metrics(SPY.loc[post.index[0] : post.index[-1]])["sharpe"]
    sel_rows.append(dict(sleeve=nm, alone_pre=alone_pre, blend_pre=s_pre, spy_pre=spy_pre, selected=ok, blend_test=s_post, spy_test=spy_post))
sel = pd.DataFrame(sel_rows).set_index("sleeve")
print("\n=== sleeve selection (pre-2016 only decides; test shown for information) ===")
print(sel.round(3))
md.append("### Blend sleeve selection (decided on pre-2016; test columns are out-of-sample)\n\n" + fmt_md(sel.assign(selected=sel.selected.astype(str)), 3) + "\n")

blends = {}
for label, names, floor in [("blend_rp_all", CANDIDATES, 0.04), ("blend_rp_survivors", survivors, 0.04), ("blend_ew_survivors", survivors, 10.0)]:
    if not names:
        continue
    sl_df = pd.concat([spy_ex_full.rename("spy")] + [sleeve_ex_series(n).rename(n) for n in names], axis=1).loc[C11:].dropna()
    res, wts = A.risk_parity_blend(sl_df, P["rf"], vol_floor=floor, cost_bps=HEAD_COST)
    blends[label] = (res, wts)
    VARIANTS.append(dict(name=label, fam="6 blend", desc=f"{'inverse-vol' if floor < 1 else 'equal-weight'} blend of SPY + {names}", fn=None, kind="blend", headline=True, drift=False))
    RES[(label, HEAD_COST)] = res
brows = []
for label, (res, wts) in blends.items():
    for w in ["all", "pre2016", "test2016+"]:
        r = sl(res, w)
        if len(r) < 100:
            continue
        s = SPY.loc[r.index[0] : r.index[-1]]
        m = A.metrics(r)
        bs = A.block_bootstrap_sharpe(ex(r), ex(s), seed=3)
        brows.append(dict(variant=label, window=w, start=str(r.index[0].date()), sharpe=m["sharpe"], cagr=m["cagr"], ann_vol=m["ann_vol"], max_dd=m["max_dd"], worst_day=m["worst_day"],
                          turnover_pa=m["turnover_pa"], spy_sharpe=bs["sharpe_b"], diff=bs["diff"], lo=bs["diff_lo"], hi=bs["diff_hi"], p_le0=bs["p_diff_le0"]))
        # also against SVXY buy and hold
blend_tbl = pd.DataFrame(brows).set_index("variant")
blend_tbl.to_csv(OUT / "summary_blend.csv")
print("\n=== blends ===")
print(survivors)
print(blend_tbl.round(3))
md.append(f"### Blends (survivors selected on pre-2016: {survivors})\n\n" + fmt_md(blend_tbl, 3) + "\n")
for label, (res, wts) in blends.items():
    print(label, "mean weights", wts.mean().round(3).to_dict())

# ---- SVXY sleeves vs SVXY buy-and-hold, by SVXY leverage era, and without Feb 2018
SV_HEADS = ["ts_bin_cash", "ts_ramp_cash", "ts_schmitt_cash", "ts_bin_spy", "spike_buy", "derisk_highvol", "ts_vvix", "ts_skew", "svxy_either"]
ERAS = {"all(2011-10+)": (C11, None), "-1x era (to 2018-02-27)": (C11, pd.Timestamp("2018-02-27")), "-0.5x era (2018-03+)": (pd.Timestamp("2018-03-01"), None)}
era_rows = []
for nm in SV_HEADS + ["BH_SVXY"]:
    base = SVXY if nm == "BH_SVXY" else RES[(nm, HEAD_COST)]
    for en, (a, b) in ERAS.items():
        r = base.loc[a:b]
        if len(r) < 100:
            continue
        m = A.metrics(r)
        row = dict(variant=nm, era=en, sharpe=m["sharpe"], max_dd=m["max_dd"], worst_day=m["worst_day"], ann_vol=m["ann_vol"])
        if nm != "BH_SVXY":
            bs = A.block_bootstrap_sharpe(ex(r), ex(SVXY.loc[r.index[0] : r.index[-1]]), seed=4)
            row.update(svxy_sharpe=bs["sharpe_b"], diff_vs_svxy=bs["diff"], lo=bs["diff_lo"], hi=bs["diff_hi"])
        era_rows.append(row)
era_tbl = pd.DataFrame(era_rows).set_index("variant")
era_tbl.to_csv(OUT / "summary_svxy_eras.csv")
print("\n=== SVXY sleeves vs SVXY buy-and-hold by era ===")
print(era_tbl.round(3))
md.append("### SVXY sleeves vs SVXY buy-and-hold, by SVXY leverage era (net 2 bps)\n\n" + fmt_md(era_tbl, 3) + "\n")

gap = (P.index >= "2018-01-25") & (P.index <= "2018-03-31")
ro = []
for nm in SV_HEADS + ["BH_SVXY"]:
    base = SVXY if nm == "BH_SVXY" else RES[(nm, HEAD_COST)]
    r = base.loc[C11:]
    r2 = r[~((r.index >= "2018-01-25") & (r.index <= "2018-03-31"))]
    ro.append(dict(variant=nm, sharpe_full=A.metrics(r)["sharpe"], sharpe_ex_JanMar2018=A.metrics(r2)["sharpe"],
                   ret_on_2018_02_05_06=float((1 + r.loc["2018-02-05":"2018-02-06", "ret"]).prod() - 1)))
rob = pd.DataFrame(ro).set_index("variant")
print("\n=== robustness: dropping 2018-01-25..2018-03-31 ===")
print(rob.round(3))
md.append("### Dropping the Feb 2018 volmageddon window (2018-01-25 to 2018-03-31)\n\n" + fmt_md(rob, 3) + "\n")

# ---- trials and expected-max Sharpe
n_trials = len(VARIANTS) - len(blends)  # blends added above are trials too, counted below
n_all = len(VARIANTS)
n_test_obs = int(((P.index >= SPLIT)).sum())
e_max = A.expected_max_sharpe(n_all, n_test_obs)
print(f"\ntrials = {n_all} (strategy variants {n_trials} + blends {len(blends)}); expected max Sharpe of {n_all} zero-skill trials over the test window = {e_max:.2f}")
md.append(f"Trials: {n_all} variants evaluated in total ({n_trials} sleeves/portfolios + {len(blends)} blends), each at 3 cost levels. "
          f"Expected best Sharpe of {n_all} independent zero-skill trials over the 2016+ window (SR0): {e_max:.2f}.\n")

# ---- save per-variant CSVs
for v in VARIANTS:
    res = RES[(v["name"], HEAD_COST)]
    keep = ["exposure", "ret"] + [c for c in ["ret_gross", "cost", "rf", "turnover"] if c in res] + [c for c in res.columns if c.startswith("w_")]
    out = res[keep].copy().round(8)
    out.index.name = "date"
    out.to_csv(OUT / f"{v['name']}.csv")
pd.DataFrame([{k: v[k] for k in ("name", "fam", "desc", "kind", "headline")} for v in VARIANTS]).to_csv(OUT / "variants.csv", index=False)

# ---- equity plot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
show = ["ts_bin_cash", "ts_ramp_spy", "derisk_highvol", "ts_vvix", "vm_implied_m", "spy_lv5", "putwrite", "blend_rp_survivors"]
for nm in show:
    if (nm, HEAD_COST) not in RES:
        continue
    r = RES[(nm, HEAD_COST)].loc[C11:]["ret"]
    eq = (1 + r).cumprod()
    ax[0].plot(eq.index, eq / eq.iloc[0], lw=1, label=nm)
    ax[1].plot(eq.index, eq / eq.cummax() - 1, lw=0.8)
for nm, b in BENCH.items():
    r = b.loc[C11:]["ret"]
    eq = (1 + r).cumprod()
    ax[0].plot(eq.index, eq / eq.iloc[0], lw=2, color="k" if nm == "SPY" else "gray", ls="-" if nm == "SPY" else "--", label="BH " + nm)
    ax[1].plot(eq.index, eq / eq.cummax() - 1, lw=1.2, color="k" if nm == "SPY" else "gray", ls="-" if nm == "SPY" else "--")
ax[0].set_yscale("log")
ax[0].set_title("Net-of-2bp growth of $1 (log), from 2011-10")
ax[0].legend(ncol=3, fontsize=7)
ax[1].set_title("Drawdown")
fig.tight_layout()
fig.savefig(OUT / "equity.png", dpi=110)

(OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
print("\nwrote", OUT)
