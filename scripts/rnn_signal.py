"""Reproduce every number in reports/03_rnn_signal.md.

    python scripts/rnn_signal.py dev [--shard i/n]   # config search on data truncated at 2016-01-01
    python scripts/rnn_signal.py select              # freeze configs from dev results
    python scripts/rnn_signal.py final               # full walk-forward with the frozen configs
    python scripts/rnn_signal.py report              # tables + figures from the saved predictions

Stages are separate so the 2016+ period is never touched before `select` has frozen the
architecture (docs/EVAL_PROTOCOL.md rule 2).  Outputs go to results/rnn/.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from vrp_strategy import data as D
from vrp_strategy import rnn_signal as R

OUT = ROOT / "results" / "rnn"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = D.SPLIT
SEEDS = (0, 1, 2, 3, 4)

# Every RNN configuration tried (the whole search space; nothing is dropped from the count).
RNN_GRID = [
    R.RNNConfig("gru", 16, 20, 0.2, 1e-3),
    R.RNNConfig("gru", 16, 40, 0.2, 1e-3),
    R.RNNConfig("gru", 32, 20, 0.2, 1e-3),
    R.RNNConfig("gru", 32, 40, 0.2, 1e-3),
    R.RNNConfig("lstm", 16, 20, 0.2, 1e-3),
    R.RNNConfig("lstm", 16, 40, 0.2, 1e-3),
    R.RNNConfig("lstm", 32, 20, 0.2, 1e-3),
    R.RNNConfig("lstm", 32, 40, 0.2, 1e-3),
    R.RNNConfig("gru", 16, 20, 0.2, 1e-2),
    R.RNNConfig("gru", 32, 20, 0.4, 1e-2),
    R.RNNConfig("gru", 64, 20, 0.4, 1e-2),
    R.RNNConfig("lstm", 64, 40, 0.4, 1e-2),
]
HGB_GRID = [dict(max_depth=2, max_iter=100), dict(max_depth=3, max_iter=150)]


def datasets():
    raw = R.load_raw_table()
    feats = R.build_features(raw)
    pay = R.variance_payoff(raw["vix"], raw["spx"])
    ds21 = R.make_dataset(feats, pay["scaled"], R.H21)
    svxy = D.load_raw("svxy").reindex(raw.index).ffill(limit=3)
    y5 = R.forward_return(svxy, R.H5)
    ds5 = R.make_dataset(feats, y5, R.H5)
    return raw, pay, ds21, ds5, y5


def dev_metrics(ds, pred, sig):
    lab = ds.y_reg[ds.index.get_indexer(pred.dates)]
    m = np.isfinite(lab)
    out = {"n_dates": int(m.sum()),
           "ic": R.spearman(sig[m], lab[m]),
           "hit": R.hit_rate(sig[m], lab[m])}
    return out


def stage_dev(shard):
    torch.set_num_threads(2)
    _, _, ds21, _, _ = datasets()
    dev = R.truncate(ds21, SPLIT)
    i, n = shard
    rows = []
    jobs = [("rnn", c, None) for c in RNN_GRID] + [("hgb", None, kw) for kw in HGB_GRID] + [("ridge", None, None), ("har", None, None)]
    for k, (kind, cfg, kw) in enumerate(jobs):
        if k % n != i:
            continue
        tag = cfg.tag() if cfg else (str(kw) if kw else kind)
        print("dev", kind, tag, flush=True)
        p = R.walk_forward(dev, kind, cfg=cfg, hgb_kw=kw, seeds=SEEDS)
        row = {"kind": kind, "config": tag, "job": k}
        if p.prob is not None:
            row.update({f"cls_{a}": b for a, b in dev_metrics(dev, p, R.signal_from_prob(p.mean_prob())).items()})
        row.update({f"reg_{a}": b for a, b in dev_metrics(dev, p, R.signal_from_reg(p.mean_reg(), p.scale)).items()})
        if p.prob is not None:
            lab = dev.y_cls[dev.index.get_indexer(p.dates)]
            m = np.isfinite(lab)
            row["brier_skill"] = R.brier_skill(p.mean_prob()[m], lab[m])
            # seed dispersion on dev
            ics = [R.spearman(R.signal_from_prob(q)[m], dev.y_reg[dev.index.get_indexer(p.dates)][m]) for q in p.prob]
            row["ic_seed_mean"], row["ic_seed_sd"] = float(np.mean(ics)), float(np.std(ics))
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / f"dev_trials_shard{i}.csv", index=False)


def stage_select():
    files = sorted(OUT.glob("dev_trials_shard*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files]).sort_values("job").reset_index(drop=True)
    df.to_csv(OUT / "dev_trials.csv", index=False)
    rnn = df[df.kind == "rnn"]
    hgb = df[df.kind == "hgb"]
    best_rnn = RNN_GRID[int(rnn.loc[rnn.cls_ic.idxmax(), "job"])]
    best_hgb = HGB_GRID[int(hgb.loc[hgb.cls_ic.idxmax(), "job"]) - len(RNN_GRID)]
    frozen = {"rnn": best_rnn.__dict__, "hgb": best_hgb, "n_trials_dev": int(len(df)),
              "criterion": "mean Spearman IC of (2P-1) vs scaled 21d payoff, walk-forward dates < 2016 with realised labels"}
    (OUT / "frozen_config.json").write_text(json.dumps(frozen, indent=2))
    print(df.to_string())
    print(json.dumps(frozen, indent=2))


def stage_final():
    torch.set_num_threads(4)
    frozen = json.loads((OUT / "frozen_config.json").read_text())
    cfg = R.RNNConfig(**frozen["rnn"])
    hgb_kw = frozen["hgb"]
    raw, pay, ds21, ds5, y5 = datasets()
    store = {}
    for name, kind in [("rnn", "rnn"), ("ridge", "ridge"), ("har", "har"), ("hgb", "hgb")]:
        print("final 21d", name, flush=True)
        p = R.walk_forward(ds21, kind, cfg=cfg, hgb_kw=hgb_kw, seeds=SEEDS, verbose=False)
        store[f"d21_{name}_dates"] = p.dates.values.astype("datetime64[ns]").astype("int64")
        store[f"d21_{name}_reg"] = p.reg
        store[f"d21_{name}_scale"] = p.scale
        if p.prob is not None:
            store[f"d21_{name}_prob"] = p.prob
    # 5-day SVXY target: labelled only from 2011-10; same frozen architecture (no separate tuning)
    first_lab = int(np.argmax(np.isfinite(ds5.y_reg)))
    min_row5 = max(R.MIN_SAMPLE_ROW, first_lab)
    for name, kind in [("rnn", "rnn"), ("ridge", "ridge"), ("har", "har"), ("hgb", "hgb")]:
        print("final 5d", name, flush=True)
        p = R.walk_forward(ds5, kind, cfg=cfg, hgb_kw=hgb_kw, seeds=SEEDS, first_t0=min_row5 + 600, min_row=min_row5)
        store[f"d5_{name}_dates"] = p.dates.values.astype("datetime64[ns]").astype("int64")
        store[f"d5_{name}_reg"] = p.reg
        store[f"d5_{name}_scale"] = p.scale
        if p.prob is not None:
            store[f"d5_{name}_prob"] = p.prob
    np.savez_compressed(OUT / "preds.npz", **store)
    print("saved preds.npz")


# ----------------------------------------------------------------------------- report
BLOCK, REPS = 42, 1000
MODELS = ["rnn", "rnn_reg", "ridge", "har", "hgb", "hgb_reg", "legacy_vrp_rank", "always_short"]
LABELS = {"rnn": "GRU/LSTM ensemble (cls head)", "rnn_reg": "GRU/LSTM ensemble (reg head)", "ridge": "Ridge",
          "har": "HAR-RV OLS", "hgb": "HistGB (cls)", "hgb_reg": "HistGB (reg)",
          "legacy_vrp_rank": "Legacy VRP-rank rule", "always_short": "Always short"}
EVENTS = {"2018-02 Volmageddon": ("2018-01-02", "2018-03-29"),
          "2020-03 COVID": ("2020-02-03", "2020-04-30"),
          "2022 bear market": ("2022-01-03", "2022-12-30")}


def _dates(z, key):
    return pd.DatetimeIndex(pd.to_datetime(z[key]))


def load_signal_sets(prefix, z, vrp):
    """-> dict model -> (signal Series, raw Series), plus member-level arrays for the RNN."""
    sigs, raws, members = {}, {}, {}
    for m in ("rnn", "ridge", "har", "hgb"):
        idx = _dates(z, f"{prefix}_{m}_dates")
        reg = z[f"{prefix}_{m}_reg"]
        scale = z[f"{prefix}_{m}_scale"]
        raw = pd.Series(reg.mean(0), idx)
        if f"{prefix}_{m}_prob" in z:
            prob = z[f"{prefix}_{m}_prob"]
            sigs[m] = pd.Series(R.signal_from_prob(prob.mean(0)), idx)
            sigs[m + "_reg"] = pd.Series(R.signal_from_reg(reg.mean(0), scale), idx)
            raws[m + "_reg"] = raw
            if m == "rnn":
                members = {"prob": pd.DataFrame(prob.T, idx), "reg": pd.DataFrame(reg.T, idx), "scale": pd.Series(scale, idx)}
        else:
            sigs[m] = pd.Series(R.signal_from_reg(reg.mean(0), scale), idx)
        raws[m] = raw
    idx = sigs["rnn"].index
    leg = R.legacy_rank_signal(vrp)
    sigs["legacy_vrp_rank"] = leg.reindex(idx)
    raws["legacy_vrp_rank"] = ((leg + 1) / 2).reindex(idx)
    sigs["always_short"] = pd.Series(1.0, idx)
    raws["always_short"] = pd.Series(1.0, idx)
    keep = sigs["legacy_vrp_rank"].dropna().index
    for k in sigs:
        sigs[k], raws[k] = sigs[k].reindex(keep), raws[k].reindex(keep)
    return sigs, raws, members


def sharpe_np(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan


def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frozen = json.loads((OUT / "frozen_config.json").read_text())
    z = np.load(OUT / "preds.npz")
    raw, pay, ds21, ds5, y5 = datasets()
    sig_tab = D.load_signals()
    S21, RAW21, mem = load_signal_sets("d21", z, sig_tab["vrp"])
    S5, RAW5, _ = load_signal_sets("d5", z, sig_tab["vrp"])
    scaled, payoff_vp = pay["scaled"], pay["payoff"] * 1e4
    svxy_ret = D.asset_returns("svxy")
    spy_ret = D.asset_returns("spy")
    test0 = pd.Timestamp(SPLIT)
    results = {}

    # ---- save signals ---------------------------------------------------------
    rows = []
    for m in MODELS:
        d = pd.DataFrame({"model": m, "date": S21[m].index, "signal": S21[m].values, "raw_forecast": RAW21[m].values})
        d["is_test_2016plus"] = d["date"] >= test0
        rows.append(d)
    pd.concat(rows).to_csv(OUT / "signal.csv", index=False)
    rows = []
    for m in MODELS:
        d = pd.DataFrame({"model": m, "date": S5[m].index, "signal": S5[m].values, "raw_forecast": RAW5[m].values})
        d["is_test_2016plus"] = d["date"] >= test0
        rows.append(d)
    pd.concat(rows).to_csv(OUT / "signal_svxy5d.csv", index=False)

    # ---- A. forecasting quality, 21d payoff ------------------------------------
    def forecast_table(S, target, label_sign, period):
        out = []
        for m in MODELS:
            s = S[m]
            idx = s.index
            lo, hi = period
            idx = idx[(idx >= lo) & (idx <= hi)]
            y = target.reindex(idx)
            ok = y.notna()
            sv, yv = s.reindex(idx)[ok].to_numpy(), y[ok].to_numpy()
            row = {"model": LABELS[m], "n": int(ok.sum())}
            if m == "always_short":
                row.update({"ic": np.nan, "ic_lo": np.nan, "ic_hi": np.nan})
            else:
                e, l, h_, _ = R.block_bootstrap(R.spearman, [sv, yv], BLOCK, REPS, seed=1)
                row.update({"ic": e, "ic_lo": l, "ic_hi": h_})
            e, l, h_, _ = R.block_bootstrap(lambda a, b: R.hit_rate(a, b), [sv, yv], BLOCK, REPS, seed=2)
            row.update({"hit": e, "hit_lo": l, "hit_hi": h_})
            out.append(row)
        base = float(np.mean(np.sign(1.0) == np.sign(target.reindex(idx).dropna())))
        return pd.DataFrame(out), base

    test_p = (test0, pd.Timestamp("2100-01-01"))
    # dev window = dates whose 21d label was fully realised before SPLIT (same mask as model selection)
    dev_p = (pd.Timestamp("1990-01-01"), ds21.index[ds21.index < test0][-(R.H21 + 1)])
    fa_test, base_test = forecast_table(S21, scaled, True, test_p)
    fa_dev, base_dev = forecast_table(S21, scaled, True, dev_p)
    fa_test.to_csv(OUT / "forecast_21d_test.csv", index=False)
    fa_dev.to_csv(OUT / "forecast_21d_dev.csv", index=False)
    results["base_rate_test"], results["base_rate_dev"] = base_test, base_dev
    print("\n21d payoff forecast quality, TEST 2016+ (always-short hit rate = %.3f)" % base_test)
    print(fa_test.round(3).to_string())
    print("\nsame, DEV 2011-05..2015 (always-short hit rate = %.3f)" % base_dev)
    print(fa_dev.round(3).to_string())

    # paired IC differences, test period
    cmp_rows = []
    idx = S21["rnn"].index[S21["rnn"].index >= test0]
    y = scaled.reindex(idx)
    ok = y.notna().to_numpy()
    for a, b in [("rnn", "ridge"), ("rnn", "har"), ("rnn", "hgb"), ("rnn", "legacy_vrp_rank"), ("rnn_reg", "ridge"), ("hgb", "ridge"), ("ridge", "har")]:
        sa, sb = S21[a].reindex(idx).to_numpy()[ok], S21[b].reindex(idx).to_numpy()[ok]
        f = lambda x, w, t: R.spearman(x, t) - R.spearman(w, t)
        e, l, h_, _ = R.block_bootstrap(f, [sa, sb, y.to_numpy()[ok]], BLOCK, REPS, seed=3)
        cmp_rows.append({"a": a, "b": b, "ic_diff": e, "lo": l, "hi": h_})
    ic_diff = pd.DataFrame(cmp_rows)
    ic_diff.to_csv(OUT / "ic_differences_test.csv", index=False)
    print("\nIC differences (test)\n", ic_diff.round(3).to_string())

    # calibration of probability heads
    cal = {}
    pr = {"rnn": pd.Series(mem["prob"].mean(1))}
    pr["hgb"] = (S21["hgb"] + 1) / 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m, c in (("rnn", "C0"), ("hgb", "C1")):
        p = ((S21[m] + 1) / 2)[lambda s: s.index >= test0]
        yy = (scaled.reindex(p.index) > 0).astype(float)
        ok2 = scaled.reindex(p.index).notna()
        t = R.calibration_table(p[ok2].to_numpy(), yy[ok2].to_numpy(), 10)
        t.to_csv(OUT / f"calibration_{m}_test.csv", index=False)
        cal[m] = {"brier_skill": R.brier_skill(p[ok2].to_numpy(), yy[ok2].to_numpy()), "table": t}
        ax[0].plot(t.mean_pred, t.freq, "o-", color=c, label=f"{LABELS[m]}  BSS={cal[m]['brier_skill']:.3f}")
        ax[1].hist(p, bins=30, alpha=.5, color=c, label=m)
    ax[0].plot([0, 1], [0, 1], "k:")
    ax[0].set(xlabel="predicted P(payoff>0)", ylabel="realised frequency", title="Calibration, 2016+ (deciles)")
    ax[0].legend(fontsize=8)
    ax[1].set(title="Predicted probability distribution")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=120); plt.close(fig)
    print("\nBrier skill (test):", {k: round(v["brier_skill"], 4) for k, v in cal.items()})
    print(cal["rnn"]["table"].round(3).to_string())

    # ---- seed stability (RNN) --------------------------------------------------
    pidx = mem["prob"].index
    tmask = pidx >= test0
    yv = scaled.reindex(pidx)
    okm = (yv.notna() & tmask).to_numpy()
    seed_ic = [R.spearman((2 * mem["prob"][k].to_numpy() - 1)[okm], yv.to_numpy()[okm]) for k in mem["prob"].columns]
    seed_sh = []
    for k in mem["prob"].columns:
        e = R.exposure_from_signal(pd.Series(2 * mem["prob"][k].to_numpy() - 1, pidx))
        seed_sh.append(R.econ_summary(R.exposure_pnl(e, svxy_ret, 2.0, start=SPLIT))["sharpe_net"])
    corr = np.corrcoef(mem["prob"].to_numpy()[okm].T)
    seed = {"member_ic": seed_ic, "member_sharpe_net": seed_sh,
            "ensemble_ic": float(fa_test.loc[fa_test.model == LABELS["rnn"], "ic"].iloc[0]),
            "mean_pairwise_corr": float((corr.sum() - len(corr)) / (len(corr) ** 2 - len(corr)))}
    print("\nseed stability", seed)

    # ---- B. economics on real SVXY ----------------------------------------------
    def econ_table(S, ret, mode="linear", label="", extra_bh=True):
        rows, pnls = [], {}
        for m in MODELS:
            e = R.exposure_from_signal(S[m], mode)
            if m == "legacy_vrp_rank" and mode != "linear":
                e = R.exposure_from_signal(S[m], mode)
            p = R.exposure_pnl(e, ret, 2.0, start=SPLIT)
            pnls[m] = p
            rows.append({"strategy": LABELS[m], **R.econ_summary(p)})
        return pd.DataFrame(rows), pnls

    econ_lin, pn_lin = econ_table(S21, svxy_ret, "linear")
    econ_lf, pn_lf = econ_table(S21, svxy_ret, "longflat")
    econ_sg, pn_sg = econ_table(S21, svxy_ret, "sign")
    last = S21["rnn"].index[-1]
    # buy and hold SPY over the same dates
    spy_pos = pd.Series(1.0, S21["rnn"].index)
    bh_spy = R.econ_summary(R.exposure_pnl(spy_pos, spy_ret, 0.0, start=SPLIT))
    bh_spy_row = pd.DataFrame([{"strategy": "Buy & hold SPY", **bh_spy}])
    econ_lin = pd.concat([econ_lin, bh_spy_row])
    for nm, t in (("linear", econ_lin), ("longflat", econ_lf), ("sign", econ_sg)):
        t.to_csv(OUT / f"econ_svxy_21d_{nm}.csv", index=False)
    print("\nECON SVXY (21d-target signals), exposure=(s+1)/2, 2bps, 2016+")
    print(econ_lin.round(3).to_string())
    print("\nECON longflat\n", econ_lf.round(3).to_string())
    print("\nECON sign\n", econ_sg.round(3).to_string())

    # constant exposure matched to each model's average exposure
    cm = []
    for m in MODELS:
        c = float(pn_lin[m]["exposure"].mean())
        p = R.exposure_pnl(pd.Series(c, S21["rnn"].index), svxy_ret, 2.0, start=SPLIT)
        cm.append({"strategy": LABELS[m], "avg_exposure": c, **{k: v for k, v in R.econ_summary(p).items() if k in ("sharpe_net", "max_drawdown", "worst_day")}})
    const_match = pd.DataFrame(cm)
    const_match.to_csv(OUT / "econ_const_exposure_matched.csv", index=False)
    print("\nconstant exposure with same avg exposure\n", const_match.round(3).to_string())

    # paired Sharpe-difference bootstrap (net daily excess returns)
    sd_rows = []
    pairs = [("rnn", "always_short"), ("rnn", "ridge"), ("rnn", "har"), ("rnn", "hgb"), ("rnn", "legacy_vrp_rank"),
             ("ridge", "always_short"), ("har", "always_short"), ("hgb", "always_short"), ("legacy_vrp_rank", "always_short")]
    for a, b in pairs:
        ra, rb = pn_lin[a]["net_excess"], pn_lin[b]["net_excess"]
        ix = ra.index.intersection(rb.index)
        e, l, h_, _ = R.block_bootstrap(lambda x, w: sharpe_np(x) - sharpe_np(w), [ra[ix].to_numpy(), rb[ix].to_numpy()], BLOCK, REPS, seed=4)
        sd_rows.append({"a": a, "b": b, "sharpe_diff": e, "lo": l, "hi": h_})
    sharpe_diff = pd.DataFrame(sd_rows)
    sharpe_diff.to_csv(OUT / "sharpe_differences_test.csv", index=False)
    print("\nPaired Sharpe differences (net, linear exposure)\n", sharpe_diff.round(3).to_string())

    # same paired test for the other exposure maps, and how often each signal is bearish at all
    sd2 = []
    for nm, pn in (("longflat", pn_lf), ("sign", pn_sg)):
        for m in ("rnn", "rnn_reg", "ridge", "har", "hgb", "hgb_reg", "legacy_vrp_rank"):
            ra, rb = pn[m]["net_excess"], pn["always_short"]["net_excess"]
            ix = ra.index.intersection(rb.index)
            e, l, h_, _ = R.block_bootstrap(lambda x, w: sharpe_np(x) - sharpe_np(w), [ra[ix].to_numpy(), rb[ix].to_numpy()], BLOCK, REPS, seed=7)
            sd2.append({"map": nm, "model": m, "sharpe_diff_vs_always_short": e, "lo": l, "hi": h_})
    pd.DataFrame(sd2).to_csv(OUT / "sharpe_differences_other_maps_test.csv", index=False)
    print("\nPaired Sharpe diff vs always-short, other exposure maps\n", pd.DataFrame(sd2).round(3).to_string())
    bear = pd.DataFrame({LABELS[m]: {"pct_days_signal_below_0_test": float((S21[m][S21[m].index >= test0] < 0).mean()),
                                     "pct_days_signal_below_0_dev": float((S21[m][S21[m].index < test0] < 0).mean()),
                                     "signal_mean_test": float(S21[m][S21[m].index >= test0].mean()),
                                     "signal_sd_test": float(S21[m][S21[m].index >= test0].std())} for m in MODELS}).T
    bear.to_csv(OUT / "signal_summary.csv")
    print("\nSignal summary\n", bear.round(3).to_string())

    # Sharpe CI for each strategy
    ci = []
    for m in MODELS:
        e, l, h_, _ = R.block_bootstrap(sharpe_np, [pn_lin[m]["net_excess"].to_numpy()], BLOCK, REPS, seed=5)
        ci.append({"strategy": LABELS[m], "sharpe_net": e, "lo": l, "hi": h_})
    sharpe_ci = pd.DataFrame(ci)
    sharpe_ci.to_csv(OUT / "sharpe_ci_test.csv", index=False)
    print("\nSharpe CIs\n", sharpe_ci.round(3).to_string())

    # ---- C. held-to-maturity payoff economics (variance-swap view) ----------------
    vr = []
    for m in MODELS:
        s = S21[m]
        s = s[s.index >= test0]
        pv = payoff_vp.reindex(s.index)
        ok3 = pv.notna().to_numpy()
        e = ((s + 1) / 2).to_numpy()[ok3]
        p = pv.to_numpy()[ok3]
        cov_stat = lambda ee, pp: float(np.mean((ee - ee.mean()) * (pp - pp.mean())))
        c, l, h_, _ = R.block_bootstrap(cov_stat, [e, p], BLOCK, REPS, seed=6)
        vr.append({"strategy": LABELS[m], "avg_exposure": e.mean(), "mean_payoff_always_short": p.mean(),
                   "mean_payoff_weighted": float(np.mean(e * p)), "ratio_vs_always_short": float(np.mean(e * p) / p.mean()),
                   "worst_payoff_weighted": float(np.min(e * p)), "timing_cov": c, "timing_cov_lo": l, "timing_cov_hi": h_,
                   "tranche_sharpe": float(np.mean(e * p) / np.std(e * p) * np.sqrt(12))})
    var_econ = pd.DataFrame(vr)
    var_econ.to_csv(OUT / "variance_payoff_economics_test.csv", index=False)
    print("\nVariance-swap (held-to-maturity) economics, 2016+, payoff in vol-pt^2\n", var_econ.round(3).to_string())

    # ---- D. 5-day SVXY target -----------------------------------------------------
    fa5, base5 = forecast_table(S5, y5, True, test_p)
    fa5.to_csv(OUT / "forecast_5d_test.csv", index=False)
    print("\n5d SVXY forward return sign (always-long hit rate %.3f)" % base5)
    print(fa5.round(3).to_string())
    econ5, pn5 = econ_table(S5, svxy_ret, "linear")
    econ5.to_csv(OUT / "econ_svxy_5d_linear.csv", index=False)
    print(econ5.round(3).to_string())

    # ---- E. events -------------------------------------------------------------------
    ev_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex="col", gridspec_kw={"height_ratios": [1, 1.3]})
    for j, (name, (a, b)) in enumerate(EVENTS.items()):
        w = svxy_ret[a:b]
        px = (1 + w).cumprod()
        axes[0, j].plot(px.index, px / px.iloc[0], color="k")
        axes[0, j].set_title(name)
        axes[0, j].set_ylabel("SVXY (start=1)")
        for m, c in (("rnn", "C0"), ("rnn_reg", "C9"), ("ridge", "C1"), ("har", "C2"), ("hgb", "C3"), ("legacy_vrp_rank", "C4")):
            axes[1, j].plot(S21[m][a:b].index, S21[m][a:b], color=c, label=LABELS[m], lw=1.2)
        axes[1, j].axhline(0, color="gray", lw=.5)
        axes[1, j].set_ylabel("signal")
        for m in MODELS:
            e = R.exposure_from_signal(S21[m], "linear")
            p = R.exposure_pnl(e, svxy_ret, 2.0, start=a, end=b)
            pre = S21[m][:a].iloc[-1]
            ev_rows.append({"event": name, "model": LABELS[m], "signal_before": pre, "mean_exposure": p["exposure"].mean(),
                            "min_exposure": p["exposure"].min(), "strategy_return": (1 + p["net_total"]).prod() - 1,
                            "svxy_return": (1 + w).prod() - 1})
    axes[1, 0].legend(fontsize=7, ncol=2)
    for ax in axes[1]:
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.tight_layout(); fig.savefig(OUT / "events.png", dpi=120); plt.close(fig)
    events = pd.DataFrame(ev_rows)
    events.to_csv(OUT / "events.csv", index=False)
    print("\nEvents\n", events.round(3).to_string())

    # ---- equity curves ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for m, c in (("rnn", "C0"), ("ridge", "C1"), ("har", "C2"), ("hgb", "C3"), ("legacy_vrp_rank", "C4"), ("always_short", "k")):
        cum = (1 + pn_lin[m]["net_total"]).cumprod()
        ax[0].plot(cum.index, cum, label=LABELS[m], color=c, lw=1.2)
        ax[1].plot(S21[m][S21[m].index >= test0].rolling(21).mean(), color=c, lw=1, label=LABELS[m])
    ax[0].set_yscale("log"); ax[0].set(title="Net equity, SVXY exposure=(s+1)/2, 2bps, 2016+"); ax[0].legend(fontsize=7)
    ax[1].set(title="Signal (21d mean)")
    fig.tight_layout(); fig.savefig(OUT / "equity_signals.png", dpi=120); plt.close(fig)

    # ---- trial count + metadata ----------------------------------------------------------
    meta = {"frozen": frozen, "seed": seed, "brier_skill": {k: v["brier_skill"] for k, v in cal.items()},
            "base_rate_test": base_test, "base_rate_dev": base_dev, "base5": base5,
            "first_pred": str(S21["rnn"].index[0].date()), "last_pred": str(S21["rnn"].index[-1].date()),
            "n_test_dates": int((S21["rnn"].index >= test0).sum())}
    (OUT / "report_meta.json").write_text(json.dumps(meta, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "select", "final", "report"])
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.stage == "dev":
        i, n = map(int, a.shard.split("/"))
        stage_dev((i, n))
    elif a.stage == "select":
        stage_select()
    elif a.stage == "final":
        stage_final()
    else:
        stage_report()
