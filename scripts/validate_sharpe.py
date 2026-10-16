"""Run the statistical-validation battery on the project's strategies.

    python scripts/validate_sharpe.py            # full run (~2-4 minutes)
    python scripts/validate_sharpe.py --quick    # fewer resamples, for smoke tests

Writes results/validation/*.csv and *.png.  Every number in
reports/04_statistical_validation.md comes from here.

Strategies (all 2016-01 -> 2026, i.e. data.SPLIT onward, same rebuild as
scripts/audit_sharpe.py):
  original_viterbi   3.08  non-causal regime labels, proxy P&L, vol-targeted
  causal_hard        2.74  forward-filtered regime, hard argmax
  causal_soft        2.81  forward-filtered regime, soft multiplier
  constant_short     2.89  short variance every day, no signal
  svxy_causal_soft         the same causal positions on REAL SVXY returns, unlevered, excess of rf
  svxy_buy_hold            SVXY long-only (the instrument itself), excess of rf
"""
import argparse
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

from vrp_strategy import stats_tests as st
from vrp_strategy.backtest import raw_strategy_returns, vol_target_scale
from vrp_strategy.data import ROOT, SPLIT, asset_returns, excess, load_signals
from vrp_strategy.regimes import filtered_proba, state_order_by_feature

OUT = ROOT / "results" / "validation"
FEATURES = ["vrp_vol", "slope", "rv_30"]
MULT_BY_RANK = np.array([1.0, 0.25, 0.0])
STRESSED_GRID = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
CRISIS_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# --------------------------------------------------------------------------- #
# Honest trial count (docs/EVAL_PROTOCOL.md rule 3).  Sources: `git log` messages
# and notebooks 03/04/07/09.  Everything that produced a backtest Sharpe that was
# looked at and could have influenced what was kept is counted, including failures.
# --------------------------------------------------------------------------- #
TRIALS = [
    ("Regime-multiplier grid, 12 stressed x 11 crisis values (nb 09, commit cb087c6);"
     " contains the hand-tuned 0.5 -> 0.25 change of commit 8e4a981", 132),
    ("HMM specification revisions: initial features, train/test split, then 'trained on vrp_vol instead'"
     " (commits 234b66b, 1144de8, 87d32aa)", 3),
    ("Vol-target scalar cap 150 -> 100 (commit 2a20427)", 2),
    ("Return definition revision: unit exposure vs % capital with 10% vol target (commit 2ae3e20)", 2),
    ("Half-Kelly sizing comparison (nb 07, commit 30852fc)", 1),
    ("Audit variants: causal hard, causal soft, no regime gate, constant short, real-SVXY positions (scripts/audit_sharpe.py)", 5),
]
N_LOW = 132         # the grid alone (lower bound: it is the only fully-logged sweep)
N_CENTRAL = 150     # 145 logged variants, rounded up for un-logged notebook re-runs
N_HIGH = 500        # ~3.5x: if for each logged variant two more manual tries went un-logged


def build_streams():
    sig = load_signals()
    train = sig[sig.index < SPLIT]
    test = sig[sig.index >= SPLIT]
    scaler = StandardScaler().fit(train[FEATURES])
    X_train, X_test = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])
    model = hmm.GaussianHMM(3, covariance_type="full", n_iter=1000, random_state=42).fit(X_train)
    order = list(state_order_by_feature(model, scaler, FEATURES.index("rv_30")))
    state_rank = np.array([order.index(s) for s in range(3)])

    viterbi = np.concatenate([model.predict(X_train), model.predict(X_test)])
    p_all = np.vstack([filtered_proba(model, X_train), filtered_proba(model, X_test)])
    rank_vit = pd.Series(state_rank[viterbi], index=sig.index)
    rank_hard = pd.Series(state_rank[p_all.argmax(1)], index=sig.index)
    soft = pd.Series(p_all @ MULT_BY_RANK[state_rank], index=sig.index)   # P(state) x multiplier
    vit_mult = pd.Series(MULT_BY_RANK[rank_vit.to_numpy()], index=sig.index)
    hard_mult = pd.Series(MULT_BY_RANK[rank_hard.to_numpy()], index=sig.index)
    vrp_rank = sig["vrp"].rolling(252).rank(pct=True)
    return sig, vrp_rank, rank_vit, rank_hard, vit_mult, hard_mult, soft


def proxy_score_returns(sig, position):
    """Vol-targeted proxy P&L over the OOS period for a full-sample position series
    (identical to scripts/audit_sharpe.py / notebook 06)."""
    d = sig.assign(position=position)
    d = d[d.index >= SPLIT].dropna(subset=["position"])
    r = raw_strategy_returns(d["position"], d["iv_daily"], d["rv_daily"]).dropna()
    return vol_target_scale(r, 0.10, 126, 100.0)[0].dropna()


def main(quick=False):
    n_boot = 500 if quick else 2000
    n_sims = 150 if quick else 1000
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    sig, vrp_rank, rank_vit, rank_hard, vit_mult, hard_mult, soft = build_streams()
    pos = {
        "original_viterbi": (vrp_rank * vit_mult).clip(0, 1),
        "causal_hard": (vrp_rank * hard_mult).clip(0, 1),
        "causal_soft": (vrp_rank * soft).clip(0, 1),
    }
    streams = {k: proxy_score_returns(sig, p) for k, p in pos.items()}
    streams["constant_short"] = proxy_score_returns(sig, pd.Series(1.0, index=sig.index))

    svxy = asset_returns("svxy")
    svxy_ex = excess(svxy)
    lag_pos = pos["causal_soft"].shift(1)
    sv = (lag_pos * svxy_ex).dropna()
    streams["svxy_causal_soft"] = sv[sv.index >= SPLIT]
    bh = svxy_ex[svxy_ex.index >= SPLIT]
    streams["svxy_buy_hold"] = bh

    # ------------------------------------------------------------------ trials
    trials_df = pd.DataFrame(TRIALS, columns=["source", "n_variants"])
    trials_df.loc[len(trials_df)] = ["TOTAL logged", trials_df["n_variants"].sum()]
    trials_df.to_csv(OUT / "trial_count.csv", index=False)
    print(trials_df.to_string(index=False))
    print(f"\nN used: low={N_LOW} (grid only), central={N_CENTRAL}, high={N_HIGH}\n")

    # ------------------------------------------------------------------ 132-point grid (needed for V and PBO/SPA)
    grid_ret_vit, grid_ret_causal = {}, {}
    for sm in STRESSED_GRID:
        for cm in CRISIS_GRID:
            mults = np.array([1.0, sm, cm])
            for store, rk in ((grid_ret_vit, rank_vit), (grid_ret_causal, rank_hard)):
                m = pd.Series(mults[rk.to_numpy()], index=sig.index)
                store[(sm, cm)] = proxy_score_returns(sig, (vrp_rank * m).clip(0, 1))
    G_vit = pd.DataFrame({f"s{a:.2f}_c{b:.1f}": v for (a, b), v in grid_ret_vit.items()}).dropna()
    G_cau = pd.DataFrame({f"s{a:.2f}_c{b:.1f}": v for (a, b), v in grid_ret_causal.items()}).dropna()
    sr_grid_vit = G_vit.mean() / G_vit.std() * np.sqrt(252)
    sr_grid_cau = G_cau.mean() / G_cau.std() * np.sqrt(252)
    print(f"Grid (Viterbi labels): best SR {sr_grid_vit.max():.2f} at {sr_grid_vit.idxmax()}, "
          f"median {sr_grid_vit.median():.2f}, cross-trial SD {sr_grid_vit.std():.2f}")
    print(f"Grid (causal labels):  best SR {sr_grid_cau.max():.2f} at {sr_grid_cau.idxmax()}, "
          f"median {sr_grid_cau.median():.2f}, cross-trial SD {sr_grid_cau.std():.2f}")
    print(f"Mean pairwise correlation inside the Viterbi grid: {G_vit.corr().to_numpy()[np.triu_indices(G_vit.shape[1], 1)].mean():.3f}")

    # ------------------------------------------------------------------ main validation table
    bench = {k: streams["constant_short"] for k in ("original_viterbi", "causal_hard", "causal_soft")}
    bench["svxy_causal_soft"] = streams["svxy_buy_hold"]
    tab = st.validate_table(streams, n_trials=N_CENTRAL, benchmark=bench, n_boot=n_boot, seed=1)
    tab.insert(0, "label", [{"original_viterbi": "Original (Viterbi, headline 3.08)",
                             "causal_hard": "Causal, hard argmax",
                             "causal_soft": "Causal, soft multiplier",
                             "constant_short": "Constant short, no signal",
                             "svxy_causal_soft": "Causal positions x real SVXY",
                             "svxy_buy_hold": "SVXY buy and hold"}[k] for k in tab.index])
    tab.to_csv(OUT / "validation_table.csv")
    show = ["sharpe", "sharpe_lo", "ci_low", "ci_high", "se_mertens", "se_hac", "psr0", "dsr", "sr0_hurdle",
            "min_trl_years", "skew", "excess_kurtosis", "rho1", "verdict"]
    print("\n=== validation, N =", N_CENTRAL, "===")
    print(tab[show].round(3).to_string())
    print("\nPaired Sharpe difference vs benchmark:")
    print(tab[["bench_sharpe", "sharpe_diff", "diff_ci_low", "diff_ci_high", "diff_p_value"]].dropna().round(3).to_string())

    # ------------------------------------------------------------------ DSR sensitivity to N and to V
    rows = []
    for name in ("original_viterbi", "causal_soft", "svxy_causal_soft", "svxy_buy_hold"):
        r = streams[name]
        for n in (1, 10, 50, N_LOW, N_CENTRAL, N_HIGH, 2000, 10000):
            d0 = st.deflated_sharpe_ratio(r, n, return_details=True)
            ds = st.deflated_sharpe_ratio(r, n, trial_sharpes=sr_grid_vit, return_details=True)
            rows.append({"strategy": name, "n_trials": n, "sr0_default_v": d0["sr0_annual"], "dsr_default_v": d0["dsr"],
                         "sr0_grid_v": ds["sr0_annual"], "dsr_grid_v": ds["dsr"]})
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "dsr_sensitivity.csv", index=False)
    print("\n=== DSR sensitivity ===")
    print(sens.round(3).to_string(index=False))

    # ------------------------------------------------------------------ timing nulls
    def proxy_score(position):
        d = sig.assign(position=position)
        d = d[d.index >= SPLIT]
        raw = raw_strategy_returns(d["position"], d["iv_daily"], d["rv_daily"])
        return st.sharpe_ratio(vol_target_scale(raw, 0.10, 126, 100.0)[0].dropna())

    svxy_oos = svxy_ex.reindex(sig.index)

    def svxy_score(position):
        r = (position.shift(1) * svxy_oos).dropna()
        return st.sharpe_ratio(r[r.index >= SPLIT])

    def oos_pos(p):
        return p[p.index >= SPLIT].dropna()

    null_rows, null_draws = [], {}
    for name, scorer in (("original_viterbi", proxy_score), ("causal_soft", proxy_score), ("svxy_causal_soft", svxy_score)):
        p_full = pos["causal_soft" if name.startswith("svxy") else name]
        # null positions are drawn over the OOS block only; the scorer needs the full-sample frame, so
        # embed the shuffled OOS block back into the (unshuffled) pre-OOS history.
        p_oos = oos_pos(p_full)
        pre = p_full[p_full.index < SPLIT]

        def score_fn(p_block, _scorer=scorer, _pre=pre):
            return _scorer(pd.concat([_pre, p_block]))

        for kind, kw in (("iid_shuffle", {}), ("circular_shift", {}), ("block_permute", {"mean_block": 21.0}),
                         ("block_permute", {"mean_block": 63.0})):
            res = st.timing_null_test(p_oos, score_fn, kind, n_sims=n_sims, seed=7, **kw)
            label = kind + (f"_{int(kw['mean_block'])}d" if kw else "")
            null_draws[(name, label)] = res["null"]
            null_rows.append({"strategy": name, "null": label, "actual_sharpe": res["actual"],
                              "null_mean": res["null_mean"], "null_sd": res["null_std"], "null_p95": res["null_p95"],
                              "percentile": res["percentile"], "p_value": res["p_value"],
                              "constant_exposure_sharpe": res["constant_exposure_score"], "n_sims": res["n_sims"]})
            print(f"  {name:18s} {label:20s} actual {res['actual']:.2f} | null mean {res['null_mean']:.2f} sd {res['null_std']:.2f} "
                  f"p95 {res['null_p95']:.2f} | pct {res['percentile']:.3f} p {res['p_value']:.3f} | flat-at-mean-exposure {res['constant_exposure_score']:.2f}")
    nulls = pd.DataFrame(null_rows)
    nulls.to_csv(OUT / "timing_nulls.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, name in zip(axes, ("original_viterbi", "causal_soft", "svxy_causal_soft")):
        for lab, col in (("iid_shuffle", "#9ca3af"), ("circular_shift", "#2563eb"), ("block_permute_21d", "#16a34a")):
            ax.hist(null_draws[(name, lab)], bins=40, alpha=0.55, color=col, label=lab, density=True)
        act = nulls[(nulls.strategy == name)].actual_sharpe.iloc[0]
        ax.axvline(act, color="#dc2626", lw=2, label=f"actual {act:.2f}")
        ax.set_title(name)
        ax.set_xlabel("Sharpe")
    axes[0].legend(fontsize=8)
    fig.suptitle("Timing-skill nulls: shuffled-iid (flawed) vs circular shift vs block permutation")
    fig.tight_layout()
    fig.savefig(OUT / "timing_nulls.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ PBO + RC/SPA on the 132-grid
    rows = []
    for tag, G, bm in (("viterbi_grid", G_vit, streams["constant_short"]), ("causal_hard_grid", G_cau, streams["constant_short"])):
        pbo = st.probability_of_backtest_overfitting(G, 16)
        bm_al = bm.reindex(G.index)
        rc_const = st.reality_check(G, benchmark=bm_al, n_boot=500 if quick else 1000, seed=3)
        rc_zero = st.reality_check(G, n_boot=500 if quick else 1000, seed=3)
        rows.append({"grid": tag, "n_candidates": G.shape[1], "best_sharpe": float((G.mean() / G.std()).max() * np.sqrt(252)),
                     "pbo": pbo["pbo"], "prob_oos_loss": pbo["prob_loss"], "mean_is_sr_ann": pbo["mean_is_sharpe_pp"] * np.sqrt(252),
                     "mean_oos_sr_ann": pbo["mean_oos_sharpe_pp"] * np.sqrt(252), "degradation_slope": pbo["degradation_slope"],
                     "rc_p_vs_zero": rc_zero["rc_pvalue"], "spa_p_vs_zero": rc_zero["spa_pvalue"],
                     "rc_p_vs_constant_short": rc_const["rc_pvalue"], "spa_p_vs_constant_short": rc_const["spa_pvalue"],
                     "best_rule_vs_constant": rc_const["best_rule"], "best_excess_ann_vs_constant": rc_const["best_mean_diff_ann"]})
    snoop = pd.DataFrame(rows)
    snoop.to_csv(OUT / "grid_snooping.csv", index=False)
    print("\n=== PBO / Reality check on the 132-point grid ===")
    print(snoop.T.to_string())

    # sub-period stability of the headline (4 blocks)
    sp = []
    for name in ("original_viterbi", "causal_soft", "constant_short", "svxy_causal_soft"):
        r = streams[name]
        for k, ch in enumerate(np.array_split(r.index, 4)):
            seg = r.loc[ch[0]:ch[-1]]
            ci = st.bootstrap_sharpe_ci(seg, n_boot=n_boot // 2, seed=11)
            sp.append({"strategy": name, "period": f"{ch[0].date()}..{ch[-1].date()}", "sharpe": ci["sharpe"],
                       "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    sp = pd.DataFrame(sp)
    sp.to_csv(OUT / "subperiod_sharpe.csv", index=False)
    print("\n=== sub-period Sharpe with 95% block-bootstrap CI ===")
    print(sp.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
