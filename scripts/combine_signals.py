"""Does combining the regime signal and the RNN signal add anything on real SVXY?

    python scripts/combine_signals.py

Pre-specified variants (12 combiner variants, counted as trials):
  sources H in {hmm3_core, jump3_lam20}
  rules   hmm_only, mean, product, veto, ic   (per source)  +  rnn_only, ridge_only
Sizing  exposure = combined signal in [0,1] of SVXY, no leverage; 2 bps on exposure change;
        excess of T-bills.
Control every variant is compared with a CONSTANT exposure equal to its own average
        exposure (so "holds less" is not mistaken for timing skill) and with always-short.
Writes results/combine/summary.csv and results/combine/equity.png.
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

from vrp_strategy import combiner as cb
from vrp_strategy.data import ROOT, SPLIT, asset_returns, load_raw, load_signals, rf_daily
from vrp_strategy.pnl import backtest, hold_to_maturity
from vrp_strategy.stats_tests import validate

OUT = ROOT / "results" / "combine"
OUT.mkdir(parents=True, exist_ok=True)
START = "2012-01-03"
COST_BPS = 2.0


def regime(name):
    d = pd.read_csv(ROOT / "results" / "regimes" / f"{name}.csv", index_col=0, parse_dates=True)
    return d["stress_score"], d["stress_flag"]


def rnn_model(name):
    d = pd.read_csv(ROOT / "results" / "rnn" / "signal.csv", parse_dates=["date"])
    return d[d["model"] == name].set_index("date")["signal"]


svxy = asset_returns("svxy")
svxy = svxy[svxy.index >= START]
idx = svxy.index
rf = rf_daily(idx)

sig = load_signals()
lr = np.log(sig["spx"] / sig["spx"].shift(1)).dropna()
htm = hold_to_maturity(sig["vix"], lr)
payoff = htm["payoff_var"] / htm["strike_var"]   # scale-free: 1 - RV^2/VIX^2 (vol-point payoff scales with VIX level)

a_rnn = cb.forecast_strength(rnn_model("rnn")).reindex(idx)
a_ridge = cb.forecast_strength(rnn_model("ridge")).reindex(idx)

exposures = {}
for src in ("hmm3_core", "jump3_lam20"):
    stress, flag = regime(src)
    a_h = cb.calm_score(stress).reindex(idx)
    flag = flag.reindex(idx)
    exposures[f"{src}|hmm_only"] = a_h
    exposures[f"{src}|mean"] = cb.combine_mean(a_h, a_rnn)
    exposures[f"{src}|product"] = cb.combine_product(a_h, a_rnn)
    exposures[f"{src}|veto"] = cb.combine_veto(a_rnn, flag)
    ic_comb, w = cb.combine_ic({"regime": a_h, "rnn": a_rnn}, payoff.reindex(idx))
    exposures[f"{src}|ic"] = ic_comb
    if src == "hmm3_core":
        w.to_csv(OUT / "ic_weights_hmm3_core.csv")
exposures["rnn_only"] = a_rnn
exposures["ridge_only"] = a_ridge
N_TRIALS = len(exposures)

windows = {
    "2012+": (START, None, None),
    "2016+": (SPLIT, None, None),
    "2016+ ex-Feb18": (SPLIT, None, ("2018-02-01", "2018-02-28")),
}


def sub(r, key):
    lo, hi, hole = windows[key]
    r = r[r.index >= lo]
    if hole:
        r = r[(r.index < hole[0]) | (r.index > hole[1])]
    return r.dropna()


def run(exposure):
    return backtest(exposure.fillna(0.0), svxy, cost_bps=COST_BPS, max_leverage=1.0, rf=rf)


bh = run(pd.Series(1.0, index=idx))
rows, rets = [], {"always_short": bh["returns"]}
for name, e in exposures.items():
    res = run(e)
    avg = float(res["exposure"].loc[SPLIT:].mean())
    const = run(pd.Series(avg, index=idx))["returns"]            # same average size, no timing
    rets[name] = res["returns"]
    row = {"variant": name, "avg_exposure_2016+": avg}
    for wk in windows:
        r, c, b = sub(res["returns"], wk), sub(const, wk), sub(bh["returns"], wk)
        sr = lambda x: x.mean() / x.std() * np.sqrt(252)
        row[f"sharpe {wk}"] = sr(r)
        row[f"const-same-size {wk}"] = sr(c)
        row[f"always-short {wk}"] = sr(b)
    r16, c16 = sub(res["returns"], "2016+"), sub(const, "2016+")
    v = validate(r16, n_trials=N_TRIALS, benchmark=c16, n_boot=1000)
    row.update({"diff_vs_const": v["sharpe_diff"], "diff_ci_low": v["diff_ci_low"],
                "diff_ci_high": v["diff_ci_high"], "diff_p": v["diff_p_value"],
                "ci_low": v["ci_low"], "ci_high": v["ci_high"], "dsr": v["dsr"],
                "max_dd": res["stats"]["max_drawdown"], "worst_day": float(r16.min()),
                "turnover_per_yr": res["stats"].get("turnover", np.nan)})
    # information content: does the signal rank the 21d payoff?
    ic = cb.block_bootstrap_ic(e.reindex(idx).loc[SPLIT:], payoff.reindex(idx).loc[SPLIT:], n_boot=500)
    row.update({"ic_21d": ic["ic"], "ic_lo": ic["ci_low"], "ic_hi": ic["ci_high"]})
    rows.append(row)

summary = pd.DataFrame(rows).set_index("variant")
summary.to_csv(OUT / "summary.csv")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(f"combiner variants (trials): {N_TRIALS}")
show = ["avg_exposure_2016+", "sharpe 2016+", "const-same-size 2016+", "always-short 2016+",
        "diff_vs_const", "diff_ci_low", "diff_ci_high", "diff_p", "max_dd", "ic_21d", "ic_lo", "ic_hi"]
print(summary[show].round(3).to_string())
print()
print(summary[["sharpe 2012+", "const-same-size 2012+", "sharpe 2016+ ex-Feb18",
               "const-same-size 2016+ ex-Feb18", "always-short 2016+ ex-Feb18"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
for name in ("always_short", "hmm3_core|mean", "hmm3_core|ic", "rnn_only"):
    r = rets[name]
    r = r[r.index >= SPLIT].dropna()
    ax.plot(r.index, (1 + r).cumprod(), label=name, lw=1.1)
ax.set_yscale("log")
ax.set_title("Combined signals on real SVXY, 2016+ (net 2 bps, excess of T-bills)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "equity.png", dpi=130)
