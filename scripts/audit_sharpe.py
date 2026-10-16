"""Reproduce the headline Sharpe and decompose where it comes from.

    python scripts/audit_sharpe.py

Writes results/audit/audit_table.csv. See docs/EVAL_PROTOCOL.md for the summary.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from vrp_strategy.backtest import raw_strategy_returns, vol_target_scale
from vrp_strategy.data import SPLIT, ROOT, asset_returns, load_signals
from vrp_strategy.metrics import performance_metrics
from vrp_strategy.regimes import filtered_proba, state_order_by_feature

FEATURES = ["vrp_vol", "slope", "rv_30"]
MULT_BY_RANK = np.array([1.0, 0.25, 0.0])  # calm, stressed, crisis (original notebook)

sig = load_signals()
train, test = sig[sig.index < SPLIT], sig[sig.index >= SPLIT]
scaler = StandardScaler().fit(train[FEATURES])
X_train, X_test = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])
model = hmm.GaussianHMM(3, covariance_type="full", n_iter=1000, random_state=42).fit(X_train)
order = list(state_order_by_feature(model, scaler, FEATURES.index("rv_30")))
mult_of_state = np.array([MULT_BY_RANK[order.index(s)] for s in range(3)])

rank = sig["vrp"].rolling(252).rank(pct=True)
rows = []


def vol_targeted(mult, label, cap=100.0):
    pos = (rank * mult).clip(0, 1)
    d = sig.assign(position=pos)
    d = d[d.index >= SPLIT].dropna(subset=["position"])
    r = raw_strategy_returns(d["position"], d["iv_daily"], d["rv_daily"]).dropna()
    scaled, _ = vol_target_scale(r, 0.10, 126, cap)
    rows.append({"variant": label, **performance_metrics(scaled.dropna())})
    return pos


viterbi = np.concatenate([model.predict(X_train), model.predict(X_test)])
vit_mult = pd.Series(mult_of_state[viterbi], index=sig.index)
p_all = np.vstack([filtered_proba(model, X_train), filtered_proba(model, X_test)])

vol_targeted(vit_mult, "1 original: Viterbi over full test block (non-causal)")
vol_targeted(pd.Series(mult_of_state[p_all.argmax(1)], index=sig.index), "2 causal filtered, hard argmax")
soft = pd.Series(p_all @ mult_of_state, index=sig.index)
pos_soft = vol_targeted(soft, "3 causal filtered, soft multiplier")
vol_targeted(pd.Series(1.0, index=sig.index), "4 no regime gate (VRP rank only)")
const = pd.Series(1.0, index=sig.index)
d = sig[sig.index >= SPLIT]
r = raw_strategy_returns(const[d.index], d["iv_daily"], d["rv_daily"]).dropna()
rows.append({"variant": "5 constant short variance, no signal", **performance_metrics(vol_target_scale(r)[0].dropna())})

# Ground truth: the same causal positions on a real traded instrument.
svxy = asset_returns("svxy")
spx = sig["spx"].pct_change()
lag_pos = pos_soft.shift(1)
for label, ret in [
    ("6 causal positions x real SVXY, unlevered", lag_pos * svxy),
    ("7 SVXY buy and hold", svxy),
    ("8 SPX buy and hold", spx),
]:
    ret = ret.dropna()
    rows.append({"variant": label, **performance_metrics(ret[ret.index >= SPLIT])})

proxy = (d["iv_daily"].shift(1) - d["rv_daily"])
j = pd.concat([proxy, svxy], axis=1, join="inner").dropna()
corr = float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])

table = pd.DataFrame(rows).set_index("variant")
out = ROOT / "results" / "audit"
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "audit_table.csv")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)
print(table[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].round(3))
print(f"\ncorr(proxy daily P&L, SVXY daily return) = {corr:.3f}")
print(f"Viterbi vs filtered label agreement on test = "
      f"{(np.argmax(p_all[len(X_train):], 1) == viterbi[len(X_train):]).mean():.3f}")
