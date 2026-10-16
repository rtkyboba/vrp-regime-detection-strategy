"""Walk-forward GRU/LSTM trading signal for short variance, plus the baselines it must beat.

The question is not "can a neural net fit VIX data" (it can, in-sample) but "does a
recurrent net add *out-of-sample, tradable* information over HAR-RV / ridge / gradient
boosting / the legacy VRP-rank rule / just always being short?".  Everything here is built
so that answer cannot be flattered:

* Features are strictly causal (`build_features`): row t uses data through the close of t.
* Targets overlap (21-day payoff, 5-day return), so the training set for a refit at block
  start `t0` only contains samples `i <= t0 - h - embargo - 1`.  That is purging (label
  window `(i, i+h]` is fully realised before `t0`) plus an embargo of `embargo` days.
* Standardisation, target scaling, early stopping and ridge penalties all use the
  training window only (`purged_split`).
* Model selection happens on a dataset *truncated at SPLIT* (`truncate`) so that the
  2016+ data cannot influence any choice (docs/EVAL_PROTOCOL.md rule 2).
* Seeds are a pure function of (ensemble member, refit index) so runs are reproducible
  and a truncated run reproduces the same pre-2016 predictions as the full run.

Targets
-------
`variance_payoff_scaled`: enter a 21-day short-variance position at t at strike VIX_t^2,
pay realised variance (sum of squared SPX log returns over t+1..t+21).  Scaled by the
strike: `1 - RV/IV`; its sign is "implied > realised".  `forward_return`: compounded
SVXY return over t+1..t+5.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import data as D

H21 = 21
H5 = 5
CLIP_LO, CLIP_HI = -3.0, 1.0          # winsorisation of the *training* regression target only
MIN_SAMPLE_ROW = 59                    # first row usable as a sample (longest window = 60)

FEATURE_COLS = [
    "log_vix", "ts_3m", "ts_6m", "log_vvix", "vvix_vix", "skew",
    "rv5", "rv10", "rv21", "rv63", "vrp_21", "vrp_var",
    "ret1", "ret5", "ret21", "dd63", "dvix1", "dvix5", "vix_rel", "labsret1",
]
HAR_COLS = ["labsret1", "rv5", "rv21", "rv63", "log_vix"]


# --------------------------------------------------------------------------- features
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Causal daily feature table.

    `raw` needs columns spx, vix, vix3m, vix6m, vvix, skew on a common daily index.
    Every column is a function of rows <= t only (rolling windows, diffs), so
    changing data after t cannot change row t (tested in tests/test_rnn_signal.py).
    Rows are left NaN until every feature is defined (no back-filling).
    """
    r = raw.copy()
    for c in ("vix6m", "vvix", "skew", "vix3m"):
        r[c] = r[c].ffill(limit=5)          # forward-fill only: uses the past
    lr = np.log(r["spx"]).diff()
    f = pd.DataFrame(index=r.index)
    f["log_vix"] = np.log(r["vix"])
    f["ts_3m"] = np.log(r["vix3m"] / r["vix"])
    f["ts_6m"] = np.log(r["vix6m"] / r["vix3m"])
    f["log_vvix"] = np.log(r["vvix"])
    f["vvix_vix"] = np.log(r["vvix"] / r["vix"])
    f["skew"] = r["skew"]
    for w in (5, 10, 21, 63):
        f[f"rv{w}"] = np.log(np.sqrt(252 * (lr ** 2).rolling(w).mean()) + 1e-4)
    f["vrp_21"] = f["log_vix"] - f["rv21"]
    f["vrp_var"] = (r["vix"] / 100) ** 2 - 252 * (lr ** 2).rolling(21).mean()
    f["ret1"] = lr
    f["ret5"] = lr.rolling(5).sum()
    f["ret21"] = lr.rolling(21).sum()
    f["dd63"] = np.log(r["spx"] / r["spx"].rolling(63).max())
    f["dvix1"] = f["log_vix"].diff()
    f["dvix5"] = f["log_vix"].diff(5)
    f["vix_rel"] = f["log_vix"] - f["log_vix"].rolling(126).mean()
    f["labsret1"] = np.log(lr.abs() + 1e-3)
    return f[FEATURE_COLS]


def load_raw_table() -> pd.DataFrame:
    sig = D.load_signals()
    idx = sig.index
    raw = pd.DataFrame(index=idx)
    raw["spx"] = D.load_raw("spx").reindex(idx)
    raw["vix"] = sig["vix"]
    raw["vix3m"] = sig["vix3m"]
    for n in ("vix6m", "vvix", "skew"):
        raw[n] = D.load_raw(n).reindex(idx)
    return raw


# --------------------------------------------------------------------------- targets
def variance_payoff(vix: pd.Series, spx: pd.Series, h: int = H21) -> pd.DataFrame:
    """Held-to-maturity short-variance payoff entered at t.

    strike_t = (VIX_t/100)^2 * h/252 ; realised = sum_{i=1..h} (log-return_{t+i})^2.
    Returns columns payoff (variance units), scaled (= 1 - RV/IV), NaN where the
    h future returns are not yet observed.
    """
    lr = np.log(spx).diff()
    rv = (lr ** 2).rolling(h).sum().shift(-h)
    strike = (vix / 100.0) ** 2 * h / 252.0
    return pd.DataFrame({"payoff": strike - rv, "scaled": 1.0 - rv / strike})


def forward_return(price: pd.Series, h: int = H5) -> pd.Series:
    """Compounded return over t+1..t+h (NaN if unobserved)."""
    lr1 = np.log(price).diff()
    return np.expm1(lr1.rolling(h).sum().shift(-h))


# --------------------------------------------------------------------------- dataset
@dataclass
class Dataset:
    index: pd.DatetimeIndex
    F: np.ndarray                 # (n, d) raw causal features, no NaN
    y_reg: np.ndarray             # (n,) regression target (NaN = no label)
    y_cls: np.ndarray             # (n,) 1.0 / 0.0 / NaN
    h: int
    names: List[str]

    def __len__(self):
        return len(self.index)


def make_dataset(feats: pd.DataFrame, y: pd.Series, h: int) -> Dataset:
    ok = feats.notna().all(axis=1)
    first = ok.idxmax()
    feats = feats.loc[first:]
    yy = y.reindex(feats.index)
    assert feats.notna().all().all(), "NaN features after first valid row"
    cls = np.where(yy.isna(), np.nan, (yy > 0).astype(float))
    return Dataset(feats.index, feats.to_numpy(float), yy.to_numpy(float), cls, h, list(feats.columns))


def truncate(ds: Dataset, end_exclusive: str) -> Dataset:
    """Dataset as it would look on `end_exclusive`: rows at/after it are removed and
    any label whose window reaches past the cut is set to NaN."""
    n = int((ds.index < pd.Timestamp(end_exclusive)).sum())
    y_reg, y_cls = ds.y_reg[:n].copy(), ds.y_cls[:n].copy()
    cut = max(0, n - ds.h)
    y_reg[cut:] = np.nan
    y_cls[cut:] = np.nan
    return Dataset(ds.index[:n], ds.F[:n], y_reg, y_cls, ds.h, ds.names)


def purged_split(t0: int, h: int, embargo: int, val_len: int, min_row: int = MIN_SAMPLE_ROW):
    """Train/validation sample indices for a model that first predicts at row `t0`.

    A sample i has label window (i, i+h].  It may be trained on only if
    i + h + embargo < t0, i.e. its label is fully realised, and `embargo` further days
    separate it from the first prediction.  The validation tail is the last `val_len`
    admissible samples; inner-train stops `h + embargo` samples before it starts so
    the validation labels do not overlap the inner-train labels either.
    """
    te = t0 - h - embargo - 1
    va = np.arange(max(min_row, te - val_len + 1), te + 1)
    tr_end = va[0] - h - embargo - 1
    tr = np.arange(min_row, tr_end + 1)
    full = np.arange(min_row, te + 1)
    return tr, va, full


# --------------------------------------------------------------------------- models
@dataclass(frozen=True)
class RNNConfig:
    cell: str = "gru"
    hidden: int = 16
    seq_len: int = 20
    dropout: float = 0.2
    weight_decay: float = 1e-3
    lr: float = 2e-3
    max_epochs: int = 40
    patience: int = 6
    batch: int = 128

    def tag(self):
        return f"{self.cell}_h{self.hidden}_L{self.seq_len}_do{self.dropout}_wd{self.weight_decay:g}"


@dataclass
class Prediction:
    """Raw output of a walk-forward run. Arrays are (n_members, n_dates)."""
    dates: pd.DatetimeIndex
    prob: Optional[np.ndarray] = None
    reg: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None        # (n_dates,) training std of clipped target

    def mean_prob(self):
        return None if self.prob is None else self.prob.mean(0)

    def mean_reg(self):
        return None if self.reg is None else self.reg.mean(0)


def _standardise(F: np.ndarray, upto: int):
    mu = F[: upto + 1].mean(0)
    sd = F[: upto + 1].std(0)
    sd[sd < 1e-8] = 1.0
    return np.clip((F - mu) / sd, -5, 5)


def _flat_features(F: np.ndarray) -> np.ndarray:
    """[x_t, 5-day mean, 21-day mean] of every feature - trailing, hence causal."""
    df = pd.DataFrame(F)
    return np.hstack([F, df.rolling(5, min_periods=1).mean().to_numpy(), df.rolling(21, min_periods=1).mean().to_numpy()])


def _target_scaling(y_reg, idx):
    y = np.clip(y_reg[idx], CLIP_LO, CLIP_HI)
    return float(y.mean()), float(y.std() + 1e-8)


def _windows(Z, idx, L):
    import torch
    off = np.arange(-L + 1, 1)
    return torch.from_numpy(Z[idx[:, None] + off[None, :]].astype(np.float32))


def rnn_fit_predict(ds: Dataset, tr, va, pred, cfg: RNNConfig, seeds: Sequence[int], t0: int):
    """Train one small GRU/LSTM per seed on the purged training window; early-stop on the
    purged validation tail; return member predictions for rows `pred`."""
    import torch
    import torch.nn as nn

    full = np.concatenate([tr, va])
    Z = _standardise(ds.F, full.max())
    mu, sd = _target_scaling(ds.y_reg, full)
    L = cfg.seq_len

    def pack(idx):
        x = _windows(Z, idx, L)
        yr = torch.from_numpy(((np.clip(ds.y_reg[idx], CLIP_LO, CLIP_HI) - mu) / sd).astype(np.float32))
        yc = torch.from_numpy(ds.y_cls[idx].astype(np.float32))
        return x, yr, yc

    xtr, yrtr, yctr = pack(tr)
    xva, yrva, ycva = pack(va)
    xpr = _windows(Z, pred, L)
    d = Z.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            rnn = nn.GRU if cfg.cell == "gru" else nn.LSTM
            self.rnn = rnn(d, cfg.hidden, batch_first=True)
            self.drop = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(cfg.hidden, 2)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(self.drop(o[:, -1]))

    huber, bce = nn.HuberLoss(delta=1.0), nn.BCEWithLogitsLoss()

    def loss_fn(out, yr, yc):
        return huber(out[:, 0], yr) + bce(out[:, 1], yc)

    probs, regs = [], []
    for s in seeds:
        torch.manual_seed(int(s) * 100003 + int(t0))
        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, best_state, bad = np.inf, None, 0
        n = len(xtr)
        for _ in range(cfg.max_epochs):
            net.train()
            perm = torch.randperm(n)
            for b in range(0, n, cfg.batch):
                j = perm[b:b + cfg.batch]
                opt.zero_grad()
                loss_fn(net(xtr[j]), yrtr[j], yctr[j]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss_fn(net(xva), yrva, ycva))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            out = net(xpr).numpy()
        probs.append(1.0 / (1.0 + np.exp(-out[:, 1])))
        regs.append(out[:, 0] * sd + mu)
    return np.array(probs), np.array(regs), sd


def ridge_fit_predict(ds: Dataset, tr, va, pred, t0: int, alphas=tuple(np.logspace(-1, 4, 11))):
    """Ridge on [x_t, 5d mean, 21d mean]; alpha picked on the purged validation tail,
    then refit on all admissible training samples."""
    from sklearn.linear_model import Ridge
    full = np.concatenate([tr, va])
    X = _flat_features(ds.F)
    X = _standardise(X, full.max())
    mu, sd = _target_scaling(ds.y_reg, full)
    y = (np.clip(ds.y_reg, CLIP_LO, CLIP_HI) - mu) / sd
    errs = [np.mean((Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[va]) - y[va]) ** 2) for a in alphas]
    a = alphas[int(np.argmin(errs))]
    m = Ridge(alpha=a).fit(X[full], y[full])
    return (m.predict(X[pred]) * sd + mu)[None, :], sd


def har_fit_predict(ds: Dataset, tr, va, pred, t0: int):
    """HAR-RV style OLS: daily/weekly/monthly/quarterly realised vol plus implied vol."""
    from sklearn.linear_model import LinearRegression
    full = np.concatenate([tr, va])
    cols = [ds.names.index(c) for c in HAR_COLS]
    X = ds.F[:, cols]
    mu, sd = _target_scaling(ds.y_reg, full)
    y = np.clip(ds.y_reg, CLIP_LO, CLIP_HI)
    m = LinearRegression().fit(X[full], y[full])
    return m.predict(X[pred])[None, :], sd


def hgb_fit_predict(ds: Dataset, tr, va, pred, t0: int, max_depth=3, max_iter=150, lr=0.05):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    full = np.concatenate([tr, va])
    X = _flat_features(ds.F)
    mu, sd = _target_scaling(ds.y_reg, full)
    kw = dict(max_depth=max_depth, max_iter=max_iter, learning_rate=lr, min_samples_leaf=40,
              l2_regularization=1.0, early_stopping=False, random_state=0)
    reg = HistGradientBoostingRegressor(**kw).fit(X[full], np.clip(ds.y_reg[full], CLIP_LO, CLIP_HI))
    clf = HistGradientBoostingClassifier(**kw).fit(X[full], ds.y_cls[full].astype(int))
    p = clf.predict_proba(X[pred])[:, 1]
    return p[None, :], reg.predict(X[pred])[None, :], sd


# --------------------------------------------------------------------------- walk-forward
def refit_points(n: int, first_t0: int, step: int) -> List[int]:
    return list(range(first_t0, n, step))


def walk_forward(ds: Dataset, kind: str, *, cfg: Optional[RNNConfig] = None, seeds: Sequence[int] = (0, 1, 2, 3, 4),
                 step: int = 63, first_t0: int = MIN_SAMPLE_ROW + 800, embargo: Optional[int] = None,
                 val_len: int = 252, min_row: int = MIN_SAMPLE_ROW, hgb_kw: Optional[dict] = None,
                 verbose: bool = False) -> Prediction:
    """Expanding-window walk-forward.  At each refit point t0 the model sees only samples
    from `purged_split(t0, ...)` and predicts rows t0 .. t0+step-1 (never refit in between)."""
    embargo = ds.h if embargo is None else embargo
    n = len(ds)
    dates, probs, regs, scales = [], [], [], []
    for t0 in refit_points(n, first_t0, step):
        pred = np.arange(t0, min(t0 + step, n))
        tr, va, full = purged_split(t0, ds.h, embargo, val_len, min_row)
        assert np.isfinite(ds.y_reg[full]).all(), "unlabelled sample leaked into training"
        p = r = None
        if kind == "rnn":
            p, r, sc = rnn_fit_predict(ds, tr, va, pred, cfg or RNNConfig(), seeds, t0)
        elif kind == "ridge":
            r, sc = ridge_fit_predict(ds, tr, va, pred, t0)
        elif kind == "har":
            r, sc = har_fit_predict(ds, tr, va, pred, t0)
        elif kind == "hgb":
            p, r, sc = hgb_fit_predict(ds, tr, va, pred, t0, **(hgb_kw or {}))
        else:
            raise ValueError(kind)
        dates.append(ds.index[pred])
        if p is not None:
            probs.append(p)
        regs.append(r)
        scales.append(np.full(len(pred), sc))
        if verbose:
            print(f"  [{kind}] refit {ds.index[t0].date()} train={len(full)}", flush=True)
    return Prediction(
        pd.DatetimeIndex(np.concatenate([d.values for d in dates])),
        np.hstack(probs) if probs else None,
        np.hstack(regs),
        np.concatenate(scales),
    )


# --------------------------------------------------------------------------- signals
def signal_from_prob(p):
    return 2.0 * np.asarray(p) - 1.0


def signal_from_reg(raw, scale):
    return np.tanh(np.asarray(raw) / np.asarray(scale))


def legacy_rank_signal(vrp: pd.Series) -> pd.Series:
    """Legacy rule: trailing-252d percentile rank of VRP in [0,1] was the position.
    As a signal in [-1,1]: 2*rank-1 (so exposure=(s+1)/2 recovers the legacy position)."""
    return 2.0 * vrp.rolling(252).rank(pct=True) - 1.0


def exposure_from_signal(sig: pd.Series, mode: str = "linear") -> pd.Series:
    """Map signal strength s in [-1,1] to SVXY exposure in [0,1] (long SVXY = short vol).
    linear: (s+1)/2 ; longflat: clip(s,0,1) ; sign: 1[s>0]."""
    if mode == "linear":
        return (sig + 1.0) / 2.0
    if mode == "longflat":
        return sig.clip(0.0, 1.0)
    if mode == "sign":
        return (sig > 0).astype(float).where(sig.notna())
    raise ValueError(mode)


# --------------------------------------------------------------------------- statistics
def _rank(a):
    return pd.Series(a).rank().to_numpy()


def spearman(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    ra, rb = _rank(np.asarray(a)[m]), _rank(np.asarray(b)[m])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap index vector of length n."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def block_bootstrap(stat: Callable[..., float], arrays: Sequence[np.ndarray], block: int = 42,
                    reps: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Point estimate and (1-alpha) percentile CI of stat(*arrays) under a moving-block
    bootstrap of aligned arrays. Block >= 2x the target horizon because labels overlap."""
    arrays = [np.asarray(a, float) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for k in range(reps):
        ix = block_indices(n, block, rng)
        draws[k] = stat(*[a[ix] for a in arrays])
    draws = draws[np.isfinite(draws)]
    return float(stat(*arrays)), float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)), draws


def hit_rate(sig, y) -> float:
    """Share of days where sign(signal) agrees with sign(target) (signal==0 counted wrong)."""
    sig, y = np.asarray(sig, float), np.asarray(y, float)
    return float(np.mean(np.sign(sig) == np.sign(y)))


def calibration_table(p, y01, bins: int = 10) -> pd.DataFrame:
    p, y01 = np.asarray(p, float), np.asarray(y01, float)
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y01, "bin": q}).groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), freq=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True)


def brier_skill(p, y01) -> float:
    """1 - Brier(model)/Brier(constant base rate): positive = better than the base rate."""
    p, y01 = np.asarray(p, float), np.asarray(y01, float)
    b = np.mean((p - y01) ** 2)
    b0 = np.mean((y01.mean() - y01) ** 2)
    return float(1.0 - b / b0)


# --------------------------------------------------------------------------- economics
def exposure_pnl(exposure: pd.Series, asset_ret: pd.Series, cost_bps: float = 2.0, start=None, end=None) -> pd.DataFrame:
    """Daily P&L of holding `exposure` (decided at close t-1) in `asset_ret` (return of day t),
    remainder in cash at the risk-free rate.  Exposure changes are charged `cost_bps`
    per unit of exposure traded.  Returns columns gross_excess, net_excess, net_total, rf, exposure."""
    # shift within the signal calendar (never forward-fill past the last signal date)
    pos = exposure.shift(1).reindex(asset_ret.index)
    rf = D.rf_daily(asset_ret.index)
    cost = pos.diff().abs().fillna(0.0) * cost_bps / 1e4
    gross = pos * (asset_ret - rf)
    out = pd.DataFrame({"gross_excess": gross, "net_excess": gross - cost, "rf": rf, "exposure": pos})
    out["net_total"] = out["net_excess"] + out["rf"]
    out = out.dropna(subset=["exposure"])
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.index <= pd.Timestamp(end)]
    return out


def econ_summary(pnl: pd.DataFrame) -> dict:
    """Excess Sharpe (gross and net of costs), Sortino, drawdown of total net return, tails."""
    from .metrics import performance_metrics
    net, gross = pnl["net_excess"], pnl["gross_excess"]
    m = performance_metrics(net)
    cum = (1 + pnl["net_total"]).cumprod()
    mdd = float((cum / cum.cummax() - 1).min())
    return {
        "sharpe_gross": float(gross.mean() / gross.std() * np.sqrt(252)),
        "sharpe_net": m["sharpe"],
        "sortino_net": m["sortino"],
        "ann_excess_net": m["ann_return"],
        "ann_vol": m["ann_vol"],
        "max_drawdown": mdd,
        "worst_day": float(net.min()),
        "skew": float(net.skew()),
        "ex_kurt": float(net.kurt()),
        "avg_exposure": float(pnl["exposure"].mean()),
        "turnover_ann": float(pnl["exposure"].diff().abs().sum() / len(pnl) * 252),
        "pct_invested": float((pnl["exposure"] > 0.01).mean()),
        "n": int(len(pnl)),
    }
