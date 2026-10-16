"""Single entry point for loading project data.

Everything downstream should load through here so every experiment sees the
same dates, the same train/test split and the same risk-free series.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Out-of-sample start. Anything fitted, selected or tuned must use data
# strictly before this date, or be re-fitted walk-forward (see docs/EVAL_PROTOCOL.md).
SPLIT = "2016-01-01"


def load_signals() -> pd.DataFrame:
    """Daily signal table built by notebook 02 (2006-08 -> present).

    Columns: spx, vix, vix3m, rv_30, vrp, vrp_vol, slope, log_ret,
    rv_daily, iv_daily.
    """
    return pd.read_csv(PROCESSED / "signals.csv", index_col=0, parse_dates=True)


def load_raw(name: str) -> pd.Series:
    """One raw close series by file stem, e.g. 'svxy', 'vix9d', 'vvix', 'vix6m',
    'skew', 'spy', 'vxx' (VXX only from 2018-01), 'irx' (13-week T-bill, % yield).

    Handles both the yfinance multi-header files from notebook 01 (spx, vix,
    vix3m) and the single-header files fetched later.
    """
    df = pd.read_csv(RAW / f"{name}.csv", index_col=0)
    df = df[pd.to_datetime(df.index, errors="coerce").notna()]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.iloc[:, 0].astype(float).rename(name).sort_index()


def rf_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free return from the 13-week T-bill yield (^IRX, in %),
    aligned to `index`, forward-filled. Use to turn returns into excess returns."""
    irx = load_raw("irx").reindex(index, method="ffill").bfill()
    return (irx / 100.0) / 252.0


def excess(returns: pd.Series) -> pd.Series:
    """Subtract the daily risk-free rate from a daily return series."""
    return returns - rf_daily(returns.index)


def asset_returns(name: str) -> pd.Series:
    """Simple daily returns of a traded instrument (svxy, vxx, spy)."""
    return load_raw(name).pct_change().dropna().rename(f"{name}_ret")
