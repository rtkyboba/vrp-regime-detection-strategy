"""Signal construction (the code of notebook 02, as a tested function).

Inputs are the raw SPX, VIX and VIX3M closes. VIX3M has the shortest history, so the
inner join starts in 2006-07 and the 30-day realised-vol warm-up drops the first 30 rows.
"""
import numpy as np
import pandas as pd

from .data import load_raw

TRADING_DAYS = 252
RV_WINDOW = 30


def build_master(spx: pd.Series, vix: pd.Series, vix3m: pd.Series) -> pd.DataFrame:
    """Inner-join the three closes on date (notebook 01)."""
    out = pd.concat({"spx": spx, "vix": vix, "vix3m": vix3m}, axis=1, join="inner").sort_index()
    out.index.name = "date"
    return out


def construct_signals(master: pd.DataFrame) -> pd.DataFrame:
    """Notebook 02 on a master frame with columns spx, vix, vix3m.

    * ``rv_30``      30-day realised vol of SPX log returns, annualised
    * ``vrp``        (VIX/100)^2 - rv_30^2   variance-space premium (sizing and P&L units)
    * ``vrp_vol``    VIX/100 - rv_30         vol-space premium (HMM feature)
    * ``slope``      (VIX3M - VIX) / VIX     term-structure slope; < 0 is backwardation
    * ``rv_daily``   squared daily log return; ``iv_daily`` (VIX/100)^2 / 252
    """
    d = master.copy()
    d["log_ret"] = np.log(d["spx"] / d["spx"].shift(1))
    d["rv_30"] = d["log_ret"].rolling(RV_WINDOW).std() * np.sqrt(TRADING_DAYS)
    d["vrp"] = (d["vix"] / 100) ** 2 - d["rv_30"] ** 2
    d["vrp_vol"] = d["vix"] / 100 - d["rv_30"]
    d["slope"] = (d["vix3m"] - d["vix"]) / d["vix"]
    d["rv_daily"] = d["log_ret"] ** 2
    d["iv_daily"] = (d["vix"] / 100) ** 2 / TRADING_DAYS
    cols = ["spx", "vix", "vix3m", "rv_30", "vrp", "vrp_vol", "slope", "log_ret", "rv_daily", "iv_daily"]
    return d[cols].dropna()


def build_signals() -> pd.DataFrame:
    """Signal table from the raw files in data/raw.

    Notebook 02 read ``master.csv`` with ``skiprows=[1]``, which was meant to skip yfinance's
    extra header row but, on the single-header master file, silently drops the first data row
    (2006-07-17). That row is dropped here too so every number in reports/ reproduces exactly;
    its only effect is that the table starts one trading day later.
    """
    master = build_master(load_raw("spx"), load_raw("vix"), load_raw("vix3m"))
    return construct_signals(master.iloc[1:])
