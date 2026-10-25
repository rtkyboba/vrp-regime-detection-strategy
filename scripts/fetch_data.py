"""Download the raw price series into data/raw (not stored in git).

    python scripts/fetch_data.py                 # fetch everything up to the pinned end date
    python scripts/fetch_data.py --verify        # compare local files with data/MANIFEST.json
    python scripts/fetch_data.py --write-manifest

The end date is pinned so reports stay comparable; Yahoo occasionally revises adjusted prices,
so --verify reports (but does not fail on) hash differences and always prints row counts and
date ranges, which are what the results actually depend on.
"""
import argparse
import hashlib
import json
import sys

import pandas as pd

from vrp_strategy.data import RAW, ROOT, TICKERS

MANIFEST = ROOT / "data" / "MANIFEST.json"
PINNED_END = "2026-05-23"  # exclusive in yfinance, so the last row is 2026-05-22
START = "1990-01-01"


def fetch_one(name: str, ticker: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=START, end=end, auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"no data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].rename(columns={"Close": name})
    out.index.name = "date"
    return out


def describe(path) -> dict:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return {
        "rows": int(len(df)),
        "first": str(df.index[0].date()),
        "last": str(df.index[-1].date()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def current_manifest() -> dict:
    return {n: describe(RAW / f"{n}.csv") for n in TICKERS if (RAW / f"{n}.csv").exists()}


def verify() -> int:
    if not MANIFEST.exists():
        print("no data/MANIFEST.json; run with --write-manifest")
        return 1
    want, have = json.loads(MANIFEST.read_text()), current_manifest()
    bad = 0
    for name in TICKERS:
        w, h = want.get(name), have.get(name)
        if h is None:
            print(f"MISSING  {name}")
            bad += 1
        elif w["rows"] != h["rows"] or w["last"] != h["last"]:
            print(f"DIFFERS  {name}: manifest {w['rows']} rows to {w['last']}, local {h['rows']} rows to {h['last']}")
            bad += 1
        elif w["sha256"] != h["sha256"]:
            print(f"values   {name}: same shape, different bytes (Yahoo revised prices?); results may shift slightly")
        else:
            print(f"ok       {name}: {h['rows']} rows {h['first']} to {h['last']}")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--end", default=PINNED_END)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--write-manifest", action="store_true")
    a = p.parse_args()
    if a.verify:
        return verify()
    if a.write_manifest:
        MANIFEST.write_text(json.dumps(current_manifest(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST}")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    for name, ticker in TICKERS.items():
        df = fetch_one(name, ticker, a.end)
        df.to_csv(RAW / f"{name}.csv")
        print(f"{name:6s} {ticker:8s} {df.index[0].date()} to {df.index[-1].date()} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
