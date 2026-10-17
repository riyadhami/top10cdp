"""Build the real-data demo from the public UCI Online Retail II dataset.

Online Retail II: transactions of a UK online gift retailer/wholesaler, Dec 2009 - Dec 2011
(~1.07M rows). Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0).

The full file is ~90 MB as CSV, so the demo keeps every order of 900 randomly chosen customers
plus a sample of guest orders (~12 MB), which fits the app's 20 MB upload limit.

Usage:
    uv run python scripts/fetch_demo_data.py                 # download from UCI
    uv run python scripts/fetch_demo_data.py --xlsx path.xlsx  # use an already downloaded copy
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "demo" / "online_retail_ii_sample.csv"


def load(xlsx: str | None) -> pd.DataFrame:
    if xlsx:
        src = xlsx
    else:
        print(f"Downloading {URL} (~45 MB)…")
        with urllib.request.urlopen(URL, timeout=300) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
        src = io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith(".xlsx"))))
    print("Reading workbook (takes a minute)…")
    sheets = pd.read_excel(src, sheet_name=None, dtype=str)
    return pd.concat(sheets.values(), ignore_index=True)


def sample(df: pd.DataFrame, customers: int = 900, guest_orders: int = 150, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df["Customer ID"].dropna().unique()
    keep = set(rng.choice(ids, customers, replace=False))
    guest = df[df["Customer ID"].isna()]
    guest_keep = rng.choice(guest["Invoice"].unique(), guest_orders, replace=False)
    return pd.concat([df[df["Customer ID"].isin(keep)], guest[guest["Invoice"].isin(guest_keep)]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", help="path to online_retail_II.xlsx if already downloaded")
    args = ap.parse_args()
    demo = sample(load(args.xlsx))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(OUT, index=False)
    print(f"Wrote {len(demo):,} rows ({OUT.stat().st_size / 1e6:.1f} MB) to {OUT}")
