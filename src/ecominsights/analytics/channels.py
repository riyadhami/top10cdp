"""Multi-channel / multi-store comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kpis import metrics, slice_period, windows


def channel_table(orders: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    ref = orders["order_date"].max()
    w = windows(ref, days)
    rows = []
    for ch, o in orders.groupby("channel"):
        cur = metrics(slice_period(o, *w["current"]))
        prev = metrics(slice_period(o, *w["previous"]))
        all_m = metrics(o)
        rows.append({"channel": ch, "revenue_total": all_m["revenue"], "orders_total": all_m["orders"],
                     "aov": all_m["aov"], "repeat_rate": all_m["repeat_rate"],
                     "revenue_current": cur["revenue"], "revenue_previous": prev["revenue"],
                     "growth": cur["revenue"] / prev["revenue"] - 1 if prev["revenue"] else np.nan,
                     "gross_margin": all_m.get("gross_margin"),
                     "identified_customers": float(o["customer_key"].notna().mean())})
    t = pd.DataFrame(rows).sort_values("revenue_total", ascending=False)
    t["share"] = t["revenue_total"] / t["revenue_total"].sum()
    return t.reset_index(drop=True)


def monthly_by_channel(orders: pd.DataFrame) -> pd.DataFrame:
    o = orders.assign(month=orders["order_date"].dt.to_period("M").dt.to_timestamp())
    return o.groupby(["month", "channel"])["revenue"].sum().reset_index()
