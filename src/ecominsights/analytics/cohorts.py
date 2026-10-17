"""Acquisition cohorts and "what did the first order predict" cohorts.

- Retention matrix: % of a monthly cohort that ordered again in month k after acquisition.
- Revenue LTV curve: cumulative revenue per acquired customer by month k.
- First-product cohorts: repeat rate and 180-day value by the product of the first order.
- Discount-acquired cohorts: repeat rate when the first order used a discount vs not,
  with a two-proportion z-test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

MIN_COHORT = 15


def retention(orders: pd.DataFrame, max_months: int = 12) -> dict:
    o = orders[orders["customer_key"].notna()].copy()
    if o.empty:
        return {}
    o["cohort"] = o["first_order_date"].dt.to_period("M")
    o["period"] = o["order_date"].dt.to_period("M")
    o["k"] = (o["period"] - o["cohort"]).apply(lambda x: x.n)
    sizes = o[o["k"] == 0].groupby("cohort")["customer_key"].nunique()
    active = o.groupby(["cohort", "k"])["customer_key"].nunique().unstack(fill_value=0)
    revenue = o.groupby(["cohort", "k"])["revenue"].sum().unstack(fill_value=0.0)
    last_period = o["period"].max()
    # Blank out cells that are in the future for a cohort.
    for coh in active.index:
        age = (last_period - coh).n
        active.loc[coh, active.columns > age] = np.nan
        revenue.loc[coh, revenue.columns > age] = np.nan
    cols = [k for k in active.columns if k <= max_months]
    ret = active[cols].div(sizes, axis=0)
    ltv = revenue[cols].cumsum(axis=1).div(sizes, axis=0)
    ret.index = ret.index.astype(str)
    ltv.index = ltv.index.astype(str)
    # Size-weighted average curve across cohorts that are old enough for each k.
    def weighted(mat):
        out = {}
        for k in cols:
            col = mat[k]
            m = col.notna()
            w = sizes.reindex(pd.PeriodIndex(col.index, freq="M"))[m.values]
            out[int(k)] = float(np.average(col[m], weights=w)) if m.any() else None
        return out
    return {"sizes": {str(k): int(v) for k, v in sizes.items()},
            "retention": ret, "ltv": ltv,
            "avg_retention": weighted(ret), "avg_ltv": weighted(ltv)}


def _value_after(orders: pd.DataFrame, customers: pd.DataFrame, days: int) -> pd.Series:
    o = orders[orders["customer_key"].notna()]
    first = customers["first_order"]
    within = o["order_date"] <= o["customer_key"].map(first) + pd.Timedelta(days=days)
    return o[within].groupby("customer_key")["revenue"].sum()


def first_product(orders: pd.DataFrame, customers: pd.DataFrame, days: int = 180) -> pd.DataFrame:
    if "first_product" not in customers:
        return pd.DataFrame()
    ref = orders["order_date"].max()
    c = customers[customers["first_order"] <= ref - pd.Timedelta(days=days)].copy()
    if c.empty:
        return pd.DataFrame()
    c[f"value_{days}d"] = _value_after(orders, c, days).reindex(c.index).fillna(0)
    c["repeated"] = c["orders"] > 1
    g = c.groupby("first_product")
    t = pd.DataFrame({"customers": g.size(), "repeat_rate": g["repeated"].mean(),
                      f"avg_value_{days}d": g[f"value_{days}d"].mean()})
    t = t[t["customers"] >= MIN_COHORT]
    base = c[f"value_{days}d"].mean()
    t["value_index"] = t[f"avg_value_{days}d"] / base if base else np.nan
    return t.sort_values(f"avg_value_{days}d", ascending=False).reset_index()


def two_prop_ztest(x1: int, n1: int, x2: int, n2: int) -> float:
    if min(n1, n2) == 0:
        return 1.0
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def discount_cohorts(orders: pd.DataFrame, customers: pd.DataFrame, days: int = 180) -> dict:
    ref = orders["order_date"].max()
    c = customers[customers["first_order"] <= ref - pd.Timedelta(days=days)].copy()
    if c.empty or c["first_order_discounted"].nunique() < 2:
        return {}
    c["value"] = _value_after(orders, c, days).reindex(c.index).fillna(0)
    c["repeated"] = c["orders"] > 1
    d, f = c[c["first_order_discounted"]], c[~c["first_order_discounted"]]
    out = {"days": days,
           "discounted": {"customers": len(d), "repeat_rate": float(d["repeated"].mean()),
                          "avg_value": float(d["value"].mean())},
           "full_price": {"customers": len(f), "repeat_rate": float(f["repeated"].mean()),
                          "avg_value": float(f["value"].mean())},
           "p_value": two_prop_ztest(int(d["repeated"].sum()), len(d), int(f["repeated"].sum()), len(f))}
    if "first_discount_code" in c:
        g = c[c["first_order_discounted"]].groupby("first_discount_code")
        codes = pd.DataFrame({"customers": g.size(), "repeat_rate": g["repeated"].mean(),
                              "avg_value": g["value"].mean()})
        out["codes"] = codes[codes["customers"] >= MIN_COHORT].reset_index().to_dict("records")
    return out
