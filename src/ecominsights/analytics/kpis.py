"""KPI snapshots over 30/90/365-day horizons, KPI-tree decomposition and change attribution.

KPI tree:  Revenue = Orders x AOV
           Orders  = Customers x Orders per customer        (identified customers only)
           AOV     = Units per order x Price per unit

Contributions use LMDI (log-mean Divisia) so the parts sum exactly to the revenue change:
    contribution(x) = L(R1, R0) * ln(x1 / x0),   L(a, b) = (a - b) / (ln a - ln b)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

HORIZONS = {"30d": 30, "90d": 90, "365d": 365}


def windows(ref: pd.Timestamp, days: int) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    end = ref.normalize() + pd.Timedelta(days=1)
    d = pd.Timedelta(days=days)
    return {"current": (end - d, end), "previous": (end - 2 * d, end - d),
            "yoy": (end - d - pd.Timedelta(days=365), end - pd.Timedelta(days=365))}


def covered(start: pd.Timestamp, first: pd.Timestamp, days: int) -> bool:
    """Does the data reach back to `start`? Compared by calendar day, allowing up to 3% of the
    window to be missing so a dataset beginning mid-day (or a few days late) still compares."""
    slack = pd.Timedelta(days=max(1, round(days * 0.03)))
    return start >= first.normalize() - slack


def slice_period(df: pd.DataFrame, start, end, col: str = "order_date") -> pd.DataFrame:
    return df[(df[col] >= start) & (df[col] < end)]


def metrics(o: pd.DataFrame) -> dict:
    rev = float(o["revenue"].sum())
    n = len(o)
    ident = o[o["customer_key"].notna()]
    cust = ident["customer_key"].nunique()
    units = float(o["units"].sum())
    ret = o[o["is_returning"]]
    new_cust = ident.loc[ident["order_seq"] == 1, "customer_key"].nunique()
    out = {
        "revenue": rev, "orders": n, "customers": cust, "aov": rev / n if n else np.nan,
        "units_per_order": units / n if n else np.nan, "price_per_unit": rev / units if units else np.nan,
        "orders_per_customer": len(ident) / cust if cust else np.nan,
        "new_customers": new_cust, "returning_customers": ret["customer_key"].nunique(),
        "returning_revenue_share": float(ret["revenue"].sum()) / rev if rev else np.nan,
        "repeat_rate": ret["customer_key"].nunique() / cust if cust else np.nan,
        "discount_rate": float(o["discount"].sum()) / float(o["gross_revenue"].sum() + o["discount"].sum())
        if len(o) and (o["gross_revenue"].sum() + o["discount"].sum()) else np.nan,
    }
    if "cogs" in o and o["cogs"].notna().mean() > 0.5:
        fees = o["fees"].fillna(0).sum() if "fees" in o else 0.0
        gp = rev - float(o["cogs"].fillna(0).sum()) - float(fees)
        out["gross_profit"] = gp
        out["gross_margin"] = gp / rev if rev else np.nan
    return out


def _pct(a, b):
    if b is None or a is None or not np.isfinite(b) or b == 0 or not np.isfinite(a):
        return None
    return a / b - 1


def snapshot(orders: pd.DataFrame) -> dict:
    """Current vs previous period and vs same period last year, for each horizon."""
    ref = orders["order_date"].max()
    first = orders["order_date"].min()
    out = {"reference_date": str(ref.date())}
    for name, days in HORIZONS.items():
        w = windows(ref, days)
        cur = metrics(slice_period(orders, *w["current"]))
        prev = metrics(slice_period(orders, *w["previous"])) if covered(w["previous"][0], first, days) else None
        yoy = metrics(slice_period(orders, *w["yoy"])) if covered(w["yoy"][0], first, days) else None
        rows = {}
        for k, v in cur.items():
            rows[k] = {"value": v,
                       "prev": prev.get(k) if prev else None, "pop": _pct(v, prev.get(k)) if prev else None,
                       "yoy_value": yoy.get(k) if yoy else None, "yoy": _pct(v, yoy.get(k)) if yoy else None}
        cur_rows = slice_period(orders, *w["current"])
        out[name] = {"window": [str(w["current"][0].date()), str((w["current"][1] - pd.Timedelta(days=1)).date())],
                     "metrics": rows,
                     "tree": kpi_tree(slice_period(orders, *w["previous"]), cur_rows) if prev else None,
                     # Same period last year removes seasonality; preferred for driver insights when available.
                     "tree_yoy": kpi_tree(slice_period(orders, *w["yoy"]), cur_rows)
                     if yoy and days < 365 else None}
    return out


def _logmean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return (a - b) / 2 if a != b else a
    return a if math.isclose(a, b) else (a - b) / (math.log(a) - math.log(b))


def _contrib(L: float, x1: float, x0: float) -> float:
    if not (x1 and x0) or x1 <= 0 or x0 <= 0:
        return 0.0
    return L * math.log(x1 / x0)


def kpi_tree(prev: pd.DataFrame, cur: pd.DataFrame) -> dict | None:
    """Decompose the revenue change between two periods into driver contributions ($)."""
    m0, m1 = metrics(prev), metrics(cur)
    r0, r1 = m0["revenue"], m1["revenue"]
    if r0 <= 0 or r1 <= 0:
        return None
    L = _logmean(r1, r0)
    c_orders = _contrib(L, m1["orders"], m0["orders"])
    c_aov = _contrib(L, m1["aov"], m0["aov"])
    node = lambda k, c: {"metric": k, "prev": m0[k], "cur": m1[k], "change": _pct(m1[k], m0[k]), "contribution": c}
    children_orders = []
    if m0["customers"] and m1["customers"]:
        lc = math.log(m1["customers"] / m0["customers"])
        lf = math.log(m1["orders_per_customer"] / m0["orders_per_customer"])
        lo = lc + lf
        # Split the orders contribution in proportion to the log changes (identified customers).
        share_c = lc / lo if abs(lo) > 1e-9 else 0.5
        children_orders = [node("customers", c_orders * share_c),
                           node("orders_per_customer", c_orders * (1 - share_c))]
    children_aov = [node("units_per_order", _contrib(L, m1["units_per_order"], m0["units_per_order"])),
                    node("price_per_unit", _contrib(L, m1["price_per_unit"], m0["price_per_unit"]))]
    return {"metric": "revenue", "prev": r0, "cur": r1, "change": _pct(r1, r0), "contribution": r1 - r0,
            "children": [dict(node("orders", c_orders), children=children_orders),
                         dict(node("aov", c_aov), children=children_aov)]}


def attribution(orders: pd.DataFrame, lines: pd.DataFrame, days: int = 90) -> dict:
    """Which segments explain the revenue change (mix-shift). Compares with the same period last
    year when the data goes back far enough (removes seasonality), else with the previous period."""
    ref = orders["order_date"].max()
    w = windows(ref, days)
    first = orders["order_date"].min()
    if covered(w["yoy"][0], first, days):
        base_key, base_label = "yoy", "the same period last year"
    elif covered(w["previous"][0], first, days):
        base_key, base_label = "previous", f"the previous {days} days"
    else:
        return {}
    w = {"current": w["current"], "previous": w[base_key]}
    o = orders.assign(customer_type=np.where(orders["is_returning"], "Returning",
                                             np.where(orders["customer_key"].notna(), "New", "Unknown")),
                      used_discount=np.where(orders["discount"] > 0, "Discounted", "Full price"))
    dims = {d: o for d in ("channel", "country", "customer_type", "used_discount") if d in o}
    l = lines.rename(columns={"line_revenue": "revenue"})
    for d in ("category", "product"):
        if d in l and l[d].notna().any():
            dims[d] = l
    total_delta = (slice_period(orders, *w["current"])["revenue"].sum()
                   - slice_period(orders, *w["previous"])["revenue"].sum())
    out = {"days": days, "base": base_key, "base_label": base_label, "total_delta": float(total_delta),
           "dimensions": {}}
    for dim, df in dims.items():
        cur = slice_period(df, *w["current"]).groupby(dim)["revenue"].sum()
        prev = slice_period(df, *w["previous"]).groupby(dim)["revenue"].sum()
        t = pd.DataFrame({"current": cur, "previous": prev}).fillna(0.0)
        t["delta"] = t["current"] - t["previous"]
        t["pct_change"] = np.where(t["previous"] > 0, t["current"] / t["previous"].replace(0, np.nan) - 1, np.nan)
        denom = t["delta"].sum()
        t["share_of_change"] = t["delta"] / denom if denom else np.nan
        t = t.sort_values("delta")
        out["dimensions"][dim] = t.reset_index().rename(columns={dim: "segment"}).to_dict("records")
    return out


def monthly(orders: pd.DataFrame) -> pd.DataFrame:
    o = orders.assign(month=orders["order_date"].dt.to_period("M").dt.to_timestamp())
    g = o.groupby("month")
    df = pd.DataFrame({
        "revenue": g["revenue"].sum(), "orders": g.size(),
        "customers": g["customer_key"].nunique(),
        "new_revenue": o[~o["is_returning"]].groupby("month")["revenue"].sum(),
        "returning_revenue": o[o["is_returning"]].groupby("month")["revenue"].sum(),
    }).fillna(0.0)
    df["aov"] = df["revenue"] / df["orders"]
    return df.reset_index()
