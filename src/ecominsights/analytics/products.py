"""Product analytics: Pareto/ABC, momentum, basket affinity, margin, stock-outs, replenishment.

Basket affinity for a pair (A, B) over multi-item orders:
    support    = P(A and B)
    confidence = P(B | A)
    lift       = P(A and B) / (P(A) * P(B))      (>1 means bought together more than chance)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def product_table(lines: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    if "product" not in lines or lines["product"].isna().all():
        return pd.DataFrame()
    ref = lines["order_date"].max()
    cur = lines["order_date"] > ref - pd.Timedelta(days=days)
    prev = (lines["order_date"] > ref - pd.Timedelta(days=2 * days)) & ~cur
    g = lines.groupby("product")
    t = pd.DataFrame({
        "revenue": g["line_revenue"].sum(), "units": g["quantity"].sum(),
        "orders": g["order_key"].nunique(), "customers": g["customer_key"].nunique(),
        "rev_current": lines[cur].groupby("product")["line_revenue"].sum(),
        "rev_previous": lines[prev].groupby("product")["line_revenue"].sum(),
    }).fillna({"rev_current": 0.0, "rev_previous": 0.0})
    if "category" in lines:
        t["category"] = g["category"].first()
    t["growth"] = np.where(t["rev_previous"] > 0, t["rev_current"] / t["rev_previous"].replace(0, np.nan) - 1, np.nan)
    t = t.sort_values("revenue", ascending=False)
    t["revenue_share"] = t["revenue"] / t["revenue"].sum()
    t["cum_share"] = t["revenue_share"].cumsum()
    t["abc"] = np.select([t["cum_share"] <= 0.8, t["cum_share"] <= 0.95], ["A", "B"], "C")
    t.loc[t.index[0], "abc"] = "A"
    if "line_cost" in lines and lines["line_cost"].notna().mean() > 0.5:
        cost = g["line_cost"].sum(min_count=1)
        t["gross_profit"] = t["revenue"] - cost
        t["margin"] = t["gross_profit"] / t["revenue"]
    return t.reset_index()


def pareto_summary(t: pd.DataFrame) -> dict:
    if t.empty:
        return {}
    n80 = int((t["cum_share"] < 0.8).sum()) + 1
    return {"products": int(len(t)), "products_for_80pct": n80, "share_for_80pct": n80 / len(t),
            "top_product": t.iloc[0]["product"], "top_product_share": float(t.iloc[0]["revenue_share"])}


def basket_pairs(lines: pd.DataFrame, min_pair_orders: int = 8, max_products: int = 150) -> pd.DataFrame:
    if "product" not in lines:
        return pd.DataFrame()
    top = lines.groupby("product")["line_revenue"].sum().nlargest(max_products).index
    l = lines[lines["product"].isin(top)][["order_key", "product"]].drop_duplicates()
    sizes = l.groupby("order_key").size()
    total_orders = lines["order_key"].nunique()
    multi = l[l["order_key"].isin(sizes[sizes > 1].index)]
    if multi.empty:
        return pd.DataFrame()
    item_orders = l.groupby("product")["order_key"].nunique()
    counts: dict[tuple[str, str], int] = {}
    for _, prods in multi.groupby("order_key")["product"]:
        for a, b in combinations(sorted(prods), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    rows = []
    for (a, b), n in counts.items():
        if n < min_pair_orders:
            continue
        pa, pb = item_orders[a] / total_orders, item_orders[b] / total_orders
        sup = n / total_orders
        lift = sup / (pa * pb)
        # Report the direction with the higher confidence: "customers who buy A also buy B".
        conf_ab, conf_ba = n / item_orders[a], n / item_orders[b]
        if conf_ba > conf_ab:
            a, b, conf_ab = b, a, conf_ba
        rows.append({"antecedent": a, "consequent": b, "pair_orders": n, "support": sup,
                     "confidence": conf_ab, "lift": lift,
                     "antecedent_orders_without_consequent": int(item_orders[a] - n)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["lift", "pair_orders"], ascending=False).reset_index(drop=True)


def stockouts(lines: pd.DataFrame, top_n: int = 30, min_gap_days: int = 10,
              min_expected_sales_days: float = 6.0) -> pd.DataFrame:
    """Zero-sales runs that are very unlikely given the product's local selling rate.

    For each gap, the local rate is the share of days with a sale in the 60 days around it.
    Under a Poisson model P(no sale for g days) = exp(-rate * g); we flag gaps where the
    expected number of selling days is >= 6 (p < 0.25%)."""
    if "product" not in lines:
        return pd.DataFrame()
    ref = lines["order_date"].max().normalize()
    top = lines.groupby("product")["line_revenue"].sum().nlargest(top_n).index
    rows = []
    for p in top:
        pl = lines[lines["product"] == p]
        d = pl["order_date"].dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)
        if len(d) < 20:
            continue
        rev_by_day = pl.groupby(pl["order_date"].dt.normalize())["line_revenue"].sum()
        ends = list(zip(d.iloc[:-1], d.iloc[1:])) + [(d.iloc[-1], ref + pd.Timedelta(days=1))]
        for start, end in ends:
            g = (end - start).days - 1
            if g < min_gap_days:
                continue
            ongoing = end > ref
            lo, hi = start - pd.Timedelta(days=60), end + pd.Timedelta(days=60)
            window = d[((d > lo) & (d <= start)) | ((d >= end) & (d < hi))]
            window_days = 60 + (0 if ongoing else min(60, (ref - end).days + 1))
            rate = len(window) / max(window_days, 1)
            if rate * g < min_expected_sales_days:
                continue
            local = rev_by_day[(rev_by_day.index > lo) & (rev_by_day.index < hi)].sum() / max(window_days, 1)
            rows.append({"product": p, "gap_start": (start + pd.Timedelta(days=1)).date(),
                         "gap_end": (min(end, ref) - pd.Timedelta(days=0 if ongoing else 1)).date(),
                         "gap_days": g, "expected_sales_days": round(rate * g, 1),
                         "typical_gap_days": round(1 / rate, 1) if rate else None,
                         "ongoing": bool(ongoing), "est_lost_revenue": float(local * g)})
    return pd.DataFrame(rows).sort_values("est_lost_revenue", ascending=False) if rows else pd.DataFrame()


def replenishment(lines: pd.DataFrame, min_customers: int = 15) -> pd.DataFrame:
    """Median days between repeat purchases of the same product by the same customer."""
    if "product" not in lines:
        return pd.DataFrame()
    l = lines[lines["customer_key"].notna()][["customer_key", "product", "order_date"]].drop_duplicates()
    l = l.sort_values("order_date")
    l["gap"] = l.groupby(["customer_key", "product"])["order_date"].diff().dt.days
    g = l.dropna(subset=["gap"]).groupby("product")
    t = pd.DataFrame({"repeat_customers": g["customer_key"].nunique(), "median_gap_days": g["gap"].median(),
                      "iqr_days": g["gap"].quantile(0.75) - g["gap"].quantile(0.25)})
    buyers = l.groupby("product")["customer_key"].nunique()
    t["repurchase_rate"] = t["repeat_customers"] / buyers.reindex(t.index)
    t = t[t["repeat_customers"] >= min_customers]
    # Regular cadence = tight spread relative to the median gap.
    t["regularity"] = 1 - (t["iqr_days"] / t["median_gap_days"]).clip(0, 1)
    return t.sort_values(["repeat_customers", "regularity"], ascending=False).reset_index()


def due_for_reorder(lines: pd.DataFrame, repl: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    """Customers whose last purchase of a replenishable product is ~one cadence ago."""
    if repl.empty:
        return pd.DataFrame()
    ref = lines["order_date"].max()
    regular = repl[(repl["regularity"] >= 0.15) & (repl["repurchase_rate"] >= 0.1)].set_index("product")["median_gap_days"]
    l = lines[lines["customer_key"].notna() & lines["product"].isin(regular.index)]
    last = l.groupby(["customer_key", "product"])["order_date"].max().reset_index()
    last["days_since"] = (ref - last["order_date"]).dt.days
    last["cadence"] = last["product"].map(regular)
    due = last[(last["days_since"] >= last["cadence"] - window_days) & (last["days_since"] <= last["cadence"] + window_days)]
    return due.sort_values("days_since")


def returns_table(returns: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if returns is None or returns.empty or "product" not in returns or table.empty:
        return pd.DataFrame()
    r = returns.groupby("product").agg(returned_units=("quantity", "sum"), returned_value=("value", "sum"))
    t = table.set_index("product")[["units", "revenue"]].join(r, how="inner")
    t["return_rate"] = t["returned_units"] / (t["units"] + t["returned_units"])
    return t.sort_values("returned_value", ascending=False).reset_index()
