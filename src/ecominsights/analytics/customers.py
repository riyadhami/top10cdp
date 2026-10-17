"""Customer table, RFM segmentation, a lightweight CLV / churn estimate and top-10 lists.

RFM: Recency (days since last order), Frequency (orders), Monetary (revenue), each scored 1-5
by quintile rank. Segments follow the widely used 11-segment R x FM grid.

CLV: BG/NBD + Gamma-Gamma (see clv.py). If the model can't be fitted (too few customers or
repeat buyers), a heuristic is used: P(active) = exp(-recency / (2 x expected gap)) and
predicted_12m = P(active) x expected orders per year x customer AOV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_ACTIONS = {
    "Champions": "Reward and ask for reviews/referrals; early access to launches. Do not discount.",
    "Loyal": "Upsell higher-value products; loyalty program; ask for reviews.",
    "Potential Loyalist": "Membership / subscription offer; recommend complementary products.",
    "New Customers": "Onboarding sequence; drive the 2nd purchase within the typical gap.",
    "Promising": "Brand-story content and a small incentive for the next order.",
    "Need Attention": "Limited-time personalised offer based on past purchases.",
    "About to Sleep": "Re-engage with bestsellers and social proof before they lapse.",
    "At Risk": "Personal win-back email; strong offer on their favourite category.",
    "Can't Lose Them": "Highest-priority win-back: personal outreach, survey, best offer.",
    "Hibernating": "Low-cost reactivation; suppress from paid audiences if no response.",
    "Lost": "Exclude from most campaigns; one last-chance offer.",
}


def customer_table(orders: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    o = orders[orders["customer_key"].notna()].sort_values("order_date")
    if o.empty:
        return pd.DataFrame()
    ref = orders["order_date"].max()
    g = o.groupby("customer_key")
    c = pd.DataFrame({
        "first_order": g["order_date"].min(), "last_order": g["order_date"].max(),
        "orders": g.size(), "revenue": g["revenue"].sum(),
        "discount_total": g["discount"].sum(),
        "channels": g["channel"].nunique(),
    })
    c["aov"] = c["revenue"] / c["orders"]
    c["recency_days"] = (ref - c["last_order"]).dt.days
    c["tenure_days"] = (ref - c["first_order"]).dt.days
    gaps = o.groupby("customer_key")["order_date"].diff().dt.days
    c["avg_gap_days"] = gaps.groupby(o["customer_key"]).mean()
    for col in ("country", "channel"):
        if col in o:
            c[col] = g[col].first()
    first_orders = o[o["order_seq"] == 1].set_index("customer_key")
    c["first_order_discounted"] = first_orders["discount"].reindex(c.index).fillna(0) > 0
    if "discount_code" in first_orders:
        c["first_discount_code"] = first_orders["discount_code"].reindex(c.index)
    c["first_order_value"] = first_orders["revenue"].reindex(c.index)
    if "product" in lines and lines["product"].notna().any():
        fl = lines[lines["order_id"].isin(first_orders["order_id"]) & lines["customer_key"].notna()]
        top = fl.sort_values("line_revenue", ascending=False).drop_duplicates("customer_key")
        c["first_product"] = top.set_index("customer_key")["product"].reindex(c.index)
        fav = (lines[lines["customer_key"].notna()].groupby(["customer_key", "product"])["line_revenue"]
               .sum().reset_index().sort_values("line_revenue", ascending=False)
               .drop_duplicates("customer_key").set_index("customer_key")["product"])
        c["favorite_product"] = fav.reindex(c.index)
    if "cogs" in o and o["cogs"].notna().mean() > 0.5:
        c["gross_profit"] = c["revenue"] - g["cogs"].sum()
    return c


def _score(s: pd.Series, ascending: bool = True) -> pd.Series:
    """Quintile score 1..5 using ranks (robust to the many ties in frequency)."""
    r = s.rank(method="first", ascending=ascending, pct=True)
    return np.ceil(r * 5).clip(1, 5).astype(int)


def _segment(r: int, f: int, m: int) -> str:
    fm = round((f + m) / 2)
    if r >= 4 and fm >= 4:
        return "Champions"
    if r == 5 and f == 1:
        return "New Customers"
    if r >= 3 and fm >= 4:
        return "Loyal"
    if r >= 4 and fm >= 2:
        return "Potential Loyalist"
    if r == 4 and fm == 1:
        return "Promising"
    if r == 3 and fm == 3:
        return "Need Attention"
    if r == 3:
        return "About to Sleep"
    if r <= 2 and fm == 5:
        return "Can't Lose Them"
    if r <= 2 and fm >= 3:
        return "At Risk"
    if r == 2:
        return "Hibernating"
    return "Lost"


def rfm(customers: pd.DataFrame) -> pd.DataFrame:
    c = customers.copy()
    c["R"] = _score(c["recency_days"], ascending=False)
    c["F"] = _score(c["orders"])
    c["M"] = _score(c["revenue"])
    c["segment"] = [_segment(r, f, m) for r, f, m in zip(c["R"], c["F"], c["M"])]
    return c


def clv(customers: pd.DataFrame, orders: pd.DataFrame | None = None) -> pd.DataFrame:
    """BG/NBD + Gamma-Gamma when it can be fitted, else the recency/gap heuristic."""
    c = customers.copy()
    if orders is not None:
        from . import clv as clv_model
        try:
            fitted = clv_model.predict(orders)
        except Exception:  # noqa: BLE001
            fitted = None
        if fitted is not None:
            pred, params = fitted
            c = c.join(pred, how="left")
            c["churn_risk"] = 1 - c["p_active"]
            c["clv_method"] = "bg_nbd_gamma_gamma"
            c.attrs["clv_params"] = params
            return c
    store_gap = c["avg_gap_days"].median()
    if not np.isfinite(store_gap):
        store_gap = 90.0
    gap = c["avg_gap_days"].fillna(store_gap * 1.5).clip(lower=7)
    c["p_active"] = np.exp(-c["recency_days"] / (2 * gap)).clip(0, 1)
    c.loc[c["orders"] == 1, "p_active"] = np.exp(-c["recency_days"] / (2 * store_gap * 1.5)).clip(0, 1) * 0.6
    orders_per_year = (365 / gap).clip(upper=24)
    c["predicted_12m_value"] = c["p_active"] * orders_per_year * c["aov"]
    c["churn_risk"] = 1 - c["p_active"]
    c["clv_method"] = "heuristic"
    return c


def segment_summary(c: pd.DataFrame) -> pd.DataFrame:
    g = c.groupby("segment")
    s = pd.DataFrame({"customers": g.size(), "revenue": g["revenue"].sum(),
                      "avg_orders": g["orders"].mean(), "avg_recency_days": g["recency_days"].mean(),
                      "aov": g["aov"].mean(), "predicted_12m_value": g["predicted_12m_value"].sum()})
    s["share_customers"] = s["customers"] / s["customers"].sum()
    s["share_revenue"] = s["revenue"] / s["revenue"].sum()
    s["action"] = s.index.map(SEGMENT_ACTIONS)
    return s.sort_values("revenue", ascending=False).reset_index()


def top_lists(c: pd.DataFrame, n: int = 10) -> dict[str, pd.DataFrame]:
    cols = [x for x in ("segment", "orders", "revenue", "aov", "recency_days", "predicted_12m_value",
                        "expected_orders_12m", "p_active", "favorite_product", "country") if x in c]
    vip = c.sort_values("predicted_12m_value", ascending=False)
    at_risk = c[c["segment"].isin(["At Risk", "Can't Lose Them", "About to Sleep"])].sort_values(
        "revenue", ascending=False)
    winback = c[c["segment"].isin(["Hibernating", "Lost"]) & (c["orders"] > 1)].sort_values(
        "revenue", ascending=False)
    rising = c[(c["orders"] <= 2) & (c["recency_days"] <= 60)].sort_values("first_order_value", ascending=False)
    return {name: df.head(n)[cols].reset_index() for name, df in
            {"vip": vip, "at_risk": at_risk, "winback": winback, "rising": rising}.items()}


def repeat_behavior(c: pd.DataFrame, orders: pd.DataFrame) -> dict:
    ident = orders[orders["customer_key"].notna()].sort_values("order_date")
    second = ident[ident["order_seq"] == 2].set_index("customer_key")["order_date"]
    days_to_second = (second - c["first_order"].reindex(second.index)).dt.days
    ref = orders["order_date"].max()
    eligible = c[c["tenure_days"] >= 90]
    within90 = (days_to_second.reindex(eligible.index) <= 90).mean() if len(eligible) else np.nan
    return {
        "customers": int(len(c)),
        "repeat_customer_rate": float((c["orders"] > 1).mean()),
        "one_time_buyers": int((c["orders"] == 1).sum()),
        "median_days_to_second_order": float(days_to_second.median()) if len(days_to_second) else None,
        "p75_days_to_second_order": float(days_to_second.quantile(0.75)) if len(days_to_second) else None,
        "second_order_within_90d": float(within90) if np.isfinite(within90) else None,
        "top20_revenue_share": float(c["revenue"].sort_values(ascending=False)
                                     .head(max(1, int(len(c) * 0.2))).sum() / c["revenue"].sum()),
        "days_to_second_hist": days_to_second.dropna().astype(int).tolist(),
        "multi_channel_customers": int((c["channels"] > 1).sum()),
        "reference_date": str(ref.date()),
    }
