"""Turn analysis output into insight candidates and rank them.

Every insight carries an estimated annual $ impact computed with a transparent formula
(shown in `impact_formula`), a confidence in [0, 1] driven by sample size / significance,
and actionability and effort ratings (1-3).

    priority score = impact x confidence x actionability / effort
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .analytics.cohorts import MIN_COHORT
from .benchmarks import BENCHMARKS

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class Insight:
    id: str
    category: str          # customers | products | revenue | operations | marketing | risk
    kind: str              # opportunity | risk | win
    title: str
    finding: str
    action: str
    impact: float          # estimated annual $ impact
    impact_formula: str
    confidence: float
    actionability: int = 2
    effort: int = 2
    evidence: dict = field(default_factory=dict)
    chart: str | None = None
    score: float = 0.0
    rank: int | None = None


def size_confidence(n: float, half: float = 40) -> float:
    """Saturating confidence from sample size: ~0.6 at n=half, -> 0.95."""
    return float(min(0.95, 0.3 + 0.65 * (1 - math.exp(-max(n, 0) / half))))


def money(x: float, cur: str = "USD") -> str:
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "CAD": "C$", "AUD": "A$"}.get(cur, "")
    a = abs(x)
    s = f"{a / 1e6:.1f}M" if a >= 1e6 else f"{a / 1e3:.1f}k" if a >= 1e4 else f"{a:,.0f}"
    return ("-" if x < 0 else "") + (sym + s if sym else f"{s} {cur}")


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


def _df(r: dict, key: str) -> pd.DataFrame:
    v = r.get(key)
    return v if isinstance(v, pd.DataFrame) else pd.DataFrame()


def generate_insights(r: dict, top_n: int = 10) -> list[dict]:
    cur = r["meta"]["currency"]
    m = lambda x: money(x, cur)
    out: list[Insight] = []
    snap = r.get("snapshot") or {}
    m90 = (snap.get("90d") or {}).get("metrics", {})
    m365 = (snap.get("365d") or {}).get("metrics", {})
    rev365 = (m365.get("revenue") or {}).get("value") or 0.0
    cust = _df(r, "customers")
    rep = r.get("repeat") or {}

    # ---- Revenue drivers (KPI tree, 90d) ----
    tree = (snap.get("90d") or {}).get("tree")
    if tree:
        leaves = [c for ch in tree["children"] for c in (ch.get("children") or [ch])]
        worst = min(leaves, key=lambda x: x["contribution"])
        best = max(leaves, key=lambda x: x["contribution"])
        label = {"customers": "active customers", "orders_per_customer": "orders per customer",
                 "units_per_order": "units per order", "price_per_unit": "average price per unit"}
        if worst["contribution"] < -0.02 * tree["prev"]:
            out.append(Insight(
                "driver_down", "revenue", "risk",
                f"{label.get(worst['metric'], worst['metric']).capitalize()} is the biggest drag on revenue",
                f"Revenue moved {pct(tree['change'] or 0, 1)} vs the previous 90 days ({m(tree['contribution'])}). "
                f"The KPI tree attributes {m(worst['contribution'])} to {label.get(worst['metric'])} "
                f"({pct(worst['change'] or 0, 1)}).",
                {"customers": "Fix acquisition: check paid traffic, conversion rate and top landing pages.",
                 "orders_per_customer": "Re-engage existing customers: post-purchase and win-back flows.",
                 "units_per_order": "Bundles, 'frequently bought together' and free-shipping thresholds.",
                 "price_per_unit": "Check discount depth and product mix shift toward cheaper items."}.get(worst["metric"], ""),
                abs(worst["contribution"]) * 4 * 0.5, "|90-day contribution| x 4 (annualised) x 50% recoverable",
                size_confidence(m90.get("orders", {}).get("value", 0), 200), 3, 2,
                {"tree": tree}, "kpi_tree"))
        if best["contribution"] > 0.05 * tree["prev"]:
            out.append(Insight(
                "driver_up", "revenue", "win",
                f"Growth is coming from {label.get(best['metric'], best['metric'])}",
                f"{label.get(best['metric']).capitalize()} changed {pct(best['change'] or 0, 1)}, adding "
                f"{m(best['contribution'])} over the last 90 days vs the prior 90.",
                "Double down on what drove it and protect it (inventory, budget, messaging).",
                best["contribution"] * 4 * 0.2, "90-day contribution x 4 x 20% further upside",
                size_confidence(m90.get("orders", {}).get("value", 0), 200), 2, 2, {"tree": tree}, "kpi_tree"))

    # ---- Mix-shift attribution ----
    attr = r.get("attribution") or {}
    for dim, rows in (attr.get("dimensions") or {}).items():
        if dim == "product" or not rows:
            continue
        worst = rows[0]
        prev_total = sum(x["previous"] for x in rows) or 1
        if worst["delta"] < -0.05 * prev_total and worst["previous"] > 0:
            out.append(Insight(
                f"segment_drop_{dim}", "revenue", "risk",
                f"{dim.replace('_', ' ').title()} '{worst['segment']}' lost {m(-worst['delta'])} in 90 days",
                f"Revenue from {worst['segment']} fell {pct(-(worst['pct_change'] or 0))} vs the prior 90 days "
                f"({m(worst['previous'])} -> {m(worst['current'])}).",
                f"Investigate {worst['segment']}: traffic, pricing, stock and campaign changes in that segment.",
                -worst["delta"] * 4 * 0.3, "|90-day segment drop| x 4 x 30% recoverable",
                size_confidence(worst["previous"] / max(m90.get("aov", {}).get("value") or 1, 1), 60), 2, 2,
                {"dimension": dim, "segment": worst}, f"attribution:{dim}"))

    # ---- Customers ----
    if not cust.empty:
        risk = cust[cust["segment"].isin(["At Risk", "Can't Lose Them"])]
        if len(risk) >= 5:
            annual = (risk["revenue"] / (risk["tenure_days"].clip(lower=365) / 365)).sum()
            out.append(Insight(
                "at_risk", "customers", "risk",
                f"{len(risk)} high-value customers are slipping away",
                f"'At Risk' and 'Can't Lose Them' customers spent {m(risk['revenue'].sum())} historically "
                f"({pct(risk['revenue'].sum() / cust['revenue'].sum())} of customer revenue) but have not "
                f"ordered in a median of {int(risk['recency_days'].median())} days.",
                "Personal win-back sequence (3 emails over 10 days) with their favourite product; "
                "survey why they stopped; exclude them from generic promos.",
                annual * 0.15, "their annualised spend x 15% win-back rate",
                size_confidence(len(risk), 30), 3, 1,
                {"customers": len(risk), "historic_revenue": float(risk["revenue"].sum())}, "segments"))

        champions = cust[cust["segment"] == "Champions"]
        if len(champions) >= 5:
            out.append(Insight(
                "champions", "customers", "opportunity",
                f"{len(champions)} Champions drive {pct(champions['revenue'].sum() / cust['revenue'].sum())} of revenue",
                f"Champions are {pct(len(champions) / len(cust))} of customers with an AOV of "
                f"{m(champions['aov'].mean())} and {champions['orders'].mean():.1f} orders on average.",
                "Launch a referral / VIP program for Champions (early access, no discounts needed).",
                champions["predicted_12m_value"].sum() * 0.10, "Champions' predicted 12-month value x 10% referral lift",
                size_confidence(len(champions), 30), 2, 2, {"customers": len(champions)}, "segments"))

        rr = rep.get("repeat_customer_rate")
        s90 = rep.get("second_order_within_90d")
        bench = BENCHMARKS["repeat_customer_rate"]["typical"]
        new_per_year = (m365.get("new_customers") or {}).get("value") or 0
        ret_aov = cust.loc[cust["orders"] > 1, "aov"].mean()
        if rr is not None and rr < bench and new_per_year:
            gap = bench - rr
            out.append(Insight(
                "repeat_rate", "customers", "opportunity",
                f"Only {pct(rr)} of customers come back for a second order",
                f"The typical DTC repeat rate is ~{pct(bench)}. Median time to a 2nd order is "
                f"{int(rep.get('median_days_to_second_order') or 0)} days"
                + (f"; {pct(s90)} reorder within 90 days." if s90 is not None else "."),
                f"Post-purchase flow timed around day {int(rep.get('median_days_to_second_order') or 30)}: "
                "usage tips, cross-sell, then a time-boxed 2nd-order incentive.",
                gap * new_per_year * (ret_aov if np.isfinite(ret_aov) else 0),
                "(benchmark - repeat rate) x new customers / yr x returning AOV",
                size_confidence(len(cust), 150), 3, 2, {"repeat_rate": rr, "benchmark": bench}, "repeat_hist"))

    dc = r.get("discount_cohorts") or {}
    if dc and dc.get("p_value", 1) < 0.1:
        d, f = dc["discounted"], dc["full_price"]
        if d["repeat_rate"] < f["repeat_rate"]:
            per_year = d["customers"] / _years(r)
            out.append(Insight(
                "discount_cohort", "marketing", "risk",
                "Discount-acquired customers are less loyal",
                f"Customers whose first order used a discount repeat at {pct(d['repeat_rate'])} vs "
                f"{pct(f['repeat_rate'])} for full-price buyers (p = {dc['p_value']:.3f}); their "
                f"{dc['days']}-day value is {m(d['avg_value'])} vs {m(f['avg_value'])}.",
                "Swap %-off welcome codes for a free gift or free-shipping threshold; reserve discounts for win-back.",
                (f["avg_value"] - d["avg_value"]) * per_year * 0.3,
                "(full-price - discounted 180d value) x discounted customers / yr x 30% closable",
                min(0.95, 1 - dc["p_value"]), 3, 1, dc, "discount_cohorts"))

    fp = _df(r, "first_product")
    if not fp.empty and len(fp) >= 2:
        top = fp.iloc[0]
        val_col = [c for c in fp.columns if c.startswith("avg_value_")][0]
        if top["value_index"] >= 1.25 and top["customers"] >= MIN_COHORT:
            base = fp[val_col].mean()
            new_per_year = (m365.get("new_customers") or {}).get("value") or 0
            out.append(Insight(
                "gateway_product", "marketing", "opportunity",
                f"'{top['first_product']}' is your best gateway product",
                f"Customers whose first purchase is {top['first_product']} are worth {m(top[val_col])} in their "
                f"first {val_col.split('_')[-1]} ({top['value_index']:.1f}x the average) and repeat at "
                f"{pct(top['repeat_rate'])}.",
                f"Lead acquisition ads, landing pages and the welcome offer with {top['first_product']}.",
                0.10 * new_per_year * max(top[val_col] - base, 0),
                "10% of new customers shifted x (gateway value - average value)",
                size_confidence(top["customers"], 40), 3, 2, {"row": top.to_dict()}, "first_product"))

    # ---- Products ----
    ptab = _df(r, "products")
    basket = _df(r, "basket")
    if not basket.empty:
        avg_line = (ptab.set_index("product")["revenue"] / ptab.set_index("product")["orders"]) if not ptab.empty \
            else pd.Series(dtype=float)
        cand = basket[basket["lift"] >= 1.5].head(10).copy()
        if len(cand):
            cand["impact"] = (cand["antecedent_orders_without_consequent"] * 0.10
                              * cand["consequent"].map(avg_line).fillna(0) / _years(r))
            b = cand.sort_values("impact", ascending=False).iloc[0]
            out.append(Insight(
                "basket_pair", "products", "opportunity",
                f"Bundle '{b['antecedent']}' with '{b['consequent']}'",
                f"{pct(b['confidence'])} of orders with {b['antecedent']} also include {b['consequent']} "
                f"(lift {b['lift']:.1f}x). {int(b['antecedent_orders_without_consequent'])} orders bought "
                f"{b['antecedent']} without it.",
                "Add a 'frequently bought together' bundle at a small discount and a cart upsell.",
                b["impact"], "orders / yr with A but not B x 10% attach uplift x avg B line value",
                size_confidence(b["pair_orders"], 30), 3, 1, {"pair": b.to_dict()}, "basket"))

    so = _df(r, "stockouts")
    if not so.empty:
        s = so.iloc[0]
        out.append(Insight(
            "stockout", "operations", "risk",
            f"'{s['product']}' looks out of stock{' right now' if s['ongoing'] else ''}",
            f"No sales for {s['gap_days']} days ({s['gap_start']} -> {s['gap_end']}) although at its usual pace it "
            f"would have sold on ~{s['expected_sales_days']:.0f} of those days. Estimated lost revenue: "
            f"{m(s['est_lost_revenue'])}.",
            "Set reorder points from sales velocity and lead time; enable back-in-stock alerts.",
            s["est_lost_revenue"] * (2 if s["ongoing"] else 1), "gap days x average daily revenue (x2 if ongoing)",
            0.7, 3, 2, {"stockouts": so.head(5).to_dict("records")}, "stockouts"))

    if not ptab.empty:
        dec = ptab[ptab["abc"].isin(["A", "B"]) & (ptab["growth"] < -0.3) & (ptab["rev_previous"] > 0)]
        if len(dec):
            d = dec.sort_values("growth").iloc[0]
            out.append(Insight(
                f"declining_{d['product']}", "products", "risk",
                f"Key product '{d['product']}' is declining fast",
                f"{d['product']} revenue fell {pct(-d['growth'])} in the last 90 days vs the prior 90 "
                f"({m(d['rev_previous'])} -> {m(d['rev_current'])}).",
                "Check stock, price changes, reviews and competitor pricing; refresh creative or retire it.",
                (d["rev_previous"] - d["rev_current"]) * 4 * 0.4, "90-day drop x 4 x 40% recoverable",
                size_confidence(d["orders"], 60), 2, 2, {"product": d.to_dict()}, "product_momentum"))
        rising = ptab[(ptab["growth"] > 0.5) & (ptab["rev_current"] > 0.02 * ptab["rev_current"].sum())]
        if len(rising):
            u = rising.sort_values("growth", ascending=False).iloc[0]
            out.append(Insight(
                f"rising_{u['product']}", "products", "win",
                f"'{u['product']}' is breaking out",
                f"{u['product']} grew {pct(u['growth'])} vs the prior 90 days to {m(u['rev_current'])}.",
                "Secure inventory, feature it on the homepage and in ads.",
                u["rev_current"] * 4 * 0.2, "current 90-day revenue x 4 x 20% upside",
                size_confidence(u["orders"], 60), 2, 1, {"product": u.to_dict()}, "product_momentum"))
        if "margin" in ptab:
            low = ptab[(ptab["margin"] < 0.3) & (ptab["revenue_share"] > 0.03)]
            if len(low):
                lw = low.sort_values("revenue", ascending=False).iloc[0]
                out.append(Insight(
                    f"low_margin_{lw['product']}", "products", "risk",
                    f"'{lw['product']}' sells well but earns little",
                    f"{lw['product']} is {pct(lw['revenue_share'])} of revenue at a {pct(lw['margin'])} gross margin.",
                    "Raise price, renegotiate COGS, or use it as a bundle anchor rather than a promo item.",
                    lw["revenue"] / _years(r) * (0.4 - lw["margin"]) * 0.5,
                    "annual revenue x (40% target - margin) x 50% closable",
                    0.8, 2, 2, {"product": lw.to_dict()}, "margin"))
        top_share = ptab.iloc[0]["revenue_share"]
        if top_share > 0.3:
            out.append(Insight(
                "concentration", "risk", "risk",
                f"{pct(top_share)} of revenue depends on one product",
                f"{ptab.iloc[0]['product']} alone drives {pct(top_share)} of revenue.",
                "Diversify: develop the #2-#5 products and secure a second supplier.",
                rev365 * top_share * 0.05, "exposed annual revenue x 5% risk",
                0.8, 1, 3, {}, "pareto"))

    due = _df(r, "due_reorder")
    repl = _df(r, "replenishment")
    if not due.empty and not repl.empty:
        top = repl.iloc[0]
        avg_line = ptab.set_index("product")["revenue"] / ptab.set_index("product")["orders"] if not ptab.empty else None
        val = float(due["product"].map(avg_line).fillna(0).mean()) if avg_line is not None else 0
        out.append(Insight(
            "replenishment", "marketing", "opportunity",
            f"{len(due)} customers are due to reorder now",
            f"Repeat buyers of {top['product']} reorder every ~{top['median_gap_days']:.0f} days. "
            f"{len(due)} customer-product pairs are inside their reorder window today.",
            f"Automated replenishment reminder ~{max(int(top['median_gap_days']) - 5, 1)} days after purchase; offer subscribe & save.",
            len(due) * val * 0.1 * 26, "customers due x avg line value x 10% incremental conversion x 26 two-week cycles",
            size_confidence(top["repeat_customers"], 40), 3, 1,
            {"cadence": repl.head(5).to_dict("records")}, "replenishment"))

    # ---- Channels ----
    ch = _df(r, "channels")
    if len(ch) > 1:
        g = ch.dropna(subset=["growth"]).sort_values("growth")
        if len(g):
            lo, hi = g.iloc[0], g.iloc[-1]
            if lo["growth"] < -0.1:
                out.append(Insight(
                    f"channel_down_{lo['channel']}", "revenue", "risk",
                    f"{lo['channel']} revenue is down {pct(-lo['growth'])}",
                    f"{lo['channel']} made {m(lo['revenue_current'])} in the last 90 days vs {m(lo['revenue_previous'])}.",
                    f"Audit {lo['channel']}: listings, ads, buy box / ranking and stock levels.",
                    (lo["revenue_previous"] - lo["revenue_current"]) * 4 * 0.3, "90-day drop x 4 x 30%",
                    0.7, 2, 2, {"channel": lo.to_dict()}, "channels"))
            if hi["growth"] > 0.15:
                out.append(Insight(
                    f"channel_up_{hi['channel']}", "revenue", "win",
                    f"{hi['channel']} is your fastest-growing channel (+{pct(hi['growth'])})",
                    f"{hi['channel']} grew to {m(hi['revenue_current'])} in the last 90 days "
                    f"({pct(hi['share'])} of all-time revenue).",
                    f"Shift budget and inventory allocation toward {hi['channel']} while it compounds.",
                    hi["revenue_current"] * 4 * 0.15, "current 90-day revenue x 4 x 15% upside",
                    0.7, 2, 2, {"channel": hi.to_dict()}, "channels"))

    # ---- Seasonality / timing ----
    seas = r.get("seasonality") or {}
    ref = pd.Timestamp(snap.get("reference_date")) if snap.get("reference_date") else None
    if seas and ref is not None:
        idx = seas["month_index"]
        upcoming = [(ref.month + k - 1) % 12 + 1 for k in (1, 2)]
        peak = max(upcoming, key=lambda mo: idx[mo - 1] or 0)
        if (idx[peak - 1] or 0) >= 1.2:
            out.append(Insight(
                "season_peak", "operations", "opportunity",
                f"Peak season ahead: {MONTHS[peak - 1]} runs {pct(idx[peak - 1] - 1)} above average",
                f"Historically {MONTHS[peak - 1]} revenue is {idx[peak - 1]:.2f}x an average month.",
                "Lock inventory for top sellers, pre-build campaigns and gift bundles 4-6 weeks ahead.",
                rev365 / 12 * (idx[peak - 1] - 1) * 0.15, "avg month x seasonal uplift x 15% captured",
                0.75, 3, 2, {"month_index": idx}, "seasonality"))

    wd = r.get("weekday") or {}
    if wd.get("dow_index"):
        di = wd["dow_index"]
        best = int(np.argmax(di))
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if di[best] >= 1.15:
            out.append(Insight(
                "best_day", "marketing", "opportunity",
                f"{days[best]} is your strongest day ({di[best]:.2f}x average)",
                f"Average revenue on {days[best]} is {pct(di[best] - 1)} above the daily mean"
                + (" — see the hour heatmap for the peak hours." if wd.get("has_time") else "."),
                f"Schedule email/SMS sends and ad budget boosts for {days[best]} peaks.",
                rev365 / 7 * 0.03, "annual revenue / 7 x 3% timing uplift", 0.6, 2, 1, {}, "weekday"))

    an = _df(r, "anomalies")
    if not an.empty and ref is not None:
        recent = an[(an["kind"] == "dip") & (pd.to_datetime(an["date"]) >= ref - pd.Timedelta(days=30))]
        if len(recent):
            lost = float(-recent["delta"].sum())
            out.append(Insight(
                "recent_dip", "operations", "risk",
                f"{len(recent)} abnormal revenue dip(s) in the last 30 days",
                f"Daily revenue fell well below the expected level on "
                f"{', '.join(pd.to_datetime(recent['date']).dt.strftime('%b %d'))} ({m(lost)} below expected).",
                "Check site uptime, checkout errors, payment failures, ad account and stock on those dates.",
                lost, "sum of (expected - actual) on dip days", 0.6, 2, 1, {}, "anomalies"))

    # ---- Benchmarks ----
    dr = (m365.get("discount_rate") or {}).get("value")
    if dr is not None and dr > BENCHMARKS["discount_rate"]["high"]:
        out.append(Insight(
            "discount_depth", "marketing", "risk", f"Discounts eat {pct(dr, 1)} of gross sales",
            f"That is above the ~{pct(BENCHMARKS['discount_rate']['typical'])} typical for DTC stores.",
            "Tighten code eligibility, cap stacking, move to threshold offers (spend X get Y).",
            (dr - BENCHMARKS["discount_rate"]["typical"]) * rev365 * 0.5,
            "(discount rate - typical) x annual revenue x 50%", 0.8, 3, 1, {}, None))

    fc = _df(r, "forecast")
    wk = _df(r, "weekly")
    if not fc.empty and len(wk) >= 13:
        nxt, last = fc["forecast"].sum(), wk["revenue"].tail(13).sum()
        out.append(Insight(
            "forecast", "revenue", "win" if nxt >= last else "risk",
            f"Next 13 weeks forecast: {m(nxt)} ({'+' if nxt >= last else ''}{pct(nxt / last - 1)} vs last 13)",
            f"Holt-Winters forecast on weekly revenue; 80% interval {m(fc['lower'].sum())} - {m(fc['upper'].sum())}.",
            "Use it to set inventory and cash plans; investigate if actuals fall below the lower band.",
            abs(nxt - last) * 0.1, "|forecast change| x 10% (planning value)", 0.5, 1, 1, {}, "forecast"))

    for i in out:
        i.impact = float(max(i.impact, 0.0)) if np.isfinite(i.impact) else 0.0
        i.score = i.impact * i.confidence * i.actionability / i.effort
    out.sort(key=lambda i: i.score, reverse=True)
    for k, i in enumerate(out, 1):
        i.rank = k
    return [asdict(i) for i in out]


def _years(r: dict) -> float:
    q = r["meta"]["quality"]
    try:
        days = (pd.Timestamp(q["date_max"]) - pd.Timestamp(q["date_min"])).days
    except Exception:  # noqa: BLE001
        return 1.0
    return max(days / 365, 0.25)
