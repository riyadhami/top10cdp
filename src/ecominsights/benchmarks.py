"""Indicative DTC ecommerce benchmarks.

These are rough, commonly cited ranges for small/mid direct-to-consumer stores, meant as a
sanity check rather than a target. Edit them to match your category.
"""

from __future__ import annotations

BENCHMARKS = {
    "repeat_customer_rate": {"label": "Repeat customer rate", "low": 0.20, "typical": 0.27, "high": 0.35,
                             "higher_is_better": True, "fmt": "pct"},
    "second_order_within_90d": {"label": "2nd order within 90 days", "low": 0.10, "typical": 0.15, "high": 0.25,
                                "higher_is_better": True, "fmt": "pct"},
    "returning_revenue_share": {"label": "Revenue from returning customers", "low": 0.25, "typical": 0.35,
                                "high": 0.50, "higher_is_better": True, "fmt": "pct"},
    "top20_revenue_share": {"label": "Revenue from top 20% of customers", "low": 0.50, "typical": 0.65,
                            "high": 0.80, "higher_is_better": None, "fmt": "pct"},
    "discount_rate": {"label": "Discount rate (of gross sales)", "low": 0.03, "typical": 0.07, "high": 0.12,
                      "higher_is_better": False, "fmt": "pct"},
    "gross_margin": {"label": "Gross margin", "low": 0.40, "typical": 0.50, "high": 0.65,
                     "higher_is_better": True, "fmt": "pct"},
    "return_rate": {"label": "Return rate (units)", "low": 0.05, "typical": 0.10, "high": 0.20,
                    "higher_is_better": False, "fmt": "pct"},
}


def compare(values: dict[str, float | None]) -> list[dict]:
    rows = []
    for key, b in BENCHMARKS.items():
        v = values.get(key)
        if v is None:
            continue
        if b["higher_is_better"] is True:
            status = "below" if v < b["low"] else "above" if v > b["high"] else "in range"
            good = v >= b["typical"]
        elif b["higher_is_better"] is False:
            status = "above" if v > b["high"] else "below" if v < b["low"] else "in range"
            good = v <= b["typical"]
        else:
            status = "above" if v > b["high"] else "below" if v < b["low"] else "in range"
            good = status == "in range"
        rows.append({"metric": key, "label": b["label"], "value": v, "low": b["low"],
                     "typical": b["typical"], "high": b["high"], "status": status, "good": bool(good)})
    return rows
