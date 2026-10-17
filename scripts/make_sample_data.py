"""Generate realistic sample exports (Shopify orders, Amazon orders, a COGS file) with planted
patterns the engine should find:

- Nov/Dec seasonality and a weekend lift
- "Starter Kit" as a gateway first product (much higher repeat rate)
- customers acquired with WELCOME20 repeat less
- Coffee Grinder + Whole Bean Coffee bought together (high lift)
- Whole Bean Coffee replenishment every ~30 days
- "Cold Brew Bottle" stock-out for ~5 weeks
- "Ceramic Mug" declining over the last 6 months

Usage: uv run python scripts/make_sample_data.py [out_dir]
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
START, END = datetime(2024, 10, 1), datetime(2026, 9, 30)

PRODUCTS = {  # name: (sku, category, price, unit_cost)
    "Whole Bean Coffee": ("BEAN-01", "Coffee", 18.0, 6.5),
    "Ground Coffee": ("GRND-01", "Coffee", 16.0, 6.0),
    "Coffee Grinder": ("GRIND-01", "Equipment", 79.0, 41.0),
    "Starter Kit": ("KIT-01", "Bundles", 59.0, 27.0),
    "Pour Over Set": ("POUR-01", "Equipment", 45.0, 19.0),
    "French Press": ("PRESS-01", "Equipment", 39.0, 17.0),
    "Cold Brew Bottle": ("COLD-01", "Equipment", 29.0, 11.0),
    "Ceramic Mug": ("MUG-01", "Merch", 22.0, 7.0),
    "Travel Tumbler": ("TUMB-01", "Merch", 32.0, 22.5),
    "Paper Filters": ("FILT-01", "Accessories", 7.0, 1.5),
    "Decaf Blend": ("DECAF-01", "Coffee", 19.0, 7.0),
    "Espresso Roast": ("ESP-01", "Coffee", 20.0, 7.5),
    "Gift Card": ("GIFT-50", "Gift", 50.0, 50.0),
    "Milk Frother": ("FROTH-01", "Equipment", 35.0, 24.0),
    "Tote Bag": ("TOTE-01", "Merch", 15.0, 4.0),
}
NAMES = list(PRODUCTS)
BASE_W = np.array([10, 6, 3, 4, 3, 3, 3, 4, 2, 4, 2, 4, 1, 2, 1], dtype=float)
STOCKOUT = (datetime(2026, 2, 10), datetime(2026, 3, 18))
COUNTRIES = ["US", "US", "US", "US", "CA", "GB", "AU", "DE"]
PROVINCES = {"US": ["CA", "NY", "TX", "WA", "IL"], "CA": ["ON", "BC"], "GB": ["ENG"], "AU": ["NSW"], "DE": ["BE"]}


def season(d: datetime) -> float:
    m = {11: 1.6, 12: 1.9, 1: 0.8, 2: 0.85, 7: 0.9, 8: 0.9}.get(d.month, 1.0)
    return m * (1.25 if d.weekday() >= 5 else 1.0)


def product_weights(d: datetime) -> np.ndarray:
    w = BASE_W.copy()
    if STOCKOUT[0] <= d <= STOCKOUT[1]:
        w[NAMES.index("Cold Brew Bottle")] = 0
    months_to_end = (END - d).days / 30
    if months_to_end < 6:  # mug decline
        w[NAMES.index("Ceramic Mug")] *= max(0.15, months_to_end / 6)
    return w / w.sum()


def random_date(lo: datetime, hi: datetime) -> datetime:
    while True:
        d = lo + timedelta(seconds=float(rng.uniform(0, (hi - lo).total_seconds())))
        growth = 0.6 + 0.8 * (d - START).days / (END - START).days
        if rng.uniform() < season(d) * growth / 3.0:
            hour = int(np.clip(rng.normal(20, 3), 6, 23))
            return d.replace(hour=hour, minute=int(rng.integers(0, 60)))


def basket(d: datetime, first_product: str | None = None) -> list[tuple[str, int]]:
    items = [first_product] if first_product else []
    n = 1 + rng.poisson(0.6)
    w = product_weights(d)
    while len(items) < n:
        p = NAMES[rng.choice(len(NAMES), p=w)]
        if p not in items:
            items.append(p)
    if "Coffee Grinder" in items and "Whole Bean Coffee" not in items and rng.uniform() < 0.7:
        items.append("Whole Bean Coffee")
    if "Pour Over Set" in items and "Paper Filters" not in items and rng.uniform() < 0.6:
        items.append("Paper Filters")
    return [(p, int(1 + rng.poisson(0.3 if PRODUCTS[p][2] > 30 else 0.8))) for p in items]


def shopify(n_customers: int = 880) -> pd.DataFrame:
    rows, order_no = [], 1001
    for c in range(n_customers):
        email = f"customer{c:04d}@mail{c % 7}.com"
        country = COUNTRIES[rng.integers(len(COUNTRIES))]
        province = PROVINCES[country][rng.integers(len(PROVINCES[country]))]
        d = random_date(START, END)
        discounted = rng.uniform() < 0.3
        first = "Starter Kit" if rng.uniform() < 0.15 else None
        p_continue = 0.55 + (0.25 if first else 0) - (0.3 if discounted else 0)
        k = 0
        while d <= END:
            items = basket(d, first if k == 0 else None)
            beans = any(p == "Whole Bean Coffee" for p, _ in items)
            subtotal = sum(PRODUCTS[p][2] * q for p, q in items)
            code, disc = "", 0.0
            if k == 0 and discounted:
                code, disc = "WELCOME20", round(subtotal * 0.2, 2)
            elif d.month == 11 and d.day >= 24 and rng.uniform() < 0.5:
                code, disc = "BFCM25", round(subtotal * 0.25, 2)
            refund = round(subtotal - disc, 2) if rng.uniform() < 0.03 else 0.0
            status = "refunded" if refund else ("voided" if rng.uniform() < 0.01 else "paid")
            net = round(subtotal - disc, 2)
            shipping = 0.0 if net >= 50 else 6.95
            tax = round(net * 0.08, 2)
            for j, (p, q) in enumerate(items):
                sku, _, price, _ = PRODUCTS[p]
                head = j == 0
                rows.append({
                    "Name": f"#{order_no}", "Email": email,
                    "Financial Status": status if head else "",
                    "Paid at": d.strftime("%Y-%m-%d %H:%M:%S -0400") if head else "",
                    "Currency": "USD" if head else "",
                    "Subtotal": net if head else "", "Shipping": shipping if head else "",
                    "Taxes": tax if head else "", "Total": round(net + shipping + tax, 2) if head else "",
                    "Discount Code": code if head else "", "Discount Amount": disc if head else "",
                    "Created at": d.strftime("%Y-%m-%d %H:%M:%S -0400"),
                    "Lineitem quantity": q, "Lineitem name": p, "Lineitem price": price,
                    "Lineitem sku": sku, "Billing Name": f"Customer {c}" if head else "",
                    "Shipping Country": country if head else "", "Shipping Province": province if head else "",
                    "Refunded Amount": refund if head else "", "Source": "web" if head else "",
                })
            order_no += 1
            k += 1
            if rng.uniform() > p_continue + (0.15 if beans else 0):
                break
            gap = rng.normal(30, 5) if beans else rng.exponential(60) + 7
            d = d + timedelta(days=float(max(gap, 3)))
    return pd.DataFrame(rows)


def amazon(n_orders: int = 450) -> pd.DataFrame:
    rows = []
    for i in range(n_orders):
        d = random_date(datetime(2025, 3, 1), END)
        order_id = f"114-{rng.integers(1e6, 9e6)}-{i:07d}"
        for p, q in basket(d):
            sku, _, price, _ = PRODUCTS[p]
            if p == "Gift Card":
                continue
            rows.append({
                "amazon-order-id": order_id,
                "purchase-date": d.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "order-status": "Cancelled" if rng.uniform() < 0.02 else "Shipped",
                "sales-channel": "Amazon.com", "product-name": p, "sku": sku,
                "quantity": q, "currency": "USD", "item-price": round(price * 1.1 * q, 2),
                "item-promotion-discount": 0.0, "ship-country": "US",
                "ship-state": ["CA", "NY", "TX", "FL"][rng.integers(4)],
            })
    return pd.DataFrame(rows)


def cogs() -> pd.DataFrame:
    return pd.DataFrame([{"sku": v[0], "product": k, "unit_cost": v[3]} for k, v in PRODUCTS.items()])


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/samples")
    out.mkdir(parents=True, exist_ok=True)
    shopify().to_csv(out / "shopify_orders.csv", index=False)
    amazon().to_csv(out / "amazon_orders.csv", index=False)
    cogs().to_csv(out / "product_costs.csv", index=False)
    print(f"Wrote samples to {out}")
