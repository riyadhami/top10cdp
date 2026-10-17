"""Turn mapped raw rows into canonical `orders` and `lines` tables, plus a data-quality report.

Revenue definition used everywhere: order revenue net of discounts and refunds,
before tax and shipping. Cancelled orders and test orders are excluded.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import FIELDS, PLATFORM_RULES, ColumnMapping

ORDER_FIELDS = [f for f, v in FIELDS.items() if v["level"] == "order"]


@dataclass
class Dataset:
    orders: pd.DataFrame          # one row per order
    lines: pd.DataFrame           # one row per line item (may be synthetic if no line data)
    contacts: pd.DataFrame        # customer_key -> email (only kept when retention is allowed)
    returns: pd.DataFrame = field(default_factory=pd.DataFrame)  # returned line items, if any
    quality: dict = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)


def _salt() -> str:
    return os.environ.get("APP_SALT", "change-me")


def hash_id(value: str) -> str:
    return hashlib.sha256((_salt() + value.strip().lower()).encode()).hexdigest()[:16]


_NUM_RE = re.compile(r"[^0-9.\-]")


def to_number(s: pd.Series) -> pd.Series:
    """Parse money-like strings: '$1,234.50', '(12.00)', '1.234,50' (EU) -> float."""
    if s.dtype.kind in "if":
        return s.astype(float)
    raw = s.astype("string").str.strip()
    neg = raw.str.startswith("(") & raw.str.endswith(")")
    eu = raw.str.contains(r"^\-?[\d.]+,\d{1,2}$", regex=True, na=False)
    raw = raw.where(~eu, raw.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    cleaned = raw.str.replace(",", "", regex=False).str.replace(_NUM_RE, "", regex=True)
    out = pd.to_numeric(cleaned, errors="coerce")
    return out.where(~neg.fillna(False), -out.abs())


_TZ_SUFFIX = re.compile(r"\s*(?:[+-]\d{2}:?\d{2}|Z|UTC)$")


def to_datetime(s: pd.Series) -> pd.Series:
    """Parse to naive local wall-clock time (offsets dropped so hour-of-day stays meaningful)."""
    s = s.astype("string").str.strip().str.replace(_TZ_SUFFIX, "", regex=True)
    out = pd.to_datetime(s, errors="coerce", format="mixed")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    return out


def standardize(raw: pd.DataFrame, mapping: ColumnMapping, channel: str | None = None) -> pd.DataFrame:
    """Rename mapped columns to canonical names and parse types. One row per raw row."""
    df = pd.DataFrame(index=raw.index)
    for fld, col in mapping.fields.items():
        if col in raw.columns:
            df[fld] = raw[col]
    for num in ("order_total", "discount", "refund", "fees", "quantity", "unit_price",
                "line_total", "unit_cost"):
        if num in df:
            df[num] = to_number(df[num])
    df["order_date"] = to_datetime(df["order_date"])
    df["order_id"] = df["order_id"].astype("string").str.strip()
    if channel:
        df["channel"] = channel
    elif "channel" not in df:
        df["channel"] = mapping.platform
    df["platform"] = mapping.platform
    return df


def build_dataset(frames: list[tuple[pd.DataFrame, ColumnMapping, str | None]],
                  cogs: pd.DataFrame | None = None, keep_contacts: bool = True) -> Dataset:
    """Combine one or more standardized uploads (multi-channel) into the canonical model."""
    all_orders, all_lines, all_returns, contacts, sources = [], [], [], [], []
    stats = {"raw_rows": 0, "bad_dates": 0, "missing_order_id": 0, "duplicate_rows": 0,
             "cancelled_orders": 0, "test_orders": 0, "return_rows": 0, "missing_customer": 0,
             "negative_revenue_orders": 0, "non_merch_lines": 0, "non_merch_value": 0.0,
             "returns_matched": 0, "fully_refunded_orders": 0}

    for i, (raw, mapping, channel) in enumerate(frames):
        df = standardize(raw, mapping, channel)
        rules = PLATFORM_RULES.get(mapping.platform, PLATFORM_RULES["generic"])
        stats["raw_rows"] += len(df)

        dup = df.duplicated(keep="first")
        stats["duplicate_rows"] += int(dup.sum())
        df = df[~dup]

        stats["missing_order_id"] += int(df["order_id"].isna().sum())
        df = df[df["order_id"].notna()]
        # Shopify and similar: order-level fields are only filled on the first row of each order.
        present_order_fields = [f for f in ORDER_FIELDS if f in df and f != "order_id"]
        df[present_order_fields] = df.groupby("order_id")[present_order_fields].transform("first")

        stats["bad_dates"] += int(df["order_date"].isna().sum())
        df = df[df["order_date"].notna()]

        # Cancellations / returns.
        if "status" in df:
            st = df["status"].astype("string").str.lower().fillna("")
            cancel = st.isin(rules["cancel_values"])
            stats["cancelled_orders"] += int(df.loc[cancel, "order_id"].nunique())
            df = df[~cancel]
        # Line revenue (computed for every row, including returns).
        if "line_total" in df:
            line_rev = df["line_total"]
        elif "unit_price" in df:
            qty = df["quantity"] if "quantity" in df else 1
            line_rev = df["unit_price"] * qty if rules["line_amount_is_unit_price"] else df["unit_price"]
        else:
            line_rev = pd.Series(np.nan, index=df.index)
        df = df.assign(line_revenue=line_rev.astype(float))
        if "quantity" not in df:
            df["quantity"] = 1.0

        # Customer key: hashed so analytics never touches raw identifiers.
        cust_src = df["customer_id"] if "customer_id" in df else pd.Series(pd.NA, index=df.index)
        df["customer_key"] = cust_src.astype("string").map(
            lambda v: hash_id(v) if isinstance(v, str) and v.strip() else pd.NA)

        # Non-merchandise lines (postage, fees, manual adjustments, test items) are not product revenue.
        kind = classify_lines(df)
        nm = kind != "merch"
        stats["non_merch_lines"] += int(nm.sum())
        stats["non_merch_value"] += round(float(df.loc[nm & (df["quantity"] > 0), "line_revenue"].sum()), 2)
        stats["test_orders"] += int(df.loc[kind == "test", "order_id"].nunique())
        df = df[~nm]

        test = pd.Series(False, index=df.index)
        if "email" in df:
            test |= df["email"].astype("string").str.contains(r"@(?:example|test)\.", regex=True, na=False)
        stats["test_orders"] += int(df.loc[test, "order_id"].nunique())
        df = df[~test]

        # Returns: Online Retail style "C" invoices or negative quantities.
        ret = df["order_id"].str.upper().str.startswith("C") if mapping.platform == "online_retail" \
            else pd.Series(False, index=df.index)
        ret = ret | (df["quantity"] < 0)
        stats["return_rows"] += int(ret.sum())
        returns = df[ret].copy()
        df = df[~ret]
        refund_by_order = pd.Series(dtype=float)
        if len(returns):
            returns["value"] = returns["line_revenue"].abs()
            matched = match_returns(df, returns)
            stats["returns_matched"] += int(len(matched))
            if len(matched):
                refund_by_order = df.loc[matched, "line_revenue"].groupby(df.loc[matched, "order_id"]).sum()
                df = df.drop(index=matched)
            r = returns[[c for c in ("product", "sku", "order_date", "value") if c in returns]].copy()
            r["quantity"] = returns["quantity"].abs()
            r["channel"] = channel or mapping.platform
            all_returns.append(r)

        stats["missing_customer"] += int(df["customer_key"].isna().sum())
        if keep_contacts and "email" in df:
            c = df[["customer_key", "email"]].dropna().drop_duplicates("customer_key")
            contacts.append(c)

        orders = _orders_from_lines(df)
        if "order_total" in orders and len(refund_by_order):
            # Order-level totals don't shrink when a line is removed, so book matched returns as refunds.
            orders["refund"] = orders["refund"].add(orders["order_id"].map(refund_by_order).fillna(0))
        orders["revenue"] = orders["gross_revenue"] - orders["refund"].fillna(0)
        full_refund = (orders["gross_revenue"] > 0) & (orders["revenue"] <= 0.005 * orders["gross_revenue"])
        stats["fully_refunded_orders"] += int(full_refund.sum())
        orders = orders[~full_refund].reset_index(drop=True)
        df = df[df["order_id"].isin(orders["order_id"])]
        stats["negative_revenue_orders"] += int((orders["revenue"] < 0).sum())
        orders["source_file"] = i

        lines = df[["order_id", "order_date", "customer_key", "channel", "quantity", "line_revenue"]
                   + [c for c in ("product", "sku", "category", "unit_cost") if c in df]].copy()
        if "product" not in lines and "sku" in lines:
            lines["product"] = lines["sku"]
        all_orders.append(orders)
        all_lines.append(lines)
        sources.append({"file": i, "platform": mapping.platform, "channel": channel or mapping.platform,
                        "rows": len(raw), "orders": len(orders)})

    orders = pd.concat(all_orders, ignore_index=True)
    lines = pd.concat(all_lines, ignore_index=True)
    # Same order ID from two channels is two different orders.
    orders["order_key"] = orders["channel"].astype(str) + ":" + orders["order_id"].astype(str)
    lines["order_key"] = lines["channel"].astype(str) + ":" + lines["order_id"].astype(str)

    if cogs is not None and len(cogs):
        lines = _apply_cogs(lines, cogs)
    if "unit_cost" in lines and lines["unit_cost"].notna().any():
        lines["line_cost"] = lines["unit_cost"] * lines["quantity"]
        cost = lines.groupby("order_key")["line_cost"].sum(min_count=1)
        orders["cogs"] = orders["order_key"].map(cost)

    orders = orders[orders["revenue"].notna()].sort_values("order_date").reset_index(drop=True)
    _mark_new_vs_returning(orders)
    contacts_df = (pd.concat(contacts).drop_duplicates("customer_key") if contacts
                   else pd.DataFrame(columns=["customer_key", "email"]))
    quality = quality_report(orders, lines, stats, frames)
    returns_df = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()
    if len(returns_df) and "product" not in returns_df and "sku" in returns_df:
        returns_df["product"] = returns_df["sku"]
    return Dataset(orders=orders, lines=lines, contacts=contacts_df, quality=quality,
                   sources=sources, returns=returns_df)


_SHIPPING = r"postage|carriage|shipping|delivery|freight"
_FEE = r"bank charges?|\bfees?\b|commission|transaction charge"
_ADJUST = r"^manual$|adjust|bad debt|^discount$|^samples?$|^update$|^\?|damaged|^lost$|^missing$"
_TEST = r"^test\b|this is a test product"
_NON_MERCH_SKUS = {"POST": "shipping", "DOT": "shipping", "C2": "shipping", "M": "adjustment",
                   "D": "adjustment", "S": "adjustment", "B": "adjustment", "BANK CHARGES": "fee",
                   "AMAZONFEE": "fee", "CRUK": "fee"}


def classify_lines(df: pd.DataFrame) -> pd.Series:
    """merch | shipping | fee | adjustment | test, from the SKU and product name."""
    kind = pd.Series("merch", index=df.index, dtype=object)
    name = df["product"].astype("string").str.strip().str.lower() if "product" in df else None
    if name is not None:
        for k, pat in (("shipping", _SHIPPING), ("fee", _FEE), ("adjustment", _ADJUST), ("test", _TEST)):
            kind = kind.mask(name.str.contains(pat, regex=True, na=False) & (kind == "merch"), k)
    if "sku" in df:
        sku = df["sku"].astype("string").str.strip().str.upper()
        kind = kind.mask(sku.isin(_NON_MERCH_SKUS) & (kind == "merch"), sku.map(_NON_MERCH_SKUS))
        kind = kind.mask(sku.str.match(r"^(ADJUST|TEST)", na=False) & (kind == "merch"), "adjustment")
    return kind


def match_returns(purchases: pd.DataFrame, returns: pd.DataFrame) -> list:
    """Index labels of purchase lines cancelled by a return line: same customer, same product,
    same quantity, most recent purchase on or before the return date. Each purchase matches once."""
    item = "sku" if "sku" in returns and returns["sku"].notna().any() else "product"
    if item not in returns or returns["customer_key"].isna().all():
        return []
    p = purchases[purchases["customer_key"].notna()]
    p = p.assign(_item=p[item].astype(str), _qty=p["quantity"].abs(), _idx=p.index)
    r = returns[returns["customer_key"].notna()]
    r = r.assign(_item=r[item].astype(str), _qty=r["quantity"].abs())
    p = p.sort_values("order_date")
    taken: set = set()
    groups = {k: g for k, g in p.groupby(["customer_key", "_item", "_qty"], sort=False)}
    for key, rr in r.sort_values("order_date").groupby(["customer_key", "_item", "_qty"], sort=False):
        cand = groups.get(key)
        if cand is None:
            continue
        for d in rr["order_date"]:
            prior = cand[(cand["order_date"] <= d) & ~cand["_idx"].isin(taken)]
            if len(prior):
                taken.add(prior["_idx"].iloc[-1])
    return list(taken)


def _orders_from_lines(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("order_id", sort=False)
    first_cols = [c for c in ORDER_FIELDS if c in df and c != "order_id"] + ["platform", "customer_key"]
    orders = g[list(dict.fromkeys(first_cols))].first().reset_index()
    orders["units"] = g["quantity"].sum().values
    orders["n_lines"] = g.size().values
    line_sum = g["line_revenue"].sum(min_count=1).values
    discount = orders["discount"].fillna(0).abs() if "discount" in orders else 0
    if "order_total" in orders:
        # Order-level net revenue (e.g. Shopify Subtotal is already net of discounts).
        orders["gross_revenue"] = orders["order_total"].fillna(pd.Series(line_sum) - discount)
    else:
        orders["gross_revenue"] = pd.Series(line_sum) - discount
    orders["discount"] = discount if "discount" in orders else 0.0
    orders["refund"] = orders["refund"].fillna(0).abs() if "refund" in orders else 0.0
    if "fees" in orders:
        orders["fees"] = orders["fees"].abs()
    return orders


def _apply_cogs(lines: pd.DataFrame, cogs: pd.DataFrame) -> pd.DataFrame:
    """Join a user-supplied cost file (sku or product -> unit_cost)."""
    cols = {c.lower().strip(): c for c in cogs.columns}
    key_col = next((cols[k] for k in ("sku", "variant sku", "product", "product name", "name", "item")
                    if k in cols), None)
    cost_col = next((cols[k] for k in ("unit_cost", "unit cost", "cost", "cogs", "cost per item", "product cost")
                     if k in cols), None)
    if key_col is None or cost_col is None:
        return lines
    costs = to_number(cogs[cost_col])
    keys = cogs[key_col].astype(str).str.strip()
    table = dict(zip(keys[costs.notna()], costs[costs.notna()]))
    join = "sku" if key_col.lower() == "sku" and "sku" in lines else "product"
    if join not in lines:
        return lines
    mapped = lines[join].astype(str).str.strip().map(table)
    lines["unit_cost"] = lines["unit_cost"].fillna(mapped) if "unit_cost" in lines else mapped
    return lines


def _mark_new_vs_returning(orders: pd.DataFrame) -> None:
    has = orders["customer_key"].notna()
    orders["order_seq"] = np.nan
    orders.loc[has, "order_seq"] = orders[has].groupby("customer_key").cumcount() + 1
    orders["is_returning"] = orders["order_seq"] > 1
    first = orders[has].groupby("customer_key")["order_date"].transform("min")
    orders.loc[has, "first_order_date"] = first


def quality_report(orders: pd.DataFrame, lines: pd.DataFrame, stats: dict, frames) -> dict:
    """0-100 score plus the analyses each gap blocks."""
    n = max(stats["raw_rows"], 1)
    checks = []

    def check(name, ok_ratio, weight, detail, blocks=None):
        checks.append({"check": name, "score": round(max(0.0, min(1.0, ok_ratio)) * 100),
                       "weight": weight, "detail": detail, "blocks": blocks or []})

    cust_cov = orders["customer_key"].notna().mean() if len(orders) else 0
    check("Customer identifiers", cust_cov, 3,
          f"{cust_cov:.0%} of orders have a customer ID or email",
          ["RFM segments", "Cohorts", "CLV", "Top customers"] if cust_cov < 0.5 else [])
    check("Valid dates", 1 - stats["bad_dates"] / n, 2, f"{stats['bad_dates']} rows with unreadable dates")
    check("Duplicates", 1 - stats["duplicate_rows"] / n, 1, f"{stats['duplicate_rows']} duplicate rows removed")
    has_product = "product" in lines and lines["product"].notna().mean() > 0.5
    check("Product detail", 1.0 if has_product else 0.0, 2,
          "Product names / SKUs present" if has_product else "No product column",
          [] if has_product else ["Product mix", "Basket analysis", "First-product cohorts"])
    span_days = (orders["order_date"].max() - orders["order_date"].min()).days if len(orders) else 0
    check("History length", min(span_days / 365, 1.0), 2, f"{span_days} days of history",
          ["Year-over-year", "Seasonality"] if span_days < 365 else [])
    has_cost = "cogs" in orders and orders["cogs"].notna().mean() > 0.5
    check("Product costs", 1.0 if has_cost else 0.0, 1,
          "Costs available" if has_cost else "No COGS; profit analysis disabled (upload a cost file)",
          [] if has_cost else ["Margin analysis"])
    neg = stats["negative_revenue_orders"] / max(len(orders), 1)
    check("Revenue sanity", 1 - neg * 5, 1, f"{stats['negative_revenue_orders']} orders with negative net revenue")
    currencies = orders["currency"].dropna().unique().tolist() if "currency" in orders else []
    check("Single currency", 1.0 if len(currencies) <= 1 else 0.3, 1,
          f"Currencies: {', '.join(map(str, currencies)) or 'not specified'}")

    total_w = sum(c["weight"] for c in checks)
    score = round(sum(c["score"] * c["weight"] for c in checks) / total_w)
    blocked = sorted({b for c in checks for b in c["blocks"]})
    return {"score": score, "checks": checks, "stats": stats, "blocked_analyses": blocked,
            "orders": int(len(orders)), "customers": int(orders["customer_key"].nunique()),
            "date_min": str(orders["order_date"].min().date()) if len(orders) else None,
            "date_max": str(orders["order_date"].max().date()) if len(orders) else None,
            "currency": currencies[0] if len(currencies) == 1 else None}
