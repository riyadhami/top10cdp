import math
from pathlib import Path

import pandas as pd
import pytest

from ecominsights.ai import rule_based_narrative, verify_numbers
from ecominsights.analytics import kpis
from ecominsights.clean import build_dataset, to_number
from ecominsights.ingest import auto_map, detect_platform, read_csv
from ecominsights.pipeline import run_analysis, to_json

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def sample_result():
    if not (SAMPLES / "shopify_orders.csv").exists():
        import subprocess, sys
        subprocess.run([sys.executable, "scripts/make_sample_data.py"], check=True)
    frames = []
    for name, ch in (("shopify_orders.csv", "Shopify"), ("amazon_orders.csv", "Amazon")):
        raw = read_csv(str(SAMPLES / name))
        frames.append((raw, auto_map(list(raw.columns)), ch))
    ds = build_dataset(frames, cogs=pd.read_csv(SAMPLES / "product_costs.csv"))
    return ds, run_analysis(ds)


def test_platform_detection_and_mapping():
    shop = ["Name", "Email", "Financial Status", "Created at", "Subtotal", "Lineitem quantity",
            "Lineitem name", "Lineitem price", "Billing Name", "Billing Phone"]
    assert detect_platform(shop) == "shopify"
    m = auto_map(shop)
    assert m.fields["order_id"] == "Name" and m.fields["order_total"] == "Subtotal"
    assert m.fields["customer_id"] == "Email"
    assert "Billing Name" in m.dropped_pii and "Billing Phone" in m.dropped_pii
    amz = auto_map(["amazon-order-id", "purchase-date", "item-price", "quantity", "sku"])
    assert amz.platform == "amazon" and amz.fields["line_total"] == "item-price"
    retail = auto_map(["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID"])
    assert retail.platform == "online_retail" and retail.fields["customer_id"] == "Customer ID"


def test_money_parsing():
    s = pd.Series(["$1,234.50", "(12.00)", "1.234,50", "", "abc"])
    out = to_number(s)
    assert out[0] == 1234.5 and out[1] == -12.0 and out[2] == 1234.5
    assert pd.isna(out[3]) and pd.isna(out[4])


def test_shopify_line_items_dedupe_to_orders():
    raw = pd.DataFrame({
        "Name": ["#1", "#1", "#2"], "Email": ["a@x.com", "a@x.com", "b@x.com"],
        "Financial Status": ["paid", "", "paid"], "Created at": ["2025-01-01 10:00:00 -0500"] * 2 + ["2025-01-03"],
        "Subtotal": ["90", "", "20"], "Discount Amount": ["10", "", "0"],
        "Lineitem quantity": ["2", "1", "1"], "Lineitem name": ["A", "B", "A"], "Lineitem price": ["30", "40", "20"],
    })
    ds = build_dataset([(raw, auto_map(list(raw.columns)), None)])
    o = ds.orders.set_index("order_id")
    assert len(o) == 2
    assert o.loc["#1", "revenue"] == 90  # Subtotal is order-level and net of discount; not summed per line
    assert o.loc["#1", "units"] == 3
    assert o.loc["#1", "order_date"].hour == 10  # local wall-clock time kept
    assert ds.lines["line_revenue"].sum() == 120


def test_kpi_tree_contributions_sum_to_change():
    d = pd.date_range("2025-01-01", periods=60, freq="D")
    o = pd.DataFrame({"order_date": d, "revenue": [100.0] * 30 + [130.0] * 30, "units": [2] * 30 + [3] * 30,
                      "customer_key": [f"c{i % 10}" for i in range(60)], "discount": 0.0,
                      "gross_revenue": [100.0] * 30 + [130.0] * 30})
    o["is_returning"] = o.groupby("customer_key").cumcount() > 0
    o["order_seq"] = o.groupby("customer_key").cumcount() + 1
    tree = kpis.kpi_tree(o.iloc[:30], o.iloc[30:])
    leaves = [c for ch in tree["children"] for c in ch["children"]]
    assert math.isclose(sum(c["contribution"] for c in leaves), tree["cur"] - tree["prev"], rel_tol=1e-9)


def test_pipeline_finds_planted_patterns(sample_result):
    ds, r = sample_result
    assert r["meta"]["module_errors"] == []
    ids = {i["id"] for i in r["insights"]}
    assert {"discount_cohort", "gateway_product", "stockout", "at_risk"} <= ids
    so = r["stockouts"]
    assert "Cold Brew Bottle" in set(so["product"])
    top_pair = r["basket"].iloc[:5]
    assert ((top_pair["antecedent"] + top_pair["consequent"]).str.contains("Grinder").any())
    assert r["first_product"].iloc[0]["first_product"] in {"Coffee Grinder", "Starter Kit"}
    assert len(r["insights"]) >= 10
    assert [i["rank"] for i in r["insights"]] == sorted(i["rank"] for i in r["insights"])
    assert len(to_json(r)) > 1000


def test_rule_narrative_and_number_verification(sample_result):
    _, r = sample_result
    md = rule_based_narrative(r)
    assert "Top priorities" in md
    from ecominsights.ai import build_facts
    facts = build_facts(r)
    assert verify_numbers("Revenue grew to $999,999,999 this year.", facts) == ["$999,999,999"]
    assert verify_numbers(md, facts) == []
