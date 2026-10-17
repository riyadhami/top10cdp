"""Canonical data model and the header vocabularies used to map raw CSVs onto it."""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical fields. "order" level fields describe the whole order and may be repeated
# (or only present on the first row) when the export is one row per line item.
FIELDS: dict[str, dict] = {
    "order_id":      {"level": "order", "required": True,  "desc": "Order / invoice identifier"},
    "order_date":    {"level": "order", "required": True,  "desc": "Order creation timestamp"},
    "customer_id":   {"level": "order", "required": False, "desc": "Customer identifier (ID or email)"},
    "email":         {"level": "order", "required": False, "desc": "Customer email (used for exports, hashed for analytics)"},
    "order_total":   {"level": "order", "required": False, "desc": "Order revenue net of discounts, before tax & shipping"},
    "discount":      {"level": "order", "required": False, "desc": "Discount amount"},
    "refund":        {"level": "order", "required": False, "desc": "Refunded amount"},
    "status":        {"level": "order", "required": False, "desc": "Order / financial status"},
    "currency":      {"level": "order", "required": False, "desc": "Currency code"},
    "country":       {"level": "order", "required": False, "desc": "Shipping / billing country"},
    "region":        {"level": "order", "required": False, "desc": "State / province"},
    "channel":       {"level": "order", "required": False, "desc": "Sales channel / source"},
    "fees":          {"level": "order", "required": False, "desc": "Marketplace / payment fees"},
    "discount_code": {"level": "order", "required": False, "desc": "Discount / coupon code"},
    "product":       {"level": "line",  "required": False, "desc": "Product / item name"},
    "sku":           {"level": "line",  "required": False, "desc": "SKU / product code"},
    "category":      {"level": "line",  "required": False, "desc": "Product type / category"},
    "quantity":      {"level": "line",  "required": False, "desc": "Units in the line"},
    "unit_price":    {"level": "line",  "required": False, "desc": "Price per unit"},
    "line_total":    {"level": "line",  "required": False, "desc": "Line revenue (price x qty)"},
    "unit_cost":     {"level": "line",  "required": False, "desc": "Cost of goods per unit"},
}

REQUIRED = [k for k, v in FIELDS.items() if v["required"]]

# Header synonyms, most specific first. Matching is done on normalized headers
# (lowercase, alphanumerics only), so "Lineitem quantity" -> "lineitemquantity".
SYNONYMS: dict[str, list[str]] = {
    "order_id": ["ordernumber", "orderid", "amazonorderid", "invoice", "invoiceno", "invoiceid",
                 "orderno", "order", "transactionid", "name", "id"],
    "order_date": ["createdat", "purchasedate", "orderdate", "invoicedate", "date", "paidat",
                   "orderdatetime", "transactiondate", "datecreated", "timestamp"],
    "customer_id": ["customerid", "customer", "buyerid", "clientid", "userid", "customerno"],
    "email": ["email", "customeremail", "billingemail", "buyeremail", "emailaddress"],
    "order_total": ["subtotal", "ordersubtotal", "netsales", "ordertotal", "ordertotalamount",
                    "total", "totalsales", "grandtotal"],
    "discount": ["discountamount", "itempromotiondiscount", "cartdiscount", "discount",
                 "discounts", "orderdiscount", "promotiondiscount"],
    "refund": ["refundedamount", "refund", "refunds", "refundamount", "returns"],
    "status": ["financialstatus", "orderstatus", "status", "paymentstatus"],
    "currency": ["currency", "currencycode"],
    "country": ["shippingcountry", "shipcountry", "billingcountry", "country", "countrycode"],
    "region": ["shippingprovince", "shipstate", "billingprovince", "state", "province", "region"],
    "channel": ["saleschannel", "channel", "source", "orderchannel", "marketplace"],
    "fees": ["fees", "fee", "amazonfees", "sellingfees", "transactionfees", "paymentfees"],
    "discount_code": ["discountcode", "couponcode", "coupon", "promocode", "promotionids"],
    "product": ["lineitemname", "productname", "itemname", "description", "product", "title",
                "producttitle"],
    "sku": ["lineitemsku", "sku", "stockcode", "productid", "asin", "itemsku", "productcode"],
    "category": ["producttype", "category", "productcategory", "type"],
    "quantity": ["lineitemquantity", "quantity", "qty", "quantitypurchased", "units", "itemquantity"],
    "unit_price": ["lineitemprice", "itemprice", "unitprice", "price", "priceeach"],
    "line_total": ["linetotal", "lineitemtotal", "itemtotal", "sales", "revenue", "amount",
                   "linerevenue"],
    "unit_cost": ["unitcost", "cost", "cogs", "costperitem", "productcost"],
}

# Columns that hold personal data we never keep (beyond hashed IDs and the opt-in email list).
PII_HINTS = ["phone", "address", "street", "zip", "postal", "billingname", "shippingname",
             "firstname", "lastname", "fullname", "buyername", "recipientname", "company"]

# Header fingerprints for platform detection: (platform, headers that must all be present).
PLATFORM_FINGERPRINTS: list[tuple[str, set[str]]] = [
    ("shopify", {"name", "financialstatus", "lineitemname", "lineitemquantity"}),
    ("amazon", {"amazonorderid", "purchasedate", "itemprice"}),
    ("woocommerce", {"ordernumber", "orderdate", "itemname"}),
    ("online_retail", {"invoice", "stockcode", "invoicedate"}),
]

# Platform specific behaviour for building orders.
PLATFORM_RULES = {
    "shopify": {"line_amount_is_unit_price": True, "cancel_values": ["voided"],
                "refund_status_values": ["refunded"]},
    # Amazon "item-price" is already the line total (price x quantity).
    "amazon": {"line_amount_is_unit_price": False, "cancel_values": ["cancelled", "canceled"]},
    "woocommerce": {"line_amount_is_unit_price": False,
                    "cancel_values": ["cancelled", "failed", "refunded", "trash"]},
    "online_retail": {"line_amount_is_unit_price": True, "cancel_values": []},
    "generic": {"line_amount_is_unit_price": True,
                "cancel_values": ["cancelled", "canceled", "voided", "failed"]},
}


def normalize_header(h: str) -> str:
    return "".join(ch for ch in str(h).lower() if ch.isalnum())


@dataclass
class ColumnMapping:
    """canonical field -> source column, plus how each match was made."""

    platform: str = "generic"
    fields: dict[str, str] = field(default_factory=dict)
    method: dict[str, str] = field(default_factory=dict)  # exact | fuzzy | ai | user
    dropped_pii: list[str] = field(default_factory=list)

    def missing_required(self) -> list[str]:
        missing = [f for f in REQUIRED if f not in self.fields]
        has_revenue = any(f in self.fields for f in ("order_total", "line_total", "unit_price"))
        if not has_revenue:
            missing.append("order_total | line_total | unit_price")
        return missing

    def to_dict(self) -> dict:
        return {"platform": self.platform, "fields": self.fields, "method": self.method,
                "dropped_pii": self.dropped_pii}

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnMapping":
        return cls(d.get("platform", "generic"), dict(d.get("fields", {})),
                   dict(d.get("method", {})), list(d.get("dropped_pii", [])))
