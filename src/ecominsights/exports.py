"""Audience exports: turn segments and top-10 lists into files marketing tools accept."""

from __future__ import annotations

import hashlib

import pandas as pd

FORMATS = {
    "generic": "Generic CSV (email + customer metrics)",
    "klaviyo": "Klaviyo list import (Email + profile properties)",
    "mailchimp": "Mailchimp audience import (Email Address + Tags)",
    "meta": "Meta Custom Audience (SHA-256 hashed emails)",
    "google": "Google Ads Customer Match (SHA-256 hashed emails)",
}


def _sha256_email(e: str) -> str:
    return hashlib.sha256(e.strip().lower().encode()).hexdigest()


def audience(customers: pd.DataFrame, contacts: pd.DataFrame, segments: list[str] | None = None,
             keys: list[str] | None = None) -> pd.DataFrame:
    c = customers
    if segments:
        c = c[c["segment"].isin(segments)]
    if keys is not None:
        c = c[c.index.isin(keys)]
    cols = [x for x in ("segment", "orders", "revenue", "aov", "last_order", "recency_days",
                        "predicted_12m_value", "p_active", "favorite_product", "country") if x in c]
    out = c[cols].join(contacts.set_index("customer_key")["email"], how="inner")
    return out.reset_index(drop=True)


def export(aud: pd.DataFrame, fmt: str = "generic") -> bytes:
    if aud.empty:
        return b""
    aud = aud.copy()
    for col in aud.select_dtypes("float").columns:
        aud[col] = aud[col].round(2)
    if fmt == "klaviyo":
        df = aud.rename(columns={"email": "Email", "segment": "RFM Segment", "orders": "Total Orders",
                                 "revenue": "Total Revenue", "favorite_product": "Favorite Product",
                                 "predicted_12m_value": "Predicted 12m Value"})
        df = df[[c for c in ("Email", "RFM Segment", "Total Orders", "Total Revenue", "Favorite Product",
                             "Predicted 12m Value") if c in df]]
    elif fmt == "mailchimp":
        df = pd.DataFrame({"Email Address": aud["email"], "Tags": aud["segment"]})
    elif fmt == "meta":
        df = pd.DataFrame({"email": aud["email"].map(_sha256_email)})
        if "country" in aud:
            df["country"] = aud["country"].str.lower()
        if "predicted_12m_value" in aud:
            df["value"] = aud["predicted_12m_value"].round(2)  # for value-based lookalikes
    elif fmt == "google":
        df = pd.DataFrame({"Email": aud["email"].map(_sha256_email)})
    else:
        df = aud
    return df.to_csv(index=False).encode()
