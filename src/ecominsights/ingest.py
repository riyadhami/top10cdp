"""Read raw CSVs, detect the source platform and map headers onto the canonical schema."""

from __future__ import annotations

import csv
import io

import pandas as pd
from rapidfuzz import fuzz

from .schema import (FIELDS, PII_HINTS, PLATFORM_FINGERPRINTS, SYNONYMS, ColumnMapping,
                     normalize_header)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FUZZY_THRESHOLD = 88


class IngestError(ValueError):
    pass


def read_csv(data: bytes | str) -> pd.DataFrame:
    """Read a CSV from bytes, sniffing encoding and delimiter. Everything is read as text;
    typing happens during cleaning so bad values can be counted instead of crashing."""
    if isinstance(data, str):
        with open(data, "rb") as fh:
            data = fh.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(f"File is {len(data) / 1e6:.1f} MB; the limit is 20 MB.")
    if not data.strip():
        raise IngestError("File is empty.")

    text = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        sep = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False,
                     na_values=["", "NA", "N/A", "null", "NULL", "None"])
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    if df.empty:
        raise IngestError("No data rows found.")
    return df


def detect_platform(columns: list[str]) -> str:
    norm = {normalize_header(c) for c in columns}
    for platform, required in PLATFORM_FINGERPRINTS:
        if required <= norm:
            return platform
    return "generic"


def auto_map(columns: list[str], platform: str | None = None) -> ColumnMapping:
    """Map headers to canonical fields: exact synonym match first, then fuzzy match."""
    platform = platform or detect_platform(columns)
    mapping = ColumnMapping(platform=platform)
    norm_to_col = {normalize_header(c): c for c in columns}
    used: set[str] = set()

    known = {s for syns in SYNONYMS.values() for s in syns}
    for col in columns:
        n = normalize_header(col)
        if n not in known and any(h in n for h in PII_HINTS):
            mapping.dropped_pii.append(col)
            used.add(col)

    # Exact pass, in synonym priority order.
    for fld, syns in SYNONYMS.items():
        for s in syns:
            col = norm_to_col.get(s)
            if col and col not in used:
                mapping.fields[fld] = col
                mapping.method[fld] = "exact"
                used.add(col)
                break

    # Fuzzy pass for anything still unmapped.
    for fld, syns in SYNONYMS.items():
        if fld in mapping.fields:
            continue
        best, best_score = None, 0
        for col in columns:
            if col in used:
                continue
            n = normalize_header(col)
            score = max(fuzz.ratio(n, s) for s in syns)
            if score > best_score:
                best, best_score = col, score
        if best and best_score >= FUZZY_THRESHOLD:
            mapping.fields[fld] = best
            mapping.method[fld] = "fuzzy"
            used.add(best)

    # Amazon's "item-price" is the line total, not a unit price.
    if platform == "amazon" and "unit_price" in mapping.fields and "line_total" not in mapping.fields:
        mapping.fields["line_total"] = mapping.fields.pop("unit_price")
        mapping.method["line_total"] = mapping.method.pop("unit_price")

    # An email column doubles as the customer identifier when there is no explicit ID.
    if "customer_id" not in mapping.fields and "email" in mapping.fields:
        mapping.fields["customer_id"] = mapping.fields["email"]
        mapping.method["customer_id"] = "derived"
    return mapping


def unmapped_columns(columns: list[str], mapping: ColumnMapping) -> list[str]:
    used = set(mapping.fields.values()) | set(mapping.dropped_pii)
    return [c for c in columns if c not in used]


def field_choices() -> list[str]:
    return list(FIELDS)
