"""Azure OpenAI layer: column mapping, executive narrative, and Q&A over the computed facts.

The model never sees raw order rows or customer identifiers. It receives:
  - mapping: column headers + a few masked sample values
  - narrative / Q&A: the aggregated facts JSON produced by the deterministic engine
Every number in the narrative is checked against the facts; unverified numbers are reported.
Without Azure credentials everything falls back to rule-based output.
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd

from .insights import money, pct
from .schema import FIELDS, ColumnMapping

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass


def _client():
    from openai import AzureOpenAI
    return AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                       api_key=os.environ["AZURE_OPENAI_API_KEY"],
                       api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))


def ai_enabled() -> bool:
    return all(os.environ.get(k) for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"))


def _chat(messages: list[dict], deployment: str | None = None, json_mode: bool = False) -> str:
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _client().chat.completions.create(
        model=deployment or os.environ["AZURE_OPENAI_DEPLOYMENT"], messages=messages, **kw)
    return resp.choices[0].message.content or ""


# ---------------- column mapping ----------------

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+")
_LONG_DIGITS = re.compile(r"\d{7,}")


def mask_value(v) -> str:
    s = str(v)[:40]
    s = _EMAIL.sub("user@example.com", s)
    return _LONG_DIGITS.sub("#######", s)


def ai_map_columns(raw: pd.DataFrame, current: ColumnMapping) -> ColumnMapping:
    """Ask the model to fill in unmapped canonical fields. Existing exact matches are kept."""
    if not ai_enabled():
        return current
    samples = {c: [mask_value(v) for v in raw[c].dropna().head(5)] for c in raw.columns
               if c not in current.dropped_pii}
    spec = {k: v["desc"] for k, v in FIELDS.items()}
    prompt = (
        "Map ecommerce export columns to canonical fields. Return JSON {\"fields\": {canonical: column}} "
        "using only column names that exist. Omit fields with no good match. Prefer order-level net "
        "revenue (subtotal after discounts) for order_total, not totals including tax/shipping.\n\n"
        f"Canonical fields: {json.dumps(spec)}\n\nAlready mapped: {json.dumps(current.fields)}\n\n"
        f"Columns with sample values: {json.dumps(samples)}")
    try:
        out = json.loads(_chat([{"role": "user", "content": prompt}],
                               os.environ.get("AZURE_OPENAI_DEPLOYMENT_MINI") or None, json_mode=True))
    except Exception:  # noqa: BLE001
        return current
    mapping = ColumnMapping.from_dict(current.to_dict())
    used = set(mapping.fields.values())
    for fld, col in (out.get("fields") or {}).items():
        if fld in FIELDS and fld not in mapping.fields and col in raw.columns and col not in used:
            mapping.fields[fld] = col
            mapping.method[fld] = "ai"
            used.add(col)
    if "customer_id" not in mapping.fields and "email" in mapping.fields:
        mapping.fields["customer_id"] = mapping.fields["email"]
        mapping.method["customer_id"] = "derived"
    return mapping


# ---------------- facts ----------------

def build_facts(r: dict, top_n: int = 10) -> dict:
    """Compact, PII-free summary handed to the model."""
    def metric_block(h):
        mm = ((r.get("snapshot") or {}).get(h) or {}).get("metrics", {})
        keep = ("revenue", "orders", "customers", "aov", "new_customers", "repeat_rate",
                "returning_revenue_share", "discount_rate", "gross_margin")
        return {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in mm[k].items()}
                for k in keep if k in mm}
    seg = r.get("segments")
    ch = r.get("channels")
    facts = {
        "currency": r["meta"]["currency"],
        "data": {k: r["meta"]["quality"].get(k) for k in ("score", "orders", "customers", "date_min",
                                                          "date_max", "blocked_analyses")},
        "kpis": {h: metric_block(h) for h in ("30d", "90d", "365d")},
        "top_insights_total_annual_impact": round(sum(i["impact"] for i in r.get("insights", [])[:top_n])),
        "top_insights": [{"id": i["id"], "rank": i["rank"], "title": i["title"], "finding": i["finding"],
                          "action": i["action"], "est_annual_impact": round(i["impact"]),
                          "confidence": round(i["confidence"], 2), "impact_formula": i["impact_formula"]}
                         for i in r.get("insights", [])[:top_n]],
        "segments": seg[["segment", "customers", "revenue", "share_revenue"]].round(3).to_dict("records")
        if isinstance(seg, pd.DataFrame) and not seg.empty else [],
        "benchmarks": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()}
                       for b in r.get("benchmarks", [])],
        "repeat": {k: v for k, v in (r.get("repeat") or {}).items() if not isinstance(v, list)},
        "channels": ch[["channel", "revenue_total", "growth", "share"]].round(3).to_dict("records")
        if isinstance(ch, pd.DataFrame) and not ch.empty else [],
    }
    return facts


# ---------------- narrative ----------------

SYSTEM = (
    "You are a senior ecommerce growth analyst writing for a busy store owner. "
    "Use ONLY the numbers in the provided facts JSON; never invent, extrapolate or recompute numbers. "
    "When you reference an insight, cite it as [id]. Be specific, plain-spoken and action-oriented."
)

NARRATIVE_PROMPT = """Write a markdown business review from these facts.

Sections:
## Executive summary  (4-5 bullets: what is happening and the 3 biggest moves)
## How the business is doing  (30/90/365-day KPIs, what changed and why, short)
## Top priorities  (the ranked insights, grouped by theme; cite [id] and the estimated impact)
## 30-day action plan  (markdown table: Week | Action | Owner | Success metric | Guardrail)
## Data caveats  (data quality score and blocked analyses, if any)

Facts JSON:
{facts}
"""


def write_narrative(r: dict) -> dict:
    facts = build_facts(r)
    if not ai_enabled():
        return {"markdown": rule_based_narrative(r), "source": "rules", "unverified_numbers": []}
    try:
        md = _chat([{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": NARRATIVE_PROMPT.format(facts=json.dumps(facts, default=str))}])
    except Exception as exc:  # noqa: BLE001
        return {"markdown": rule_based_narrative(r), "source": "rules",
                "unverified_numbers": [], "error": str(exc)}
    unverified = verify_numbers(md, facts)
    return {"markdown": md, "source": "azure-openai", "unverified_numbers": unverified}


def ask(question: str, r: dict, history: list[dict] | None = None) -> str:
    if not ai_enabled():
        return "AI is not configured. Add Azure OpenAI settings to .env to enable Q&A."
    facts = build_facts(r, top_n=20)
    msgs = [{"role": "system", "content": SYSTEM + " If the facts cannot answer the question, say which "
             "data or analysis would be needed."},
            {"role": "system", "content": "Facts JSON:\n" + json.dumps(facts, default=str)}]
    msgs += (history or [])[-6:] + [{"role": "user", "content": question}]
    try:
        return _chat(msgs)
    except Exception as exc:  # noqa: BLE001
        return f"AI request failed: {exc}"


# ---------------- verification ----------------

_NUM = re.compile(r"(?<![\w.])[-+]?[$€£₹]?\d[\d,]*(?:\.\d+)?\s?[kKmM%]?(?![\w])")


def _parse(tok: str) -> tuple[float, bool] | None:
    t = tok.strip().replace(",", "").lstrip("+$€£₹")
    is_pct = t.endswith("%")
    mult = 1.0
    if t[-1:] in "kK":
        mult, t = 1e3, t[:-1]
    elif t[-1:] in "mM":
        mult, t = 1e6, t[:-1]
    t = t.rstrip("%").strip()
    try:
        return float(t) * mult, is_pct
    except ValueError:
        return None


def _collect(obj, out: list[float]):
    if isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for tok in _NUM.findall(obj):
            p = _parse(tok)
            if p:
                out.append(p[0] / 100 if p[1] else p[0])


def verify_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in `text` that don't match any number in `facts` (within rounding)."""
    known: list[float] = []
    _collect(facts, known)
    unverified = []
    body = re.sub(r"\[[^\]]+\]", "", text)  # ignore [insight_id] citations
    for tok in set(_NUM.findall(body)):
        p = _parse(tok)
        if not p:
            continue
        val, is_pct = p
        if not is_pct and float(val).is_integer() and (0 <= val <= 31 or val == 100 or 1990 <= val <= 2100):
            continue  # small counts, week numbers, days, years
        target = val / 100 if is_pct else val
        tol = 0.006 if is_pct else max(abs(target) * 0.02, 0.6)
        if not any(abs(abs(k) - abs(target)) <= tol for k in known):
            unverified.append(tok.strip())
    return sorted(unverified)


# ---------------- fallback ----------------

def rule_based_narrative(r: dict) -> str:
    cur = r["meta"]["currency"]
    m = lambda x: money(x, cur)
    snap = r.get("snapshot") or {}
    lines = ["## Executive summary", ""]
    for h in ("30d", "90d", "365d"):
        mm = (snap.get(h) or {}).get("metrics")
        if not mm:
            continue
        rev = mm["revenue"]
        chg = f" ({'+' if (rev['pop'] or 0) >= 0 else ''}{pct(rev['pop'], 1)} vs previous)" if rev["pop"] is not None else ""
        yoy = f", {'+' if (rev['yoy'] or 0) >= 0 else ''}{pct(rev['yoy'], 1)} YoY" if rev["yoy"] is not None else ""
        lines.append(f"- **Last {h}:** revenue {m(rev['value'])}{chg}{yoy}; {mm['orders']['value']:,} orders, "
                     f"AOV {m(mm['aov']['value'] or 0)}.")
    ins = r.get("insights", [])[:10]
    if ins:
        total = sum(i["impact"] for i in ins)
        lines.append(f"- The top {len(ins)} priorities below are worth an estimated **{m(total)} / year** combined.")
    lines += ["", "## Top priorities", ""]
    for i in ins:
        lines.append(f"**{i['rank']}. {i['title']}** [{i['id']}] — est. {m(i['impact'])}/yr, "
                     f"confidence {pct(i['confidence'])}")
        lines.append(f"- *What:* {i['finding']}")
        lines.append(f"- *Do:* {i['action']}")
        lines.append("")
    lines += ["## 30-day action plan", "", "| Week | Action | Success metric |", "|---|---|---|"]
    for k, i in enumerate(ins[:4], 1):
        lines.append(f"| {k} | {i['action']} | Track: {i['title']} |")
    q = r["meta"]["quality"]
    lines += ["", "## Data caveats", "",
              f"- Data quality score: **{q['score']}/100** ({q['orders']:,} orders, {q['customers']:,} customers, "
              f"{q['date_min']} to {q['date_max']})."]
    if q.get("blocked_analyses"):
        lines.append(f"- Not available with this data: {', '.join(q['blocked_analyses'])}.")
    lines.append("- *Narrative generated by rules. Configure Azure OpenAI for an AI-written review.*")
    return "\n".join(lines)
