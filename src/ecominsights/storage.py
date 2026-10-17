"""Workspace storage (local Parquet + JSON), run history, re-upload comparison and retention.

Layout:  DATA_DIR/<workspace>/
            mappings.json                    saved column mappings by header signature
            runs/<run_id>/orders.parquet     canonical orders (hashed customer keys)
                         lines.parquet
                         customers.parquet   RFM / CLV table
                         contacts.parquet    customer_key -> email (only if the user allows it)
                         result.json         aggregated results (no per-customer rows)
                         run.json            run metadata
Postgres + object storage replace this for multi-user deployment; the interface stays the same.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .clean import Dataset
from .pipeline import to_json


def data_dir() -> Path:
    p = Path(os.environ.get("DATA_DIR", "data/workspaces"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ws(workspace: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", workspace):
        raise ValueError("Workspace id must be 3-64 letters, digits, '-' or '_'.")
    p = data_dir() / workspace
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_workspace_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------- mapping templates ----------

def header_signature(columns: list[str]) -> str:
    return hashlib.sha1("|".join(sorted(columns)).encode()).hexdigest()[:12]


def save_mapping(workspace: str, columns: list[str], mapping: dict) -> None:
    f = _ws(workspace) / "mappings.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[header_signature(columns)] = mapping
    f.write_text(json.dumps(data, indent=1))


def load_mapping(workspace: str, columns: list[str]) -> dict | None:
    f = _ws(workspace) / "mappings.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get(header_signature(columns))


# ---------- runs ----------

def save_run(workspace: str, ds: Dataset, result: dict, label: str = "") -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = _ws(workspace) / "runs" / run_id
    d.mkdir(parents=True)
    ds.orders.to_parquet(d / "orders.parquet", index=False)
    ds.lines.to_parquet(d / "lines.parquet", index=False)
    if isinstance(result.get("customers"), pd.DataFrame) and not result["customers"].empty:
        result["customers"].to_parquet(d / "customers.parquet")
    if len(ds.contacts):
        ds.contacts.to_parquet(d / "contacts.parquet", index=False)
    (d / "result.json").write_text(to_json(result))
    q = ds.quality
    meta = {"run_id": run_id, "label": label, "created_at": datetime.now().isoformat(timespec="seconds"),
            "orders": q["orders"], "customers": q["customers"], "date_min": q["date_min"],
            "date_max": q["date_max"], "quality_score": q["score"], "sources": ds.sources}
    (d / "run.json").write_text(json.dumps(meta, indent=1))
    return run_id


def list_runs(workspace: str) -> list[dict]:
    root = _ws(workspace) / "runs"
    if not root.exists():
        return []
    runs = [json.loads((p / "run.json").read_text()) for p in root.iterdir() if (p / "run.json").exists()]
    return sorted(runs, key=lambda r: r["created_at"], reverse=True)


def load_run(workspace: str, run_id: str) -> dict:
    d = _ws(workspace) / "runs" / run_id
    out = {"meta": json.loads((d / "run.json").read_text()),
           "result": json.loads((d / "result.json").read_text())}
    if (d / "customers.parquet").exists():
        out["customers"] = pd.read_parquet(d / "customers.parquet")
    return out


def compare_runs(workspace: str, prev_id: str, cur_id: str) -> dict:
    """What changed between two uploads: KPIs, insights, and customer segment migration."""
    prev, cur = load_run(workspace, prev_id), load_run(workspace, cur_id)
    kpi = []
    for h in ("30d", "90d", "365d"):
        pm = ((prev["result"].get("snapshot") or {}).get(h) or {}).get("metrics", {})
        cm = ((cur["result"].get("snapshot") or {}).get(h) or {}).get("metrics", {})
        for k in ("revenue", "orders", "customers", "aov", "repeat_rate", "gross_margin"):
            if k in cm and k in pm and cm[k]["value"] is not None and pm[k]["value"] is not None:
                kpi.append({"horizon": h, "metric": k, "previous_upload": pm[k]["value"],
                            "latest_upload": cm[k]["value"],
                            "change": (cm[k]["value"] / pm[k]["value"] - 1) if pm[k]["value"] else None})
    p_ids = {i["id"]: i for i in prev["result"].get("insights", [])[:10]}
    c_ids = {i["id"]: i for i in cur["result"].get("insights", [])[:10]}
    migration = None
    if "customers" in prev and "customers" in cur:
        j = prev["customers"][["segment"]].join(cur["customers"][["segment"]], lsuffix="_prev",
                                                 rsuffix="_now", how="inner")
        migration = pd.crosstab(j["segment_prev"], j["segment_now"])
    return {"kpis": kpi, "new_insights": [c_ids[k] for k in c_ids if k not in p_ids],
            "resolved_insights": [p_ids[k] for k in p_ids if k not in c_ids],
            "migration": migration, "previous": prev["meta"], "latest": cur["meta"]}


# ---------- privacy ----------

def delete_workspace(workspace: str) -> None:
    shutil.rmtree(_ws(workspace), ignore_errors=True)


def purge_expired(retention_days: int | None = None) -> int:
    """Delete runs older than RETENTION_DAYS. Returns the number of runs removed."""
    days = retention_days or int(os.environ.get("RETENTION_DAYS", "90"))
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    for ws in data_dir().iterdir():
        runs = ws / "runs"
        if not runs.is_dir():
            continue
        for r in runs.iterdir():
            meta = r / "run.json"
            if meta.exists() and datetime.fromisoformat(json.loads(meta.read_text())["created_at"]) < cutoff:
                shutil.rmtree(r, ignore_errors=True)
                removed += 1
    return removed
