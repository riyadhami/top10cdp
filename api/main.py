"""FastAPI service exposing the same engine the Streamlit app uses.

Run:  uv run uvicorn api.main:app --reload
Analysis is synchronous (files are <= 20 MB); move to a job queue (RQ/Celery) when traffic grows.
"""

from __future__ import annotations

import json
import os
import secrets

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ecominsights import ai, exports, storage
from ecominsights.clean import build_dataset
from ecominsights.ingest import IngestError, auto_map, is_cost_file, read_csv
from ecominsights.pipeline import run_analysis, to_jsonable
from ecominsights.report import build_html
from ecominsights.schema import ColumnMapping



def require_api_key(x_api_key: str | None = Header(None)) -> None:
    """When API_KEY is set, every /v1 call must send it as X-API-Key."""
    expected = os.environ.get("API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(401, "Missing or invalid X-API-Key")


app = FastAPI(title="Store Growth Insights API", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True, "ai": ai.ai_enabled()}


@app.post("/v1/mapping/suggest", dependencies=[Depends(require_api_key)])
async def suggest_mapping(file: UploadFile = File(...), use_ai: bool = Form(False)):
    try:
        raw = read_csv(await file.read())
    except IngestError as e:
        raise HTTPException(400, str(e)) from e
    mapping = auto_map(list(raw.columns))
    if use_ai:
        mapping = ai.ai_map_columns(raw, mapping)
    return {"columns": list(raw.columns), "rows": len(raw), "mapping": mapping.to_dict(),
            "missing_required": mapping.missing_required()}


@app.post("/v1/analyze", dependencies=[Depends(require_api_key)])
async def analyze(files: list[UploadFile] = File(..., description="Order exports and optional product cost "
                                                                     "files; cost files are detected automatically"),
                  channels: str = Form("", description="Comma-separated channel label per file"),
                  mappings: str = Form("", description="Optional JSON list of mappings, one per file"),
                  currency: str = Form("", description="Currency code if the exports don't include one"),
                  workspace: str = Form(""),
                  keep_contacts: bool = Form(True),
                  narrative: bool = Form(True)):
    labels = [c.strip() for c in channels.split(",")] if channels else []
    given = json.loads(mappings) if mappings else []
    frames, cost_frames, order_files = [], [], []
    for k, f in enumerate(files):
        try:
            raw = read_csv(await f.read())
        except IngestError as e:
            raise HTTPException(400, f"{f.filename}: {e}") from e
        if not (k < len(given) and given[k]) and is_cost_file(list(raw.columns)):
            cost_frames.append(raw)
            continue
        order_files.append(f.filename or "")
        mapping = ColumnMapping.from_dict(given[k]) if k < len(given) and given[k] else auto_map(list(raw.columns))
        if mapping.missing_required():
            raise HTTPException(422, {"file": f.filename, "missing_required": mapping.missing_required(),
                                      "mapping": mapping.to_dict()})
        frames.append((raw, mapping, labels[k] if k < len(labels) and labels[k] else None))
    if not frames:
        raise HTTPException(422, "No order export found; product cost files alone can't be analysed.")
    cogs_df = pd.concat(cost_frames, ignore_index=True) if cost_frames else None
    ds = build_dataset(frames, cogs=cogs_df, keep_contacts=keep_contacts)
    result = run_analysis(ds, currency=currency or None)
    ws = workspace or storage.new_workspace_id()
    run_id = storage.save_run(ws, ds, result, label=", ".join(order_files))
    body = {"workspace": ws, "run_id": run_id,
            **to_jsonable({k: v for k, v in result.items() if k != "customers"})}
    if narrative:
        body["narrative"] = ai.write_narrative(result)
    return JSONResponse(body)


@app.get("/v1/workspaces/{ws}/runs", dependencies=[Depends(require_api_key)])
def runs(ws: str):
    return storage.list_runs(ws)


@app.get("/v1/workspaces/{ws}/runs/{run_id}", dependencies=[Depends(require_api_key)])
def run(ws: str, run_id: str):
    try:
        return storage.load_run(ws, run_id)["result"]
    except FileNotFoundError as e:
        raise HTTPException(404, "Run not found") from e


@app.get("/v1/workspaces/{ws}/compare", dependencies=[Depends(require_api_key)])
def compare(ws: str, previous: str, latest: str):
    c = storage.compare_runs(ws, previous, latest)
    if c["migration"] is not None:
        c["migration"] = c["migration"].to_dict()
    return to_jsonable(c)


@app.get("/v1/workspaces/{ws}/runs/{run_id}/report.html", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
def report(ws: str, run_id: str):
    # Rebuilt from the stored JSON (no raw data), so per-insight charts that need order rows are omitted.
    res = storage.load_run(ws, run_id)["result"]
    for key in ("segments", "products", "monthly", "weekly", "forecast"):
        if isinstance(res.get(key), list):
            res[key] = pd.DataFrame(res[key])
    return build_html(res, None)


@app.get("/v1/workspaces/{ws}/runs/{run_id}/audience", dependencies=[Depends(require_api_key)])
def audience(ws: str, run_id: str, segments: str = "", fmt: str = "generic"):
    d = storage._ws(ws) / "runs" / run_id
    if not (d / "contacts.parquet").exists() or not (d / "customers.parquet").exists():
        raise HTTPException(404, "No contacts stored for this run")
    cust = pd.read_parquet(d / "customers.parquet")
    contacts = pd.read_parquet(d / "contacts.parquet")
    aud = exports.audience(cust, contacts, [s for s in segments.split(",") if s] or None)
    return Response(exports.export(aud, fmt), media_type="text/csv")


@app.delete("/v1/workspaces/{ws}", dependencies=[Depends(require_api_key)])
def delete(ws: str):
    storage.delete_workspace(ws)
    return {"deleted": ws}
