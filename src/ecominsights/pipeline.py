"""Orchestrates the deterministic analysis: Dataset -> result dict -> ranked insights."""

from __future__ import annotations

import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import benchmarks
from .analytics import channels, cohorts, customers, kpis, products, timeseries
from .clean import Dataset
from .insights import generate_insights


def _safe(fn, *args, default=None, errors: list | None = None, **kw):
    """Run one analysis module; a failure in one module must not sink the whole report."""
    try:
        return fn(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        if errors is not None:
            errors.append(f"{fn.__module__.split('.')[-1]}.{fn.__name__}: {exc}")
        return default


def run_analysis(ds: Dataset) -> dict:
    o, l = ds.orders, ds.lines
    errs: list[str] = []
    r: dict = {"meta": {"generated_at": datetime.now().isoformat(timespec="seconds"),
                        "sources": ds.sources, "quality": ds.quality,
                        "currency": ds.quality.get("currency") or "USD"}}

    r["snapshot"] = _safe(kpis.snapshot, o, default={}, errors=errs)
    r["attribution"] = _safe(kpis.attribution, o, l, default={}, errors=errs)
    r["monthly"] = _safe(kpis.monthly, o, default=pd.DataFrame(), errors=errs)

    cust = _safe(customers.customer_table, o, l, default=pd.DataFrame(), errors=errs)
    if len(cust) >= 20:
        cust = customers.clv(customers.rfm(cust))
        r["customers"] = cust
        r["segments"] = _safe(customers.segment_summary, cust, default=pd.DataFrame(), errors=errs)
        r["top_lists"] = _safe(customers.top_lists, cust, default={}, errors=errs)
        r["repeat"] = _safe(customers.repeat_behavior, cust, o, default={}, errors=errs)
        r["cohorts"] = _safe(cohorts.retention, o, default={}, errors=errs)
        r["first_product"] = _safe(cohorts.first_product, o, cust, default=pd.DataFrame(), errors=errs)
        r["discount_cohorts"] = _safe(cohorts.discount_cohorts, o, cust, default={}, errors=errs)
    else:
        r["customers"] = pd.DataFrame()

    ptab = _safe(products.product_table, l, default=pd.DataFrame(), errors=errs)
    r["products"] = ptab
    if not ptab.empty:
        r["pareto"] = products.pareto_summary(ptab)
        r["basket"] = _safe(products.basket_pairs, l, default=pd.DataFrame(), errors=errs)
        r["stockouts"] = _safe(products.stockouts, l, default=pd.DataFrame(), errors=errs)
        repl = _safe(products.replenishment, l, default=pd.DataFrame(), errors=errs)
        r["replenishment"] = repl
        r["due_reorder"] = _safe(products.due_for_reorder, l, repl, default=pd.DataFrame(), errors=errs)
        r["returns"] = _safe(products.returns_table, ds.returns, ptab, default=pd.DataFrame(), errors=errs)

    r["weekly"] = _safe(timeseries.weekly, o, default=pd.DataFrame(), errors=errs)
    r["anomalies"] = _safe(timeseries.anomalies, o, default=pd.DataFrame(), errors=errs)
    r["weekday"] = _safe(timeseries.weekday_hour, o, default={}, errors=errs)
    r["seasonality"] = _safe(timeseries.seasonality, o, default={}, errors=errs)
    r["trend"] = _safe(timeseries.trend, o, default={}, errors=errs)
    r["forecast"] = _safe(timeseries.forecast, o, default=pd.DataFrame(), errors=errs)

    if o["channel"].nunique() > 1:
        r["channels"] = _safe(channels.channel_table, o, default=pd.DataFrame(), errors=errs)
        r["channel_monthly"] = _safe(channels.monthly_by_channel, o, default=pd.DataFrame(), errors=errs)

    r["benchmarks"] = benchmarks.compare(_benchmark_values(r, o))
    r["insights"] = _safe(generate_insights, r, default=[], errors=errs)
    r["meta"]["module_errors"] = errs
    return r


def _benchmark_values(r: dict, o: pd.DataFrame) -> dict:
    rep = r.get("repeat") or {}
    m365 = ((r.get("snapshot") or {}).get("365d") or {}).get("metrics", {})
    v = {"repeat_customer_rate": rep.get("repeat_customer_rate"),
         "second_order_within_90d": rep.get("second_order_within_90d"),
         "top20_revenue_share": rep.get("top20_revenue_share"),
         "returning_revenue_share": (m365.get("returning_revenue_share") or {}).get("value"),
         "discount_rate": (m365.get("discount_rate") or {}).get("value"),
         "gross_margin": (m365.get("gross_margin") or {}).get("value")}
    ret = r.get("returns")
    if isinstance(ret, pd.DataFrame) and not ret.empty:
        v["return_rate"] = float(ret["returned_units"].sum() / (ret["units"].sum() + ret["returned_units"].sum()))
    return {k: (float(x) if x is not None and np.isfinite(x) else None) for k, x in v.items()}


# ---------- serialization ----------

def to_jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
        df.columns = [str(c) for c in df.columns]
        return [to_jsonable(rec) for rec in df.to_dict("records")]
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def to_json(result: dict, drop: tuple[str, ...] = ("customers",)) -> str:
    """Serialize a result; the per-customer table is dropped by default (it is PII-adjacent)."""
    return json.dumps(to_jsonable({k: v for k, v in result.items() if k not in drop}), indent=1)
