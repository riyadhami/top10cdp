# Store Growth Insights (MVP)

Upload ecommerce order exports (Shopify, Amazon, WooCommerce, Online Retail or any CSV) and get a
ranked **Top 10 action list**, customer segments, cohorts, product and channel analytics, an
AI-written review, and audience exports.

**Principle: math decides, AI explains.** Every number is computed deterministically in Python.
Azure OpenAI only maps messy columns, writes the narrative from a facts JSON, and answers questions,
and every number it writes is checked against those facts.

## Run

```bash
uv sync
uv run python scripts/make_sample_data.py      # sample Shopify + Amazon + COGS files
uv run streamlit run app/streamlit_app.py      # UI  -> http://localhost:8501
uv run uvicorn api.main:app --port 8000        # API -> http://localhost:8000/docs
uv run pytest -q
```

Copy `.env.example` to `.env` and fill in the `AZURE_OPENAI_*` settings to enable AI. Without them
the app runs in rule-based mode.

## Pipeline

```
CSV(s) ─► ingest.py      sniff encoding/delimiter, detect platform, map headers (exact → fuzzy → AI)
       ─► clean.py       dedupe line items → orders, cancellations/returns/test orders, money/date parsing,
                         hash customer IDs, drop PII columns, join COGS, data-quality score
       ─► analytics/     kpis (30/90/365d, KPI tree with LMDI, mix-shift attribution)
                         customers (RFM segments, CLV/churn heuristic, top-10 lists)
                         cohorts (retention, LTV curve, first-product & discount cohorts + z-test)
                         products (ABC/Pareto, momentum, basket lift, margin, Poisson stock-out test,
                                   replenishment cadence, returns)
                         timeseries (seasonality, weekday×hour, robust-z anomalies, Holt-Winters forecast)
                         channels (multi-store comparison)
       ─► insights.py    candidates with $ impact formula + confidence → score = impact×conf×action/effort
       ─► ai.py          Azure OpenAI narrative / Q&A on facts only, with number verification
       ─► report.py      HTML report · exports.py audience CSVs · storage.py runs, history, retention
```

## Add-ons included

1. Profit: upload a cost file (sku/product, unit_cost) to get margin by product, channel and order.
3. Audience exports: Klaviyo, Mailchimp, Meta and Google (hashed) CSVs per segment.
5. Data-quality score, listing which analyses each gap blocks.
6. Re-upload history: KPI deltas, new and resolved priorities, segment migration matrix.
7. Multi-channel: upload several files with a channel label each.
8. Indicative DTC benchmarks (edit `src/ecominsights/benchmarks.py`).
9. Privacy: PII columns dropped, salted hashes for IDs, emails kept only with opt-in and never sent
   to AI, delete-workspace control, `RETENTION_DAYS` purge.

## Next phases

- Replace the CLV heuristic with BG/NBD + Gamma-Gamma.
- PDF export (static chart images).
- Auth and multi-tenant storage (Postgres + object storage) for the self-serve SaaS.
- Background jobs, scheduled digests, Shopify/Amazon API connectors.
