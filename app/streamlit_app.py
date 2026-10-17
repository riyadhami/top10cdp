"""Streamlit MVP: upload ecommerce CSVs -> map columns -> ranked insights, dashboards, report.

Run:  uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ecominsights import ai, charts, exports, storage
from ecominsights.clean import build_dataset
from ecominsights.ingest import IngestError, auto_map, is_cost_file, read_csv, unmapped_columns
from ecominsights.insights import money, pct
from ecominsights.pipeline import run_analysis
from ecominsights.report import build_html, to_pdf
from ecominsights.schema import FIELDS, ColumnMapping

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
NONE = "— not in file —"


def esc(text: str) -> str:
    """Escape '$' so Streamlit markdown doesn't render currency amounts as LaTeX."""
    return str(text).replace("$", "\\$")


st.set_page_config(page_title="Store Growth Insights", page_icon="📈", layout="wide")
ss = st.session_state


# ---------------- workspace & sidebar ----------------

if "ws" not in st.query_params:
    st.query_params["ws"] = storage.new_workspace_id()
ws = st.query_params["ws"]

with st.sidebar:
    st.markdown("### 📈 Store Growth Insights")
    st.caption(f"Workspace `{ws}` — bookmark this URL to come back to your history.")
    keep_contacts = st.toggle("Keep customer emails for audience exports", value=True,
                              help="Emails are stored only in your workspace and never sent to AI. "
                                   "Turn off to keep hashed IDs only (exports disabled).")
    st.markdown(f"**AI:** {'🟢 Azure OpenAI connected' if ai.ai_enabled() else '⚪ Not configured (rule-based mode)'}")
    st.divider()
    if st.button("Load sample data (Shopify + Amazon + costs)", width="stretch"):
        ss["sample"] = True
        ss.pop("result", None)
    with st.expander("Privacy & data"):
        st.caption("Names, phones and addresses are dropped on upload. Customer IDs/emails are hashed "
                   "for analysis. Runs are deleted after the retention period.")
        if st.button("Delete all my data", type="secondary"):
            storage.delete_workspace(ws)
            for k in list(ss.keys()):
                del ss[k]
            st.success("Workspace deleted.")


# ---------------- upload & mapping ----------------

def mapping_editor(name: str, raw: pd.DataFrame, mapping: ColumnMapping) -> ColumnMapping:
    cols = list(raw.columns)
    opts = [NONE] + cols
    new = ColumnMapping(platform=mapping.platform, dropped_pii=mapping.dropped_pii)
    grid = st.columns(3)
    for k, (fld, spec) in enumerate(FIELDS.items()):
        cur = mapping.fields.get(fld)
        label = f"{fld}{' *' if spec['required'] else ''}"
        method = mapping.method.get(fld)
        choice = grid[k % 3].selectbox(label, opts, index=opts.index(cur) if cur in opts else 0,
                                       key=f"map_{name}_{fld}",
                                       help=spec["desc"] + (f" (matched: {method})" if method else ""))
        if choice != NONE:
            new.fields[fld] = choice
            new.method[fld] = method if choice == cur else "user"
    return new


uploads: list[tuple[str, bytes]] = []
if ss.get("sample"):
    uploads = [(p.name, p.read_bytes()) for p in (SAMPLES / "shopify_orders.csv", SAMPLES / "amazon_orders.csv",
                                                  SAMPLES / "product_costs.csv") if p.exists()]

if "result" not in ss:
    st.title("Turn your order exports into a growth plan")
    st.write("Upload order CSVs from **Shopify, Amazon, WooCommerce** or any store (up to 20 MB each). "
             "Add several files to compare channels, and optionally a **product costs** file (SKU or product "
             "+ unit cost) for profit analysis. We detect each file's type, map the columns, clean the data "
             "and rank the **top 10 things to do next**.")
    files = st.file_uploader("Order exports and optional product costs (CSV)", type=["csv", "tsv", "txt"],
                             accept_multiple_files=True)
    uploads += [(f.name, f.getvalue()) for f in files or []]

    currency = st.selectbox("Store currency", ["Detect from file", "USD", "EUR", "GBP", "INR", "CAD", "AUD"],
                            help="Used when the export has no currency column.")
    frames, cost_frames, ok = [], [], True
    for name, data in uploads:
        try:
            raw = read_csv(data)
        except IngestError as e:
            st.error(f"{name}: {e}")
            ok = False
            continue
        detected = "Product costs" if is_cost_file(list(raw.columns)) else "Orders"
        if detected == "Product costs":
            with st.expander(f"**{name}** — detected **product costs**, {len(raw):,} rows"):
                kind = st.segmented_control("File type", ["Orders", "Product costs"], default=detected,
                                            key=f"kind_{name}") or detected
                st.caption("Joined to order lines by SKU (or product name) to compute margins.")
                st.dataframe(raw.head(5), hide_index=True)
            if kind == "Product costs":
                cost_frames.append(raw)
                continue
        saved = storage.load_mapping(ws, list(raw.columns))
        mapping = ColumnMapping.from_dict(saved) if saved else auto_map(list(raw.columns))
        with st.expander(f"**{name}** — detected **{mapping.platform}** orders, {len(raw):,} rows "
                         f"{'(saved mapping)' if saved else ''}", expanded=bool(mapping.missing_required())):
            if detected == "Orders":
                kind = st.segmented_control("File type", ["Orders", "Product costs"], default="Orders",
                                            key=f"kind_{name}") or "Orders"
                if kind == "Product costs":
                    cost_frames.append(raw)
                    st.caption("Will be used as a product cost file.")
                    continue
            c1, c2 = st.columns([2, 1])
            channel = c1.text_input("Channel / store label", value=mapping.platform.replace("_", " ").title(),
                                    key=f"ch_{name}")
            if ai.ai_enabled() and c2.button("✨ AI-suggest mapping", key=f"ai_{name}"):
                with st.spinner("Asking Azure OpenAI (headers + masked samples only)…"):
                    ss[f"aimap_{name}"] = ai.ai_map_columns(raw, mapping).to_dict()
            if f"aimap_{name}" in ss:
                mapping = ColumnMapping.from_dict(ss[f"aimap_{name}"])
            mapping = mapping_editor(name, raw, mapping)
            if mapping.dropped_pii:
                st.caption("🔒 Dropped personal columns: " + ", ".join(mapping.dropped_pii))
            extra = unmapped_columns(list(raw.columns), mapping)
            if extra:
                st.caption("Ignored columns: " + ", ".join(extra[:25]) + (" …" if len(extra) > 25 else ""))
            st.dataframe(raw.drop(columns=mapping.dropped_pii).head(5), hide_index=True)
            missing = mapping.missing_required()
            if missing:
                st.error("Missing required fields: " + ", ".join(missing))
                ok = False
        frames.append((raw, mapping, channel, name))

    cogs_df = pd.concat(cost_frames, ignore_index=True) if cost_frames else None
    if uploads and not frames:
        st.warning("Add at least one order export — product cost files alone can't be analysed.")
    elif cost_frames:
        st.caption(f"💰 Profit analysis on: {len(cost_frames)} product cost file(s).")

    if frames and st.button("Analyze", type="primary", disabled=not ok):
        with st.status("Analyzing…", expanded=True) as status:
            st.write("Cleaning and combining files…")
            for raw, mapping, _, _ in frames:
                storage.save_mapping(ws, list(raw.columns), mapping.to_dict())
            ds = build_dataset([(r, m, c) for r, m, c, _ in frames], cogs=cogs_df, keep_contacts=keep_contacts)
            st.write(f"{ds.quality['orders']:,} orders, {ds.quality['customers']:,} customers. Running analytics…")
            result = run_analysis(ds, currency=None if currency == "Detect from file" else currency)
            st.write("Saving run…")
            run_id = storage.save_run(ws, ds, result, label=", ".join(n for *_, n in frames))
            ss.update(result=result, ds=ds, run_id=run_id)
            ss.pop("narrative", None)
            ss.pop("pdf", None)
            status.update(label="Done", state="complete")
        st.rerun()
    st.stop()


# ---------------- results ----------------

r, ds = ss["result"], ss["ds"]
cur = r["meta"]["currency"]
sym = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "CAD": "C$", "AUD": "A$"}.get(cur, "")
q = r["meta"]["quality"]
m = lambda x: money(x, cur)


_chart_n = [0]


def show(fig):
    if fig is not None:
        _chart_n[0] += 1
        st.plotly_chart(fig, config={"displayModeBar": False}, key=f"chart_{_chart_n[0]}")


head = st.columns([4, 1])
head[0].title("Growth review")
head[0].caption(f"{q['date_min']} → {q['date_max']} · {q['orders']:,} orders · {q['customers']:,} customers · "
                f"channels: {', '.join(s['channel'] for s in r['meta']['sources'])} · data quality {q['score']}/100")
if head[1].button("↺ New upload"):
    for k in ("result", "ds", "narrative", "sample", "run_id", "pdf"):
        ss.pop(k, None)
    st.rerun()
if r["meta"].get("module_errors"):
    st.warning("Some analyses were skipped: " + "; ".join(r["meta"]["module_errors"]))

tabs = st.tabs(["Overview", "Top 10 actions", "Customers", "Cohorts", "Products", "Channels",
                "Timing & trends", "AI review & report", "History", "Data quality"])

# ---- Overview ----
with tabs[0]:
    horizon = st.segmented_control("Horizon", ["30d", "90d", "365d"], default="90d", key="hz") or "90d"
    snap = r["snapshot"].get(horizon) or {}
    mm = snap.get("metrics", {})
    st.caption(f"Window {snap.get('window', ['', ''])[0]} → {snap.get('window', ['', ''])[1]}, "
               "compared with the previous period of the same length.")
    tiles = [("revenue", "Revenue", "money"), ("orders", "Orders", "int"), ("customers", "Customers", "int"),
             ("aov", "Avg order value", "money"), ("repeat_rate", "Repeat customers", "pct"),
             ("gross_margin", "Gross margin", "pct")]
    present = [t for t in tiles if t[0] in mm]
    cols = st.columns(3) + st.columns(3)
    for col, (k, label, fmt) in zip(cols, present):
        v = mm[k]["value"]
        val = m(v) if fmt == "money" else pct(v or 0) if fmt == "pct" else f"{int(v):,}"
        d = mm[k]["pop"]
        col.metric(label, val, f"{d * 100:+.1f}%" if d is not None else None,
                   help=f"YoY: {mm[k]['yoy'] * 100:+.1f}%" if mm[k].get("yoy") is not None else None)
    if snap.get("tree"):
        show(charts.kpi_waterfall(snap["tree"], sym))
        st.caption("Revenue = Customers × Orders/customer × Units/order × Price/unit. Bars show how much each "
                   "driver added or removed (LMDI decomposition; parts sum exactly to the change).")
    c1, c2 = st.columns(2)
    with c1:
        show(charts.monthly_revenue(r["monthly"], sym))
    with c2:
        show(charts.weekly_forecast(r["weekly"], r["forecast"], sym))
    attr = (r.get("attribution") or {}).get("dimensions") or {}
    if attr:
        dim = st.selectbox("Explain the 90-day revenue change by", list(attr), index=0)
        show(charts.figure_for_insight(f"attribution:{dim}", r, ds, sym))

# ---- Top 10 ----
with tabs[1]:
    ins = r["insights"][:10]
    if not ins:
        st.info("Not enough data to rank insights.")
    else:
        st.subheader(esc(f"Your top {len(ins)} priorities — est. {m(sum(i['impact'] for i in ins))}/yr combined"))
        st.caption("Ranked by impact × confidence × actionability ÷ effort. Impact estimates are directional; "
                   "each card shows its formula.")
        kind_icon = {"risk": "🔴", "opportunity": "🟢", "win": "🏆"}
        for i in ins:
            with st.expander(esc(f"{kind_icon.get(i['kind'], '')} **{i['rank']}. {i['title']}** — "
                                 f"{m(i['impact'])}/yr · {pct(i['confidence'])} confidence"), expanded=i["rank"] <= 3):
                a, b = st.columns([3, 2])
                a.markdown(esc(f"**What we found:** {i['finding']}\n\n**What to do:** {i['action']}"))
                b.markdown(f"Category: `{i['category']}`  \nEffort: {'●' * i['effort']}{'○' * (3 - i['effort'])}  \n"
                           f"Impact formula: _{i['impact_formula']}_")
                show(charts.figure_for_insight(i.get("chart"), r, ds, sym))
        if len(r["insights"]) > 10:
            with st.expander(f"{len(r['insights']) - 10} more insights"):
                for i in r["insights"][10:]:
                    st.markdown(esc(f"- **{i['title']}** — {i['finding']}"))

# ---- Customers ----
with tabs[2]:
    cust = r.get("customers")
    if cust is None or cust.empty:
        st.info("Customer analyses need a customer ID or email column.")
    else:
        rep = r.get("repeat") or {}
        c = st.columns(4)
        c[0].metric("Customers", f"{rep.get('customers', 0):,}")
        c[1].metric("Repeat customer rate", pct(rep.get("repeat_customer_rate") or 0))
        c[2].metric("Median days to 2nd order", f"{rep.get('median_days_to_second_order') or 0:.0f}")
        c[3].metric("Revenue from top 20%", pct(rep.get("top20_revenue_share") or 0))
        cm = r.get("clv_model") or {}
        bt = cm.get("backtest")
        with st.expander("How reliable are the customer value predictions?"):
            if cm.get("method") == "bg_nbd_gamma_gamma":
                st.markdown("Predicted 12-month value uses **BG/NBD** (how often each customer buys and the "
                            "chance they're still active) × **Gamma-Gamma** (how much they spend per order).")
            else:
                st.markdown("Not enough repeat buyers to fit the statistical model; using a recency-based heuristic.")
            if bt:
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Predicted orders (last {bt['holdout_days']}d, held out)", f"{bt['predicted_total']:,.0f}")
                c2.metric("Actual orders", f"{bt['actual_total']:,.0f}",
                          f"{bt['total_error'] * 100:+.0f}% error" if bt.get("total_error") is not None else None,
                          delta_color="off")
                c3.metric("Per-customer correlation", f"{bt['correlation']:.2f}" if bt.get("correlation") else "—")
                weak = abs(bt.get("total_error") or 0) > 0.3 or (bt.get("correlation") or 0) < 0.5
                (st.warning if weak else st.success)(
                    "Backtest is weak: treat predicted values as rough rankings, not forecasts." if weak else
                    "Backtest is good: predictions tracked actual repeat orders in the held-out period.")
        show(charts.segments_bar(r["segments"], sym))
        st.dataframe(r["segments"], hide_index=True, column_config={
            "revenue": st.column_config.NumberColumn(format="%.0f"),
            "share_customers": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "share_revenue": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "predicted_12m_value": st.column_config.NumberColumn(format="%.0f")})
        st.subheader("Top 10 customer lists")
        lists = r["top_lists"]
        names = {"vip": "💎 VIPs (highest predicted 12-month value)", "at_risk": "⚠️ High-value at risk",
                 "winback": "🔁 Win-back (lapsed repeat buyers)", "rising": "🌱 Rising new customers"}
        contacts = ds.contacts.set_index("customer_key")["email"] if len(ds.contacts) else pd.Series(dtype=str)
        for key, label in names.items():
            df = lists.get(key)
            if df is None or df.empty:
                continue
            with st.expander(label, expanded=key == "at_risk"):
                show_df = df.copy()
                show_df.insert(0, "customer", show_df["customer_key"].map(contacts).fillna(show_df["customer_key"]))
                st.dataframe(show_df.drop(columns=["customer_key"]), hide_index=True)
        st.subheader("Export audiences")
        if not len(ds.contacts):
            st.info("Exports need emails. Enable 'Keep customer emails' in the sidebar and re-run.")
        else:
            c1, c2 = st.columns(2)
            segs = c1.multiselect("Segments", list(r["segments"]["segment"]), default=["At Risk", "Can't Lose Them"])
            fmt = c2.selectbox("Format", list(exports.FORMATS), format_func=lambda k: exports.FORMATS[k])
            aud = exports.audience(cust, ds.contacts, segs)
            st.download_button(f"Download {len(aud):,} contacts", exports.export(aud, fmt),
                               file_name=f"audience_{'_'.join(s.replace(' ', '') for s in segs) or 'all'}_{fmt}.csv",
                               mime="text/csv", disabled=aud.empty)
        show(charts.days_to_second_hist(rep.get("days_to_second_hist", [])))

# ---- Cohorts ----
with tabs[3]:
    co = r.get("cohorts") or {}
    if not co:
        st.info("Cohorts need customer IDs.")
    else:
        show(charts.retention_heatmap(co["retention"]))
        c1, c2 = st.columns(2)
        with c1:
            show(charts.ltv_curve(co["avg_ltv"], sym))
        with c2:
            if r.get("discount_cohorts"):
                show(charts.discount_cohort_bar(r["discount_cohorts"]))
                dc = r["discount_cohorts"]
                st.caption(f"Two-proportion z-test p = {dc['p_value']:.4f} "
                           f"({'significant' if dc['p_value'] < 0.05 else 'not significant'} at 5%).")
        fp = r.get("first_product")
        if isinstance(fp, pd.DataFrame) and not fp.empty:
            show(charts.first_product_bar(fp, sym))
            st.caption("Which first purchase creates the most valuable customers — use the winners in acquisition.")

# ---- Products ----
with tabs[4]:
    pt = r.get("products")
    if pt is None or pt.empty:
        st.info("Product analyses need a product name or SKU column.")
    else:
        p = r.get("pareto") or {}
        c = st.columns(3)
        c[0].metric("Products sold", p.get("products"))
        c[1].metric("Products making 80% of revenue", f"{p.get('products_for_80pct')} ({pct(p.get('share_for_80pct') or 0)})")
        c[2].metric("Top product share", pct(p.get("top_product_share") or 0), help=p.get("top_product"))
        c1, c2 = st.columns(2)
        with c1:
            show(charts.pareto_bar(pt, sym))
        with c2:
            show(charts.momentum_bar(pt))
        if "margin" in pt:
            show(charts.margin_scatter(pt, sym))
        if isinstance(r.get("basket"), pd.DataFrame) and not r["basket"].empty:
            show(charts.basket_bar(r["basket"]))
        so = r.get("stockouts")
        if isinstance(so, pd.DataFrame) and not so.empty:
            st.subheader("Likely stock-outs")
            st.dataframe(so, hide_index=True)
            pick = st.selectbox("Inspect product", so["product"].unique())
            s = so[so["product"] == pick].iloc[0]
            show(charts.stockout_timeline(ds.lines, pick, s["gap_start"], s["gap_end"]))
        rp = r.get("replenishment")
        if isinstance(rp, pd.DataFrame) and not rp.empty:
            st.subheader("Replenishment cadence")
            st.dataframe(rp, hide_index=True)
        rt = r.get("returns")
        if isinstance(rt, pd.DataFrame) and not rt.empty:
            st.subheader("Returns")
            st.dataframe(rt, hide_index=True)
        with st.expander("All products"):
            st.dataframe(pt, hide_index=True)

# ---- Channels ----
with tabs[5]:
    ch = r.get("channels")
    if ch is None or (isinstance(ch, pd.DataFrame) and ch.empty):
        st.info("Upload files from more than one channel or store (e.g. Shopify + Amazon) to compare them.")
    else:
        st.dataframe(ch, hide_index=True, column_config={
            "share": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "growth": st.column_config.NumberColumn(format="percent"),
            "identified_customers": st.column_config.NumberColumn("customer IDs present", format="percent")})
        show(charts.channel_lines(r["channel_monthly"], sym))
        st.caption(f"{(r.get('repeat') or {}).get('multi_channel_customers', 0)} customers bought in more than one channel "
                   "(matched on hashed email/ID; marketplace orders without buyer IDs can't be matched).")

# ---- Timing ----
with tabs[6]:
    wd = r.get("weekday") or {}
    if wd.get("has_time"):
        show(charts.weekday_hour_heatmap(wd["heatmap"], sym))
    if r.get("seasonality"):
        show(charts.seasonality_bar(r["seasonality"]["month_index"]))
    show(charts.anomalies_chart(ds.orders, r.get("anomalies", pd.DataFrame()), sym))
    an = r.get("anomalies")
    if isinstance(an, pd.DataFrame) and not an.empty:
        st.dataframe(an.sort_values("date", ascending=False), hide_index=True)

# ---- AI review & report ----
with tabs[7]:
    if "narrative" not in ss:
        with st.spinner("Writing the review…" if ai.ai_enabled() else "Building the rule-based review…"):
            ss["narrative"] = ai.write_narrative(r)
    n = ss["narrative"]
    c1, c2 = st.columns([3, 1])
    c1.caption(f"Written by {'Azure OpenAI from the computed facts only' if n['source'] != 'rules' else 'rules (AI not configured)'}.")
    if c2.button("Regenerate"):
        ss.pop("narrative")
        ss.pop("pdf", None)
        st.rerun()
    if n.get("error"):
        st.warning(f"AI call failed, showing rule-based review: {n['error']}")
    if n.get("unverified_numbers"):
        st.warning(esc("These numbers could not be matched to the computed facts — double-check them: "
                       + ", ".join(n["unverified_numbers"])))
    st.markdown(esc(n["markdown"]))
    d1, d2, d3 = st.columns(3)
    if d1.button("📄 Prepare PDF report", type="primary"):
        with st.spinner("Rendering PDF…"):
            try:
                ss["pdf"] = to_pdf(build_html(r, n, ds, for_print=True))
            except Exception as exc:  # noqa: BLE001
                st.error(f"PDF rendering failed ({exc}). Run `uv run playwright install chromium` once.")
    if ss.get("pdf"):
        d1.download_button("⬇️ Download PDF", ss["pdf"], file_name="growth_review.pdf", mime="application/pdf")
    d2.download_button("⬇️ Interactive report (HTML)", build_html(r, n, ds).encode(),
                       file_name="growth_review.html", mime="text/html")
    d3.download_button("⬇️ Review (Markdown)", n["markdown"].encode(), file_name="REVIEW.md")
    st.divider()
    st.subheader("Ask about your data")
    if not ai.ai_enabled():
        st.caption("Configure Azure OpenAI in `.env` to ask questions.")
    else:
        hist = ss.setdefault("chat", [])
        for msg in hist:
            st.chat_message(msg["role"]).markdown(esc(msg["content"]))
        if qn := st.chat_input("e.g. Which segment should I email first and with what offer?"):
            st.chat_message("user").markdown(esc(qn))
            ans = ai.ask(qn, r, hist)
            st.chat_message("assistant").markdown(esc(ans))
            hist += [{"role": "user", "content": qn}, {"role": "assistant", "content": ans}]

# ---- History ----
with tabs[8]:
    runs = storage.list_runs(ws)
    st.dataframe(pd.DataFrame(runs).drop(columns=["sources"], errors="ignore"), hide_index=True)
    if len(runs) < 2:
        st.info("Upload again next week/month to see what changed between uploads.")
    else:
        ids = [x["run_id"] for x in runs]
        c1, c2 = st.columns(2)
        new_id = c1.selectbox("Latest upload", ids, index=0)
        old_id = c2.selectbox("Compare with", ids, index=1)
        if new_id != old_id:
            cmp = storage.compare_runs(ws, old_id, new_id)
            st.dataframe(pd.DataFrame(cmp["kpis"]), hide_index=True,
                         column_config={"change": st.column_config.NumberColumn(format="percent")})
            c1, c2 = st.columns(2)
            c1.markdown(esc("**New priorities**\n" + ("\n".join(f"- {i['title']}" for i in cmp["new_insights"]) or "- none")))
            c2.markdown(esc("**No longer in top 10**\n" + ("\n".join(f"- {i['title']}" for i in cmp["resolved_insights"]) or "- none")))
            if cmp["migration"] is not None:
                st.markdown("**Customer segment migration** (rows: previous segment, columns: now)")
                st.dataframe(cmp["migration"])

# ---- Data quality ----
with tabs[9]:
    st.metric("Data quality score", f"{q['score']}/100")
    st.dataframe(pd.DataFrame(q["checks"]).drop(columns=["weight"]), hide_index=True,
                 column_config={"score": st.column_config.ProgressColumn(min_value=0, max_value=100)})
    if q["blocked_analyses"]:
        st.warning("Not available with this data: " + ", ".join(q["blocked_analyses"]))
    st.json(q["stats"])
    st.dataframe(pd.DataFrame(r["meta"]["sources"]), hide_index=True)
