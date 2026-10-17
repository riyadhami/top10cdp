"""HTML report (charts embedded via Plotly) and PDF export (the same HTML printed by headless Chromium)."""

from __future__ import annotations

import html

import markdown as md
import pandas as pd

from . import charts
from .insights import money, pct

CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--accent:#2a78d6;--good:#006300;--bad:#d03b3b}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--line:#2c2c2a;--accent:#3987e5;--good:#0ca30c;--bad:#e66767}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 12px;border-bottom:1px solid var(--line);
padding-bottom:6px}h3{font-size:16px;margin:0 0 6px}.sub{color:var(--ink2)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .l{color:var(--ink2);font-size:13px}.kpi .v{font-size:22px;font-weight:600}
.kpi .d{font-size:13px}.up{color:var(--good)}.down{color:var(--bad)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
.meta{color:var(--muted);font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px;
font-variant-numeric:tabular-nums}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--ink2);font-weight:600}.narr table{margin:8px 0}
@media print{.card{break-inside:avoid}}
"""

# PDFs are always light, with page breaks between sections.
PRINT_CSS = """
:root{--bg:#fff;--surface:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#6b6a66;--line:#e1e0d9;
--accent:#2a78d6;--good:#006300;--bad:#d03b3b}
main{max-width:none;padding:0}h2{break-after:avoid}.card{break-inside:avoid}
.kpi{break-inside:avoid}.card.narr{break-inside:auto}.narr table,.narr tr{break-inside:avoid}.js-plotly-plot,.plotly-graph-div{break-inside:avoid}
"""


def _kpi_tiles(r: dict, cur: str) -> str:
    mm = ((r.get("snapshot") or {}).get("90d") or {}).get("metrics", {})
    tiles = []
    for key, label, fmt in (("revenue", "Revenue (90d)", "money"), ("orders", "Orders (90d)", "int"),
                            ("customers", "Customers (90d)", "int"), ("aov", "AOV (90d)", "money"),
                            ("repeat_rate", "Repeat rate (90d)", "pct"), ("gross_margin", "Gross margin (90d)", "pct")):
        if key not in mm or mm[key]["value"] is None:
            continue
        v = mm[key]["value"]
        val = money(v, cur) if fmt == "money" else pct(v) if fmt == "pct" else f"{int(v):,}"
        d = mm[key]["pop"]
        delta = (f'<div class="d {"up" if d >= 0 else "down"}">{"▲" if d >= 0 else "▼"} {pct(abs(d), 1)} vs prev 90d</div>'
                 if d is not None else "")
        tiles.append(f'<div class="kpi"><div class="l">{label}</div><div class="v">{val}</div>{delta}</div>')
    return '<div class="kpis">' + "".join(tiles) + "</div>"


def build_html(r: dict, narrative: dict | None, ds=None, store_name: str = "Your store",
               for_print: bool = False) -> str:
    cur = r["meta"]["currency"]
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}.get(cur, "")
    q = r["meta"]["quality"]
    first_fig = [True]

    def fig_html(fig):
        if fig is None:
            return ""
        if for_print:
            fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#0b0b0b")
        # The PDF renderer may have no network, so print mode inlines plotly.js once.
        js = (True if for_print else "cdn") if first_fig[0] else False
        out = fig.to_html(full_html=False, include_plotlyjs=js, config={"displayModeBar": False, "staticPlot": for_print},
                          default_width="100%" if not for_print else 900)
        first_fig[0] = False
        return out

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>Growth Review</title>"
             f"<style>{CSS}{PRINT_CSS if for_print else ''}</style>"
             f"</head><body><main>",
             f"<h1>{html.escape(store_name)} — Growth Review</h1>",
             f"<div class='sub'>Data {q['date_min']} to {q['date_max']} · {q['orders']:,} orders · "
             f"{q['customers']:,} customers · data quality {q['score']}/100 · generated {r['meta']['generated_at']}</div>",
             "<h2>Key numbers</h2>", _kpi_tiles(r, cur)]

    if narrative:
        parts.append("<h2>Executive review</h2><div class='card narr'>"
                     + md.markdown(narrative["markdown"], extensions=["tables"]) + "</div>")
        if narrative.get("unverified_numbers"):
            parts.append(f"<div class='meta'>Numbers not found in the computed facts (check before sharing): "
                         f"{html.escape(', '.join(narrative['unverified_numbers']))}</div>")

    parts.append("<h2>Top 10 priorities</h2>")
    for i in r.get("insights", [])[:10]:
        parts.append(
            f"<div class='card'><h3>{i['rank']}. {html.escape(i['title'])}</h3>"
            f"<div class='meta'>Est. impact {money(i['impact'], cur)}/yr · confidence {pct(i['confidence'])} · "
            f"{i['category']} · formula: {html.escape(i['impact_formula'])}</div>"
            f"<p>{html.escape(i['finding'])}</p><p><b>Do:</b> {html.escape(i['action'])}</p>"
            + fig_html(charts.figure_for_insight(i.get("chart"), r, ds, sym)) + "</div>")

    if isinstance(r.get("monthly"), pd.DataFrame) and not r["monthly"].empty:
        parts += ["<h2>Trends</h2>", fig_html(charts.monthly_revenue(r["monthly"], sym))]
        if isinstance(r.get("forecast"), pd.DataFrame):
            parts.append(fig_html(charts.weekly_forecast(r["weekly"], r["forecast"], sym)))

    if isinstance(r.get("segments"), pd.DataFrame) and not r["segments"].empty:
        seg = r["segments"]
        rows = "".join(f"<tr><td>{s['segment']}</td><td>{s['customers']:,}</td><td>{money(s['revenue'], cur)}</td>"
                       f"<td>{pct(s['share_revenue'])}</td><td>{html.escape(s['action'])}</td></tr>"
                       for s in seg.to_dict("records"))
        parts += ["<h2>Customer segments</h2>", fig_html(charts.segments_bar(seg, sym)),
                  "<table><tr><th>Segment</th><th>Customers</th><th>Revenue</th><th>Share</th><th>Play</th></tr>"
                  + rows + "</table>"]

    if r.get("benchmarks"):
        parts += ["<h2>Benchmarks</h2>", fig_html(charts.benchmark_chart(r["benchmarks"])),
                  "<div class='meta'>Indicative ranges for small/mid DTC stores; adjust for your category.</div>"]

    parts.append("<h2>Method</h2><div class='meta'>All metrics are computed deterministically from your data "
                 "(KPI tree with LMDI decomposition, RFM quintiles, cohort retention, FP pair lift, Poisson "
                 "stock-out test, robust z-score anomalies, Holt-Winters forecast). Impact estimates are directional "
                 "and use the formula shown on each card.</div></main></body></html>")
    return "\n".join(parts)


def to_pdf(html_doc: str) -> bytes:
    """Render the print version of the report to PDF with headless Chromium (Playwright).

    One-time setup: `uv run playwright install chromium`."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1000, "height": 1400})
            page.emulate_media(media="print", color_scheme="light")
            page.set_content(html_doc, wait_until="load")
            n = html_doc.count('class="plotly-graph-div"')
            if n:
                page.wait_for_function(f"document.querySelectorAll('.main-svg').length >= {n}", timeout=30000)
            return page.pdf(format="A4", print_background=True,
                            margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
                            display_header_footer=True, header_template="<span></span>",
                            footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'>"
                                            "Growth Review · page <span class='pageNumber'></span> of "
                                            "<span class='totalPages'></span></div>")
        finally:
            browser.close()
