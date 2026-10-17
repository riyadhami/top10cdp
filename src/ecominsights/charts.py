"""Plotly figures shared by the Streamlit app and the HTML report.

Colors follow a validated categorical order (fixed, never cycled), one-hue sequential ramps
for magnitude, and a blue/red diverging pair with a gray midpoint. Single-axis charts only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
POS, NEG, NEUTRAL = "#2a78d6", "#e34948", "#c3c2b7"
MUTED = "#898781"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _layout(fig: go.Figure, title: str = "", height: int = 360, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height,
                      margin=dict(l=8, r=8, t=48 if title else 16, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None),
                      hovermode="x unified", bargap=0.25, **kw)
    fig.update_xaxes(showgrid=False, linecolor="rgba(137,135,129,0.5)", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.18)", zeroline=False, tickfont=dict(color=MUTED))
    return fig


def _bar_style(fig):
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig


def monthly_revenue(monthly: pd.DataFrame, cur: str = "$") -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["new_revenue"], name="New customers", marker_color=SERIES[0])
    fig.add_bar(x=monthly["month"], y=monthly["returning_revenue"], name="Returning customers",
                marker_color=SERIES[1])
    fig.update_layout(barmode="stack")
    return _bar_style(_layout(fig, "Monthly revenue: new vs returning", yaxis_tickprefix=cur))


def weekly_forecast(weekly: pd.DataFrame, fc: pd.DataFrame, cur: str = "$") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=weekly["week"], y=weekly["revenue"], name="Weekly revenue", mode="lines",
                    line=dict(color=SERIES[0], width=2))
    if not fc.empty:
        fig.add_scatter(x=list(fc["week"]) + list(fc["week"][::-1]), y=list(fc["upper"]) + list(fc["lower"][::-1]),
                        fill="toself", fillcolor="rgba(42,120,214,0.12)", line=dict(width=0),
                        name="80% interval", hoverinfo="skip")
        fig.add_scatter(x=fc["week"], y=fc["forecast"], name="Forecast", mode="lines",
                        line=dict(color=SERIES[0], width=2, dash="dot"))
    return _layout(fig, "Weekly revenue and 13-week forecast", yaxis_tickprefix=cur)


def kpi_waterfall(tree: dict, cur: str = "$", title: str = "What moved revenue (last 90 days vs previous 90)",
                  start_label: str = "Previous") -> go.Figure:
    label = {"customers": "Customers", "orders_per_customer": "Orders / customer",
             "units_per_order": "Units / order", "price_per_unit": "Price / unit", "orders": "Orders", "aov": "AOV"}
    leaves = [c for ch in tree["children"] for c in (ch.get("children") or [ch])]
    x = [start_label] + [label.get(c["metric"], c["metric"]) for c in leaves] + ["Current"]
    y = [tree["prev"]] + [c["contribution"] for c in leaves] + [tree["cur"]]
    fig = go.Figure(go.Waterfall(x=x, y=y, measure=["absolute"] + ["relative"] * len(leaves) + ["total"],
                                 increasing=dict(marker_color=POS), decreasing=dict(marker_color=NEG),
                                 totals=dict(marker_color=NEUTRAL), connector=dict(line=dict(color=NEUTRAL, width=1)),
                                 text=[("-" if v < 0 else "") + f"{cur}{abs(v):,.0f}" for v in y], textposition="outside"))
    fig = _layout(fig, title, yaxis_tickprefix=cur)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def segments_bar(seg: pd.DataFrame, cur: str = "$") -> go.Figure:
    s = seg.sort_values("revenue")
    fig = go.Figure(go.Bar(y=s["segment"], x=s["revenue"], orientation="h", marker_color=SERIES[0],
                           customdata=np.stack([s["customers"], s["share_revenue"] * 100], axis=-1),
                           hovertemplate="%{y}<br>" + cur + "%{x:,.0f} revenue<br>%{customdata[0]} customers"
                                         "<br>%{customdata[1]:.0f}% of revenue<extra></extra>",
                           text=[f"{c} customers" for c in s["customers"]], textposition="auto"))
    fig = _layout(fig, "Revenue by RFM segment", height=420, xaxis_tickprefix=cur)
    fig.update_layout(hovermode="closest")
    return _bar_style(fig)


def retention_heatmap(ret: pd.DataFrame) -> go.Figure:
    z = ret.values * 100
    fig = go.Figure(go.Heatmap(z=z, x=[f"M{c}" for c in ret.columns], y=ret.index, colorscale=SEQ[1:],
                               zmin=0, zmax=float(np.nanpercentile(z[:, 1:], 95)) if z.shape[1] > 1 else 100,
                               xgap=2, ygap=2, hovertemplate="Cohort %{y}, %{x}: %{z:.0f}% active<extra></extra>",
                               colorbar=dict(title="% active", thickness=10)))
    fig = _layout(fig, "Monthly cohort retention (% of cohort ordering in month k)",
                  height=max(320, 22 * len(ret) + 80))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(hovermode="closest")
    return fig


def ltv_curve(avg_ltv: dict, cur: str = "$") -> go.Figure:
    k = [int(x) for x in avg_ltv if avg_ltv[x] is not None]
    v = [avg_ltv[x] for x in avg_ltv if avg_ltv[x] is not None]
    fig = go.Figure(go.Scatter(x=k, y=v, mode="lines+markers", line=dict(color=SERIES[0], width=2),
                               marker=dict(size=8), name="Avg cumulative revenue per customer"))
    return _layout(fig, "Average cumulative revenue per acquired customer", xaxis_title="Months since first order",
                   yaxis_tickprefix=cur)


def first_product_bar(fp: pd.DataFrame, cur: str = "$") -> go.Figure:
    val_col = [c for c in fp.columns if c.startswith("avg_value_")][0]
    s = fp.sort_values(val_col)
    avg = float(np.average(fp[val_col], weights=fp["customers"]))
    fig = go.Figure(go.Bar(y=s["first_product"], x=s[val_col], orientation="h",
                           marker_color=[SERIES[0] if v >= avg else NEUTRAL for v in s[val_col]],
                           customdata=np.stack([s["customers"], s["repeat_rate"] * 100], axis=-1),
                           hovertemplate="%{y}<br>" + cur + "%{x:,.0f} avg value<br>%{customdata[0]} customers"
                                         "<br>%{customdata[1]:.0f}% repeat<extra></extra>"))
    fig.add_vline(x=avg, line=dict(color=MUTED, dash="dot"), annotation_text="average",
                  annotation_font_color=MUTED)
    fig = _layout(fig, f"Customer value in first {val_col.split('_')[-1]} by first product purchased",
                  height=max(300, 28 * len(s) + 80), xaxis_tickprefix=cur)
    fig.update_layout(hovermode="closest")
    return _bar_style(fig)


def discount_cohort_bar(dc: dict) -> go.Figure:
    groups = ["Full-price first order", "Discounted first order"]
    vals = [dc["full_price"]["repeat_rate"] * 100, dc["discounted"]["repeat_rate"] * 100]
    fig = go.Figure(go.Bar(x=groups, y=vals, marker_color=[SERIES[0], SERIES[1]],
                           text=[f"{v:.0f}%" for v in vals], textposition="outside",
                           hovertemplate="%{x}: %{y:.1f}% repeat<extra></extra>"))
    fig = _layout(fig, "Repeat rate by how the customer was acquired", yaxis_ticksuffix="%", showlegend=False)
    fig.update_layout(hovermode="closest")
    return _bar_style(fig)


def pareto_bar(ptab: pd.DataFrame, cur: str = "$", top: int = 20) -> go.Figure:
    t = ptab.head(top).iloc[::-1]
    color = {"A": SERIES[0], "B": SERIES[1], "C": SERIES[2]}
    fig = go.Figure()
    for cls in ("A", "B", "C"):
        s = t[t["abc"] == cls]
        if len(s):
            fig.add_bar(y=s["product"], x=s["revenue"], orientation="h", name=f"Class {cls}",
                        marker_color=color[cls], customdata=s["cum_share"] * 100,
                        hovertemplate="%{y}<br>" + cur + "%{x:,.0f}<br>cumulative %{customdata:.0f}%<extra></extra>")
    fig = _layout(fig, "Top products by revenue (ABC classes: A = first 80% of revenue)",
                  height=max(320, 24 * len(t) + 90), xaxis_tickprefix=cur)
    fig.update_layout(hovermode="closest", barmode="overlay")
    return _bar_style(fig)


def momentum_bar(ptab: pd.DataFrame, n: int = 15) -> go.Figure:
    t = ptab.dropna(subset=["growth"])
    t = t[t["rev_previous"] > 0.005 * t["rev_previous"].sum()].sort_values("growth")
    t = pd.concat([t.head(n // 2), t.tail(n - n // 2)]).drop_duplicates("product")
    fig = go.Figure(go.Bar(y=t["product"], x=t["growth"] * 100, orientation="h",
                           marker_color=[POS if g >= 0 else NEG for g in t["growth"]],
                           hovertemplate="%{y}: %{x:+.0f}% vs prior 90 days<extra></extra>"))
    fig = _layout(fig, "Product momentum: last 90 days vs prior 90", height=max(320, 24 * len(t) + 80),
                  xaxis_ticksuffix="%")
    fig.update_layout(hovermode="closest")
    return _bar_style(fig)


def margin_scatter(ptab: pd.DataFrame, cur: str = "$") -> go.Figure:
    fig = go.Figure(go.Scatter(x=ptab["revenue"], y=ptab["margin"] * 100, mode="markers+text",
                               text=ptab["product"], textposition="top center", textfont=dict(size=10, color=MUTED),
                               marker=dict(size=10, color=SERIES[0], line=dict(color="#fcfcfb", width=2)),
                               hovertemplate="%{text}<br>" + cur + "%{x:,.0f} revenue<br>%{y:.0f}% margin<extra></extra>"))
    fig = _layout(fig, "Revenue vs gross margin by product", xaxis_tickprefix=cur, yaxis_ticksuffix="%", height=420)
    fig.update_layout(hovermode="closest")
    return fig


def basket_bar(basket: pd.DataFrame, n: int = 10) -> go.Figure:
    t = basket.head(n).iloc[::-1]
    labels = t["antecedent"] + " → " + t["consequent"]
    fig = go.Figure(go.Bar(y=labels, x=t["lift"], orientation="h", marker_color=SERIES[0],
                           customdata=np.stack([t["confidence"] * 100, t["pair_orders"]], axis=-1),
                           hovertemplate="%{y}<br>lift %{x:.2f}x<br>%{customdata[0]:.0f}% confidence"
                                         "<br>%{customdata[1]} orders together<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=MUTED, dash="dot"))
    fig = _layout(fig, "Products bought together (lift > 1 = more than chance)", height=max(300, 28 * len(t) + 80))
    fig.update_layout(hovermode="closest")
    return _bar_style(fig)


def stockout_timeline(lines: pd.DataFrame, product: str, start, end) -> go.Figure:
    d = lines[lines["product"] == product].set_index("order_date")["quantity"].resample("W").sum()
    fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=SERIES[0], name="Units / week"))
    fig.add_vrect(x0=pd.Timestamp(start), x1=pd.Timestamp(end), fillcolor=NEG, opacity=0.12, line_width=0,
                  annotation_text="no sales", annotation_font_color=MUTED)
    return _bar_style(_layout(fig, f"{product}: weekly units", height=300))


def weekday_hour_heatmap(heat: pd.DataFrame, cur: str = "$") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=heat.values, x=list(range(24)), y=DAYS, colorscale=SEQ[1:], xgap=2, ygap=2,
                               hovertemplate="%{y} %{x}:00 - " + cur + "%{z:,.0f}<extra></extra>",
                               colorbar=dict(title="Revenue", thickness=10)))
    fig = _layout(fig, "When customers buy: revenue by weekday and hour", height=320, xaxis_title="Hour of day")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(hovermode="closest")
    return fig


def seasonality_bar(idx: list) -> go.Figure:
    vals = [(v or 0) * 100 - 100 for v in idx]
    fig = go.Figure(go.Bar(x=MONTHS, y=vals, marker_color=[POS if v >= 0 else NEG for v in vals],
                           hovertemplate="%{x}: %{y:+.0f}% vs average month (trend removed)<extra></extra>"))
    fig = _layout(fig, "Seasonality: each month vs an average month", yaxis_ticksuffix="%", showlegend=False)
    fig.update_layout(hovermode="closest")
    return _bar_style(fig)


def anomalies_chart(orders: pd.DataFrame, an: pd.DataFrame, cur: str = "$") -> go.Figure:
    d = orders.set_index("order_date")["revenue"].resample("D").sum()
    fig = go.Figure(go.Scatter(x=d.index, y=d.values, mode="lines", line=dict(color=NEUTRAL, width=1.5),
                               name="Daily revenue"))
    if not an.empty:
        for kind, color in (("spike", POS), ("dip", NEG)):
            a = an[an["kind"] == kind]
            if len(a):
                fig.add_scatter(x=a["date"], y=a["revenue"], mode="markers", name=kind.capitalize() + "s",
                                marker=dict(size=9, color=color, line=dict(color="#fcfcfb", width=2)))
    return _layout(fig, "Daily revenue with detected anomalies", yaxis_tickprefix=cur)


def channel_lines(cm: pd.DataFrame, cur: str = "$") -> go.Figure:
    fig = go.Figure()
    for k, (ch, g) in enumerate(cm.groupby("channel", sort=True)):
        fig.add_scatter(x=g["month"], y=g["revenue"], name=str(ch), mode="lines+markers",
                        line=dict(color=SERIES[k % 3], width=2), marker=dict(size=6))
    return _layout(fig, "Monthly revenue by channel", yaxis_tickprefix=cur)


def days_to_second_hist(days: list[int]) -> go.Figure:
    fig = go.Figure(go.Histogram(x=days, xbins=dict(size=7), marker_color=SERIES[0],
                                 hovertemplate="%{x} days: %{y} customers<extra></extra>"))
    fig = _layout(fig, "Days from first to second order", xaxis_title="Days", yaxis_title="Customers")
    fig.update_layout(hovermode="closest")
    return _bar_style(fig)


def benchmark_chart(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    labels = [r["label"] for r in rows]
    fig.add_bar(y=labels, x=[(r["high"] - r["low"]) * 100 for r in rows], base=[r["low"] * 100 for r in rows],
                orientation="h", marker_color="rgba(195,194,183,0.45)", name="Typical range", hoverinfo="skip")
    fig.add_scatter(y=labels, x=[r["value"] * 100 for r in rows], mode="markers", name="Your store",
                    marker=dict(size=12, color=SERIES[0], symbol="diamond", line=dict(color="#fcfcfb", width=2)),
                    hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
    fig = _layout(fig, "Your store vs indicative DTC benchmarks", height=max(260, 46 * len(rows) + 80),
                  xaxis_ticksuffix="%")
    fig.update_layout(hovermode="closest")
    return _bar_style(fig)


CHARTS_FOR_INSIGHT = {
    "kpi_tree", "segments", "first_product", "discount_cohorts", "basket", "product_momentum", "margin",
    "pareto", "seasonality", "weekday", "anomalies", "channels", "forecast", "repeat_hist", "stockouts",
}


def figure_for_insight(key: str | None, r: dict, ds=None, cur: str = "$") -> go.Figure | None:
    """Chart that backs a given insight (used in the Top-10 view and the report)."""
    try:
        if key == "kpi_tree":
            s90 = r["snapshot"].get("90d") or {}
            if s90.get("tree_yoy"):
                return kpi_waterfall(s90["tree_yoy"], cur, "What moved revenue (last 90 days vs same period last year)",
                                     "Last year")
            return kpi_waterfall(s90["tree"], cur) if s90.get("tree") else None
        if key == "segments":
            return segments_bar(r["segments"], cur)
        if key == "first_product":
            return first_product_bar(r["first_product"], cur)
        if key == "discount_cohorts":
            return discount_cohort_bar(r["discount_cohorts"])
        if key == "basket":
            return basket_bar(r["basket"])
        if key == "product_momentum":
            return momentum_bar(r["products"])
        if key == "margin":
            return margin_scatter(r["products"], cur)
        if key == "pareto":
            return pareto_bar(r["products"], cur)
        if key == "seasonality":
            return seasonality_bar(r["seasonality"]["month_index"])
        if key == "weekday":
            wd = r["weekday"]
            return weekday_hour_heatmap(wd["heatmap"], cur) if wd.get("has_time") else None
        if key == "anomalies" and ds is not None:
            return anomalies_chart(ds.orders, r["anomalies"], cur)
        if key == "channels":
            return channel_lines(r["channel_monthly"], cur)
        if key == "forecast":
            return weekly_forecast(r["weekly"], r["forecast"], cur)
        if key == "repeat_hist":
            return days_to_second_hist(r["repeat"]["days_to_second_hist"])
        if key == "stockouts" and ds is not None:
            s = r["stockouts"].iloc[0]
            return stockout_timeline(ds.lines, s["product"], s["gap_start"], s["gap_end"])
        if key and key.startswith("attribution:"):
            dim = key.split(":", 1)[1]
            rows = pd.DataFrame(r["attribution"]["dimensions"][dim])
            rows = pd.concat([rows.head(5), rows.tail(5)]).drop_duplicates("segment")
            fig = go.Figure(go.Bar(y=rows["segment"].astype(str), x=rows["delta"], orientation="h",
                                   marker_color=[POS if v >= 0 else NEG for v in rows["delta"]],
                                   hovertemplate="%{y}: " + cur + "%{x:+,.0f}<extra></extra>"))
            fig = _layout(fig, f"Revenue change by {dim.replace('_', ' ')} (last 90 days vs "
                               f"{r['attribution'].get('base_label', 'previous 90 days')})",
                          xaxis_tickprefix=cur)
            fig.update_layout(hovermode="closest")
            return _bar_style(fig)
    except Exception:  # noqa: BLE001
        return None
    return None
