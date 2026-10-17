"""Trend, seasonality, weekday/hour patterns, anomaly detection and a short forecast.

Anomalies: robust z-score of daily revenue against a centred 28-day rolling median, scaled by
the rolling MAD (z = 0.6745 * (x - median) / MAD); |z| >= 3.5 is flagged.
Seasonality: ratio of monthly revenue to a log-linear trend, averaged by calendar month.
Forecast: Holt-Winters exponential smoothing on weekly revenue (additive trend, yearly
seasonality when there are 2+ years of data), with an 80% interval from residuals.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def daily(orders: pd.DataFrame) -> pd.DataFrame:
    s = orders.set_index("order_date")["revenue"].resample("D").sum()
    n = orders.set_index("order_date")["revenue"].resample("D").count()
    return pd.DataFrame({"revenue": s, "orders": n})


def anomalies(orders: pd.DataFrame, z_threshold: float = 3.5) -> pd.DataFrame:
    d = daily(orders)
    if len(d) < 60:
        return pd.DataFrame()
    med = d["revenue"].rolling(28, center=True, min_periods=14).median()
    mad = (d["revenue"] - med).abs().rolling(28, center=True, min_periods=14).median()
    z = 0.6745 * (d["revenue"] - med) / mad.replace(0, np.nan)
    out = d.assign(expected=med, z=z)
    out = out[out["z"].abs() >= z_threshold].copy()
    out["kind"] = np.where(out["z"] > 0, "spike", "dip")
    out["delta"] = out["revenue"] - out["expected"]
    return out.reset_index().rename(columns={"order_date": "date"})


def weekday_hour(orders: pd.DataFrame) -> dict:
    o = orders.assign(dow=orders["order_date"].dt.dayofweek, hour=orders["order_date"].dt.hour)
    has_time = (o["hour"] != 0).mean() > 0.2
    dow = o.groupby("dow")["revenue"].sum()
    n_weeks = max((o["order_date"].max() - o["order_date"].min()).days / 7, 1)
    dow_avg = (dow / n_weeks).reindex(range(7), fill_value=0)
    out = {"dow_avg_revenue": dow_avg.tolist(),
           "dow_index": (dow_avg / dow_avg.mean()).tolist() if dow_avg.mean() else None,
           "has_time": bool(has_time)}
    if has_time:
        heat = o.pivot_table(index="dow", columns="hour", values="revenue", aggfunc="sum", fill_value=0)
        out["heatmap"] = heat.reindex(index=range(7), columns=range(24), fill_value=0)
    return out


def seasonality(orders: pd.DataFrame) -> dict:
    """Month-of-year index after removing the growth trend (log-linear fit on monthly revenue)."""
    mth = orders.set_index("order_date")["revenue"].resample("MS").sum().iloc[1:-1]  # drop partial months
    mth = mth[mth > 0]
    if len(mth) < 12:
        return {}
    t = np.arange(len(mth))
    b, a = np.polyfit(t, np.log(mth.values), 1)
    ratio = pd.Series(mth.values / np.exp(a + b * t), index=mth.index)
    idx = ratio.groupby(ratio.index.month).mean().reindex(range(1, 13))
    idx = idx / idx.mean()
    return {"month_index": [None if not np.isfinite(v) else round(float(v), 3) for v in idx],
            "peak_month": int(idx.idxmax()), "low_month": int(idx.idxmin()),
            "monthly_growth_trend": float(np.exp(b) - 1)}


def trend(orders: pd.DataFrame, weeks: int = 26) -> dict:
    w = orders.set_index("order_date")["revenue"].resample("W").sum()
    w = w.iloc[1:-1]  # drop partial first / last weeks
    w = w.tail(weeks)
    if len(w) < 8:
        return {}
    x = np.arange(len(w))
    slope, intercept = np.polyfit(x, w.values, 1)
    return {"weeks": len(w), "weekly_slope": float(slope), "weekly_mean": float(w.mean()),
            "slope_pct_per_week": float(slope / w.mean()) if w.mean() else None}


def forecast(orders: pd.DataFrame, horizon_weeks: int = 13) -> pd.DataFrame:
    w = orders.set_index("order_date")["revenue"].resample("W").sum().iloc[1:-1]
    if len(w) < 16:
        return pd.DataFrame()
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if len(w) >= 104 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(w.values, trend="add", damped_trend=True, seasonal=seasonal,
                                         seasonal_periods=52 if seasonal else None,
                                         initialization_method="estimated").fit()
        fc = model.forecast(horizon_weeks)
        resid_sd = float(np.std(model.resid[-52:]))
    except Exception:
        fc = np.repeat(w.tail(8).mean(), horizon_weeks)
        resid_sd = float(w.tail(26).std())
    idx = pd.date_range(w.index[-1] + pd.Timedelta(weeks=1), periods=horizon_weeks, freq="W")
    fc = np.clip(fc, 0, None)
    return pd.DataFrame({"week": idx, "forecast": fc, "lower": np.clip(fc - 1.28 * resid_sd, 0, None),
                         "upper": fc + 1.28 * resid_sd})


def weekly(orders: pd.DataFrame) -> pd.DataFrame:
    return (orders.set_index("order_date")["revenue"].resample("W").sum().iloc[1:-1]
            .rename("revenue").reset_index().rename(columns={"order_date": "week"}))
