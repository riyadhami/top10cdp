"""Probabilistic customer lifetime value: BG/NBD (purchase frequency + dropout) and Gamma-Gamma
(spend per order), fitted by maximum likelihood with SciPy.

Per customer, with time in weeks:
    x   = repeat purchases (orders - 1)
    t_x = time of last purchase, measured from the first purchase
    T   = age (reference date - first purchase)
    m   = average value of the repeat orders

BG/NBD (Fader, Hardie & Lee 2005): while alive a customer buys at rate λ ~ Gamma(r, α); after each
purchase they drop out with probability p ~ Beta(a, b).
Gamma-Gamma (Fader & Hardie 2013): spend per order ~ Gamma(p, ν), ν ~ Gamma(q, γ).

predicted_12m_value = E[orders in next 52 weeks | x, t_x, T] x E[order value | x, m]   (undiscounted)

`backtest` refits on data up to a cut-off and compares predicted vs actual orders in the holdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

MIN_CUSTOMERS = 50
PENALTY = 1e-3


def rfm_summary(orders: pd.DataFrame, ref: pd.Timestamp | None = None) -> pd.DataFrame:
    """x, t_x, T (weeks) and m for each identified customer, using orders on or before `ref`.
    Orders on the same day are merged, as is standard for these models."""
    o = orders[orders["customer_key"].notna()]
    ref = ref if ref is not None else o["order_date"].max()
    o = o[o["order_date"] <= ref]
    daily = (o.assign(day=o["order_date"].dt.normalize())
             .groupby(["customer_key", "day"])["revenue"].sum().reset_index())
    g = daily.groupby("customer_key")
    first, last = g["day"].min(), g["day"].max()
    n = g.size()
    first_value = daily.sort_values("day").drop_duplicates("customer_key").set_index("customer_key")["revenue"]
    total = g["revenue"].sum()
    s = pd.DataFrame({"x": n - 1, "t_x": (last - first).dt.days / 7,
                      "T": (ref.normalize() - first).dt.days / 7})
    s["m"] = np.where(s["x"] > 0, (total - first_value.reindex(s.index)) / s["x"].clip(lower=1), np.nan)
    return s


# ---------------- BG/NBD ----------------

def _bgnbd_ll(params, x, tx, T):
    r, alpha, a, b = np.exp(params)
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    A3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        A4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-12)) - (r + x) * np.log(alpha + tx),
                      -np.inf)
    ll = A1 + A2 + np.logaddexp(A3, A4)
    return -(ll.sum()) + PENALTY * np.sum(np.exp(params) ** 2)


def fit_bgnbd(s: pd.DataFrame) -> dict:
    x, tx, T = s["x"].values.astype(float), s["t_x"].values, s["T"].values
    best = None
    for init in ([0.0, np.log(max(T.mean(), 1)), 0.0, 0.0], [np.log(0.5), np.log(10), np.log(0.5), np.log(2)]):
        res = minimize(_bgnbd_ll, np.array(init), args=(x, tx, T), method="L-BFGS-B",
                       bounds=[(-8, 8)] * 4)
        if best is None or res.fun < best.fun:
            best = res
    r, alpha, a, b = np.exp(best.x)
    return {"r": r, "alpha": alpha, "a": a, "b": b, "converged": bool(best.success), "nll": float(best.fun)}


def p_alive(p: dict, x, tx, T):
    r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = (a / np.maximum(b + x - 1, 1e-12)) * ((alpha + T) / (alpha + tx)) ** (r + x)
    return np.where(x > 0, 1 / (1 + ratio), 1.0)


def expected_purchases(p: dict, t: float, x, tx, T):
    """E[number of purchases in (T, T+t] | x, t_x, T]."""
    r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
        den = 1 + np.where(x > 0, (a / np.maximum(b + x - 1, 1e-12)) * ((alpha + T) / (alpha + tx)) ** (r + x), 0)
    return num / den


# ---------------- Gamma-Gamma ----------------

def _gg_ll(params, x, m):
    p, q, v = np.exp(params)
    ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v)
          + (p * x - 1) * np.log(m) + (p * x) * np.log(x) - (p * x + q) * np.log(x * m + v))
    return -(ll.sum()) + PENALTY * np.sum(np.exp(params) ** 2)


def fit_gamma_gamma(s: pd.DataFrame) -> dict | None:
    rep = s[(s["x"] > 0) & (s["m"] > 0)]
    if len(rep) < 20:
        return None
    x, m = rep["x"].values.astype(float), rep["m"].values
    res = minimize(_gg_ll, np.array([0.0, 0.5, np.log(max(m.mean(), 1))]), args=(x, m), method="L-BFGS-B",
                   bounds=[(-8, 10)] * 3)
    p, q, v = np.exp(res.x)
    return {"p": p, "q": q, "v": v, "converged": bool(res.success),
            "freq_value_corr": float(np.corrcoef(x, m)[0, 1]) if len(rep) > 2 else None}


def expected_order_value(gg: dict, x, m, fallback: float):
    p, q, v = gg["p"], gg["q"], gg["v"]
    if q <= 1:
        return np.where(np.asarray(x) > 0, m, fallback)
    pop = v * p / (q - 1)
    x = np.asarray(x, float)
    m = np.nan_to_num(np.asarray(m, float), nan=pop)
    w = (q - 1) / (p * x + q - 1)
    return np.where(x > 0, w * pop + (1 - w) * m, pop)


# ---------------- public API ----------------

def predict(orders: pd.DataFrame, horizon_weeks: float = 52) -> tuple[pd.DataFrame, dict] | None:
    """Per-customer p_alive, expected orders and predicted value; None if the model can't be fitted."""
    s = rfm_summary(orders)
    if len(s) < MIN_CUSTOMERS or (s["x"] > 0).sum() < 10:
        return None
    bg = fit_bgnbd(s)
    exp_n = expected_purchases(bg, horizon_weeks, s["x"].values, s["t_x"].values, s["T"].values)
    pa = p_alive(bg, s["x"].values, s["t_x"].values, s["T"].values)
    if not (np.isfinite(exp_n).all() and (exp_n >= 0).all()):
        return None
    gg = fit_gamma_gamma(s)
    first_vals = orders[orders["customer_key"].notna()].groupby("customer_key")["revenue"].mean()
    fallback = float(first_vals.mean())
    ev = expected_order_value(gg, s["x"].values, s["m"].values, fallback) if gg else \
        np.where(s["x"] > 0, s["m"].fillna(fallback), fallback)
    out = pd.DataFrame({"p_active": pa, "expected_orders_12m": exp_n, "expected_order_value": ev},
                       index=s.index)
    out["predicted_12m_value"] = out["expected_orders_12m"] * out["expected_order_value"]
    params = {"bgnbd": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bg.items()},
              "gamma_gamma": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in gg.items()} if gg else None}
    return out, params


def backtest(orders: pd.DataFrame, holdout_days: int = 180) -> dict | None:
    """Fit on everything before the last `holdout_days`, predict orders in the holdout, compare."""
    ref = orders["order_date"].max()
    cut = ref - pd.Timedelta(days=holdout_days)
    o = orders[orders["customer_key"].notna()]
    if (cut - o["order_date"].min()).days < 180:
        return None
    s = rfm_summary(o, cut)
    if len(s) < MIN_CUSTOMERS or (s["x"] > 0).sum() < 10:
        return None
    bg = fit_bgnbd(s)
    pred = pd.Series(expected_purchases(bg, holdout_days / 7, s["x"].values, s["t_x"].values, s["T"].values),
                     index=s.index)
    hold = o[(o["order_date"] > cut) & o["customer_key"].isin(s.index)]
    actual = (hold.assign(day=hold["order_date"].dt.normalize()).drop_duplicates(["customer_key", "day"])
              .groupby("customer_key").size().reindex(s.index, fill_value=0))
    by_freq = pd.DataFrame({"x": s["x"].clip(upper=6), "predicted": pred, "actual": actual}) \
        .groupby("x")[["predicted", "actual"]].mean().reset_index()
    return {"holdout_days": holdout_days, "customers": int(len(s)),
            "predicted_total": float(pred.sum()), "actual_total": float(actual.sum()),
            "total_error": float(pred.sum() / actual.sum() - 1) if actual.sum() else None,
            "mae": float((pred - actual).abs().mean()),
            "correlation": float(np.corrcoef(pred, actual)[0, 1]) if actual.std() > 0 else None,
            "by_repeat_purchases": by_freq.to_dict("records")}
