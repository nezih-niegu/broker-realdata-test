"""Client flow from public trade data.

A trade print says who was the aggressor (taker). Fills of the same aggressive order share an
order id, so they are grouped into one parent order. Each parent order is treated as a client order
sent to a hypothetical broker:

* informed client   = parent orders at or above the training-set size quantile `q_informed`
* uninformed client = all other parent orders

This is an ex-ante rule (it only uses the order's own size, decided on training days). It is a proxy:
public exchange data carry no client identities. `markouts` checks whether the proxy is informative.
"""
import numpy as np
import pandas as pd


def parent_orders(trades: pd.DataFrame) -> pd.DataFrame:
    """trades: columns time (datetime64), price, qty, buyer_maker (bool), buy_oid, sell_oid.
    Returns one row per aggressive order: time (first fill), side (+1 buy / -1 sell), qty, vwap."""
    t = trades
    side = np.where(t["buyer_maker"].values, -1, 1)          # buyer is maker -> seller aggressed
    oid = np.where(side > 0, t["buy_oid"].values, t["sell_oid"].values)
    df = pd.DataFrame({"oid": oid, "side": side, "time": t["time"].values,
                       "qty": t["qty"].values, "pq": t["price"].values * t["qty"].values})
    g = df.groupby(["oid", "side"], sort=False)
    out = g.agg(time=("time", "min"), qty=("qty", "sum"), pq=("pq", "sum")).reset_index()
    out["vwap"] = out["pq"] / out["qty"]
    return out.drop(columns="pq").sort_values("time").reset_index(drop=True)


def classify(orders: pd.DataFrame, threshold: float) -> pd.DataFrame:
    o = orders.copy()
    o["informed"] = o["qty"] >= threshold
    return o


def markouts(orders: pd.DataFrame, trades: pd.DataFrame, horizons_s=(10, 60, 300)):
    """Average signed price move after each parent order, in basis points:
    side * (P(t+h) - vwap) / vwap. Positive = the order was 'right' (toxic for its counterparty)."""
    tt = trades["time"].values.astype("datetime64[ms]").astype(np.int64)
    pp = trades["price"].values
    ot = orders["time"].values.astype("datetime64[ms]").astype(np.int64)
    res = {}
    for h in horizons_s:
        idx = np.searchsorted(tt, ot + h * 1000, side="right") - 1
        idx = np.clip(idx, 0, len(pp) - 1)
        m = orders["side"].values * (pp[idx] - orders["vwap"].values) / orders["vwap"].values * 1e4
        res[h] = m
    return pd.DataFrame(res, index=orders.index)


def binned_flows(orders: pd.DataFrame, trades: pd.DataFrame, start, end, dt_s: float):
    """Per-bin signed flow rates (qty per hour) for informed and uninformed clients, and the last
    trade price at the start of each bin. Bins: [start + k dt, start + (k+1) dt)."""
    n = int(round((end - start) / pd.Timedelta(seconds=dt_s)))
    edges = start + pd.to_timedelta(np.arange(n + 1) * dt_s, unit="s")
    dt_h = dt_s / 3600.0
    sel = (orders["time"] >= start) & (orders["time"] < end)
    o = orders[sel]
    k = ((o["time"] - start) / pd.Timedelta(seconds=dt_s)).astype(int).clip(0, n - 1).values
    signed = (o["side"] * o["qty"]).values
    inf = o["informed"].values
    eta = np.bincount(k[inf], weights=signed[inf], minlength=n) / dt_h
    xi = np.bincount(k[~inf], weights=signed[~inf], minlength=n) / dt_h
    # price at bin starts (last trade before the edge), forward filled
    tt = trades["time"].values
    idx = np.searchsorted(tt, edges.values, side="right") - 1
    idx = np.clip(idx, 0, len(trades) - 1)
    S = trades["price"].values[idx]
    return {"eta": eta, "xi": xi, "S": S, "edges": edges}
