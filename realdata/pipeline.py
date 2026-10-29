"""Shared data preparation for the real-data experiments.

Reproduces scripts/run_test.py exactly (proxy, episodes, signal, calibration, gamma*), and adds a
validation split carved out of the training days for model selection:
  train_fit = first 11 training days, val = last 3 training days, test = the 6 held-out days.
Calibration and gamma* still use all 14 training days, as in run_test.py.
"""
import numpy as np
import pandas as pd

from .loaders import load_folder
from .flow import parent_orders, classify, binned_flows
from .calibrate import calibrate, signal
from . import backtest as BT
from brokergame.config import GameConfig
from brokergame.riccati import RiccatiSolution
from brokergame.policies import RiccatiPolicy

PHI_RATIO = 1.0 / 1.2e-3
GAMMAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]


def _episodes(orders, trades, days, dt_s):
    E = {"eta": [], "xi": [], "S": []}
    for d in days:
        for h in range(24):
            st = d + pd.Timedelta(hours=h)
            b = binned_flows(orders, trades, st, st + pd.Timedelta(hours=1), dt_s)
            if np.isnan(b["S"]).any():
                continue
            for k in E:
                E[k].append(b[k])
    return {k: np.array(v) for k, v in E.items()}


def params_for(p0, gamma):
    q = dict(p0)
    q["phi"] = q["psi"] = PHI_RATIO * q["a"]
    q["rB"] = q["rI"] = 0.5 * gamma * q["sig_S"] ** 2
    return q


def nash_policy(p0, gamma, n):
    q = params_for(p0, gamma)
    R = RiccatiSolution(GameConfig(n=n, **q))
    return q, RiccatiPolicy(R.gains(0)), R


def prepare(data_dir, fee_bps=4.0, train_days=14, val_days=3, dt_s=30.0, q_informed=0.95):
    trades, files = load_folder(data_dir)
    orders = parent_orders(trades)
    days = pd.date_range(trades["time"].min().normalize(), trades["time"].max().normalize(), freq="D")
    days = [d for d in days if ((trades["time"] >= d) & (trades["time"] < d + pd.Timedelta(days=1))).sum() > 1000]
    tr_days, te_days = days[:train_days], days[train_days:]
    split = te_days[0]
    thr = float(orders.loc[orders["time"] < split, "qty"].quantile(q_informed))
    orders = classify(orders, thr)
    dt = dt_s / 3600.0
    tr = _episodes(orders, trades, tr_days, dt_s)
    fit = _episodes(orders, trades, tr_days[:train_days - val_days], dt_s)
    val = _episodes(orders, trades, tr_days[train_days - val_days:], dt_s)
    te = _episodes(orders, trades, te_days, dt_s)
    n = tr["eta"].shape[1]
    tick_tr = trades.loc[trades["time"] < split, "price"].values[:200000]
    best = None
    for hl in (1, 2, 4, 8):
        for h in (1, 2, 4, 10):
            _, c, r2 = signal(tr["eta"], tr["S"], dt, hl, h)
            if best is None or r2 > best[2]:
                best = (hl, h, r2)
    p0, diag = calibrate(tr, dt, fee_rate=fee_bps * 1e-4, tick_prices=tick_tr,
                         half_life_steps=best[0], horizon_steps=best[1])
    for D in (tr, fit, val, te):
        D["alpha"], _, _ = signal(D["eta"], D["S"], dt, best[0], best[1], coef=diag["signal_coef"])
    sharpe = {}
    for g in GAMMAS:
        q, pol, _ = nash_policy(p0, g, n)
        r = BT.run(BT.TorchPolicy(pol), tr["eta"], tr["xi"], tr["alpha"], tr["S"], q, dt)
        sharpe[g] = float(r["mtm"].mean() / r["mtm"].std(ddof=1))
    g_star = max(sharpe, key=sharpe.get)
    p = params_for(p0, g_star)
    return {"train": tr, "fit": fit, "val": val, "test": te, "p0": p0, "p": p, "gamma_star": g_star,
            "dt": dt, "n": n, "diag": diag, "files": files, "threshold": thr}
