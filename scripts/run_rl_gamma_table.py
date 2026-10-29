"""Objective-by-objective comparison across risk aversions.

For each gamma, every strategy is scored on test hours by J_gamma (P&L minus the running penalty
gamma*sigma^2/2*int Q^2 minus the terminal penalty). Strategies with a free parameter are tuned on
validation J_gamma; DPS checkpoints and seeds are selected on validation J_gamma. Paired differences
against the closed form at the same gamma. Output: results/rl_benchmark/gamma_table.json
Usage: python scripts/run_rl_gamma_table.py GAMMA
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realdata.pipeline import nash_policy, params_for
from realdata import backtest as BT
from realdata import rl as RL

torch.set_num_threads(1)
R = ROOT / "results" / "rl_benchmark"
P = pickle.loads((R / "prepared_fee4.pkl").read_bytes())
dt, n = P["dt"], P["n"]
g = float(sys.argv[1])
q = params_for(P["p0"], g)


def run(st, D):
    return BT.run(st, D["eta"], D["xi"], D["alpha"], D["S"], q, dt)


def tune(make, grid):
    best = max(grid, key=lambda x: run(make(x), P["val"])["objective"].mean())
    return best, run(make(best), P["test"])


res = {}
_, pol, _ = nash_policy(P["p0"], g, n)
res["Nash closed form"] = run(BT.TorchPolicy(pol), P["test"])
res["Nash closed form, Q^I off"] = run(BT.NoInformedInventory(pol), P["test"])
kap, res["Linear control (kappa tuned)"] = tune(lambda k: (lambda o, kk, e, x: -k * o[:, 0]),
                                                (0.5, 1, 2, 5, 10, 20, 50, 100, 200))
tau, res["Inventory decay (tau tuned)"] = tune(lambda t: BT.inventory_decay(n=n, tau0=t),
                                               (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0))
sc = RL.Scales(P["fit"], q, dt)
for lab, lin in (("DPS, linear", True), ("DPS, MLP", False)):
    best = None
    for seed in (0, 1):
        pol_d, v = RL.train_dps(P["fit"], P["val"], q, dt, sc, linear=lin, seed=seed,
                                tag=f"{lab} gamma={g} seed={seed}", log=lambda *a: None)
        if best is None or v > best[1]:
            best = (pol_d, v)
    res[lab] = run(RL.DPSStrategy(best[0], sc, n), P["test"])

J0, M0 = res["Nash closed form"]["objective"], res["Nash closed form"]["mtm"]
out = {"gamma": g, "kappa": kap, "tau0": tau, "rows": {}}
for k, r in res.items():
    J, M = r["objective"], r["mtm"]
    d, dm = J - J0, M - M0
    out["rows"][k] = {"J": float(J.mean()), "J_se": float(J.std(ddof=1) / np.sqrt(len(J))),
                      "dJ": float(d.mean()), "dJ_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                      "pnl": float(M.mean()), "pnl_sd": float(M.std(ddof=1)),
                      "dPnL": float(dm.mean()), "dPnL_se": float(dm.std(ddof=1) / np.sqrt(len(dm)))}
    print(f"gamma={g:g} {k:32s} J {J.mean():+10.0f}  dJ {d.mean():+10.0f} ± {d.std(ddof=1)/12:7.0f}  "
          f"P&L {M.mean():+7.0f} sd {M.std(ddof=1):6.0f}", flush=True)
f = R / "gamma_table.json"
allr = json.loads(f.read_text()) if f.exists() else {}
allr[str(g)] = out
f.write_text(json.dumps(allr, indent=2))
