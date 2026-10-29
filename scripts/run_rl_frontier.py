"""Risk-return frontier on test hours: closed-form Nash (gamma sweep), linear control (kappa sweep) and
DPS trained at each gamma (checkpoint and seed chosen on validation J_gamma).

A single objective at gamma* can favour whichever strategy happens to match that penalty; the frontier
asks the stronger question: at equal P&L risk, which strategy earns more?
Output: results/rl_benchmark/frontier.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realdata.pipeline import nash_policy, params_for, GAMMAS
from realdata import backtest as BT
from realdata import rl as RL

torch.set_num_threads(1)
R = ROOT / "results" / "rl_benchmark"
P = pickle.loads((R / "prepared_fee4.pkl").read_bytes())
dt, n, Te, Va = P["dt"], P["n"], P["test"], P["val"]


def pt(r, **kw):
    m = r["mtm"]
    return {"pnl": float(m.mean()), "pnl_sd": float(m.std(ddof=1)), "sharpe": float(m.mean() / m.std(ddof=1)),
            "abs_QT": float(np.abs(r["terminal_inventory"]).mean()), **kw}


def run(st, D, p):
    return BT.run(st, D["eta"], D["xi"], D["alpha"], D["S"], p, dt)


out = json.loads((R / "frontier.json").read_text()) if (R / "frontier.json").exists() else {}
out["nash"] = []
for g in GAMMAS:
    q, pol, _ = nash_policy(P["p0"], g, n)
    out["nash"].append(pt(run(BT.TorchPolicy(pol), Te, q), gamma=g))
out["linear"] = [pt(run(lambda o, k, e, x, kap=kap: -kap * o[:, 0], Te, P["p"]), kappa=kap)
                 for kap in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)]
print("nash", [(d["gamma"], round(d["pnl"]), round(d["pnl_sd"])) for d in out["nash"]], flush=True)
print("linear", [(d["kappa"], round(d["pnl"]), round(d["pnl_sd"])) for d in out["linear"]], flush=True)
(R / "frontier.json").write_text(json.dumps(out, indent=2))

done = {d["gamma"] for d in out.get("dps", [])}
out.setdefault("dps", [])
for g in (1e-3, 1e-2, 0.1, 1.0, 10.0):
    if g in done:
        continue
    q = params_for(P["p0"], g)
    sc = RL.Scales(P["fit"], q, dt)
    best = None
    for seed in (0, 1):
        pol, v = RL.train_dps(P["fit"], Va, q, dt, sc, seed=seed, tag=f"DPS gamma={g} seed={seed}")
        if best is None or v > best[1]:
            best = (pol, v, seed)
    r = run(RL.DPSStrategy(best[0], sc, n), Te, q)
    out["dps"].append(pt(r, gamma=g, seed=best[2], test_J_gamma=float(r["objective"].mean())))
    print("dps", out["dps"][-1], flush=True)
    (R / "frontier.json").write_text(json.dumps(out, indent=2))
