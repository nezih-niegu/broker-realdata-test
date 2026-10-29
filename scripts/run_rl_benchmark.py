"""Benchmark RL brokers against the paper's closed-form Nash broker on held-out real hours.

Usage:
  python scripts/run_rl_benchmark.py --algo baselines
  python scripts/run_rl_benchmark.py --algo PPO --seed 0 [--extended] [--steps 300000]
  algos: PPO, RecurrentPPO (LSTM), A2C, SAC, TQC, TD3, DDPG, DPS (differentiable policy search, MLP), DPS-linear
Protocol: train on 11 training days, select the checkpoint with the best validation objective on the
last 3 training days, evaluate once on the 6 test days. Objective = the paper's broker objective
(P&L - terminal and running inventory penalties) at the risk aversion gamma* chosen on training days.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realdata.pipeline import prepare, nash_policy, params_for, GAMMAS
from realdata import backtest as BT
from realdata import rl as RL

OUT = ROOT / "results" / "rl_benchmark"
TAG_SUFFIX = ""
LEARNING_STARTS = 5000   # off-policy uniform-random warm-up steps (overridable with --learning_starts)
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def summarise(r):
    o, m = r["objective"], r["mtm"]
    return {"objective_mean": float(o.mean()), "objective_se": float(o.std(ddof=1) / np.sqrt(len(o))),
            "pnl_mean": float(m.mean()), "pnl_sd": float(m.std(ddof=1)),
            "sharpe": float(m.mean() / m.std(ddof=1)),
            "abs_QT": float(np.abs(r["terminal_inventory"]).mean()),
            "lit_volume": float(r["lit_volume"].mean())}


def run_on(strategy, D, P):
    return BT.run(strategy, D["eta"], D["xi"], D["alpha"], D["S"], P["p"], P["dt"])


def save(name, val_score, rtest, extra=None):
    out = {"name": name, "val_objective": float(val_score), "test": summarise(rtest), **(extra or {})}
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=2))
    np.savez(OUT / f"{name}.npz", objective=rtest["objective"], mtm=rtest["mtm"])
    print(f"{name:34s} val {val_score:+9.1f} | test objective {out['test']['objective_mean']:+9.1f} "
          f"± {out['test']['objective_se']:6.1f}  P&L {out['test']['pnl_mean']:+8.1f}  "
          f"Sharpe {out['test']['sharpe']:+.3f}  |Q_T| {out['test']['abs_QT']:.1f}", flush=True)


def baselines(P):
    n = P["n"]
    q, pol, _ = nash_policy(P["p0"], P["gamma_star"], n)
    s = BT.TorchPolicy(pol)
    save("Nash_closed_form_gamma_star", run_on(s, P["val"], P)["objective"].mean(), run_on(s, P["test"], P),
         {"gamma": P["gamma_star"]})
    # give the paper its best shot: choose gamma on validation objective (evaluated with the
    # paper objective at gamma*, so all strategies are scored by the same criterion)
    best = None
    for g in GAMMAS:
        _, pol_g, _ = nash_policy(P["p0"], g, n)
        v = run_on(BT.TorchPolicy(pol_g), P["val"], P)["objective"].mean()
        if best is None or v > best[1]:
            best = (g, v, pol_g)
    save("Nash_closed_form_gamma_tuned_on_val", best[1], run_on(BT.TorchPolicy(best[2]), P["test"], P),
         {"gamma": best[0]})
    # model-based repair: the same closed form with the informed-inventory channel switched off
    # (on real data Q^I does not mean-revert within the hour, so following it builds inventory)
    s = BT.NoInformedInventory(pol)
    save("Nash_closed_form_no_QI_channel", run_on(s, P["val"], P)["objective"].mean(), run_on(s, P["test"], P),
         {"gamma": P["gamma_star"]})
    s = BT.inventory_decay(n=n, tau0=0.05)
    save("Inventory_decay_tau0.05", run_on(s, P["val"], P)["objective"].mean(), run_on(s, P["test"], P))
    best = None
    for tau in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        v = run_on(BT.inventory_decay(n=n, tau0=tau), P["val"], P)["objective"].mean()
        if best is None or v > best[1]:
            best = (tau, v)
    save("Inventory_decay_tau_tuned_on_val", best[1],
         run_on(BT.inventory_decay(n=n, tau0=best[0]), P["test"], P), {"tau0": best[0]})
    # one-parameter benchmark: constant-rate inventory control v = -kappa Q^B, kappa tuned on val.
    # If an RL agent cannot beat this, its edge over the closed form is just tighter inventory control.
    best = None
    for kap in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        v = run_on(linear_control(kap), P["val"], P)["objective"].mean()
        if best is None or v > best[1]:
            best = (kap, v)
    save("Linear_control_kappa_tuned_on_val", best[1], run_on(linear_control(best[0]), P["test"], P),
         {"kappa": best[0]})


def linear_control(kappa):
    return lambda obs, k, eta, xi: -kappa * obs[:, 0]


def sb3_run(P, algo, seed, steps, extended, reward_scale=None):
    from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
    from sb3_contrib import RecurrentPPO, TQC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    sc = RL.Scales(P["fit"], P["p"], P["dt"])
    # off-policy critics regress raw returns; at scale 100 the returns of an untrained policy reach
    # 1e3-1e5 and the critics diverge (first run, kept as *-rs100). 1e4 keeps them O(1-10).
    off = algo in ("SAC", "TQC", "TD3", "DDPG")
    rs = reward_scale or (1e4 if off else RL.REWARD_SCALE)
    env = RL.ReplayBrokerEnv(P["fit"], P["p"], P["dt"], sc, extended, seed, reward_scale=rs)
    common = dict(seed=seed, verbose=0, gamma=0.999, policy_kwargs=dict(net_arch=[64, 64]))
    # small initial exploration: typical speeds are ~1-5% of V_SCALE
    onp = dict(seed=seed, verbose=0, gamma=0.999, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-2.0))
    if algo == "PPO":
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, learning_rate=3e-4, ent_coef=0.0, **onp)
    elif algo == "RecurrentPPO":
        onp["policy_kwargs"].update(lstm_hidden_size=64)
        model = RecurrentPPO("MlpLstmPolicy", env, n_steps=2048, batch_size=256, learning_rate=3e-4, **onp)
    elif algo == "A2C":
        model = A2C("MlpPolicy", env, n_steps=120, learning_rate=7e-4, **onp)
    else:
        noise = NormalActionNoise(np.zeros(1), 0.02 * np.ones(1)) if algo in ("TD3", "DDPG") else None
        cls = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG, "TQC": TQC}[algo]
        kw = dict(ent_coef="auto_0.01", target_entropy=-3.0) if algo in ("SAC", "TQC") else {}
        model = cls("MlpPolicy", env, learning_starts=LEARNING_STARTS, batch_size=256, learning_rate=3e-4,
                    action_noise=noise, buffer_size=200000, **kw, **common)
    tag = f"{algo}{'-rs%g' % rs if reward_scale else ''}{TAG_SUFFIX}{'_ext' if extended else ''}_seed{seed}"

    class ValCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best, self.best_params = -np.inf, None

        def _on_step(self):
            if self.num_timesteps % 10000 == 0:
                st = RL.SB3Strategy(self.model, sc, P["n"], extended, P["val"]["S"])
                v = run_on(st, P["val"], P)["objective"].mean()
                if v > self.best:
                    self.best, self.best_params = v, self.model.get_parameters()
                print(f"[{tag}] {self.num_timesteps:7d} steps  val objective {v:+.1f}  (best {self.best:+.1f})",
                      flush=True)
            return True

    cb = ValCallback()
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb)
    model.set_parameters(cb.best_params)
    st = RL.SB3Strategy(model, sc, P["n"], extended, P["test"]["S"])
    save(tag, cb.best, run_on(st, P["test"], P), {"algo": algo, "seed": seed, "steps": steps,
                                                  "extended": extended, "reward_scale": rs,
                                                  "train_seconds": time.time() - t0})


def dps_run(P, seed, extended, linear):
    sc = RL.Scales(P["fit"], P["p"], P["dt"])
    tag = f"DPS{'-linear' if linear else ''}{'_ext' if extended else ''}_seed{seed}"
    t0 = time.time()
    pol, _ = RL.train_dps(P["fit"], P["val"], P["p"], P["dt"], sc, extended, linear, seed, tag=tag)
    val = run_on(RL.DPSStrategy(pol, sc, P["n"], extended, P["val"]["S"]), P["val"], P)["objective"].mean()
    rt = run_on(RL.DPSStrategy(pol, sc, P["n"], extended, P["test"]["S"]), P["test"], P)
    save(tag, val, rt, {"algo": "DPS-linear" if linear else "DPS", "seed": seed, "extended": extended,
                        "train_seconds": time.time() - t0})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--fee_bps", type=float, default=4.0)
    ap.add_argument("--reward_scale", type=float, default=None)
    ap.add_argument("--learning_starts", type=int, default=None)
    ap.add_argument("--tag_suffix", default="")
    a = ap.parse_args()
    if a.learning_starts is not None:
        LEARNING_STARTS = a.learning_starts
    TAG_SUFFIX = a.tag_suffix
    cache = OUT / f"prepared_fee{a.fee_bps:g}.pkl"
    if cache.exists():
        P = pickle.loads(cache.read_bytes())
    else:
        P = prepare(ROOT / "data" / "BTCUSDT", fee_bps=a.fee_bps)
        cache.write_bytes(pickle.dumps(P))
    if a.algo == "baselines":
        print(f"gamma* = {P['gamma_star']}; episodes fit/val/test = {len(P['fit']['eta'])}/"
              f"{len(P['val']['eta'])}/{len(P['test']['eta'])}")
        baselines(P)
    elif a.algo in ("DPS", "DPS-linear"):
        dps_run(P, a.seed, a.extended, a.algo == "DPS-linear")
    else:
        steps = a.steps or (200000 if a.algo in ("PPO", "A2C", "RecurrentPPO") else 100000)
        sb3_run(P, a.algo, a.seed, steps, a.extended, a.reward_scale)
