"""Real-data test of the broker-trader game on Binance BTC/USDT trades.

Pipeline (all choices fixed on training days only):
 1. Load trades, group fills into parent (aggressive) orders.
 2. Informed client = parent orders >= the 95th size percentile of TRAINING days; others uninformed.
    Check on TEST days with markouts.
 3. Cut the data into one-hour episodes on a 30-second grid; client flow rates per bin.
 4. Calibrate the game (signal, OU flow, Obizhaeva-Wang impact, volatility, costs) on training hours.
 5. Solve the closed-form Nash (Riccati) and train learned agents in the calibrated simulator.
 6. Backtest every broker strategy on the TEST hours: real client flow, real prices, modelled own impact.
 7. Descriptive test: does real informed flow move like the model's equilibrium eta*?

Usage: python scripts/run_test.py [--fee_bps 4] [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realdata.loaders import load_folder
from realdata.flow import parent_orders, classify, markouts, binned_flows
from realdata.calibrate import calibrate, signal, roll_half_spread
from realdata import backtest as BT
import brokergame.env as ENV
from brokergame.config import GameConfig
from brokergame.riccati import RiccatiSolution
from brokergame.policies import RiccatiPolicy, TimeGainPolicy, MLPPolicy
from brokergame.learn import train, evaluate

torch.set_num_threads(2)
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "log.txt", "a") as f:
        f.write(s + "\n")


def episodes(orders, trades, days, dt_s):
    E = {"eta": [], "xi": [], "S": [], "start": []}
    for d in days:
        for h in range(24):
            st = d + pd.Timedelta(hours=h)
            b = binned_flows(orders, trades, st, st + pd.Timedelta(hours=1), dt_s)
            if np.isnan(b["S"]).any():
                continue
            E["eta"].append(b["eta"]); E["xi"].append(b["xi"]); E["S"].append(b["S"])
            E["start"].append(str(st))
    return {k: (np.array(v) if k != "start" else v) for k, v in E.items()}


def stats(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))),
            "sd": float(x.std(ddof=1)), "p5": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "BTCUSDT"))
    ap.add_argument("--fee_bps", type=float, default=4.0)
    ap.add_argument("--train_days", type=int, default=14)
    ap.add_argument("--dt_s", type=float, default=30.0)
    ap.add_argument("--q_informed", type=float, default=0.95)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--no_learned", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"fee{args.fee_bps:g}bps"
    t0 = time.time()

    # ---------------------------------------------------------------- data
    trades, files = load_folder(args.data)
    orders = parent_orders(trades)
    days = pd.date_range(trades["time"].min().normalize(), trades["time"].max().normalize(), freq="D")
    days = [d for d in days if ((trades["time"] >= d) & (trades["time"] < d + pd.Timedelta(days=1))).sum() > 1000]
    tr_days, te_days = days[:args.train_days], days[args.train_days:]
    split = te_days[0]
    log(f"\n===== {tag}: {len(trades):,} trades, {len(orders):,} parent orders, "
        f"{len(tr_days)} train days ({tr_days[0].date()}..{tr_days[-1].date()}), "
        f"{len(te_days)} test days ({te_days[0].date()}..{te_days[-1].date()})")

    thr = float(orders.loc[orders["time"] < split, "qty"].quantile(args.q_informed))
    orders = classify(orders, thr)
    mk = markouts(orders, trades)
    te_mask = orders["time"] >= split
    mk_tab = {}
    for lab, m in (("informed", orders["informed"] & te_mask), ("uninformed", ~orders["informed"] & te_mask)):
        mk_tab[lab] = {f"{h}s": {"mean_bps": float(mk.loc[m, h].mean()),
                                 "se_bps": float(mk.loc[m, h].std() / np.sqrt(m.sum()))} for h in mk.columns}
        mk_tab[lab]["n_orders"] = int(m.sum())
        mk_tab[lab]["volume_share"] = float(orders.loc[m, "qty"].sum() / orders.loc[te_mask, "qty"].sum())
    log(f"informed threshold {thr:.4f} BTC (train {args.q_informed:.0%} quantile); TEST-day markouts (bps):")
    for lab in mk_tab:
        log(f"  {lab:10s} " + "  ".join(f"{k}: {v['mean_bps']:+.2f}±{v['se_bps']:.2f}"
                                       for k, v in mk_tab[lab].items() if isinstance(v, dict))
            + f"  volume share {mk_tab[lab]['volume_share']:.0%}")

    dt = args.dt_s / 3600.0
    tr = episodes(orders, trades, tr_days, args.dt_s)
    te = episodes(orders, trades, te_days, args.dt_s)
    n = tr["eta"].shape[1]
    log(f"episodes: train {len(tr['eta'])}, test {len(te['eta'])}, n = {n} steps of {args.dt_s:.0f}s")

    # ---------------------------------------------------------------- calibration (training days only)
    tick_tr = trades.loc[trades["time"] < split, "price"].values[:200000]
    best = None
    for hl in (1, 2, 4, 8):                        # signal smoothing (steps) and horizon (steps)
        for h in (1, 2, 4, 10):
            _, c, r2 = signal(tr["eta"], tr["S"], dt, hl, h)
            if best is None or r2 > best[2]:
                best = (hl, h, r2)
    p0, diag = calibrate(tr, dt, fee_rate=args.fee_bps * 1e-4, tick_prices=tick_tr,
                         half_life_steps=best[0], horizon_steps=best[1])
    alpha_tr, _, _ = signal(tr["eta"], tr["S"], dt, best[0], best[1], coef=diag["signal_coef"])
    alpha_te, _, r2_te = signal(te["eta"], te["S"], dt, best[0], best[1], coef=diag["signal_coef"])
    diag["signal_R2_test"] = r2_te
    log("calibrated base parameters (units: hours, USDT, BTC): "
        + ", ".join(f"{k}={v:.4g}" for k, v in p0.items() if k not in ("phi", "psi", "rB", "rI")))
    log(f"signal: half-life {best[0]} steps, horizon {best[1]} steps; R^2 of forward drift: "
        f"train {diag['signal_R2_train']:.4f}, test {r2_te:.4f}; half-spread {diag['half_spread']:.3f}; "
        f"per-unit lit cost {diag['kappa_tc']:.3f} USDT/BTC")

    PHI_RATIO = 1.0 / 1.2e-3          # the paper's ratio of terminal penalty to lit cost (phi/a)

    def params(gamma):
        q = dict(p0)
        q["phi"] = q["psi"] = PHI_RATIO * q["a"]
        q["rB"] = q["rI"] = 0.5 * gamma * q["sig_S"] ** 2     # mean-variance inventory aversion
        return q

    def nash_for(gamma):
        q = params(gamma)
        cfg = GameConfig(n=n, **q)
        R = RiccatiSolution(cfg)
        return q, cfg, R, {"B": RiccatiPolicy(R.gains(0)), "I": RiccatiPolicy(R.gains(1))}

    def bt(strategy, q, data, alpha):
        return BT.run(strategy, data["eta"], data["xi"], alpha, data["S"], q, dt)

    GAMMAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]   # gamma = 1 makes the discrete-time closed form unstable (gain x dt > 2)
    # choose the central risk aversion on TRAINING hours: best training Sharpe of the Nash broker
    train_sharpe = {}
    for g_ in GAMMAS:
        q, _, R_, nsh = nash_for(g_)
        r = bt(BT.TorchPolicy(nsh["B"]), q, tr, alpha_tr)
        train_sharpe[g_] = float(r["mtm"].mean() / r["mtm"].std(ddof=1))
    g_star = max(train_sharpe, key=train_sharpe.get)
    log("training Sharpe of Nash broker by gamma: " + ", ".join(f"{k:g}: {v:.3f}" for k, v in train_sharpe.items())
        + f"  -> gamma* = {g_star:g}")
    p, cfg, R, nash = nash_for(g_star)
    log(f"parameters at gamma*: phi=psi={p['phi']:.4g}, rB=rI={p['rB']:.4g}; Riccati finite={R.ok}, "
        f"max|P|={R.max_abs:.3g}")

    # state scales for the learned agents from a simulated Nash rollout in the calibrated model
    flow_sd = float(np.concatenate([tr["eta"], tr["xi"]]).std())
    ENV.V_MAX = 50 * flow_sd
    sim = evaluate(cfg, nash, batch=4000, record=True)["rec"]
    sc = [float(sim[k].std()) + 1e-9 for k in ("QB", "QI", "Y", "alpha", "xi")]
    act = float(sim["v"].std()) + 1e-9

    strategies = {
        "Nash broker (closed form)": BT.TorchPolicy(nash["B"]),
        "Nash broker, signal off": BT.NoSignal(nash["B"]),
        "Nash broker, informed-inventory channel off": BT.NoInformedInventory(nash["B"]),
        "Internalise, liquidate at close": BT.internalise(n=n),
        "Externalise all client flow": BT.externalise(n=n),
        "Inventory decay, no signal (tau0=0.05h)": BT.inventory_decay(n=n, tau0=0.05),
    }
    sim_checks = {}
    if not args.no_learned:
        torch.manual_seed(0)
        ol = {w: TimeGainPolicy(cfg, sc, act) for w in "BI"}
        train(cfg, ol, iters=args.iters, mode="open", log=log, tag=f"{tag} learned-OL", log_every=500)
        cl = {w: MLPPolicy(cfg, sc, act) for w in "BI"}
        for w in "BI":
            cl[w].lin.load_state_dict(ol[w].state_dict())
        train(cfg, cl, iters=args.iters, mode="closed", log=log, tag=f"{tag} learned-CL", log_every=500)
        a_ = evaluate(cfg, ol, batch=4000, record=True)["rec"]
        r2 = 1 - ((a_["v"] - sim["v"]) ** 2).mean() / sim["v"].var()
        sim_checks = {"R2_learnedOL_vs_Nash_broker_in_sim": float(r2),
                      "sim_JB_nash": evaluate(cfg, nash)["JB"], "sim_JB_learnedOL": evaluate(cfg, ol)["JB"],
                      "sim_JB_learnedCL": evaluate(cfg, cl)["JB"]}
        log("in the calibrated simulator: " + ", ".join(f"{k}={v:.4f}" for k, v in sim_checks.items()))
        strategies["Learned broker, open loop (MARL)"] = BT.TorchPolicy(ol["B"])
        strategies["Learned broker, closed loop (MARL)"] = BT.TorchPolicy(cl["B"])

    # ---------------------------------------------------------------- backtest on TEST hours at gamma*
    res, runs = {}, {}
    for name, s_ in strategies.items():
        r = bt(s_, p, te, alpha_te)
        runs[name] = r
        res[name] = {"pnl": stats(r["mtm"]), "pnl_net_of_client_revenue": stats(r["mtm"] - r["client_revenue"]),
                     "objective": stats(r["objective"]),
                     "abs_terminal_inventory": float(np.abs(r["terminal_inventory"]).mean()),
                     "lit_volume_per_hour": float(r["lit_volume"].mean()),
                     "sharpe_hourly": float(r["mtm"].mean() / r["mtm"].std(ddof=1))}
    base = runs["Nash broker (closed form)"]
    for name, r in runs.items():
        d = r["mtm"] - base["mtm"]
        res[name]["pnl_minus_nash"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        log(f"  {name:44s} P&L/h {res[name]['pnl']['mean']:+8.1f} (sd {res[name]['pnl']['sd']:7.1f}) "
            f"net of client spread {res[name]['pnl_net_of_client_revenue']['mean']:+8.1f}  "
            f"Sharpe/h {res[name]['sharpe_hourly']:+.3f}  P&L-Nash {d.mean():+7.1f} ± {d.std(ddof=1)/np.sqrt(len(d)):5.1f}  "
            f"|Q_T| {res[name]['abs_terminal_inventory']:6.2f}  lit vol {res[name]['lit_volume_per_hour']:6.1f}")

    # ---------------------------------------------------------------- risk-return frontiers on TEST hours
    frontier = {"Nash broker (closed form), gamma grid": [], "Inventory decay, tau0 grid": []}
    for g_ in GAMMAS:
        q, _, _, nsh = nash_for(g_)
        r = bt(BT.TorchPolicy(nsh["B"]), q, te, alpha_te)
        frontier["Nash broker (closed form), gamma grid"].append(
            {"param": g_, "mean": float(r["mtm"].mean()), "sd": float(r["mtm"].std(ddof=1)),
             "se": float(r["mtm"].std(ddof=1) / np.sqrt(len(r["mtm"])))})
    for tau in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        r = bt(BT.inventory_decay(n=n, tau0=tau), p, te, alpha_te)
        frontier["Inventory decay, tau0 grid"].append(
            {"param": tau, "mean": float(r["mtm"].mean()), "sd": float(r["mtm"].std(ddof=1)),
             "se": float(r["mtm"].std(ddof=1) / np.sqrt(len(r["mtm"])))})
    for k, v in frontier.items():
        log(f"  frontier {k}: " + "; ".join(f"{x['param']:g}: {x['mean']:+.0f}/{x['sd']:.0f}" for x in v))
    plot_frontier(frontier, res, OUT / f"frontier_{tag}.png", tag)

    # ---------------------------------------------------------------- descriptive test of the flow model
    # Model predictions: informed flow trades on the drift (positive loading on forward drift) and
    # unwinds its inventory (negative loading on Q^I); uninformed flow loads on neither.
    h = best[1]
    descr = {}
    for lab, F in (("informed", te["eta"]), ("uninformed", te["xi"])):
        Qc = np.concatenate([np.zeros((F.shape[0], 1)), np.cumsum(F * dt, 1)[:, :-1]], 1)
        # drift from the END of bin k over the next h bins: excludes the bin's own price impact
        D = (te["S"][:, 1 + h:n + 1] - te["S"][:, 1:n + 1 - h]) / (h * dt)
        Fm, Qm = F[:, :n - h], Qc[:, :n - h]
        Xr = np.stack([(D / D.std()).ravel(), (Qm / Qm.std()).ravel(), np.ones(D.size)], 1)
        beta, *_ = np.linalg.lstsq(Xr, Fm.ravel(), rcond=None)
        e = Fm.ravel() - Xr @ beta
        se_ = np.sqrt(np.diag(np.linalg.inv(Xr.T @ Xr)) * e.var())
        descr[lab] = {"coef_fwd_drift_per_sd": float(beta[0]), "t_fwd_drift": float(beta[0] / se_[0]),
                      "coef_own_inventory_per_sd": float(beta[1]), "t_own_inventory": float(beta[1] / se_[1])}
        log(f"  {lab:10s} flow on forward drift: {beta[0]:+.1f} (t={beta[0]/se_[0]:+.1f}); "
            f"on own inventory: {beta[1]:+.1f} (t={beta[1]/se_[1]:+.1f})  [BTC/h per 1 sd]")

    out = {"tag": tag, "args": vars(args), "files": files, "informed_threshold_btc": thr,
           "markouts_test": mk_tab, "base_params": p0, "params_at_gamma_star": p, "gamma_star": g_star,
           "train_sharpe_by_gamma": train_sharpe, "diagnostics": diag, "sim_checks": sim_checks,
           "backtest_at_gamma_star": res, "frontiers": frontier, "flow_regressions": descr,
           "n_train_episodes": len(tr["eta"]), "n_test_episodes": len(te["eta"]),
           "runtime_s": time.time() - t0}
    with open(OUT / f"results_{tag}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    log(f"done in {time.time() - t0:.0f}s")


def plot_frontier(frontier, res, path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cols = {"Nash broker (closed form), gamma grid": "#1f4e79", "Inventory decay, tau0 grid": "#7f7f7f"}
    for k, v in frontier.items():
        ax.errorbar([x["sd"] for x in v], [x["mean"] for x in v], yerr=[2 * x["se"] for x in v],
                    fmt="-o", ms=4, lw=1.4, capsize=2, color=cols[k], label=k)
    marks = {"Internalise, liquidate at close": ("s", "#d62728"), "Externalise all client flow": ("D", "#9467bd"),
             "Learned broker, open loop (MARL)": ("^", "#e07b39"), "Learned broker, closed loop (MARL)": ("v", "#2ca02c")}
    for name, (m, c) in marks.items():
        if name in res:
            ax.plot(res[name]["pnl"]["sd"], res[name]["pnl"]["mean"], m, ms=8, color=c, label=name)
    ax.set_xlabel("standard deviation of hourly P&L (USDT)")
    ax.set_ylabel("mean hourly P&L (USDT)")
    ax.set_title(f"Broker strategies on held-out Binance BTC/USDT hours ({tag}); bars: ±2 s.e.", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
