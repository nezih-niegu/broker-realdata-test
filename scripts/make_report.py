"""Build the figures and the results tables from results/results_*.json.

A frontier point is flagged unstable when the closed-form feedback gain g on the broker's own inventory
satisfies g*dt > 2: the discrete inventory recursion Q <- (1 - g dt) Q then diverges. Between 1 and 2
it overshoots and oscillates but converges; those points are kept and reported with their g*dt.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brokergame.config import GameConfig
from brokergame.riccati import RiccatiSolution

OUT = ROOT / "results"
TAGS = ["fee1bps", "fee4bps", "fee10bps"]


def stable(base, gamma, n):
    q = dict(base)
    q["phi"] = q["psi"] = q["a"] / 1.2e-3
    q["rB"] = q["rI"] = 0.5 * gamma * q["sig_S"] ** 2
    R = RiccatiSolution(GameConfig(n=n, **q))
    gain = np.abs(R.gains(0)[:, 0]).max()
    return bool(gain / n < 2.0), float(gain / n)


def main():
    res = {t: json.load(open(OUT / f"results_{t}.json")) for t in TAGS if (OUT / f"results_{t}.json").exists()}
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 4.6))
    axes = np.atleast_1d(axes)
    lines = ["# Results\n"]
    for ax, (tag, r) in zip(axes, res.items()):
        n = 3600 // int(r["args"]["dt_s"])
        fr = r["frontiers"]
        nash = fr["Nash broker (closed form), gamma grid"]
        for x in nash:
            x["stable"], x["gain_dt"] = stable(r["base_params"], x["param"], n)
        ns = [x for x in nash if x["stable"] and x["gain_dt"] <= 1.5]   # g dt > 1.5: strong oscillation
        dec = fr["Inventory decay, tau0 grid"]
        ax.errorbar([x["sd"] for x in ns], [x["mean"] for x in ns], yerr=[2 * x["se"] for x in ns],
                    fmt="-o", ms=4, lw=1.5, capsize=2, color="#1f4e79", label="Nash broker (closed form), varying $\\gamma$")
        ax.errorbar([x["sd"] for x in dec], [x["mean"] for x in dec], yerr=[2 * x["se"] for x in dec],
                    fmt="-o", ms=4, lw=1.2, capsize=2, color="#8c8c8c", label="Inventory decay, varying speed")
        bt = r["backtest_at_gamma_star"]
        for name, m, c in (("Internalise, liquidate at close", "s", "#d62728"),
                           ("Externalise all client flow", "D", "#9467bd"),
                           ("Learned broker, open loop (MARL)", "^", "#e07b39"),
                           ("Learned broker, closed loop (MARL)", "v", "#2ca02c")):
            if name in bt:
                ax.plot(bt[name]["pnl"]["sd"], bt[name]["pnl"]["mean"], m, ms=9, color=c, mec="black",
                        mew=0.8, zorder=10, label=name)
        ax.set_title(f"lit fee {r['args']['fee_bps']:g} bps", fontsize=10)
        ax.set_xlabel("sd of hourly P&L (USDT)")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("mean hourly P&L (USDT)")

        # ---- tables
        lines.append(f"\n## Lit fee {r['args']['fee_bps']:g} bps (risk aversion chosen on training days: "
                     f"gamma* = {r['gamma_star']:g})\n")
        lines.append("| Strategy | Mean P&L/h | sd | Sharpe/h | P&L net of client spread | Paired vs Nash | mean abs Q_T (BTC) | Lit volume (BTC/h) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, v in bt.items():
            d = v["pnl_minus_nash"]
            lines.append(f"| {name} | {v['pnl']['mean']:+.0f} ± {v['pnl']['se']:.0f} | {v['pnl']['sd']:.0f} | "
                         f"{v['sharpe_hourly']:+.3f} | {v['pnl_net_of_client_revenue']['mean']:+.0f} | "
                         f"{d['mean']:+.0f} ± {d['se']:.0f} | {v['abs_terminal_inventory']:.1f} | {v['lit_volume_per_hour']:.0f} |")
        lines.append("\nFrontier (mean / sd of hourly P&L): Nash by gamma: "
                     + "; ".join(f"{x['param']:g}: {x['mean']:+.0f}/{x['sd']:.0f}" + f" (g dt {x['gain_dt']:.2f}" + (")" if x["stable"] else ", unstable, excluded)")
                                 for x in nash))
        lines.append("Inventory decay by tau0 (hours): " + "; ".join(f"{x['param']:g}: {x['mean']:+.0f}/{x['sd']:.0f}" for x in dec))
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Broker strategies on 144 held-out hours of Binance BTC/USDT (Jan 2019); bars ±2 s.e.; "
                 "closed-form points with gain x dt > 1.5 omitted",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / "frontiers.png", dpi=150)
    fig.savefig(OUT / "frontiers.pdf")
    plt.close(fig)

    r = res.get("fee4bps") or next(iter(res.values()))
    lines.append("\n## Data, proxy and model checks (identical across fee levels)\n")
    mk = r["markouts_test"]
    lines.append("| Client class (test days) | 10 s markout (bps) | 60 s | 300 s | Volume share | Orders |")
    lines.append("|---|---|---|---|---|---|")
    for lab, v in mk.items():
        lines.append(f"| {lab} | {v['10s']['mean_bps']:+.2f} ± {v['10s']['se_bps']:.2f} | {v['60s']['mean_bps']:+.2f} ± {v['60s']['se_bps']:.2f} | "
                     f"{v['300s']['mean_bps']:+.2f} ± {v['300s']['se_bps']:.2f} | {v['volume_share']:.0%} | {v['n_orders']:,} |")
    fl = r["flow_regressions"]
    lines.append("\n| Flow (test days) | Loading on forward drift (t) | Model predicts | Loading on own inventory (t) | Model predicts |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| informed | {fl['informed']['coef_fwd_drift_per_sd']:+.1f} ({fl['informed']['t_fwd_drift']:+.1f}) | positive | "
                 f"{fl['informed']['coef_own_inventory_per_sd']:+.1f} ({fl['informed']['t_own_inventory']:+.1f}) | negative |")
    lines.append(f"| uninformed | {fl['uninformed']['coef_fwd_drift_per_sd']:+.1f} ({fl['uninformed']['t_fwd_drift']:+.1f}) | zero | "
                 f"{fl['uninformed']['coef_own_inventory_per_sd']:+.1f} ({fl['uninformed']['t_own_inventory']:+.1f}) | zero |")
    d = r["diagnostics"]
    lines.append(f"\nSignal: R^2 of forward drift {d['signal_R2_train']:.4f} (train), {d['signal_R2_test']:.4f} (test).")
    lines.append("\nCalibrated base parameters (hours, USDT, BTC): "
                 + ", ".join(f"{k} = {v:.4g}" for k, v in r["base_params"].items() if k not in ("phi", "psi", "rB", "rI")))
    if r.get("sim_checks"):
        lines.append("\nLearned agents in the calibrated simulator: "
                     + ", ".join(f"{k} = {v:.4f}" for k, v in r["sim_checks"].items()))
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
