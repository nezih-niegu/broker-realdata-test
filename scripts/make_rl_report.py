"""Summarise the RL benchmark: results/rl_benchmark/*.json -> RESULTS_RL.md, summary.json, figure.

Per family we report (i) the mean over seeds of the test metrics and (ii) the seed chosen by validation
objective (what a practitioner would deploy). Paired differences against the closed-form Nash broker
use the same 144 test hours, so the standard error is that of the per-hour difference.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "rl_benchmark"
NASH = "Nash_closed_form_gamma_star"

FAMILY_ORDER = ["PPO", "PPO_ext", "RecurrentPPO", "A2C", "SAC", "TQC", "TD3", "DDPG",
                "DPS", "DPS_ext", "DPS-linear", "SAC-rs100", "TD3-rs100", "DDPG-rs100", "TD3-nowarmup"]
DIAGNOSTIC = ("-rs100", "-nowarmup")   # failed first-run configurations, reported but not candidates
BASE_ORDER = [NASH, "Nash_closed_form_gamma_tuned_on_val", "Nash_closed_form_no_QI_channel", "Inventory_decay_tau0.05",
              "Inventory_decay_tau_tuned_on_val", "Linear_control_kappa_tuned_on_val"]
LABEL = {NASH: "Closed-form Nash (paper), $\\gamma^*$", "Nash_closed_form_gamma_tuned_on_val": "Closed-form Nash, $\\gamma$ tuned on val",
         "Nash_closed_form_no_QI_channel": "Closed-form Nash, $Q^I$ channel off",
         "Inventory_decay_tau0.05": "Inventory decay, $\\tau_0=0.05$", "Inventory_decay_tau_tuned_on_val": "Inventory decay, $\\tau_0$ tuned",
         "Linear_control_kappa_tuned_on_val": "Linear control $v=-\\kappa Q^B$, $\\kappa$ tuned",
         "PPO": "PPO", "PPO_ext": "PPO + price features", "RecurrentPPO": "Recurrent PPO (LSTM)", "A2C": "A2C",
         "SAC": "SAC", "TQC": "TQC", "TD3": "TD3", "DDPG": "DDPG", "DPS": "DPS (MLP)",
         "DPS_ext": "DPS (MLP) + price features", "DPS-linear": "DPS (linear)",
         "SAC-rs100": "SAC, first run", "TD3-rs100": "TD3, first run", "DDPG-rs100": "DDPG, first run",
         "TD3-nowarmup": "TD3, no warm-up"}


def load(name):
    j = json.loads((R / f"{name}.json").read_text())
    z = np.load(R / f"{name}.npz")
    return j, z["objective"], z["mtm"]


def paired(a, b):
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def pval(m, se):
    """Two-sided p-value of a paired mean difference (t with 143 df ~ normal)."""
    from scipy.stats import t
    return float(2 * t.sf(abs(m / se), 143)) if se > 0 else float("nan")


def holm(ps):
    """Holm-Bonferroni adjusted p-values."""
    idx = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def main():
    _, J0, M0 = load(NASH)
    rows, fam_runs = [], {}
    for f in sorted(R.glob("*_seed*.json")):
        fam = re.sub(r"_seed\d+$", "", f.stem)
        fam_runs.setdefault(fam, []).append(f.stem)
    for name in BASE_ORDER:
        if not (R / f"{name}.json").exists():
            continue
        j, J, M = load(name)
        dJ, sJ = paired(J, J0); dM, sM = paired(M, M0)
        rows.append({"model": name, "kind": "baseline", "n_seeds": 1, "val_J": j["val_objective"],
                     "test_J": float(J.mean()), "test_J_se": float(J.std(ddof=1) / np.sqrt(len(J))),
                     "pnl": float(M.mean()), "pnl_sd": float(M.std(ddof=1)), "sharpe": float(M.mean() / M.std(ddof=1)),
                     "abs_QT": j["test"]["abs_QT"], "dJ_vs_nash": dJ, "dJ_se": sJ, "dPnL_vs_nash": dM, "dPnL_se": sM,
                     "seed_J": [float(J.mean())], **{k: j[k] for k in ("gamma", "tau0", "kappa") if k in j}})
    for fam in [f for f in FAMILY_ORDER if f in fam_runs] + [f for f in fam_runs if f not in FAMILY_ORDER]:
        runs = [load(n) for n in sorted(fam_runs[fam])]
        Js = np.stack([r[1] for r in runs]); Ms = np.stack([r[2] for r in runs])
        sel = int(np.argmax([r[0]["val_objective"] for r in runs]))
        Jm, Mm = Js.mean(0), Ms.mean(0)            # per-hour average over seeds
        Jsel, Msel = Js[sel], Ms[sel]
        dJ, sJ = paired(Jsel, J0); dM, sM = paired(Msel, M0)
        dJm, sJm = paired(Jm, J0)
        rows.append({"model": fam, "kind": "rl", "n_seeds": len(runs),
                     "val_J": runs[sel][0]["val_objective"], "selected_seed": runs[sel][0].get("seed", sel),
                     "test_J": float(Jsel.mean()), "test_J_se": float(Jsel.std(ddof=1) / np.sqrt(len(Jsel))),
                     "pnl": float(Msel.mean()), "pnl_sd": float(Msel.std(ddof=1)),
                     "sharpe": float(Msel.mean() / Msel.std(ddof=1)), "abs_QT": runs[sel][0]["test"]["abs_QT"],
                     "dJ_vs_nash": dJ, "dJ_se": sJ, "dPnL_vs_nash": dM, "dPnL_se": sM,
                     "seed_J": [float(x) for x in Js.mean(1)], "seed_pnl": [float(x) for x in Ms.mean(1)],
                     "seed_sharpe": [float(m.mean() / m.std(ddof=1)) for m in Ms],
                     "seedavg_test_J": float(Js.mean()), "seedavg_dJ_vs_nash": dJm, "seedavg_dJ_se": sJm,
                     "train_minutes": float(np.mean([r[0].get("train_seconds", np.nan) for r in runs]) / 60)})
    for r in rows:
        r["p_J"] = pval(r["dJ_vs_nash"], r["dJ_se"]); r["p_PnL"] = pval(r["dPnL_vs_nash"], r["dPnL_se"])
    oth = [r for r in rows if r["model"] != NASH]
    for key in ("p_J", "p_PnL"):
        for r, a in zip(oth, holm(np.array([r[key] for r in oth]))):
            r[key + "_holm"] = float(a)
    rl = [r for r in rows if r["kind"] == "rl"]
    cand = [r for r in rl if not r["model"].endswith(DIAGNOSTIC)]
    best = max(cand, key=lambda r: r["val_J"]) if cand else None
    lin = next((r for r in rows if r["model"] == "Linear_control_kappa_tuned_on_val"), None)
    if best and lin:
        Jb = np.stack([load(n)[1] for n in sorted(fam_runs[best["model"]])])[
            [load(n)[0]["val_objective"] for n in sorted(fam_runs[best["model"]])].index(best["val_J"])]
        _, Jl, _ = load("Linear_control_kappa_tuned_on_val")
        best["dJ_vs_linear"], best["dJ_vs_linear_se"] = paired(Jb, Jl)
        best["p_vs_linear"] = pval(best["dJ_vs_linear"], best["dJ_vs_linear_se"])
    summary = {"rows": rows, "best_by_validation": best["model"] if best else None}
    (R / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- markdown
    L = ["# RL benchmark on held-out BTCUSDT hours (4 bps fee)", "",
         "Objective J = the paper's broker objective at the training-day risk aversion gamma* "
         "(P&L - running inventory penalty - terminal penalty), per one-hour episode, 144 test hours.",
         "RL rows: test metrics of the seed with the best validation J; 'seeds' lists each seed's test J.", "",
         "| model | val J | test J (± se) | ΔJ vs Nash (± se) | P&L | sd P&L | Sharpe | ΔP&L vs Nash (± se) | E\\|Q_T\\| | test J per seed |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['val_J']:.0f} | {r['test_J']:.0f} ± {r['test_J_se']:.0f} | "
                 f"{r['dJ_vs_nash']:+.0f} ± {r['dJ_se']:.0f} | {r['pnl']:.0f} | {r['pnl_sd']:.0f} | {r['sharpe']:.3f} | "
                 f"{r['dPnL_vs_nash']:+.0f} ± {r['dPnL_se']:.0f} | {r['abs_QT']:.1f} | "
                 + ", ".join(f"{x:.0f}" for x in r["seed_J"]) + " |")
    if best:
        L += ["", f"Best RL family by validation J: **{best['model']}** (seed {best['selected_seed']})."]
        if "dJ_vs_linear" in best:
            L.append(f"Its test J minus the tuned linear rule's: {best['dJ_vs_linear']:+.0f} ± {best['dJ_vs_linear_se']:.0f}.")
    (R / "RESULTS_RL.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---------------- figure
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    order = rows[::-1]
    y = np.arange(len(order))
    for i, r in enumerate(order):
        c = "#b2182b" if r["model"].startswith("Nash") else ("#4d4d4d" if r["kind"] == "baseline" else "#2166ac")
        cost = [-x for x in r["seed_J"]]
        a1.scatter(np.maximum(cost, 1), [i] * len(cost), s=9, color=c, alpha=0.45, lw=0)
        a1.errorbar(max(-r["test_J"], 1), i, xerr=None, fmt="o", ms=4.5, color=c, mfc=c if r["kind"] == "rl" else "white")
    a1.set_xscale("log")
    a1.set_yticks(y, [LABEL.get(r["model"], r["model"]) for r in order])
    a1.set_xlabel("test cost $-J$ per hour (log scale; lower is better)")
    a1.grid(axis="x", alpha=0.3, lw=0.5, which="both")
    a1.set_title("(a) paper's objective at $\\gamma^*$", fontsize=8)
    fr = R / "frontier.json"
    if fr.exists():
        F = json.loads(fr.read_text())
        for key, c, ls, lab, keep in (
                ("nash", "#b2182b", "-", "closed form, $\\gamma$ sweep", lambda d: True),
                ("linear", "#4d4d4d", ":", "linear control, $\\kappa$ sweep", lambda d: d["kappa"] <= 100),
                ("dps", "#2166ac", "--", "DPS (MLP), $\\gamma\\leq0.1$", lambda d: d["gamma"] <= 0.1)):
            pts = sorted([d for d in F.get(key, []) if keep(d)], key=lambda d: d.get("gamma", d.get("kappa")))
            a2.plot([d["pnl_sd"] for d in pts], [d["pnl"] for d in pts], ls, color=c, lw=1, label=lab,
                    marker=".", ms=4)
    shown = {NASH: "closed form, $\\gamma^*$", "Nash_closed_form_no_QI_channel": "closed form, $Q^I$ off",
             "Linear_control_kappa_tuned_on_val": "linear, $\\kappa$ tuned", "DPS-linear": "DPS (linear)",
             "RecurrentPPO": "Rec. PPO", "PPO": "PPO", "A2C": "A2C"}
    for r in rows:
        if r["model"] not in shown:
            continue
        c = "#b2182b" if r["model"].startswith("Nash") else ("#4d4d4d" if r["kind"] == "baseline" else "#2166ac")
        mk = "s" if r["kind"] == "baseline" else "o"
        a2.scatter(r["pnl_sd"], r["pnl"], color=c, marker=mk, s=20, zorder=3)
        a2.annotate(shown[r["model"]], (r["pnl_sd"], r["pnl"]), fontsize=6, xytext=(3, -8),
                    textcoords="offset points")
    a2.set_xlim(900, 4100); a2.set_ylim(-500, 2600)
    a2.legend(frameon=False, fontsize=6.5, loc="upper left")
    a2.axhline(0, color="k", lw=0.5)
    a2.set_xlabel("sd of hourly P&L (USDT)")
    a2.set_ylabel("mean hourly P&L (USDT)")
    a2.set_title("(b) risk and return on test hours", fontsize=8)
    a2.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(R / "rl_benchmark.png", dpi=150)
    fig.savefig(R / "rl_benchmark.pdf")


if __name__ == "__main__":
    main()
