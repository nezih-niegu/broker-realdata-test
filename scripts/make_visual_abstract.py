"""Visual abstract of the study: results/visual_abstract.png (+ .pdf).

Panels 3-4 are read from this repository's results; panels 1-2 quote the synthetic-game results of the
companion repository brokergame_marl (results/E*_*.json, S3_seeds_E3.json), as reported in the paper.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BENCH, ALT, NEUTRAL, SECOND = "#e34948", "#2a78d6", "#9a9893", "#eb6834"

# ---------------------------------------------------------------- data
# panel 1: R^2 of learned vs closed-form actions in the 9 configurations of the RNE analysis
cfg = ["base", "no ξ", "$\\Xi$=1", "$\\Xi$=2", "$\\Xi$=4", "$\\Xi$=0", "$\\Xi$=1", "$\\Xi$=2", "$\\Xi$=4"]
r2_B = [0.992, 0.990, 0.993, 0.995, 0.996, 0.998, 0.998, 0.998, 0.998]
r2_I = [0.970, 0.974, 0.988, 0.987, 0.990, 0.998, 0.995, 0.994, 0.994]
# panel 2: broker's gain from a feedback deviation, % of |J^B| (3 seeds), open- vs closed-loop equilibrium
ol_gain = [19.4, 15.1, 14.0]
cl_gain = [-2.4, 0.0, 0.0]       # seed 0 measured (-0.0011/0.0461); closed-loop seeds 1-2 not re-certified -> 0 shown as n/a
# panel 3: t-statistics of informed flow on forward drift and on own inventory (test days)
rt = json.loads((ROOT / "results" / "results_fee4bps.json").read_text())["flow_regressions"]["informed"]
t_drift, t_inv = rt["t_fwd_drift"], rt["t_own_inventory"]
# panel 4: RL benchmark, test objective at gamma*
S = {r["model"]: r for r in json.loads((ROOT / "results" / "rl_benchmark" / "summary.json").read_text())["rows"]}
rows = [("Closed-form RNE\n(benchmark)", "Nash_closed_form_gamma_star", BENCH),
        ("Closed form,\nQᴵ term removed", "Nash_closed_form_no_QI_channel", NEUTRAL),
        ("Linear rule v=−κQᴮ", "Linear_control_kappa_tuned_on_val", NEUTRAL),
        ("PPO", "PPO", ALT), ("Recurrent PPO", "RecurrentPPO", ALT),
        ("DPS, linear\n(best on validation)", "DPS-linear", ALT)]

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(13.5, 5.2), facecolor=SURF)
fig.text(0.012, 0.955, "Learned broker strategies against informed order flow", fontsize=15,
         weight="bold", color=INK)
fig.text(0.012, 0.905, "Benchmark: Riccati–Nash equilibrium (RNE) of Cartea, Jaimungal & Sánchez-Betancourt (2026)"
         "   |   certified multi-agent learning → 4.3 M Binance BTC/USDT trades → 11 RL families, 144 held-out hours",
         fontsize=9, color=INK2)
gs = fig.add_gridspec(1, 4, left=0.045, right=0.975, top=0.74, bottom=0.2, wspace=0.45,
                      width_ratios=[1, 0.8, 0.8, 1.3])

def head(ax, n, title, sub):
    ax.set_title(f"{n}  {title}", loc="left", fontsize=10, weight="bold", color=INK, pad=30)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", linespacing=1.2)
    ax.set_facecolor(SURF)

# 1 recovery
a = fig.add_subplot(gs[0])
x = range(len(cfg))
a.plot(x, r2_B, "o", ms=6, color=ALT, label="broker")
a.plot(x, r2_I, "s", ms=6, color=SECOND, label="informed trader")
a.axhline(0.95, color=INK2, lw=0.8, ls=":")
a.text(8.4, 0.9515, "0.95 target", fontsize=7, color=INK2, ha="right", va="bottom")
a.axvspan(4.5, 8.5, color="#f0efec", zorder=0)
a.text(6.5, 0.9435, "strong impact:\nno RNE existence result", ha="center", fontsize=7, color=INK2)
a.set_xticks(list(x), cfg, fontsize=7, rotation=45)
a.set_ylim(0.94, 1.0); a.set_ylabel("R² learned vs closed-form actions")
a.grid(axis="y", color=GRID, lw=0.6); a.legend(frameon=False, fontsize=7, loc="lower left")
head(a, "①", "Equilibrium recovered", "R² ≥ 0.97 in all 9 RNE configurations;\nexploitability < 1% of payoffs")

# 2 concept
b = fig.add_subplot(gs[1])
seeds = [0, 1, 2]
b.bar([s - 0.2 for s in seeds], ol_gain, width=0.38, color=BENCH, label="vs open-loop RNE")
b.bar([0.2], [max(cl_gain[0], 0) + 0.3], width=0.38, color=ALT, label="vs closed-loop Nash (seed 0)")
b.text(0.2, 1.2, "≈0", ha="center", fontsize=7.5, color=INK)
for s, v in zip(seeds, ol_gain):
    b.text(s - 0.2, v + 0.5, f"{v:.0f}%", ha="center", fontsize=7.5, color=INK)
b.set_xticks(seeds, ["seed 0", "seed 1", "seed 2"]); b.set_ylim(0, 25)
b.set_ylabel("broker's gain from reacting to the state\n(% of her payoff)")
b.grid(axis="y", color=GRID, lw=0.6); b.legend(frameon=False, fontsize=6.8, loc="upper right")
head(b, "②", "Open loop is exploitable", "against the RNE, a broker\nwho reacts to the state gains 14–20%")

# 3 data
c = fig.add_subplot(gs[2])
labels = ["on next price\ndrift (β_D)", "on own\ninventory (β_Q)"]
obs = [t_drift, t_inv]
c.barh([1, 0], obs, height=0.45, color=BENCH)
c.axvline(0, color=INK2, lw=0.8)
for y, v, pred in ((1, obs[0], "model: > 0"), (0, obs[1], "model: < 0")):
    c.text(0.3 if v < 0 else v + 0.3, y, f"t = {v:+.1f}", va="center", ha="left", fontsize=8, color=INK)
    c.text(0, y + 0.33, pred, ha="center", fontsize=7, color=INK2)
c.set_yticks([1, 0], labels); c.set_xlim(-8, 10); c.set_ylim(-0.6, 1.7)
c.set_xlabel("t-statistic, informed flow (test days)")
c.grid(axis="x", color=GRID, lw=0.6)
head(c, "③", "Model assumptions fail", "large orders reverse and persist,\nopposite to the model; signal R² < 0")

# 4 RL
d = fig.add_subplot(gs[3])
ys = list(range(len(rows)))[::-1]
for y, (lab, key, col) in zip(ys, rows):
    r = S[key]
    cost = -r["test_J"]
    d.barh(y, cost, height=0.55, color=col)
    d.text(cost * 1.15, y, f"{cost:,.0f}    P&L {r['pnl']:,.0f}", va="center", fontsize=7.5, color=INK)
d.set_xscale("log"); d.set_xlim(1e3, 1.5e6)
d.set_yticks(ys, [r[0] for r in rows], fontsize=7.5)
d.set_xlabel("test cost −Ĵ at γ* per hour, USDT (log scale; lower is better)")
d.grid(axis="x", color=GRID, lw=0.6, which="major")
head(d, "④", "Out of sample, paper's own objective",
     "8 of 11 RL families beat the closed form (Holm p<0.01);\nremoving one term of the closed form closes most of the gap")

fig.text(0.012, 0.03, "Takeaway: the closed form is a sound equilibrium computation at its own parameters, but its response to the "
         "informed client's inventory fails out of sample.\nRemoving that term, or learning the controller, cuts the "
         "risk-adjusted cost about 40×; raw P&L still favours the original closed form.", fontsize=8.6, color=INK2,
         linespacing=1.4)
out = ROOT / "results" / "visual_abstract"
fig.savefig(f"{out}.png", dpi=170, facecolor=SURF)
fig.savefig(f"{out}.pdf", facecolor=SURF)
print("wrote", out)
