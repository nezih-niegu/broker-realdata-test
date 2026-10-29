"""Markouts by parent-order size decile on TEST days (deciles fixed on training days).

Motivation: the stealth-trading hypothesis (Barclay & Warner 1993) predicts that informed traders
concentrate in medium-size orders, so the largest orders need not be the most informed.
Output: results/markouts_by_size.json and .png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realdata.loaders import load_folder
from realdata.flow import parent_orders, markouts

trades, _ = load_folder(ROOT / "data" / "BTCUSDT")
orders = parent_orders(trades)
days = pd.date_range(trades["time"].min().normalize(), trades["time"].max().normalize(), freq="D")
split = days[14]
edges = orders.loc[orders["time"] < split, "qty"].quantile(np.linspace(0, 1, 11)).values.copy()
edges[0], edges[-1] = 0, np.inf
te = orders[orders["time"] >= split].copy()
te["decile"] = np.clip(np.searchsorted(edges, te["qty"].values, side="right") - 1, 0, 9)
mk = markouts(te, trades, horizons_s=(10, 60, 300, 900))
res = []
for d in range(10):
    m = te["decile"].values == d
    row = {"decile": d + 1, "min_btc": float(edges[d]), "max_btc": float(edges[d + 1]),
           "n": int(m.sum()), "volume_share": float(te.loc[m, "qty"].sum() / te["qty"].sum())}
    for h in mk.columns:
        x = mk.loc[m, h].values
        row[f"{h}s"] = float(x.mean()); row[f"{h}s_se"] = float(x.std() / np.sqrt(len(x)))
        # volume-weighted markout: what a counterparty loses per unit traded
        w = te.loc[m, "qty"].values
        row[f"{h}s_vw"] = float(np.sum(w * x) / np.sum(w))
    res.append(row)
(ROOT / "results" / "markouts_by_size.json").write_text(json.dumps(res, indent=2))
for r in res:
    print(f"decile {r['decile']:2d} [{r['min_btc']:.4f},{r['max_btc']:.4f}) n={r['n']:7d} vol {r['volume_share']:.1%}  "
          + "  ".join(f"{h}s {r[f'{h}s']:+.2f}±{r[f'{h}s_se']:.2f}" for h in (10, 60, 300, 900)))

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(6.5, 2.8))
x = np.arange(1, 11)
for h, c in ((10, "#c6dbef"), (60, "#6baed6"), (300, "#2171b5"), (900, "#08306b")):
    ax.errorbar(x + (h ** 0.25 - 3) * 0.05, [r[f"{h}s"] for r in res], yerr=[2 * r[f"{h}s_se"] for r in res],
                fmt="-o", ms=3, lw=1.1, capsize=1.5, color=c, label=f"{h} s")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x, [f"D{i}" for i in x])
ax.set_xlabel("parent-order size decile (training-day cut-offs; D10 = largest)")
ax.set_ylabel("markout (bps)")
ax.legend(title="horizon", ncol=4, frameon=False, fontsize=7, title_fontsize=7, loc="lower right")
ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout()
fig.savefig(ROOT / "results" / "markouts_by_size.png", dpi=150)
fig.savefig(ROOT / "results" / "markouts_by_size.pdf")
