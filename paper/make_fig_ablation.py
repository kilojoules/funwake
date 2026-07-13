"""fig_ablation — which component of Claude's iter_192 schedule holds
constraint feasibility at scale.

Each iter_192 variant, run through the K=50 matrix at ROWP rose N=200/300:
remove the dual LR bumps, weaken the alpha penalty (5x->1x coupling, no late
ramp), or swap Claude's low Adam betas (0.3, 0.5) for standard (0.9, 0.999).
Strict-tolerance (0.1 m) feasibility is the readout; AEP is essentially
unchanged across variants (annotated), so these knobs are constraint machinery,
not AEP.

Finding: standard Adam betas collapse feasibility to 0% at every scale -> the
LOW betas are the dominant self-feasibility mechanism the LLM discovered; the
strong alpha coupling is secondary, the bumps minor.

Data: results/matrix/ablation/*.json + full iter_192 from ms_highn_v2.
Output: paper/figs/fig_ablation.{pdf,png}
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"
MINSP = 792.0

VARIANTS = [
    ("full iter_192", "claude", "#c0392b"),
    ("$-$ dual bumps", "ab_nobumps", "#e08e0b"),
    ("$-$ strong $\\alpha$", "ab_weakalpha", "#2c6fbb"),
    ("standard betas\n(0.9, 0.999)", "ab_stdbetas", "#7f7f7f"),
]
NS = [200, 300]

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})


def _seeds(cell, who):
    if who == "claude":
        p = os.path.join(ROOT, f"results/matrix/ms_highn_v2/{cell}.json")
        return json.load(open(p))["schedules"]["claude"]["seeds"] if os.path.exists(p) else []
    p = os.path.join(ROOT, f"results/matrix/ablation/{cell}.{who}.json")
    return json.load(open(p))["schedules"][who]["seeds"] if os.path.exists(p) else []


def feas(seeds, tol=0.1):
    t = [r for r in seeds if "aep_gwh" in r]
    if not t:
        return np.nan
    return 100 * sum(1 for r in t if r["max_out_m"] <= tol
                     and r["min_dist_m"] >= MINSP - tol) / len(t)


fig, ax = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
x = np.arange(len(VARIANTS))
w = 0.36
for j, N in enumerate(NS):
    vals = [feas(_seeds(f"rowp_n{N}_roserowp", who)) for _, who, _ in VARIANTS]
    bars = ax.bar(x + (j - 0.5) * w, vals, w,
                  color=[c for _, _, c in VARIANTS],
                  alpha=0.55 if N == 200 else 1.0,
                  edgecolor=INK, linewidth=0.5,
                  label=f"$N={N}$")
    for xi, v in zip(x + (j - 0.5) * w, vals):
        ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom",
                fontsize=6.6, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([lbl for lbl, _, _ in VARIANTS], fontsize=7.6)
ax.set_ylabel("feasible restarts @ 0.1 m (%)", fontsize=8.4)
ax.set_ylim(0, 112)
ax.set_title("What holds feasibility at scale (ROWP rose)",
             fontsize=9.2, weight="bold", pad=6)
ax.grid(axis="y", alpha=0.2, lw=0.5); ax.set_axisbelow(True)
ax.tick_params(length=2.5, labelsize=7)
# lighter bars = N=200, solid = N=300
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=MUT, alpha=0.55, label="$N=200$"),
                   Patch(facecolor=MUT, label="$N=300$")],
          loc="upper right", fontsize=7.2, frameon=False)
ax.text(0.02, 0.02, "AEP unchanged across variants (~15510 GWh @5 m); "
        "these knobs are constraint machinery.",
        transform=ax.transAxes, fontsize=6.4, color=MUT, style="italic")

for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_ablation.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
