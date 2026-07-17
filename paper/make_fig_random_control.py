"""fig_random_control — the matched-budget non-LLM control (reviewer #1).

Two panels, both vs turbine count N on the ROWP farm (mean over the three
directional roses: DEI / ROWP / omnidirectional; unidirectional is degenerate
and excluded):

  (A) AEP advantage over the naive baseline at LOOSE tolerance (5 m): the
      random-search champion MATCHES the LLM schedules — on raw AEP, dumb
      search does as well.
  (B) Strict-tolerance feasibility (0.1 m): the random champion's feasibility
      COLLAPSES with scale (~40% -> 0%), while the LLM schedules stay feasible
      (54-100%). The LLM discovered schedules that do the constraint work the
      no-early-stopping skeleton no longer does; random search did not.

So automated schedule discovery must find schedules that are both high-AEP and
self-feasible; random search finds the former, the LLM finds both.

Data: results/matrix/{ms_v2,ms_highn_v2}/*.json (LLM+baseline),
results/matrix/random_scale/*.random.json (control).
Output: paper/figs/fig_random_control.{pdf,png}
"""
import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"
COL = {"claude": "#c0392b", "gemini": "#2c6fbb", "random": "#27ae60"}
LAB = {"claude": "Claude dual-bump", "gemini": "Gemini schedule",
       "random": "random search (matched budget)"}
MINSP = 792.0
ROSES = ["dei", "rowp", "omnidir"]
NS = [80, 200, 300]

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})

# cell-indexed load (files are keyed by d["cell"], not filename)
CELLS = {}
for sub in ("results/matrix/ms_v2", "results/matrix/ms_highn_v2"):
    for f in glob.glob(os.path.join(ROOT, sub, "*.json")):
        d = json.load(open(f)); CELLS.setdefault(d["cell"], {}).update(d["schedules"])
for f in glob.glob(os.path.join(ROOT, "results/matrix/random_scale/*.random.json")):
    d = json.load(open(f)); CELLS.setdefault(d["cell"], {}).update(d["schedules"])


def seeds(cell, who):
    return CELLS.get(cell, {}).get(who, {}).get("seeds", [])


def best_feas(s, tol):
    g = [r["aep_gwh"] for r in s if "aep_gwh" in r
         and r["max_out_m"] <= tol and r["min_dist_m"] >= MINSP - tol]
    return max(g) if g else None


def feas_frac(s, tol):
    t = [r for r in s if "aep_gwh" in r]
    if not t:
        return np.nan
    return 100 * sum(1 for r in t if r["max_out_m"] <= tol
                     and r["min_dist_m"] >= MINSP - tol) / len(t)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

# ---- Panel A: AEP advantage over baseline at 5 m (loose) ----
for who in ["claude", "gemini", "random"]:
    means, xs, pts = [], [], []
    for N in NS:
        vals = []
        for rose in ROSES:
            cell = f"rowp_n{N}_rose{rose}"
            b = best_feas(seeds(cell, "baseline"), 5.0)
            s = best_feas(seeds(cell, who), 5.0)
            if b and s:
                vals.append(100 * (s - b) / b); pts.append((N, 100 * (s - b) / b))
        if vals:
            xs.append(N); means.append(np.mean(vals))
    if xs:
        axA.plot(xs, means, "-o", color=COL[who], lw=1.8, ms=4, zorder=3)
    if pts:
        px, py = zip(*pts)
        axA.scatter(px, py, s=10, color=COL[who], alpha=0.35, zorder=2, linewidth=0)
axA.set_xscale("log"); axA.set_xticks(NS); axA.set_xticklabels([str(n) for n in NS])
axA.xaxis.set_minor_formatter(plt.NullFormatter())
axA.axhline(0, color=MUT, lw=0.8, ls=(0, (4, 2)))
axA.set_xlabel("turbines $N$", fontsize=8.2)
axA.set_ylabel(r"$\Delta$AEP vs baseline (%)", fontsize=8.2)
axA.set_title("(A) AEP @ 5 m tolerance", fontsize=8.8, weight="bold")
axA.grid(alpha=0.18, lw=0.5); axA.set_axisbelow(True)
axA.text(0.5, 0.06, "random matches the LLM", transform=axA.transAxes,
         ha="center", fontsize=7.4, style="italic", color=MUT)

# ---- Panel B: strict-tolerance feasibility (0.1 m) ----
for who in ["claude", "gemini", "random"]:
    means, xs, pts = [], [], []
    for N in NS:
        vals = []
        for rose in ROSES:
            f = feas_frac(seeds(f"rowp_n{N}_rose{rose}", who), 0.1)
            if not np.isnan(f):
                vals.append(f); pts.append((N, f))
        if vals:
            xs.append(N); means.append(np.mean(vals))
    if xs:
        axB.plot(xs, means, "-o", color=COL[who], lw=1.8, ms=4, zorder=3)
    if pts:
        px, py = zip(*pts)
        axB.scatter(px, py, s=10, color=COL[who], alpha=0.35, zorder=2, linewidth=0)
axB.set_xscale("log"); axB.set_xticks(NS); axB.set_xticklabels([str(n) for n in NS])
axB.xaxis.set_minor_formatter(plt.NullFormatter())
axB.set_ylim(-4, 104)
axB.set_xlabel("turbines $N$", fontsize=8.2)
axB.set_ylabel("feasible restarts @ 0.1 m (%)", fontsize=8.2)
axB.set_title("(B) strict-tolerance feasibility", fontsize=8.8, weight="bold")
axB.grid(alpha=0.18, lw=0.5); axB.set_axisbelow(True)
axB.text(0.5, 0.55, "random collapses;\nLLM holds", transform=axB.transAxes,
         ha="center", fontsize=7.4, style="italic", color=MUT)

handles = [Line2D([0], [0], color=COL[w], lw=1.9, marker="o", ms=4, label=LAB[w])
           for w in ["claude", "gemini", "random"]]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10),
           ncol=3, fontsize=7.8, frameon=False)

for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_random_control.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
