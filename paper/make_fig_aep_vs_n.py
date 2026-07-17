"""fig_aep_vs_n — deployed schedules vs a MULTISTART baseline, best-of-K each,
across three sites × four roses × turbine count N.

Fair comparison (addresses the 500-vs-1 objection): every plotted point is a
best-of-multistart, not a single run.
  DEI / ROWP:  schedule = best of 50 skeleton starts (results/matrix/ms/);
               baseline = best of 500 TopFarm-SGD starts (baselines_matrix.json,
               the established gradient baseline). The 50-vs-500 asymmetry
               HANDICAPS the schedules, so any schedule win is conservative.
  ParqueFicticio: schedule and baseline are both best of 50 starts through the
               multizone skeleton (no TopFarm solver for disconnected zones).

y = ΔAEP (schedule best − baseline best), GWh. Zero line = baseline.
Filled = schedule feasible; open = infeasible / no feasible start.

Output: paper/figs/fig_aep_vs_n.{pdf,png}
"""
import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"
C_BASE, C_CLAUDE, C_GEM = "#555555", "#c0392b", "#2c6fbb"
PARQO_TOL = 1.0

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})

# ---- matrix multistart (best-of-50 per schedule, feasible @0.1m) ----
MS = {}
for f in glob.glob(os.path.join(ROOT, "results/matrix/ms/cell*.json")):
    d = json.load(open(f)); MS[d["cell"]] = d["schedules"]
TOPFARM = json.load(open(os.path.join(ROOT, "results/matrix/baselines_matrix.json")))

# ---- parqo multistart (all seeds, re-gateable) ----
PQ = {s: json.load(open(os.path.join(ROOT, f"parqo/parqo_ms_{s}.json")))
      for s in ["baseline", "claude", "gemini"]}

ROSES = ["dei", "rowp", "omnidir", "uniform"]
ROSE_TITLE = {"dei": "DEI rose", "rowp": "ROWP rose",
              "omnidir": "omnidirectional", "uniform": "unidirectional"}
MATRIX_N = [30, 40, 50, 60, 70, 80]
PARQO_N = [10, 15, 20, 25, 30, 35]


def matrix_point(farm, rose, n, sched):
    key = f"{farm}_n{n}_rose{rose}"
    s = MS.get(key, {}).get(sched)
    if not s:
        return None, False
    if s["best"]:
        return s["best"]["aep_gwh"], True
    return s.get("best_infeas_aep"), False   # no feasible start


def matrix_baseline(farm, rose, n):
    b = TOPFARM.get(f"{farm}_n{n}_rose{rose}")
    return b["best_aep"] if b else None


def parqo_best(sched, rose, n):
    cell = PQ[sched].get(f"{rose}|n{n}")
    if not cell:
        return None, False
    good = [s for s in cell["seeds"]
            if s["max_sdf_m"] <= PARQO_TOL and s["min_dist_m"] >= 160 - PARQO_TOL]
    if good:
        return max(good, key=lambda s: s["aep_gwh"])["aep_gwh"], True
    allf = [s["aep_gwh"] for s in cell["seeds"] if "aep_gwh" in s]
    return (max(allf) if allf else None), False


def draw(ax, xs, ys, fs, color, mk):
    if not xs:
        return
    ax.plot(xs, ys, color=color, lw=1.4, alpha=0.9, zorder=2)
    xs, ys, fs = np.array(xs), np.array(ys), np.array(fs)
    ax.scatter(xs[fs], ys[fs], s=26, color=color, marker=mk,
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(xs[~fs], ys[~fs], s=26, facecolor="white", marker=mk,
               edgecolor=color, linewidth=1.1, zorder=3)


fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.0), constrained_layout=True)
SITE_LABEL = ["DEI · IEA 15 MW", "ROWP · IEA 10 MW", "ParqueFicticio · V80 · 5 zones"]

for ri, farm in enumerate(["dei", "rowp", "parqo"]):
    row_absmax = 0
    for ci, rose in enumerate(ROSES):
        ax = axes[ri, ci]
        ax.axhline(0, color=C_BASE, lw=1.2, ls=(0, (4, 2)), zorder=1)
        if farm in ("dei", "rowp"):
            NS = MATRIX_N
            for sched, color, mk in [("claude", C_CLAUDE, "o"), ("gemini", C_GEM, "^")]:
                xs, ys, fs = [], [], []
                for n in NS:
                    a, feas = matrix_point(farm, rose, n, sched)
                    b = matrix_baseline(farm, rose, n)
                    if a is not None and b is not None and b > 0:
                        xs.append(n); ys.append(100*(a - b)/b); fs.append(feas)
                draw(ax, xs, ys, fs, color, mk)
                if ys:
                    row_absmax = max(row_absmax, max(abs(v) for v in ys))
        else:
            NS = PARQO_N
            for sched, color, mk in [("claude", C_CLAUDE, "o"), ("gemini", C_GEM, "^")]:
                xs, ys, fs = [], [], []
                for n in NS:
                    a, feas = parqo_best(sched, rose, n)
                    b, _ = parqo_best("baseline", rose, n)
                    if a is not None and b is not None and b > 0:
                        xs.append(n); ys.append(100*(a - b)/b); fs.append(feas)
                draw(ax, xs, ys, fs, color, mk)
                if ys:
                    row_absmax = max(row_absmax, max(abs(v) for v in ys))
        if ri == 0:
            ax.set_title(ROSE_TITLE[rose], fontsize=8.4, weight="bold", pad=4)
        ax.grid(alpha=0.2, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(length=2.5, labelsize=6.8)
        ax.set_xticks(NS[::2])
        if ri == 2:
            ax.set_xlabel("turbines $N$", fontsize=8)
    m = row_absmax * 1.15 + 1
    for ci in range(4):
        axes[ri, ci].set_ylim(-m, m)
    axes[ri, 0].set_ylabel(f"{SITE_LABEL[ri]}\n$\\Delta$AEP vs baseline (%)",
                           fontsize=7.2)

legend = [
    Line2D([0], [0], color=C_BASE, lw=1.2, ls=(0, (4, 2)),
           label="multistart baseline (0)"),
    Line2D([0], [0], color=C_CLAUDE, marker="o", lw=1.4, label="Claude dual-bump"),
    Line2D([0], [0], color=C_GEM, marker="^", lw=1.4, label="Gemini schedule"),
    Line2D([0], [0], color=MUT, marker="o", lw=0, markerfacecolor="white",
           markeredgecolor=MUT, label="open = infeasible"),
]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.035),
           ncol=4, fontsize=7.6, frameon=False)

for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_aep_vs_n.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
