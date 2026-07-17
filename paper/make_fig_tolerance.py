"""fig_tolerance — the constraint-tolerance confound, made explicit.

For each cell, sweep the feasibility tolerance T and show two coupled
quantities the schedules and baseline trade off:
  top row:    feasibility rate  = fraction of 50 starts feasible at T
  bottom row: best-feasible AEP = best AEP among starts feasible at T
Both vs T (log). A schedule's AEP line only exists where it has a feasible
start; the baseline appears only past ~its final-LR design tolerance, while
the discovered schedules are feasible down to strict T — the funwake
constraint-precision improvement, quantified.

Prototype over dei_n50 × 4 roses. Output: paper/figs/fig_tolerance.{pdf,png}
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"
COL = {"baseline": "#555555", "claude": "#c0392b", "gemini": "#2c6fbb"}
MK = {"baseline": "s", "claude": "o", "gemini": "^"}
TOLS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})

ROSES = ["dei", "rowp", "omnidir", "uniform"]
ROSE_TITLE = {"dei": "DEI rose", "rowp": "ROWP rose",
              "omnidir": "omnidirectional", "uniform": "unidirectional"}
FARM, N = "dei", 50


def load_cell(farm, n, rose):
    d = json.load(open(os.path.join(ROOT, f"results/matrix/ms_v2/cell_lookup.json"))) \
        if False else None
    # find the cell json by content
    import glob
    for f in glob.glob(os.path.join(ROOT, "results/matrix/ms_v2/*.json")):
        j = json.load(open(f))
        if j["cell"] == f"{farm}_n{n}_rose{rose}":
            return j
    return None


def feas_at(seeds, tol, minsp):
    return [r for r in seeds if "aep_gwh" in r
            and r["max_out_m"] <= tol and r["min_dist_m"] >= minsp - tol]


fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), sharex=True,
                         constrained_layout=True)

for ci, rose in enumerate(ROSES):
    cell = load_cell(FARM, N, rose)
    prob = json.load(open(os.path.join(ROOT, f"results/matrix/problem_{FARM}_n{N}_rose{rose}.json")))
    minsp = float(prob["min_spacing_m"])
    ax_f, ax_a = axes[0, ci], axes[1, ci]
    for sname in ["baseline", "claude", "gemini"]:
        s = cell["schedules"].get(sname)
        if not s or "seeds" not in s:
            continue
        rates, aeps = [], []
        for t in TOLS:
            g = feas_at(s["seeds"], t, minsp)
            rates.append(100 * len(g) / len(s["seeds"]))
            aeps.append(max(r["aep_gwh"] for r in g) if g else np.nan)
        ax_f.plot(TOLS, rates, color=COL[sname], marker=MK[sname], ms=3.5,
                  lw=1.4, zorder=3)
        ax_a.plot(TOLS, aeps, color=COL[sname], marker=MK[sname], ms=3.5,
                  lw=1.4, zorder=3)
    ax_f.set_xscale("log"); ax_a.set_xscale("log")
    ax_f.set_ylim(-5, 105)
    ax_f.set_title(ROSE_TITLE[rose], fontsize=8.4, weight="bold", pad=3)
    for ax in (ax_f, ax_a):
        ax.grid(alpha=0.2, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(length=2.5, labelsize=6.8)
    ax_a.set_xlabel("constraint tolerance (m)", fontsize=7.6)

axes[0, 0].set_ylabel("feasible starts (%)", fontsize=7.8)
axes[1, 0].set_ylabel("best-feasible\nAEP (GWh)", fontsize=7.8)

legend = [Line2D([0], [0], color=COL[s], marker=MK[s], lw=1.4,
                 label={"baseline": "baseline (seed sched.)",
                        "claude": "Claude dual-bump",
                        "gemini": "Gemini schedule"}[s])
          for s in ["baseline", "claude", "gemini"]]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.08),
           ncol=3, fontsize=7.6, frameon=False)

for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_tolerance.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
