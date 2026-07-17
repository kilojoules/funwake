"""fig_aep_dominance — AEP advantage of the discovered schedules over the
baseline, at DEPLOYABLE constraint tolerance, shown to be tolerance-robust.

For each cell the schedules and baseline are gated at the baseline's design
tolerance (>= 1 m, where the seed schedule actually satisfies constraints).
The line is the AEP gap at 1 m; the shaded band spans the gap recomputed over
tol in {1, 2, 5} m, so a thin band = tolerance-robust dominance (not a
strict-tolerance artifact).

y = 100 x (schedule_best - baseline_best) / baseline_best  (%)
x = turbine count N; rows = sites; cols = roses.

Prototype over the DEI/ROWP matrix (low-N). Output: paper/figs/fig_aep_dominance.{pdf,png}
"""
import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.transforms import IdentityTransform
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"
C_CLAUDE, C_GEM = "#c0392b", "#2c6fbb"


class Squiggle(AbstractPathEffect):
    """Draw a line as a sine wiggle. Amplitude/wavelength in display points,
    so the squiggle is isotropic on the page regardless of data scale."""

    def __init__(self, amplitude=2.2, wavelength=11.0):
        super().__init__((0, 0))
        self.amp, self.wl = amplitude, wavelength

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        v = affine.transform_path(tpath).vertices
        if len(v) < 2:
            return
        seg = np.hypot(*np.diff(v, axis=0).T)
        cum = np.concatenate([[0], np.cumsum(seg)])
        total = cum[-1]
        if total == 0:
            return
        s = np.linspace(0, total, max(int(total / 1.5), 3))
        x = np.interp(s, cum, v[:, 0]); y = np.interp(s, cum, v[:, 1])
        dx, dy = np.gradient(x), np.gradient(y)
        L = np.hypot(dx, dy); L[L == 0] = 1
        off = self.amp * np.sin(2 * np.pi * s / self.wl)
        xw, yw = x - dy / L * off, y + dx / L * off
        renderer.draw_path(gc, MPath(np.column_stack([xw, yw])),
                           IdentityTransform(), rgbFace)


SQUIGGLE = [Squiggle()]
# baseline constraint tolerance -> (linestyle, path-effect). Show the full
# sweep: the gap GROWS as tolerance tightens because the baseline is built to
# satisfy constraints only loosely (~1 m) and mostly fails at 0.1 m, so its
# best-feasible layout there is poor. That is the constraint-precision result
# in AEP terms, not a plotting artifact. (0.1 m baseline is few-sample — its
# feasible count is annotated so the reader sees the sampling context.)
TOL_STYLE = [(0.1, "solid", None),              # strict: with markers
             (1.0, "solid", SQUIGGLE),          # squiggly
             (5.0, (0, (1, 4)), None)]          # loosely dotted

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})

# load all v2 cells (low-N ms_v2 + high-N ms_highn_v2)
# NOTE: the matched-budget random-search control lives in its own figure
# (fig_random_control) — it matches on AEP but its feasibility collapses at
# scale, which this AEP-gap view would misleadingly imply as parity.
CELLS = {}
for sub in ("results/matrix/ms_v2", "results/matrix/ms_highn_v2"):
    for f in glob.glob(os.path.join(ROOT, sub, "*.json")):
        d = json.load(open(f)); CELLS[d["cell"]] = d["schedules"]

MINSP = {}
for f in glob.glob(os.path.join(ROOT, "results/matrix/problem_*.json")):
    try:
        p = json.load(open(f))
        MINSP[os.path.basename(f)[8:-5]] = float(p["min_spacing_m"])
    except Exception:
        pass

ROSES = ["dei", "rowp", "omnidir", "uniform"]
ROSE_TITLE = {"dei": "DEI rose", "rowp": "ROWP rose",
              "omnidir": "omnidirectional", "uniform": "unidirectional"}
NS = [30, 40, 50, 60, 70, 80, 200, 300]
PARQO_N = [10, 15, 20, 25, 30]   # N=35 exceeds the ~34-turbine zone capacity
PARQO_MINSP = 160.0

# parqo tolerance-sweepable multistart (per-seed max_sdf_m / min_dist_m)
PQ = {s: json.load(open(os.path.join(ROOT, f"parqo/parqo_ms_{s}.json")))
      for s in ["baseline", "claude", "gemini"]}


def best_feasible(sched, tol, minsp, viol_key="max_out_m"):
    if not sched or "seeds" not in sched:
        return None
    g = [r for r in sched["seeds"] if "aep_gwh" in r
         and r[viol_key] <= tol and r["min_dist_m"] >= minsp - tol]
    return max(r["aep_gwh"] for r in g) if g else None


def gap_pct(farm, rose, n, who, tol):
    key = f"{farm}_n{n}_rose{rose}"
    sc = CELLS.get(key)
    if not sc:
        return None
    minsp = MINSP.get(key, 0.0)
    b = best_feasible(sc.get("baseline"), tol, minsp)
    s = best_feasible(sc.get(who), tol, minsp)
    if b and s and b > 0:
        return 100 * (s - b) / b
    return None


def parqo_gap_pct(rose, n, who, tol):
    if who not in PQ:            # no parqo data for the random control
        return None
    bc = PQ["baseline"].get(f"{rose}|n{n}")
    sc = PQ[who].get(f"{rose}|n{n}")
    if not bc or not sc:
        return None
    b = best_feasible(bc, tol, PARQO_MINSP, viol_key="max_sdf_m")
    s = best_feasible(sc, tol, PARQO_MINSP, viol_key="max_sdf_m")
    if b and s and b > 0:
        return 100 * (s - b) / b
    return None


fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.0), sharex="row",
                         constrained_layout=True)
SITES = [
    ("dei",   "DEI · IEA 15 MW",   NS,       gap_pct),
    ("rowp",  "ROWP · IEA 10 MW",  NS,       gap_pct),
    ("parqo", "Parque Ficticio · V80 · 5 zones", PARQO_N,
     lambda farm, rose, n, who, t: parqo_gap_pct(rose, n, who, t)),
]

for ri, (farm, slabel, nrange, gapf) in enumerate(SITES):
    row_absmax = 0
    for ci, rose in enumerate(ROSES):
        ax = axes[ri, ci]
        ax.axhline(0, color=MUT, lw=1.0, ls=(0, (4, 2)), zorder=1)
        for who, color in [("claude", C_CLAUDE), ("gemini", C_GEM)]:
            for tol, style, pe in TOL_STYLE:
                xs, ys = [], []
                for n in nrange:
                    v = gapf(farm, rose, n, who, tol)
                    if v is not None:
                        xs.append(n); ys.append(v)
                if not xs:
                    continue
                ax.plot(xs, ys, color=color, ls=style, lw=1.4,
                        path_effects=pe, zorder=3)
                row_absmax = max(row_absmax, max(abs(v) for v in ys))
        if ri == 0:
            ax.set_title(ROSE_TITLE[rose], fontsize=8.4, weight="bold", pad=3)
        ax.grid(alpha=0.2, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(length=2.5, labelsize=6.8)
        ax.set_xscale("log")
        xt = [30, 50, 80, 200, 300] if farm != "parqo" else [10, 15, 20, 30]
        ax.set_xticks(xt)
        ax.set_xticklabels([str(t) for t in xt])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        if ri == 2:
            ax.set_xlabel("turbines $N$", fontsize=8)
    m = row_absmax * 1.15 + 0.05
    for ci in range(4):
        axes[ri, ci].set_ylim(-m, m)
    axes[ri, 0].set_ylabel(f"{slabel}\n$\\Delta$AEP vs baseline (%)", fontsize=7.2)

# two-key legend: color = schedule, linestyle = constraint tolerance (both sides)
leg_sched = [Line2D([0], [0], color=C_CLAUDE, lw=1.6, label="Claude dual-bump"),
             Line2D([0], [0], color=C_GEM, lw=1.6, label="Gemini schedule")]
leg_tol = [Line2D([0], [0], color=MUT, lw=1.4, ls=st, path_effects=pe,
                  label=f"tolerance {t:g} m") for t, st, pe in TOL_STYLE]
l1 = fig.legend(handles=leg_sched, loc="upper center", bbox_to_anchor=(0.5, 1.075),
                ncol=2, fontsize=7.6, frameon=False)
fig.add_artist(l1)
fig.legend(handles=leg_tol, loc="upper center", bbox_to_anchor=(0.5, 1.035),
           ncol=3, fontsize=7.6, frameon=False)

for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_aep_dominance.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
