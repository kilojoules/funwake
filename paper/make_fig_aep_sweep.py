"""fig_aep_sweep — best-feasible AEP per turbine vs N. The discovered schedules
(at STRICT tolerance) are solid colour; the baseline is drawn as its own gray
tolerance-FAMILY (strict / 1 m / 5 m), NOT a constant dashed line, so the reader
sees the baseline climb as violation is forgiven. Where the baseline has no
strictly-feasible layout in 50 starts its strict curve is absent and the gap is
marked (a headline result, not a hole).

AEP/N removes the N-growth so the schedule-vs-baseline gap and the baseline's
tolerance-climb are both visible. One honest tolerance for the schedules
(strict, 0.1 m); the baseline's forgiveness is shown by its gray family.

Data: results/matrix/ms_v2 + ms_highn_v2, parqo/parqo_ms_*. Rows=farms, cols=roses.
Output: paper/figs/fig_aep_sweep.{pdf,png}
"""
import glob, json, os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"
C_CLAUDE, C_GEM = "#c0392b", "#2c6fbb"
G_STRICT, G_1, G_5 = "#3a3a3a", "#8a8a8a", "#c2c2c2"
plt.rcParams.update({"font.size": 12.8, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUT, "ytick.color": MUT, "font.family": "sans-serif"})

CELLS = {}
for sub in ("results/matrix/ms_v2", "results/matrix/ms_highn_v2"):
    for f in glob.glob(os.path.join(ROOT, sub, "*.json")):
        d = json.load(open(f)); CELLS[d["cell"]] = d["schedules"]
PQ = {s: json.load(open(os.path.join(ROOT, f"parqo/parqo_ms_{s}.json"))) for s in ["baseline", "claude", "gemini"]}

def best_feas(seeds, tol, minsp, vkey):
    g = [r for r in seeds if "aep_gwh" in r and r[vkey] <= tol and r["min_dist_m"] >= minsp - tol]
    return max(r["aep_gwh"] for r in g) if g else None

MINSP = {"dei": 960.0, "rowp": 792.0, "parqo": 160.0}
def get_seeds(farm, n, rose, who):
    if farm == "parqo":
        return PQ[who].get(f"{rose}|n{n}", {}).get("seeds", []), "max_sdf_m"
    return CELLS.get(f"{farm}_n{n}_rose{rose}", {}).get(who, {}).get("seeds", []), "max_out_m"

SITES = [("dei", "DEI · IEA 15MW", [30, 40, 50, 60, 70, 80]),
         ("rowp", "ROWP · IEA 10MW", [30, 40, 50, 60, 70, 80]),
         ("parqo", "ParqueFicticio · V80", [10, 15, 20, 25, 30, 35])]
ROSES = ["dei", "rowp", "omnidir", "uniform"]
RLAB = {"dei": "DEI rose", "rowp": "ROWP rose", "omnidir": "omnidirectional", "uniform": "unidirectional"}

fig, axes = plt.subplots(3, 4, figsize=(11.5, 7.2), constrained_layout=True)
for ri, (farm, slabel, nrange) in enumerate(SITES):
    minsp = MINSP[farm]
    for ci, rose in enumerate(ROSES):
        ax = axes[ri, ci]
        S = {k: ([], []) for k in ("claude", "gemini", "b_strict", "b_1", "b_5")}
        gaps = []
        for n in nrange:
            def bf(who, tol):
                seeds, vk = get_seeds(farm, n, rose, who)
                v = best_feas(seeds, tol, minsp, vk)
                return v / n if v is not None else None
            vals = {"claude": bf("claude", 0.1), "gemini": bf("gemini", 0.1),
                    "b_strict": bf("baseline", 0.1), "b_1": bf("baseline", 1.0), "b_5": bf("baseline", 5.0)}
            for k, v in vals.items():
                if v is not None: S[k][0].append(n); S[k][1].append(v)
            if vals["b_strict"] is None and (vals["b_1"] is not None or vals["b_5"] is not None):
                gaps.append(n)
        ax.plot(*S["b_5"], "-", color=G_5, lw=1.5, zorder=2)
        ax.plot(*S["b_1"], "-", color=G_1, lw=1.5, zorder=2)
        ax.plot(*S["b_strict"], "-", color=G_STRICT, lw=1.7, marker="o", ms=3, zorder=3)
        ax.plot(*S["claude"], "-", color=C_CLAUDE, lw=1.9, marker="o", ms=3, zorder=5)
        ax.plot(*S["gemini"], "-", color=C_GEM, lw=1.9, marker="o", ms=3, zorder=5)
        y0 = ax.get_ylim()[0]
        for n in gaps:
            ax.axvspan(n - 2.5, n + 2.5, color=G_STRICT, alpha=0.035, zorder=0)
        if gaps:
            ax.plot(gaps, [y0] * len(gaps), "x", color=G_STRICT, ms=5, zorder=6, clip_on=False)
        if ri == 0: ax.set_title(RLAB[rose], fontsize=12.9, weight="bold")
        if ri == 2: ax.set_xlabel("turbines $N$", fontsize=12.0)
        ax.grid(alpha=0.16, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2.5, labelsize=10.5)
    axes[ri, 0].set_ylabel(slabel, fontsize=11.5, weight="bold")
fig.supylabel("best-feasible AEP / turbine (GWh)", fontsize=12.0)

leg = [Line2D([], [], color=C_CLAUDE, lw=1.9, marker="o", ms=3, label="Claude (strict)"),
       Line2D([], [], color=C_GEM, lw=1.9, marker="o", ms=3, label="Gemini (strict)"),
       Line2D([], [], color=G_STRICT, lw=1.7, marker="o", ms=3, label="baseline (strict)"),
       Line2D([], [], color=G_1, lw=1.5, label="baseline (1 m tol)"),
       Line2D([], [], color=G_5, lw=1.5, label="baseline (5 m tol)"),
       Line2D([], [], color=G_STRICT, lw=0, marker="x", ms=5, label="baseline 0/50 strictly feasible")]
fig.legend(handles=leg, loc="upper center", ncol=6, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 1.05))
for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_aep_sweep.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("wrote", out)
