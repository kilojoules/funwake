"""fig_feasibility — constraint-precision trend on the validation farm (ROWP).

Fraction of the K=50 multistart restarts that are feasible vs constraint
tolerance (tol=0 = strictly inside the boundary), for three wind climates.
The discovered schedules are feasible down to strict / near-strict tolerance
while the baseline seed schedule needs ~1 m of slack — the constraint-precision
result that drives the AEP gap in fig_aep_dominance. Each panel carries a small
inset of a feasible optimized ROWP layout for that rose (boundary + turbines +
minimum-spacing disks).

Largest complete turbine count is used (upgrades to N=200/300 when the high-N
run merges). Data: results/matrix/{ms_v2,ms_highn_v2}/*.json;
layouts paper/rowp_rose_layouts.json.

Output: paper/figs/fig_feasibility.{pdf,png}
"""
import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
import numpy as np

WSCMAP = LinearSegmentedColormap.from_list(
    "wsp", ["#2c1a4a", "#3b5a8f", "#4b9bbf", "#8fcfa8", "#e8ecc0"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"
COL = {"baseline": "#555555", "claude": "#c0392b", "gemini": "#2c6fbb"}
LABEL = {"baseline": "baseline (seed schedule)", "claude": "Claude dual-bump",
         "gemini": "Gemini schedule"}
C_ROWP = "#b5542d"
TOLS = np.array([0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})

MS = {}
for sub in ("results/matrix/ms_v2", "results/matrix/ms_highn_v2"):
    for f in glob.glob(os.path.join(ROOT, sub, "*.json")):
        d = json.load(open(f)); MS[d["cell"]] = d["schedules"]
MINSP = {}
for f in glob.glob(os.path.join(ROOT, "results/matrix/problem_*.json")):
    try:
        MINSP[os.path.basename(f)[8:-5]] = float(json.load(open(f))["min_spacing_m"])
    except Exception:
        pass
FLOWFIELDS = {}
_fp = os.path.join(ROOT, "paper/rowp_flowfields.json")
if os.path.exists(_fp):
    FLOWFIELDS = json.load(open(_fp))


def feas_frac(seeds, tol, minsp, vk="max_out_m"):
    tot = [r for r in seeds if "aep_gwh" in r]
    if not tot:
        return np.nan
    return sum(1 for r in tot if r[vk] <= tol and r["min_dist_m"] >= minsp - tol) / len(tot)


def complete(key):
    return key in MS and all(MS[key].get(s, {}).get("seeds") for s in
                             ("baseline", "claude", "gemini"))


FARM = "rowp"
PANELS = [("Unidirectional", "uniform"), ("ROWP rose", "rowp"),
          ("Omnidirectional", "omnidir")]
ALLN = sorted({int(k.split("_n")[1].split("_")[0]) for k in MS if k.startswith(FARM + "_")})
NMAX = max([n for n in ALLN if all(complete(f"{FARM}_n{n}_rose{r}") for _, r in PANELS)],
           default=80)

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True,
                         constrained_layout=True)

for ax, (title, rose) in zip(axes, PANELS):
    key = f"{FARM}_n{NMAX}_rose{rose}"
    for sched in ["baseline", "claude", "gemini"]:
        sc = MS.get(key, {}).get(sched)
        if not sc or "seeds" not in sc:
            continue
        ms = MINSP.get(key, 0.0)
        y = [feas_frac(sc["seeds"], t, ms, "max_out_m") * 100 for t in TOLS]
        ax.plot(TOLS, y, color=COL[sched], lw=1.8, zorder=3,
                marker="o", ms=3.0, markevery=[0])
    ax.set_xscale("symlog", linthresh=0.001, linscale=0.4)
    ax.axvline(0.0007, color=MUT, lw=0.6, ls=":", alpha=0.7, zorder=1)
    ax.set_title(f"{title}  ($N{{=}}{NMAX}$)", fontsize=8.6, weight="bold",
                 color=INK, pad=4)
    ax.set_xlabel("constraint tolerance (m)", fontsize=7.8)
    ax.set_ylim(-4, 104); ax.set_xlim(-0.0005, 7)
    ax.set_xticks([0, 0.01, 0.1, 1]); ax.set_xticklabels(["0\n(strict)", "0.01", "0.1", "1"])
    ax.grid(alpha=0.18, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(length=2.5, labelsize=6.8)

    # ---- inset: wake flow field inside the farm boundary ----
    # transparent (clipped to boundary), frameless, tucked top-left
    ff = FLOWFIELDS.get(rose)
    if ff:
        iax = ax.inset_axes([0.02, 0.49, 0.47, 0.47], zorder=8)
        iax.patch.set_alpha(0.0)
        S = np.asarray(ff["S"]); ext = ff["extent"]
        bk = np.asarray(ff["boundary"], float)
        im = iax.imshow(S, extent=ext, origin="lower", cmap=WSCMAP,
                        vmin=4, vmax=10.2, interpolation="bilinear", zorder=2)
        clip = PathPatch(MPath(bk), transform=iax.transData)
        im.set_clip_path(clip)
        iax.plot(np.append(bk[:, 0], bk[0, 0]), np.append(bk[:, 1], bk[0, 1]),
                 color=INK, lw=0.7, zorder=3)
        iax.scatter(ff["x"], ff["y"], s=0.8, color="#c0392b", linewidth=0,
                    zorder=4)
        iax.set_xlim(ext[0], ext[1]); iax.set_ylim(ext[2], ext[3])
        iax.set_aspect("equal"); iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_visible(False)

axes[0].set_ylabel("feasible restarts (%)", fontsize=8.3)

from matplotlib.lines import Line2D
legend = [Line2D([0], [0], color=COL[s], lw=1.9, label=LABEL[s])
          for s in ["baseline", "claude", "gemini"]]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.11),
           ncol=3, fontsize=8, frameon=False)

for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_feasibility.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
