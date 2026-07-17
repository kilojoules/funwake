"""Application-section figure: the two farm boundaries (to scale) and the four
wind roses used across the train/held-out pair and the generalization matrix.

Top row:  DEI + ROWP polygons on a shared km scale, with farm parameters.
Bottom:   four polar roses (DEI, ROWP, omnidirectional, unidirectional) on a
          shared radial scale; bar length = sector energy weight, bar color =
          sector mean wind speed (single-hue sequential ramp).

Output: paper/figs/fig_farms_roses.{pdf,png}
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "paper", "figs")

# ---------------------------------------------------------------- data
def load(p):
    return json.load(open(os.path.join(ROOT, p)))

dei = load("playground/problem.json")
rowp = load("results/problem_rowp.json")
omni = load("results/matrix/problem_dei_n50_roseomnidir.json")["wind_rose"]
unif = load("results/matrix/problem_dei_n50_roseuniform.json")["wind_rose"]

# ROWP boundary ships in UTM — translate to centred local km like the harness.
def local_km(verts):
    a = np.asarray(verts, float)
    a = a - a.mean(axis=0)
    return a / 1000.0

def shoelace_km2(a):
    x, y = a[:, 0], a[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(np.roll(x, 1), y))

# ---------------------------------------------------------------- style
INK = "#333333"
MUT = "#777777"
C_DEI = "#2c5f8a"    # deep blue  (training farm)
C_ROWP = "#b5542d"   # warm rust  (held-out farm)
# single-hue sequential ramp for wind speed (light -> dark blue-teal)
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#d3e4f0", "#8fbcd9", "#4b8cbf", "#22608f", "#0e3a5c"])

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0],
                      hspace=0.32, wspace=0.38,
                      left=0.07, right=0.97, top=0.95, bottom=0.06)

# ============================================================ polygons
ax_dei = fig.add_subplot(gs[0, 0:2])
ax_rowp = fig.add_subplot(gs[0, 2:4])

farm_specs = [
    (ax_dei, dei, C_DEI, "Training — DEI (dk1d tender 9)",
     "IEA 15 MW · D = 240 m"),
    (ax_rowp, rowp, C_ROWP, "Held-out — IEA 740-10 ROWP",
     "IEA 10 MW · D = 198 m"),
]

# shared span so relative size is honest
spans = []
polys = []
for _, prob, *_ in farm_specs:
    a = local_km(prob["boundary_vertices"])
    polys.append(a)
    spans.append(max(np.ptp(a[:, 0]), np.ptp(a[:, 1])))
half = max(spans) / 2 * 1.18

for (ax, prob, color, title, turbine), a in zip(farm_specs, polys):
    area = shoelace_km2(a)
    n = prob["n_target"]
    D = prob["rotor_diameter"]
    sp = prob["min_spacing_m"]
    ax.fill(a[:, 0], a[:, 1], facecolor=color, alpha=0.13, edgecolor=color,
            linewidth=1.8, joinstyle="round", zorder=2)
    cx, cy = a[:, 0].mean(), a[:, 1].mean()
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5, color=INK, pad=6, weight="bold")
    ax.text(0.03, 0.97,
            f"{turbine}\n$N$ = {n} turbines\nmin spacing {sp/1000:.2f} km (4 D)"
            f"\narea {area:.0f} km$^2$",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.8,
            color=INK, linespacing=1.45,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                      boxstyle="round,pad=0.35"))
    # 4D spacing disk, to scale, anchored lower-left inside the panel
    disk_r = sp / 2000.0  # radius km  (diameter = min spacing)
    dx = cx - half * 0.74
    dy = cy - half * 0.80
    ax.add_patch(plt.Circle((dx, dy), disk_r, facecolor="none",
                            edgecolor=INK, linewidth=0.9, linestyle=(0, (3, 2))))
    ax.plot([dx], [dy], marker=".", ms=3, color=INK)
    ax.annotate("4 D exclusion", (dx + disk_r * 1.25, dy), fontsize=7,
                color=MUT, va="center")
    ax.set_xlabel("east (km)", fontsize=8)
    ax.set_ylabel("north (km)", fontsize=8)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, labelsize=7.5)

# ============================================================ roses
rose_specs = [
    ("DEI rose", dei["wind_rose"], "training"),
    ("ROWP rose", rowp["wind_rose"], "held-out"),
    ("Omnidirectional", omni, "matrix"),
    ("Unidirectional", unif, "matrix"),
]

# shared color scale over all sector speeds
all_speeds = np.concatenate([np.asarray(r["speeds_ms"], float)
                             for _, r, _ in rose_specs])
norm = Normalize(all_speeds.min(), all_speeds.max())

# shared radial scale over all sector weights
rmax = max(max(r["weights"]) for _, r, _ in rose_specs) * 1.12

axes_r = [fig.add_subplot(gs[1, i], projection="polar") for i in range(4)]
for ax, (title, rose, tag) in zip(axes_r, rose_specs):
    wd = np.deg2rad(np.asarray(rose["directions_deg"], float))
    ws = np.asarray(rose["speeds_ms"], float)
    w = np.asarray(rose["weights"], float)

    if len(set(rose["directions_deg"])) == 1:
        # unidirectional: all energy from one bearing — draw a single wedge
        # whose colour is the weight-averaged speed
        mean_ws = float(np.sum(ws * w) / np.sum(w))
        ax.bar(wd[0], rmax * 0.88, width=np.deg2rad(9), bottom=0,
               color=SPEED_CMAP(norm(mean_ws)), edgecolor="white",
               linewidth=0.6, zorder=3)
        ax.annotate("100 %\nfrom 0°", (np.deg2rad(38), rmax * 0.62),
                    fontsize=7, color=INK, ha="left", linespacing=1.3)
    else:
        width = np.deg2rad(360 / len(wd) * 0.92)
        ax.bar(wd, w, width=width, bottom=0,
               color=SPEED_CMAP(norm(ws)), edgecolor="white",
               linewidth=0.5, zorder=3)
        # direct-label the peak sector, placed OUTSIDE the rose near its
        # bearing — offset to whichever side lands farther from a compass
        # letter (N/E/S/W sit at the cardinals)
        k = int(np.argmax(w))
        peak_deg = float(rose["directions_deg"][k])

        def dist_to_cardinal(deg):
            deg = deg % 360
            return min(abs(deg - c) % 360 if abs(deg - c) % 360 <= 180
                       else 360 - abs(deg - c) % 360
                       for c in (0, 90, 180, 270))

        cand = [peak_deg + 32, peak_deg - 32]
        lab_deg = max(cand, key=dist_to_cardinal)
        ax.annotate(f"{peak_deg:.0f}°  ·  {100*w[k]:.0f} %",
                    (np.deg2rad(lab_deg), rmax * 1.38), fontsize=6.8,
                    color=INK, ha="center", va="center",
                    annotation_clip=False)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, rmax)
    ax.set_yticks([0.05, 0.10, 0.15])
    ax.set_yticklabels(["5%", "10%", "15%"], fontsize=5.8, color=MUT)
    ax.set_rlabel_position(200)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=7, color=MUT)
    ax.tick_params(pad=-2)
    ax.grid(alpha=0.3, linewidth=0.45)
    ax.spines["polar"].set_alpha(0.25)
    ax.set_title(title, fontsize=8.8, color=INK, pad=14, weight="bold")

# speed colorbar under the roses
sm = ScalarMappable(norm=norm, cmap=SPEED_CMAP)
cb = fig.colorbar(sm, ax=axes_r, orientation="horizontal",
                  fraction=0.055, pad=0.12, aspect=42, shrink=0.72)
cb.set_label("sector mean wind speed (m s$^{-1}$)", fontsize=8, color=INK)
cb.ax.tick_params(labelsize=7, colors=MUT)
cb.outline.set_edgecolor(MUT)
cb.outline.set_linewidth(0.5)

for ext in ("pdf", "png"):
    out = os.path.join(FIGS, f"fig_farms_roses.{ext}")
    fig.savefig(out, dpi=200)
    print("wrote", out)
