"""Benchmark-anatomy figure: the full experiment space in one graphic.

(a) The two boundary polygons to scale, filled with layouts OPTIMIZED by the
    deployed dual-bump schedule (iter_192) at the train/held-out turbine
    counts, plus packing extremes (N = 30 / N = 300) as postage stamps.
    Layouts from paper/figs/deployed_layouts.json (see gen_deployed_layouts.py).
(b) The four wind climates as compact roses (shared radial + colour scale).
(c) The turbine-count sweep converted to capacity density (MW km^-2) — the
    quantity that makes N physically comparable across the two farms — with
    the commercial offshore band shaded and train/held-out cells starred.

Output: paper/figs/fig_benchmark_anatomy.{pdf,png}
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "paper", "figs")
sys.path.insert(0, os.path.join(ROOT, "playground/pixwake/src"))
sys.path.insert(0, os.path.join(ROOT, "validation/stochastic_aep"))

# deployed-schedule (iter_192) optimized layouts, generated once by
# paper/gen_deployed_layouts.py via the schedule-only harness

# ---------------------------------------------------------------- data
def load(p):
    return json.load(open(os.path.join(ROOT, p)))

dei = load("playground/problem.json")
rowp = load("results/problem_rowp.json")
omni = load("results/matrix/problem_dei_n50_roseomnidir.json")["wind_rose"]
unif = load("results/matrix/problem_dei_n50_roseuniform.json")["wind_rose"]

# ParqueFicticio — five disconnected inclusion zones, Vestas V80 (D = 80 m)
parqo_prob = load("parqo/problem_parqo.json")
PARQO_ZONES = [np.asarray(z, float) for z in load("parqo/inclusion_zones.json")["zones"]]
PARQO_CTR = np.vstack(PARQO_ZONES).mean(axis=0)      # shared centring for zones+layout
PARQO_D = float(parqo_prob["rotor_diameter"])         # 80 m
PARQO_MINSP = float(parqo_prob["min_spacing_m"])      # 160 m = 2 D
PARQO_MW = 2.0                                         # V80 rated

def parqo_layout_km(path):
    lay = load(path)
    x = (np.asarray(lay["x"], float) - PARQO_CTR[0]) / 1000.0
    y = (np.asarray(lay["y"], float) - PARQO_CTR[1]) / 1000.0
    return x, y

def parqo_zones_km():
    return [(z - PARQO_CTR) / 1000.0 for z in PARQO_ZONES]

N_SWEEP = [30, 40, 50, 60, 70, 80, 200, 300]

def local_km(verts):
    a = np.asarray(verts, float)
    return (a - a.mean(axis=0)) / 1000.0

def shoelace_km2(a):
    x, y = a[:, 0], a[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(np.roll(x, 1), y))

DEPLOYED = json.load(open(os.path.join(FIGS, "deployed_layouts.json")))

def deployed_layout(problem, key):
    """iter_192-optimized layout for this cell in the centred local-km frame.

    The schedule-only harness optimises in the problem's own coordinates
    after the skeleton's internal centring — but its saved output is in raw
    problem coordinates, so apply the same centre shift as the boundary."""
    verts = np.asarray(problem["boundary_vertices"], float)
    cx, cy = verts.mean(axis=0)
    lay = DEPLOYED[key]
    x = (np.asarray(lay["x"], float) - cx) / 1000.0
    y = (np.asarray(lay["y"], float) - cy) / 1000.0
    bl = (verts - [cx, cy]) / 1000.0
    return x, y, bl

# ---------------------------------------------------------------- style
INK, MUT = "#333333", "#777777"
C_DEI, C_ROWP = "#2c5f8a", "#b5542d"
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#d3e4f0", "#8fbcd9", "#4b8cbf", "#22608f", "#0e3a5c"])

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})

fig = plt.figure(figsize=(7.2, 7.2))
gs = fig.add_gridspec(3, 12,
                      height_ratios=[1.55, 0.95, 0.46],
                      hspace=0.42, wspace=0.9,
                      left=0.065, right=0.975, top=0.945, bottom=0.06)

# ================================================= (a) farms w/ layouts
C_PARQO = "#4a7c59"
farm_defs = [
    (dei, C_DEI, 50, "Training — DEI", "IEA 15 MW · D = 240 m", 15.0, "dei_50"),
    (rowp, C_ROWP, 74, "Validation — ROWP", "IEA 10 MW · D = 198 m", 10.0, "rowp_74"),
]
ax_main = [fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 4:8])]

half = 0
polys = []
for prob, *_ in farm_defs:
    a = local_km(prob["boundary_vertices"])
    polys.append(a)
    half = max(half, np.ptp(a[:, 0]) / 2, np.ptp(a[:, 1]) / 2)
half *= 1.16

areas = []
for ax, (prob, color, n_show, title, turbine, mw, lkey), a in zip(ax_main, farm_defs, polys):
    area = shoelace_km2(a); areas.append(area)
    x, y, _ = deployed_layout(prob, lkey)
    ax.fill(a[:, 0], a[:, 1], facecolor=color, alpha=0.10,
            edgecolor=color, linewidth=1.7, joinstyle="round", zorder=2)
    r_km = float(prob["min_spacing_m"]) / 2000.0   # half min-spacing
    ax.add_collection(EllipseCollection(
        widths=2 * r_km, heights=2 * r_km, angles=0, units="xy",
        offsets=np.column_stack([x, y]), transOffset=ax.transData,
        facecolors=color, edgecolors="none", alpha=0.18, zorder=3))
    # redraw as light fill + solid edge: EllipseCollection applies alpha to
    # both, so overlay a second edge-only collection at full strength
    ax.add_collection(EllipseCollection(
        widths=2 * r_km, heights=2 * r_km, angles=0, units="xy",
        offsets=np.column_stack([x, y]), transOffset=ax.transData,
        facecolors="none", edgecolors=color, linewidths=0.7, zorder=4))
    ax.scatter(x, y, s=3.5, color=color, linewidth=0, zorder=4)
    cx, cy = a[:, 0].mean(), a[:, 1].mean()
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(f"{title}", fontsize=9.6, weight="bold", color=INK, pad=5)
    ax.text(0.03, 0.97,
            f"{turbine}\n$N$ = {n_show} shown · area {area:.0f} km$^2$"
            f"\ndisks: half min-spacing (2 D)",
            transform=ax.transAxes, va="top", fontsize=7.6, color=INK,
            linespacing=1.4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78,
                      boxstyle="round,pad=0.3"))
    ax.set_xlabel("east (km)", fontsize=7.8)
    if ax is ax_main[0]:
        ax.set_ylabel("north (km)", fontsize=7.8)
    ax.grid(alpha=0.16, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, labelsize=7)

# -------------------------------------- (a3) ParqueFicticio, own scale
ax_pq = fig.add_subplot(gs[0, 8:12])
zk = parqo_zones_km()
px, py = parqo_layout_km("parqo/layout_parqo.json")
pq_area = sum(shoelace_km2(z) for z in zk)
_zv = np.vstack(zk)
pq_half = max(np.ptp(_zv[:, 0]), np.ptp(_zv[:, 1])) / 2 * 1.16
r_pq = PARQO_MINSP / 2000.0
for z in zk:
    ax_pq.fill(z[:, 0], z[:, 1], facecolor=C_PARQO, alpha=0.10,
               edgecolor=C_PARQO, linewidth=1.4, joinstyle="round", zorder=2)
ax_pq.add_collection(EllipseCollection(
    widths=2 * r_pq, heights=2 * r_pq, angles=0, units="xy",
    offsets=np.column_stack([px, py]), transOffset=ax_pq.transData,
    facecolors=C_PARQO, edgecolors="none", alpha=0.18, zorder=3))
ax_pq.add_collection(EllipseCollection(
    widths=2 * r_pq, heights=2 * r_pq, angles=0, units="xy",
    offsets=np.column_stack([px, py]), transOffset=ax_pq.transData,
    facecolors="none", edgecolors=C_PARQO, linewidths=0.7, zorder=4))
ax_pq.scatter(px, py, s=3.5, color=C_PARQO, linewidth=0, zorder=4)
zc = np.vstack(zk).mean(axis=0)
ax_pq.set_xlim(zc[0] - pq_half, zc[0] + pq_half)
ax_pq.set_ylim(zc[1] - pq_half, zc[1] + pq_half)
ax_pq.set_aspect("equal")
ax_pq.set_title("Test — Parque Ficticio", fontsize=9.6, weight="bold",
                color=INK, pad=5)
ax_pq.text(0.03, 0.97,
           f"V80 · D = {PARQO_D:.0f} m\n$N$ = {len(px)} shown · 5 zones "
           f"{pq_area:.1f} km$^2$\ndisks: 2 D · own scale",
           transform=ax_pq.transAxes, va="top", fontsize=7.6, color=INK,
           linespacing=1.4, bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.78, boxstyle="round,pad=0.3"))
ax_pq.set_xlabel("east (km)", fontsize=7.8)
ax_pq.grid(alpha=0.16, linewidth=0.5); ax_pq.set_axisbelow(True)
ax_pq.tick_params(length=2.5, labelsize=7)

# ================================================= (b) roses
rose_specs = [
    ("DEI rose", dei["wind_rose"]),
    ("ROWP rose", rowp["wind_rose"]),
    ("Omnidirectional", omni),
    ("Unidirectional", unif),
]
all_speeds = np.concatenate([np.asarray(r["speeds_ms"], float)
                             for _, r in rose_specs])
norm = Normalize(all_speeds.min(), all_speeds.max())
rmax = max(max(r["weights"]) for _, r in rose_specs) * 1.12

rose_axes = []
for i, (title, rose) in enumerate(rose_specs):
    ax = fig.add_subplot(gs[1, i * 3:(i + 1) * 3], projection="polar")
    rose_axes.append(ax)
    wd = np.deg2rad(np.asarray(rose["directions_deg"], float))
    ws = np.asarray(rose["speeds_ms"], float)
    w = np.asarray(rose["weights"], float)
    if len(set(rose["directions_deg"])) == 1:
        mean_ws = float(np.sum(ws * w) / np.sum(w))
        ax.bar(wd[0], rmax * 0.88, width=np.deg2rad(9),
               color=SPEED_CMAP(norm(mean_ws)), edgecolor="white",
               linewidth=0.6, zorder=3)
    else:
        width = np.deg2rad(360 / len(wd) * 0.92)
        ax.bar(wd, w, width=width, color=SPEED_CMAP(norm(ws)),
               edgecolor="white", linewidth=0.45, zorder=3)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_ylim(0, rmax)
    ax.set_yticks([0.1]); ax.set_yticklabels(["10%"], fontsize=5.4, color=MUT)
    ax.set_rlabel_position(200)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "", "S", ""], fontsize=6.6, color=MUT)
    ax.tick_params(pad=-3)
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.spines["polar"].set_alpha(0.25)
    ax.set_title(title, fontsize=8.2, weight="bold", color=INK, pad=10)

# slim colorbar beneath the rose row (auto-placed, steals rose space)
sm = ScalarMappable(norm=norm, cmap=SPEED_CMAP)
cb = fig.colorbar(sm, ax=rose_axes, orientation="horizontal",
                  fraction=0.06, pad=0.16, aspect=48, shrink=0.62)
cb.set_label("sector mean wind speed (m s$^{-1}$)", fontsize=7, color=INK,
             labelpad=2)
cb.ax.tick_params(labelsize=6, colors=MUT, length=2)
cb.outline.set_edgecolor(MUT); cb.outline.set_linewidth(0.4)

# ================================================= (c) capacity density
axd = fig.add_subplot(gs[2, :])
axd.set_xscale("log")
COMM_LO, COMM_HI = 3.0, 10.0
axd.axvspan(COMM_LO, COMM_HI, color="#e9e2d4", alpha=0.85, zorder=1)
axd.text((COMM_LO * COMM_HI) ** 0.5, 1.72, "commercial\noffshore",
         fontsize=6.6, color="#8a7d5e", ha="center", va="top", linespacing=1.0)

# three lanes. DEI/ROWP (IEA 15/10 MW) star their train/held-out flagship
# cell; Parqo (V80, synthetic zones) has no train/test role, so no star — its
# density sits in a different regime because the turbine is far smaller.
lane_defs = [
    (C_DEI, areas[0], 15.0, N_SWEEP, 50, "DEI", 1.45),
    (C_ROWP, areas[1], 10.0, N_SWEEP, 74, "ROWP", 0.90),
    (C_PARQO, pq_area, PARQO_MW, [10, 15, 20, 25, 30], None, "Parque\nFicticio\n(V80)", 0.35),
]
for color, area, mw, nsweep, n_flag, label, y in lane_defs:
    dens = [n * mw / area for n in nsweep]
    axd.plot(dens, [y] * len(dens), color=color, alpha=0.35, lw=1.2, zorder=2)
    axd.scatter(dens, [y] * len(dens), s=20, color=color, zorder=3,
                edgecolor="white", linewidth=0.5)
    for n, dv in zip(nsweep, dens):
        axd.annotate(str(n), (dv, y), textcoords="offset points",
                     xytext=(0, 6), fontsize=5.8, color=MUT, ha="center")
    if n_flag is not None:                        # star train/held-out cell
        dstar = n_flag * mw / area
        axd.scatter([dstar], [y], marker="*", s=140, color=color,
                    edgecolor="black", linewidth=0.7, zorder=5)
        axd.annotate(f"$N$={n_flag}", (dstar, y), textcoords="offset points",
                     xytext=(0, -14), fontsize=6.6, color=color, ha="center",
                     weight="bold")
    axd.text(-0.012, y, label, transform=axd.get_yaxis_transform(),
             fontsize=7.4, color=color, ha="right", va="center", weight="bold",
             linespacing=0.95)

axd.set_xlim(1, 130)
axd.set_ylim(0, 1.85)
axd.set_yticks([])
axd.set_xlabel("capacity density (MW km$^{-2}$, log scale)", fontsize=8)
axd.tick_params(labelsize=7, length=2.5)
from matplotlib.ticker import LogFormatter, FixedLocator
axd.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
axd.set_xticklabels([1, 2, 5, 10, 20, 50, 100])
for s in ["left", "right", "top"]:
    axd.spines[s].set_visible(False)
axd.text(128, 1.80, "DEI/ROWP $N$=30–300 · Parque Ficticio $N$=10–30 (zone cap $\\approx$34)",
         fontsize=6.8, color=MUT, ha="right", va="top")

for ext in ("pdf", "png"):
    out = os.path.join(FIGS, f"fig_benchmark_anatomy.{ext}")
    fig.savefig(out, dpi=200)
    print("wrote", out)
