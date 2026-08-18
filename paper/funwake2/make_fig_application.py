"""FunWake-2 application overview: the three farm geometries and four wind resources.

Top row:  DEI (training) + ROWP (held-out) on a shared km scale, and Parque (a
          training geometry, multi-zone) on its own scale.
Bottom:   four polar roses (DEI, ROWP, omnidirectional, unidirectional).

funwake-2 roles differ from FunWake-1: DEI *and* Parque are training geometries
(a matrix of turbine counts x wind roses); ROWP is the held-out selection farm;
the pre-registered test set is specific cells (high-N ROWP, Parque real wind,
ROWP unidirectional).

Output: paper/funwake2/fig_application.{pdf,png}
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
def load(p): return json.load(open(os.path.join(ROOT, p)))

dei = load("playground/problem.json")
rowp = load("results/problem_rowp.json")
parq = load("parqo/problem_parqo.json")
omni = load("results/matrix/problem_dei_n50_roseomnidir.json")["wind_rose"]
unif = load("results/matrix/problem_dei_n50_roseuniform.json")["wind_rose"]

def local_km(verts):
    a = np.asarray(verts, float); return (a - a.mean(axis=0)) / 1000.0
def shoelace_km2(a):
    x, y = a[:, 0], a[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(np.roll(x, 1), y))

INK, MUT = "#333333", "#777777"
C_DEI, C_ROWP, C_PARQ = "#2c5f8a", "#b5542d", "#3f7d4e"
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#d3e4f0", "#8fbcd9", "#4b8cbf", "#22608f", "#0e3a5c"])
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
                     "font.family": "sans-serif"})

fig = plt.figure(figsize=(9.2, 6.6))
gs = fig.add_gridspec(2, 12, height_ratios=[1.3, 1.0], hspace=0.42, wspace=1.4,
                      left=0.06, right=0.97, top=0.90, bottom=0.07)

ax_dei = fig.add_subplot(gs[0, 0:4])
ax_rowp = fig.add_subplot(gs[0, 4:8])
ax_parq = fig.add_subplot(gs[0, 8:12])

# --- DEI + ROWP share a scale (honest relative size) ---
dei_a, rowp_a = local_km(dei["boundary_vertices"]), local_km(rowp["boundary_vertices"])
half = max(max(np.ptp(dei_a[:, 0]), np.ptp(dei_a[:, 1])),
           max(np.ptp(rowp_a[:, 0]), np.ptp(rowp_a[:, 1]))) / 2 * 1.18
for ax, a, prob, color, title, role, turb in [
    (ax_dei, dei_a, dei, C_DEI, "DEI", "training", "IEA 15 MW · $D$=240 m · 4$D$ spacing"),
    (ax_rowp, rowp_a, rowp, C_ROWP, "ROWP", "held-out (out-of-sample check)", "IEA 10 MW · $D$=198 m · 4$D$ spacing")]:
    ax.fill(a[:, 0], a[:, 1], facecolor=color, alpha=0.14, edgecolor=color, lw=1.8, zorder=2)
    cx, cy = a[:, 0].mean(), a[:, 1].mean()
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_aspect("equal")
    ax.set_title(f"{title}\n\\textit{{{role}}}" if False else title, fontsize=10, color=INK, pad=12, weight="bold")
    ax.text(0.5, 1.02, role, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, style="italic", color=color)
    ax.text(0.03, 0.97, f"{turb}\n$N$={prob['n_target']} · area {shoelace_km2(a):.0f} km$^2$",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.3, color=INK, linespacing=1.4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, boxstyle="round,pad=0.3"))
    ax.set_xlabel("east (km)", fontsize=8); ax.set_ylabel("north (km)", fontsize=8)
    ax.grid(alpha=0.18, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2.5, labelsize=7.2)

# --- Parque on its OWN (much smaller) scale, multi-zone ---
zones = [np.asarray(z, float) for z in parq["inclusion_zones"]]
allv = np.concatenate(zones); ctr = allv.mean(axis=0)
zones_km = [(z - ctr) / 1000.0 for z in zones]
zarea = sum(shoelace_km2(z) for z in zones_km)
ph = max(np.ptp(np.concatenate(zones_km)[:, 0]), np.ptp(np.concatenate(zones_km)[:, 1])) / 2 * 1.15
for z in zones_km:
    ax_parq.fill(z[:, 0], z[:, 1], facecolor=C_PARQ, alpha=0.16, edgecolor=C_PARQ, lw=1.5, zorder=2)
ax_parq.set_xlim(-ph, ph); ax_parq.set_ylim(-ph, ph); ax_parq.set_aspect("equal")
ax_parq.set_title("Parque Ficticio", fontsize=10, color=INK, pad=12, weight="bold")
ax_parq.text(0.5, 1.02, "training geometry (multi-zone)", transform=ax_parq.transAxes,
             ha="center", va="bottom", fontsize=8, style="italic", color=C_PARQ)
ax_parq.text(0.03, 0.97, f"Vestas V80 · $D$=80 m · 2$D$ spacing\n$N$={parq['n_target']} · "
             f"{len(zones)} zones · {zarea:.2f} km$^2$\n(own scale)", transform=ax_parq.transAxes,
             va="top", ha="left", fontsize=7.3, color=INK, linespacing=1.4,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, boxstyle="round,pad=0.3"))
ax_parq.set_xlabel("east (km)", fontsize=8); ax_parq.set_ylabel("north (km)", fontsize=8)
ax_parq.grid(alpha=0.18, lw=0.5); ax_parq.set_axisbelow(True); ax_parq.tick_params(length=2.5, labelsize=7.2)

# ============================================================ roses
rose_specs = [("DEI rose", dei["wind_rose"]), ("ROWP rose", rowp["wind_rose"]),
              ("Omnidirectional", omni), ("Unidirectional", unif)]
all_speeds = np.concatenate([np.asarray(r["speeds_ms"], float) for _, r in rose_specs])
norm = Normalize(all_speeds.min(), all_speeds.max())
rmax = max(max(r["weights"]) for _, r in rose_specs) * 1.12
axes_r = [fig.add_subplot(gs[1, 3*i:3*i+3], projection="polar") for i in range(4)]
for ax, (title, rose) in zip(axes_r, rose_specs):
    wd = np.deg2rad(np.asarray(rose["directions_deg"], float))
    ws = np.asarray(rose["speeds_ms"], float); w = np.asarray(rose["weights"], float)
    if len(set(rose["directions_deg"])) == 1:
        mean_ws = float(np.sum(ws * w) / np.sum(w))
        ax.bar(wd[0], rmax * 0.88, width=np.deg2rad(9), bottom=0,
               color=SPEED_CMAP(norm(mean_ws)), edgecolor="white", lw=0.6, zorder=3)
        ax.annotate("100%\nfrom 0°", (np.deg2rad(38), rmax * 0.62), fontsize=7, color=INK, ha="left")
    else:
        ax.bar(wd, w, width=np.deg2rad(360 / len(wd) * 0.92), bottom=0,
               color=SPEED_CMAP(norm(ws)), edgecolor="white", lw=0.5, zorder=3)
        k = int(np.argmax(w)); pk = float(rose["directions_deg"][k])
        ax.annotate(f"{pk:.0f}° · {100*w[k]:.0f}%", (np.deg2rad(pk + 32), rmax * 1.38),
                    fontsize=6.8, color=INK, ha="center", va="center", annotation_clip=False)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_ylim(0, rmax)
    ax.set_yticks([0.05, 0.10, 0.15]); ax.set_yticklabels(["5%", "10%", "15%"], fontsize=5.8, color=MUT)
    ax.set_rlabel_position(200); ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=7, color=MUT); ax.tick_params(pad=-2)
    ax.grid(alpha=0.3, lw=0.45); ax.spines["polar"].set_alpha(0.25)
    ax.set_title(title, fontsize=8.8, color=INK, pad=13, weight="bold")

sm = ScalarMappable(norm=norm, cmap=SPEED_CMAP)
cb = fig.colorbar(sm, ax=axes_r, orientation="horizontal", fraction=0.05, pad=0.13, aspect=42, shrink=0.7)
cb.set_label("sector mean wind speed (m s$^{-1}$)", fontsize=8, color=INK)
cb.ax.tick_params(labelsize=7, colors=MUT); cb.outline.set_edgecolor(MUT); cb.outline.set_linewidth(0.5)

fig.suptitle("Training geometries: DEI + Parque  ·  Held-out check (out-of-sample): ROWP  ·  "
             "Pre-registered test: high-$N$ ROWP, Parque real wind, ROWP unidirectional",
             fontsize=9, y=0.975, color=INK)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_application.{ext}"), dpi=200)
    print("wrote", os.path.join(OUT, f"fig_application.{ext}"))
