"""fig_flowfield — wake flow fields for the optimized ROWP layouts.

One panel per wind climate: the streamwise wind-speed field (Bastankhah k=0.04)
for the deployed schedule's N=80 ROWP layout, evaluated at each rose's
dominant direction. Shows that the optimizer routes every turbine's wake into
open space — the unidirectional case is (near) wake-free, which is why it is
the zero-gap column in the AEP comparison.

Layouts: paper/rowp_rose_layouts.json. Output: paper/figs/fig_flowfield.{pdf,png}
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "playground/pixwake/src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "playground"))
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from harness import build_sim  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"
LAYOUTS = json.load(open(os.path.join(ROOT, "paper/rowp_rose_layouts.json")))
WSCMAP = LinearSegmentedColormap.from_list(
    "wsp", ["#2c1a4a", "#3b5a8f", "#4b9bbf", "#8fcfa8", "#e8ecc0"])

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})

PANELS = [("Unidirectional", "uniform"), ("ROWP rose", "rowp"),
          ("Omnidirectional", "omnidir")]
WS0 = 10.0
GRID = 240

fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), constrained_layout=True,
                         gridspec_kw={"wspace": 0.05})
pcm = None
for ax, (title, rose) in zip(axes, PANELS):
    prob = json.load(open(os.path.join(ROOT, f"results/matrix/problem_rowp_n80_rose{rose}.json")))
    sim = build_sim(prob)
    wr = prob["wind_rose"]
    wd_arr = np.asarray(wr["directions_deg"]); w = np.asarray(wr["weights"])
    wd_dom = float(wd_arr[np.argmax(w)])
    lay = LAYOUTS[rose]["layout"]
    x = np.asarray(lay["x"]); y = np.asarray(lay["y"])
    b = np.asarray(LAYOUTS[rose]["boundary"], float); c = b.mean(axis=0)
    pad = 1500
    gx = np.linspace(x.min() - pad, x.max() + pad, GRID)
    gy = np.linspace(y.min() - pad, y.max() + pad, GRID)
    GX, GY = np.meshgrid(gx, gy)
    speeds, _ = sim.flow_map(jnp.array(x), jnp.array(y),
                             fm_x=jnp.array(GX.ravel()), fm_y=jnp.array(GY.ravel()),
                             ws=WS0, wd=wd_dom)
    S = np.asarray(speeds).reshape(GX.shape)
    pcm = ax.contourf((GX - c[0]) / 1000, (GY - c[1]) / 1000, S,
                      levels=np.linspace(4, 10.2, 32), cmap=WSCMAP, extend="both")
    bk = (b - c) / 1000
    ax.plot(np.append(bk[:, 0], bk[0, 0]), np.append(bk[:, 1], bk[0, 1]),
            color="white", lw=1.3, zorder=4)
    ax.scatter((x - c[0]) / 1000, (y - c[1]) / 1000, s=6, c="#c0392b",
               edgecolor="white", linewidth=0.25, zorder=5)
    # wind arrow, points DOWNWIND (wd = direction wind comes FROM, N=0 CW)
    th = np.deg2rad(wd_dom)
    ddx, ddy = -np.sin(th), -np.cos(th)        # downwind unit vector
    ax.annotate("", xy=(0.16 + 0.13 * ddx, 0.90 + 0.13 * ddy),
                xytext=(0.16, 0.90), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8,
                                mutation_scale=12))
    ax.text(0.05, 0.98, f"wind\n{wd_dom:.0f}°", transform=ax.transAxes,
            color="white", fontsize=7, va="top", ha="left", weight="bold",
            linespacing=0.95)
    ax.set_title(title, fontsize=8.8, weight="bold", color=INK, pad=4)
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)", fontsize=7.6)
    if ax is axes[0]:
        ax.set_ylabel("north (km)", fontsize=7.6)
    ax.tick_params(length=2.5, labelsize=6.8)

cb = fig.colorbar(pcm, ax=axes, orientation="vertical", fraction=0.03, pad=0.01,
                  aspect=32)
cb.set_label("wind speed (m s$^{-1}$)", fontsize=7.6)
cb.ax.tick_params(labelsize=6.6, colors=MUT)

for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_flowfield.{ext}")
    fig.savefig(out, dpi=200)
    print("wrote", out)
