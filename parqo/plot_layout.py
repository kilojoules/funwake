"""Plot the final dual-bump layout: five inclusion-zone polygons +
optimized turbines. Turbines and boundaries only.

Usage:
    pixi run python parqo/plot_layout.py
Writes: parqo/parqo_layout.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    problem = json.load(open(os.path.join(HERE, "problem_parqo.json")))
    opt = json.load(open(os.path.join(HERE, "layout_parqo.json")))
    res = json.load(open(os.path.join(HERE, "results_parqo.json")))
    aep = res["optimized"]["aep_gwh"]

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, z in enumerate(problem["inclusion_zones"]):
        z = np.array(z + [z[0]])
        ax.plot(z[:, 0], z[:, 1], "k-", lw=1.8,
                label="inclusion zones" if i == 0 else None)

    ax.plot(opt["x"], opt["y"], "^", color="crimson", ms=11, mec="w",
            label=f"dual-bump optimized ({aep:.2f} GWh)")
    for i, (x, y) in enumerate(zip(opt["x"], opt["y"])):
        ax.annotate(str(i + 1), (x, y), textcoords="offset points",
                    xytext=(7, 7), fontsize=9, color="crimson")

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"ParqueFicticio inclusion zones (Criado Risco et al. 2024)"
                 f" — dual-bump schedule, seed {res.get('seed', 0)}")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(HERE, "parqo_layout.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
