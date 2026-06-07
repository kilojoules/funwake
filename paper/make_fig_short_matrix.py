"""Re-render paper/figs/fig_short_matrix.{png,pdf} from
results/matrix/schedules_matrix.json + baselines_matrix.json.

The original generator was dropped in commit 988e6645; this re-creates
the short-paper variant with three lines:
    - Claude schedule (iter 192)         red squares
    - Gemini schedule                    blue circles
    - Gemini schedule (fixed beta)       teal triangles (ablation)

Row labels are "DEI polygon (training)" / "ROWP polygon (held-out)";
columns are the four wind roses. Infeasible cells render as x markers.

Usage:
    pixi run python paper/make_fig_short_matrix.py
"""
import json
import os

import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(PROJECT_ROOT, "paper", "figs")


def main():
    sched_path = os.path.join(PROJECT_ROOT, "results", "matrix",
                              "schedules_matrix.json")
    base_path = os.path.join(PROJECT_ROOT, "results", "matrix",
                             "baselines_matrix.json")
    sched = json.load(open(sched_path))
    baselines = json.load(open(base_path))

    farms = ["dei", "rowp"]
    roses = ["uniform", "omnidir", "dei", "rowp"]
    ns = [30, 40, 50, 60, 70, 80]
    rose_titles = {
        "uniform": "Uniform",
        "omnidir": "Omnidirectional",
        "dei": "DEI rose",
        "rowp": "ROWP rose",
    }
    farm_labels = {
        "dei": "DEI polygon (training)",
        "rowp": "ROWP polygon (held-out)",
    }

    labels = [
        ("Claude schedule (iter 192)", "Claude",                       "#c73a1a", "s"),
        ("Gemini schedule",            "Gemini (scheduled β)",    "#1a73c7", "o"),
        ("Gemini schedule (fixed beta)", "Gemini (fixed β=0.3,0.5)", "#0a8e7a", "^"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5), sharex=True, sharey=True)

    for i, farm in enumerate(farms):
        for j, rose in enumerate(roses):
            ax = axes[i, j]
            ax.axhline(0.0, color="black", linewidth=1.2, zorder=3)

            for key_label, leg, color, marker in labels:
                xs_f, ys_f, xs_i, ys_i = [], [], [], []
                for n in ns:
                    k = f"{key_label}|{farm}_n{n}_rose{rose}"
                    s = sched.get(k)
                    if not s or "aep_gwh" not in s:
                        continue
                    bkey = f"{farm}_n{n}_rose{rose}"
                    b = baselines.get(bkey, {}).get("best_aep")
                    if b is None:
                        continue
                    gap = (s["aep_gwh"] / b - 1.0) * 100
                    if s.get("feasible"):
                        xs_f.append(n); ys_f.append(gap)
                    else:
                        xs_i.append(n); ys_i.append(gap)

                if xs_f:
                    ax.plot(xs_f, ys_f, marker=marker, linestyle="-",
                            color=color, linewidth=1.4, markersize=6,
                            label=leg if (i, j) == (0, 0) else None)
                if xs_i:
                    ax.plot(xs_i, ys_i, marker="x", linestyle="",
                            color=color, markersize=7, markeredgewidth=1.5,
                            label=f"{leg} (infeasible)" if (i, j) == (0, 0) else None)

            if i == 0:
                ax.set_title(rose_titles[rose], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{farm_labels[farm]}\nGap over baseline (%)",
                              fontsize=9)
            if i == 1:
                ax.set_xlabel("Turbine count $N$")
            ax.set_xticks(ns)

    axes[0, 0].legend(loc="upper left", fontsize=7, framealpha=0.9)

    plt.tight_layout()
    out_png = os.path.join(FIGS, "fig_short_matrix.png")
    out_pdf = os.path.join(FIGS, "fig_short_matrix.pdf")
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
