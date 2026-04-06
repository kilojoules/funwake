#!/usr/bin/env python
"""Pareto front: Training AEP vs Held-out ROWP AEP for each run.

Shows the tradeoff between training performance and generalization.
Pareto-optimal scripts are connected and highlighted.

Usage:
    python plot_pareto.py --save pareto.png
"""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = [
    {
        "name": "Full optimize (black box)",
        "path": "results_agent_claude_7hr/attempt_log.json",
        "color": "C0",
        "marker": "o",
    },
    {
        "name": "Full optimize (exposed seed)",
        "path": "results_agent_claude_6hr/attempt_log.json",
        "color": "C1",
        "marker": "s",
    },
    {
        "name": "Suggested schedule seed",
        "path": "results_agent_schedule_5hr/attempt_log.json",
        "color": "C2",
        "marker": "^",
    },
    {
        "name": "Enforced schedule-only",
        "path": "results_agent_schedule_only_5hr/attempt_log.json",
        "color": "C3",
        "marker": "D",
    },
]

TRAIN_BASELINE = 5540.72
ROWP_BASELINE = 4246.67


def pareto_front(points):
    """Return indices of Pareto-optimal points (maximize both objectives)."""
    points = np.array(points)
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j >= i on both and j > i on at least one
            if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", default="pareto.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(10, 7))

    for run in RUNS:
        try:
            with open(run["path"]) as f:
                data = json.load(f)
        except FileNotFoundError:
            continue

        # Get feasible attempts with both scores
        feasible = [e for e in data
                    if e.get("train_aep") and e.get("rowp_feasible")
                    and e.get("rowp_aep") and e["train_aep"] > TRAIN_BASELINE * 0.9]

        if not feasible:
            continue

        train = [e["train_aep"] for e in feasible]
        rowp = [e["rowp_aep"] for e in feasible]

        # Scatter all feasible points
        ax.scatter(train, rowp, color=run["color"], marker=run["marker"],
                   alpha=0.25, s=30, linewidths=0, zorder=1)

        # Compute and plot Pareto front
        points = list(zip(train, rowp))
        pareto_idx = pareto_front(points)

        if len(pareto_idx) > 0:
            pareto_pts = sorted([(train[i], rowp[i]) for i in pareto_idx])
            px = [p[0] for p in pareto_pts]
            py = [p[1] for p in pareto_pts]

            # Pareto front line (step-like)
            ax.plot(px, py, "-", color=run["color"], linewidth=2, zorder=3,
                    label=f"{run['name']} ({len(pareto_idx)} Pareto)")
            ax.scatter(px, py, color=run["color"], marker=run["marker"],
                       s=80, edgecolors="black", linewidths=0.8, zorder=4)

    # Baselines
    ax.axvline(TRAIN_BASELINE, color="gray", ls="--", alpha=0.4)
    ax.axhline(ROWP_BASELINE, color="gray", ls="--", alpha=0.4)
    ax.annotate(f"Train baseline\n({TRAIN_BASELINE:.1f})",
                xy=(TRAIN_BASELINE, ax.get_ylim()[0]),
                fontsize=8, color="gray", ha="right", va="bottom")
    ax.annotate(f"ROWP baseline ({ROWP_BASELINE:.1f})",
                xy=(ax.get_xlim()[0], ROWP_BASELINE),
                fontsize=8, color="gray", ha="left", va="bottom")

    # Ideal corner annotation
    ax.annotate("← better generalization\n↑ better training",
                xy=(0.98, 0.98), xycoords="axes fraction",
                fontsize=8, color="gray", ha="right", va="top")

    ax.set_xlabel("Training AEP (GWh)")
    ax.set_ylabel("Held-out ROWP AEP (GWh)")
    ax.set_title("Pareto Front: Training vs Held-out Performance")
    ax.legend(loc="lower left", fontsize=9)

    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")
    plt.close(fig)


if __name__ == "__main__":
    main()
