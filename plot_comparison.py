#!/usr/bin/env python
"""Compare progress across multiple agent runs.

Usage:
    python plot_comparison.py --save comparison.png
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


def load_run(path):
    with open(path) as f:
        data = json.load(f)
    successes = [e for e in data if "train_aep" in e and e["train_aep"] > TRAIN_BASELINE * 0.9]
    t0 = data[0]["timestamp"]
    return successes, t0


def running_best(entries, key="train_aep"):
    """Compute running best over entries."""
    times, bests = [], []
    rb = -float("inf")
    for e in entries:
        val = e.get(key)
        if val is not None and val > rb:
            rb = val
        if val is not None:
            times.append(e["_time_min"])
            bests.append(rb)
    return times, bests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", default="comparison.png")
    args = p.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"hspace": 0.12})
    ax_train, ax_rowp = axes

    for run in RUNS:
        try:
            entries, t0 = load_run(run["path"])
        except FileNotFoundError:
            print(f"Skipping {run['name']}: file not found")
            continue

        # Add relative time
        for e in entries:
            e["_time_min"] = (e["timestamp"] - t0) / 60

        # Training running best
        t_train, rb_train = running_best(entries, "train_aep")
        ax_train.plot(t_train, rb_train, "-", color=run["color"],
                      linewidth=2, label=run["name"])

        # ROWP running best (only feasible)
        rowp_entries = [e for e in entries if e.get("rowp_feasible") and e.get("rowp_aep")]
        if rowp_entries:
            t_rowp, rb_rowp = running_best(rowp_entries, "rowp_aep")
            ax_rowp.plot(t_rowp, rb_rowp, "-", color=run["color"],
                         linewidth=2, label=run["name"])

    # Baselines
    ax_train.axhline(TRAIN_BASELINE, color="gray", ls="--", alpha=0.5,
                     label=f"Baseline ({TRAIN_BASELINE:.1f})")
    ax_rowp.axhline(ROWP_BASELINE, color="gray", ls="--", alpha=0.5,
                    label=f"Baseline ({ROWP_BASELINE:.1f})")

    ax_train.set_ylabel("Training AEP — Running Best (GWh)")
    ax_train.legend(loc="lower right", fontsize=9)
    ax_train.set_title("Claude Code: Design Freedom vs Constrained Schedule")

    ax_rowp.set_ylabel("ROWP AEP — Running Best (GWh)")
    ax_rowp.set_xlabel("Time (minutes)")
    ax_rowp.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")
    plt.close(fig)


if __name__ == "__main__":
    main()
