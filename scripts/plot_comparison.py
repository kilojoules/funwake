#!/usr/bin/env python
"""Compare progress across multiple agent runs.

Shows raw attempt values (thin/transparent) and "deploy" lines
(running best — the script you'd actually ship at each point in time).

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
    },
    {
        "name": "Full optimize (exposed seed)",
        "path": "results_agent_claude_6hr/attempt_log.json",
        "color": "C1",
    },
    {
        "name": "Suggested schedule seed",
        "path": "results_agent_schedule_5hr/attempt_log.json",
        "color": "C2",
    },
    {
        "name": "Enforced schedule-only",
        "path": "results_agent_schedule_only_5hr/attempt_log.json",
        "color": "C3",
    },
]

TRAIN_BASELINE = 5540.72
ROWP_BASELINE = 4246.67


def load_run(path):
    with open(path) as f:
        data = json.load(f)
    min_train = TRAIN_BASELINE * 0.9
    successes = [e for e in data if "train_aep" in e and e["train_aep"] > min_train]
    t0 = data[0]["timestamp"]
    for e in successes:
        e["_time_min"] = (e["timestamp"] - t0) / 60
    return successes


def running_best(entries, key):
    """Compute running best, return (times, values) for the deploy line."""
    times, bests = [], []
    rb = -float("inf")
    for e in entries:
        val = e.get(key)
        if val is not None:
            if val > rb:
                rb = val
            times.append(e["_time_min"])
            bests.append(rb)
    return times, bests


def deploy_line(entries):
    """Compute the deploy line: running best on ROWP AEP among feasible
    scripts. Returns (times, train_vals, rowp_vals) — the same script
    tracked on both panels. We deploy based on held-out performance."""
    times_t, vals_t = [], []
    times_r, vals_r = [], []
    best_rowp = -float("inf")
    best_train = None
    for e in entries:
        if e.get("train_aep") is None:
            continue
        if not e.get("rowp_feasible") or not e.get("rowp_aep"):
            continue
        t = e["_time_min"]
        if e["rowp_aep"] > best_rowp:
            best_rowp = e["rowp_aep"]
            best_train = e["train_aep"]
        times_r.append(t)
        vals_r.append(best_rowp)
        if best_train is not None:
            times_t.append(t)
            vals_t.append(best_train)
    return times_t, vals_t, times_r, vals_r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", default="comparison.png")
    args = p.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={"hspace": 0.15})
    ax_train, ax_rowp = axes

    for run in RUNS:
        try:
            entries = load_run(run["path"])
        except FileNotFoundError:
            print(f"Skipping {run['name']}: file not found")
            continue

        c = run["color"]
        label = run["name"]

        # ── Raw scatter on both panels ──
        t_raw = [e["_time_min"] for e in entries if "train_aep" in e]
        v_raw = [e["train_aep"] for e in entries if "train_aep" in e]
        ax_train.scatter(t_raw, v_raw, color=c, alpha=0.15, s=8,
                         linewidths=0, zorder=1)

        feas = [e for e in entries if e.get("rowp_feasible") and e.get("rowp_aep")]
        if feas:
            t_f = [e["_time_min"] for e in feas]
            v_f = [e["rowp_aep"] for e in feas]
            ax_rowp.scatter(t_f, v_f, color=c, alpha=0.15, s=8,
                            linewidths=0, zorder=1)

        # ── Deploy line: same script on both panels ──
        # Running best on training among ROWP-feasible scripts
        dt, vt, dr, vr = deploy_line(entries)
        if dt:
            ax_train.plot(dt, vt, "-", color=c, linewidth=2.5,
                          label=label, zorder=2)
        if dr:
            ax_rowp.plot(dr, vr, "-", color=c, linewidth=2.5,
                         label=label, zorder=2)

    # Baselines
    ax_train.axhline(TRAIN_BASELINE, color="gray", ls="--", alpha=0.5,
                     label=f"Baseline ({TRAIN_BASELINE:.1f})")
    ax_rowp.axhline(ROWP_BASELINE, color="gray", ls="--", alpha=0.5,
                    label=f"Baseline ({ROWP_BASELINE:.1f})")

    # Labels
    ax_train.set_ylabel("Training AEP (GWh)")
    ax_train.legend(loc="lower right", fontsize=9)
    ax_train.set_title("Claude Code: Design Freedom vs Constrained Schedule")
    ax_train.annotate("deploy line = script with best\nheld-out AEP (same script both panels)",
                      xy=(0.02, 0.95), xycoords="axes fraction",
                      fontsize=8, color="gray", va="top")

    ax_rowp.set_ylabel("Held-out ROWP AEP (GWh)")
    ax_rowp.set_xlabel("Time (minutes)")
    ax_rowp.legend(loc="lower right", fontsize=9)

    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")
    plt.close(fig)


if __name__ == "__main__":
    main()
