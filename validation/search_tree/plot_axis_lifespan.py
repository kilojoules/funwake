"""Axis-lifespan figure: per-(agent, axis) timeline showing when each schedule
family was first tried, how long it was actively explored, and when it was
last attempted within each agent's 5-hr budget.

Story: agents explore the same canonical family set but with different
emphasis. Claude's late shift into the `dual_bump` family (the deployed
winner) is visible as a bar opening around 2 hr and running to budget end;
Gemini stays in `cyclic_lr` + `warm_restart` throughout.

AXIS classification is heuristic (regex over docstring + code keywords) —
agents did not emit STRATEGY_REGISTRY HYPOTHESIS/AXIS tags despite the
prompt asking for them, so the bar ends are "last attempt in family within
budget", not explicit LESSON-tagged closures.
"""
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "search_data.json")))

AGENT_COLORS = {"Claude": "#d62728", "Gemini": "#1f77b4"}
BUDGET_HR = 5.0


def in_budget(r):
    return (r["elapsed_hr"] is not None) and (r["elapsed_hr"] <= BUDGET_HR)


def main():
    # Per-(agent, axis): first/last elapsed_hr within budget window.
    fl = defaultdict(lambda: {"first_t": None, "last_t": None,
                              "first_n": None, "last_n": None})
    for r in DATA:
        if not in_budget(r): continue
        k = (r["agent"], r["axis"])
        if fl[k]["first_t"] is None:
            fl[k]["first_t"] = r["elapsed_hr"]
            fl[k]["first_n"] = r["attempt"]
        fl[k]["last_t"] = r["elapsed_hr"]
        fl[k]["last_n"] = r["attempt"]

    counts = defaultdict(int)
    for r in DATA:
        if in_budget(r): counts[r["axis"]] += 1
    AXES_ORDERED = [a for a, _ in sorted(counts.items(), key=lambda x: -x[1])
                    if a != "other"]
    if "other" in counts:
        AXES_ORDERED.append("other")
    n_axes = len(AXES_ORDERED)

    def yrow(agent, axis):
        i = AXES_ORDERED.index(axis)
        base = n_axes - i - 1
        return base + (0.7 if agent == "Claude" else 0.3)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    # Lifespan bars (translucent, per agent color)
    for (agent, axis), info in fl.items():
        if axis not in AXES_ORDERED: continue
        y = yrow(agent, axis)
        color = AGENT_COLORS[agent]
        ax.plot([info["first_t"], info["last_t"]], [y, y],
                color=color, lw=3.0, alpha=0.45, solid_capstyle="butt", zorder=2)

    # Per-attempt dots
    for r in DATA:
        if not in_budget(r): continue
        if r["axis"] not in AXES_ORDERED: continue
        y = yrow(r["agent"], r["axis"])
        ax.scatter([r["elapsed_hr"]], [y], color=AGENT_COLORS[r["agent"]],
                   s=8, alpha=0.65, zorder=3, edgecolor="none")

    # First (◇) + last (×) markers — black so they don't double-load agent color
    for (agent, axis), info in fl.items():
        if axis not in AXES_ORDERED: continue
        y = yrow(agent, axis)
        ax.scatter([info["first_t"]], [y], marker="D", s=55, color="black",
                   zorder=5)
        ax.scatter([info["last_t"]], [y], marker="x", s=65, color="black",
                   lw=1.8, zorder=5)

    yticks = [n_axes - i - 0.5 for i in range(n_axes)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(AXES_ORDERED, fontsize=9)
    ax.set_ylim(-0.2, n_axes + 0.2)
    ax.set_xlim(0, BUDGET_HR)
    ax.set_xlabel("Wall-clock session-time (hours)")
    ax.set_ylabel("Schedule family (axis) — Claude=top row, Gemini=bottom row of each pair")
    ax.set_title(
        "Schedule-family lifespans within 5-hr budget. "
        "Bars span first→last attempt in family per agent.",
        fontsize=10, loc="left", pad=42)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    legend_items = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
               markersize=8, label="First attempt in family"),
        Line2D([0], [0], marker="x", color="black", markersize=9, lw=0,
               markeredgewidth=2, label="Last in-budget attempt"),
        mpatches.Patch(color=AGENT_COLORS["Claude"], alpha=0.55,
                       label="Claude (top half of row)"),
        mpatches.Patch(color=AGENT_COLORS["Gemini"], alpha=0.55,
                       label="Gemini (bottom half of row)"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8.5, framealpha=0.95)

    ax.text(0.0, -0.16,
            "A family can REOPEN within the budget; the × marks the LAST "
            "in-budget attempt, not an explicit LESSON closure (agents did "
            "not emit AXIS/LESSON tags).\nAXIS classification is heuristic "
            "(regex over docstring + code).",
            transform=ax.transAxes, fontsize=7.5, color="#333")

    for ext in ("pdf", "png"):
        out = os.path.join(HERE, f"fig_axis_lifespan.{ext}")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
