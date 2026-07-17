"""Performance figure: AEP-vs-wall-clock for Claude + Gemini schedule-only
5-hr runs (top panel from the original combined figure, now standalone).

Story: each agent's running-max held-out-feasible train AEP over its 5-hr
session-time budget. Deployment star marks the schedule with the highest
held-out (ROWP) AEP — NOT the highest training AEP — and the figure visibly
shows higher-training-AEP points (hollow circles) that failed held-out and
were not deployed.
"""
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "search_data.json")))

AGENT_COLORS = {"Claude": "#d62728", "Gemini": "#1f77b4"}
BASELINE = 5540.72
BUDGET_HR = 5.0
DEPLOYED_ATTEMPT = 192


def in_budget(r):
    return (r["elapsed_hr"] is not None) and (r["elapsed_hr"] <= BUDGET_HR)


def main():
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    y_lo, y_hi = 5520, 5572

    for agent, color in AGENT_COLORS.items():
        rows = sorted([r for r in DATA if r["agent"] == agent and in_budget(r)
                       and r["train_aep"] is not None],
                      key=lambda r: r["elapsed_hr"])
        t = np.array([r["elapsed_hr"] for r in rows])
        v = np.array([r["train_aep"] for r in rows])
        train_feas = np.array([r["train_feasible"] for r in rows])
        rowp_feas = np.array([r["rowp_feasible"] for r in rows])
        rowp_aep = np.array([(r.get("rowp_aep") or -np.inf) for r in rows])

        both_pass = train_feas & rowp_feas
        train_only = train_feas & ~rowp_feas

        ax.scatter(t[both_pass], v[both_pass], color=color, s=20, alpha=0.85,
                   edgecolor="none", zorder=3,
                   label=f"{agent}: train + held-out feasible")
        ax.scatter(t[train_only], v[train_only], facecolors="none",
                   edgecolors=color, s=20, alpha=0.55, lw=0.7, zorder=2,
                   label=f"{agent}: train-only feasible")

        # Running-max of held-out AEP among both-feasibles; plot the
        # corresponding training AEP value (so the line and the star are in
        # the same metric the y-axis shows).
        t_rm, v_rm = [], []
        best_rowp, best_train = -np.inf, None
        for ti, vi, m, ri in zip(t, v, both_pass, rowp_aep):
            if m and ri > best_rowp:
                best_rowp, best_train = ri, vi
            if best_train is not None:
                t_rm.append(ti); v_rm.append(best_train)
        if t_rm:
            ax.plot(t_rm, v_rm, color=color, lw=2.4, alpha=0.9, zorder=4,
                    label=f"{agent}: best-so-far held-out feasible (train AEP)")

        dep = [r for r in rows if r["attempt"] == DEPLOYED_ATTEMPT]
        if dep:
            r = dep[0]
            ax.scatter([r["elapsed_hr"]], [r["train_aep"]], marker="*",
                       s=420, color=color, edgecolor="black", lw=1.3,
                       zorder=5, label=f"{agent} deployed (iter_192)")
            rowp = r.get("rowp_aep")
            ho_label = (f"held-out AEP={rowp:.1f}" if rowp is not None
                        else "held-out feasible (no AEP logged)")
            ax.annotate(ho_label,
                        xy=(r["elapsed_hr"], r["train_aep"]),
                        xytext=(r["elapsed_hr"] + 0.18, r["train_aep"] - 5),
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle="-", color=color,
                                        alpha=0.5, lw=0.7))

    ax.axhline(BASELINE, ls="--", color="gray", alpha=0.7, lw=1.1,
               label=f"500-multistart baseline ({BASELINE:.1f})")

    ax.text(0.02, 0.04,
            "Selection rule: deployed = schedule with the highest HELD-OUT AEP "
            "among those feasible on both farms.\nThe star can sit BELOW the "
            "highest training-AEP points — those higher-train points had lower "
            "held-out\nAEP and were not deployed. Hollow circles = pass training "
            "but fail held-out (not deployable).\nOut-of-axis points "
            "(constraint-violating layouts with inflated AEP) are dropped.",
            transform=ax.transAxes, fontsize=7.5, color="#333",
            bbox=dict(facecolor="white", edgecolor="#bbb", boxstyle="round,pad=0.4"))

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(0, BUDGET_HR)
    ax.set_xlabel("Wall-clock session-time (hours)")
    ax.set_ylabel("Training AEP (GWh)")

    n_cl = sum(1 for r in DATA if r['agent']=='Claude' and in_budget(r))
    n_gm = sum(1 for r in DATA if r['agent']=='Gemini' and in_budget(r))
    ax.set_title(
        f"Schedule-only search, 5-hr session-time budget.    "
        f"Claude: {n_cl} attempts (~{3600*5/n_cl:.0f}s each)    |    "
        f"Gemini: {n_gm} attempts (~{3600*5/n_gm:.0f}s each).\n"
        "wall-clock = summed within-session time (Gemini was quota-throttled "
        "across days).",
        fontsize=9.5, loc="left", pad=45)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=7.5, ncol=4, framealpha=0.92, columnspacing=1.0)
    ax.grid(alpha=0.25)

    for ext in ("pdf", "png"):
        out = os.path.join(HERE, f"fig_search_performance.{ext}")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
