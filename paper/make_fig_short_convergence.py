"""fig_short_convergence — the schedule search for both agents, one point/attempt.

2x2 grid: columns = Claude Code (left) / Gemini CLI (right); rows = training
(DEI, N=50) AEP / validation (held-out ROWP, N=74) AEP. Each attempt is one
point: filled = feasible, hollow circle = infeasible. Dashed line = the
500-multistart baseline for that farm. Deployed schedule (iter 192) starred
where its value is available.

In-budget = attempts within 5 h of each run's first attempt (Claude 2026-04-05;
Gemini 2026-04-07/08). Later continuation attempts (Apr 16/17/25/26) are
excluded.

Validation (ROWP) AEP was held out from the agent during search; it is
backfilled by re-scoring each attempt afterwards, so the bottom row is a
retrospective evaluation, not a search signal.

Data: results_agent_schedule_only_5hr/attempt_log.json (Claude);
results_agent_gemini_cli_5hr/attempt_log.json (Gemini).
Baselines: 500-start TopFarm gated at 0.1 m max-outside tolerance
(results/baselines_01tol.json): DEI 5545.0, ROWP 4243.6. Matches the
tolerance language of fig_aep_dominance. (At strict-0 the TopFarm baseline is
essentially infeasible: 3/500 DEI, 0/500 ROWP — it lands mm-cm outside.)

Output: paper/figs/fig_short_convergence.{pdf,png}
"""
import json
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"
C_CLAUDE, C_GEM = "#d1622a", "#2c6fbb"   # agent colours: Claude orange, Gemini blue
DEPLOYED_ATTEMPT = 192

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})


def in_budget(path):
    a = json.load(open(os.path.join(ROOT, path)))
    a = a if isinstance(a, list) else a.get("attempts", [])
    t0 = min(x["timestamp"] for x in a if x.get("timestamp"))
    return [x for x in a if x.get("timestamp") and x["timestamp"] - t0 <= 5 * 3600]


CLAUDE = in_budget("results_agent_schedule_only_5hr/attempt_log.json")
GEMINI = in_budget("results_agent_gemini_cli_5hr/attempt_log.json")
# 500-start TopFarm baselines gated at a clean 0.1 m max-outside tolerance
# (same tolerance language as fig_aep_dominance). See results/baselines_01tol.json.
_B01 = json.load(open(os.path.join(ROOT, "results/baselines_01tol.json")))
DEI_BASE = _B01["dei_n50"]["best_aep_01tol"]     # 5545.0 (500/500 feasible @ 0.1 m)
ROWP_BASE = _B01["rowp_n74"]["best_aep_01tol"]   # 4243.6 (492/500 feasible @ 0.1 m)


def series(attempts, ak, fk):
    x, y, feas = [], [], []
    for a in attempts:
        v = a.get(ak)
        if v is None:
            continue
        x.append(a["attempt"]); y.append(v); feas.append(bool(a.get(fk)))
    return np.array(x), np.array(y), np.array(feas)


TRAIN_FLOOR = 5000.0   # feasible DEI schedules score ~5500; below = log glitch


def champion_traj(attempts):
    """The deployed-so-far schedule, on both farms.

    The champion is the running argmax of validation (ROWP) AEP among attempts
    feasible on BOTH farms — i.e. the schedule that would be deployed if the
    search stopped now. BOTH lines track this one schedule: bottom = its
    validation AEP (monotone, the deployment-selection criterion), top = the
    SAME schedule's training AEP. `champ` = final deployed schedule, starred.

    One log glitch is guarded: attempt 83 (Gemini) has attempt 81's ROWP value
    mis-written into its training field (4262 GWh, implausible for a feasible
    DEI layout); iter_83's script was not retained so the true value is
    unrecoverable, and the training line forward-fills the last valid champion
    training AEP through it."""
    order = sorted(attempts, key=lambda a: a["attempt"])
    xs = np.array([a["attempt"] for a in order])
    bv, champ, tline, vline, last_train = -np.inf, None, [], [], np.nan
    for a in order:
        if a.get("train_feasible") and a.get("rowp_feasible") \
                and a.get("rowp_aep") is not None and a["rowp_aep"] > bv:
            bv, champ = a["rowp_aep"], a
        vline.append(champ["rowp_aep"] if champ else np.nan)
        if champ and champ.get("train_aep") is not None \
                and champ["train_aep"] >= TRAIN_FLOOR:
            last_train = champ["train_aep"]
        tline.append(last_train)
    return xs, {"train_aep": np.array(tline), "rowp_aep": np.array(vline)}, champ


fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex="col", sharey="row",
                         constrained_layout=True)

COLS = [("Claude Code", CLAUDE, C_CLAUDE), ("Gemini CLI", GEMINI, C_GEM)]
ROWS = [("train_aep", "train_feasible", DEI_BASE, "train AEP (GWh)", (5450, 5615)),
        ("rowp_aep", "rowp_feasible", ROWP_BASE, "validation AEP (GWh)", (4150, 4340))]

for ci, (cname, attempts, color) in enumerate(COLS):
    champ_x, champ_line, champ = champion_traj(attempts)   # deployment champion
    for ri, (ak, fk, base, ylab, ylim) in enumerate(ROWS):
        ax = axes[ri, ci]
        x, y, feas = series(attempts, ak, fk)
        ax.axhline(base, ls=(0, (5, 2)), color=INK, lw=1.0, zorder=2)
        ax.plot(champ_x, champ_line[ak], color=color, lw=1.8, alpha=0.95,
                zorder=7, solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white"),
                              pe.Normal()])
        ax.scatter(x[feas], y[feas], s=13, color=color, edgecolor="white",
                   linewidth=0.3, zorder=4)
        ax.scatter(x[~feas], y[~feas], s=13, facecolor="none", edgecolor=color,
                   linewidth=0.8, zorder=4)
        # deployed = the final validation champion (marked on both rows)
        if champ and champ.get(ak) is not None:
            ax.scatter([champ["attempt"]], [champ[ak]], marker="*", s=130,
                       color=color, edgecolor="black", linewidth=0.7, zorder=6)
        n_below = int(np.sum(y < ylim[0]))
        if n_below:
            ax.annotate(f"{n_below} below axis", (x.max(), ylim[0]),
                        fontsize=6.0, color=MUT, ha="right", va="bottom",
                        xytext=(0, 1), textcoords="offset points", style="italic")
        ax.set_ylim(*ylim)
        if ri == 0:
            ax.set_title(cname, fontsize=9.5, weight="bold", color=INK, pad=5)
        if ci == 0:
            ax.set_ylabel(ylab, fontsize=8)
        if ri == 1:
            ax.set_xlabel("search attempt (in-budget)", fontsize=8)
        # baseline label only on left column
        if ci == 0:
            ax.annotate(f"baseline {base:.0f}", (x.min(), base), fontsize=6.2,
                        color=INK, ha="left", va="bottom", xytext=(1, 1),
                        textcoords="offset points")
        ax.grid(alpha=0.18, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(length=2.5, labelsize=7)

# row labels (farm) on the right edge
for ri, lab in enumerate(["Training · DEI", "Validation · held-out ROWP"]):
    axes[ri, 1].annotate(lab, (1.015, 0.5), xycoords="axes fraction",
                         rotation=270, ha="left", va="center", fontsize=8,
                         weight="bold", color=INK)

legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=MUT,
           markeredgecolor="white", markersize=6, label="feasible"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor=MUT, markersize=6, label="infeasible"),
    Line2D([0], [0], color=INK, ls=(0, (5, 2)), lw=1.0,
           label="500-start baseline (0.1 m tol)"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor=MUT,
           markeredgecolor="black", markersize=11,
           label="deployed (best validation)"),
]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.055),
           ncol=4, fontsize=7.4, frameon=False)

for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_short_convergence.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
