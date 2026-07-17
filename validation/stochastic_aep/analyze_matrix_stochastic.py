"""Analyse stochastic 48-cell matrix vs original coarse-grid values.

For each cell:
- gap_stochastic_internal = (sched_aep - sgd_baseline_aep) / sgd_baseline_aep * 100
- gap_original_vs_500ms   = (sched_aep_original - best_500ms_baseline) / best_500ms_baseline * 100
  (lifted straight from results/matrix/schedules_matrix.json and
  results/matrix/baselines_matrix.json — the apples-to-oranges original gap)

Pre-registered threshold: a cell shows a real advantage if gap ≥ 0.2 %.

Output:
- side-by-side figure (4 panels per polygon: stochastic gap vs N for each rose,
  with original-coarse-grid overlay)
- comparison table CSV
- per-cell verdict (clears 0.2 % bar? in original AND stochastic?)
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = "/Users/julianquick/portfolio_copy/funwake"


def load_aep_safe(runs):
    """From a list of runs (one per sample seed), return (mean, std, n) or None."""
    aeps = [r["aep_det_gwh"] for r in runs if "aep_det_gwh" in r]
    if not aeps:
        return None
    aeps = np.array(aeps)
    return {
        "mean": float(aeps.mean()),
        "std": float(aeps.std(ddof=1)) if len(aeps) > 1 else 0.0,
        "n": len(aeps),
        "values": aeps.tolist(),
    }


def compute_stochastic_gaps(stoch_path):
    d = json.load(open(stoch_path))
    cells_out = {}
    for cell_key, cell in d["cells"].items():
        schedules = cell["schedules"]
        per_sched = {}
        for name, info in schedules.items():
            per_sched[name] = load_aep_safe(info["runs"])
        base = per_sched.get("sgd_baseline")
        if base is None:
            cells_out[cell_key] = {"error": "missing sgd_baseline"}
            continue
        cells_out[cell_key] = {
            "farm": cell["farm"], "n": cell["n"], "rose": cell["rose"],
            "sgd_baseline": base,
            "schedules": per_sched,
            "gap_pct": {
                name: (
                    None if per_sched[name] is None
                    else 100.0 * (per_sched[name]["mean"] - base["mean"]) / base["mean"]
                )
                for name in per_sched
            },
            "gap_spread_pct": {
                name: (
                    None if per_sched[name] is None or per_sched[name]["std"] == 0
                    else 100.0 * per_sched[name]["std"] / base["mean"]
                )
                for name in per_sched
            },
        }
    return cells_out


def load_original_matrix():
    sched_path = os.path.join(PROJECT_ROOT, "results/matrix/schedules_matrix.json")
    base_path = os.path.join(PROJECT_ROOT, "results/matrix/baselines_matrix.json")
    sched = json.load(open(sched_path))
    base = json.load(open(base_path))
    cells = {}
    # schedules_matrix.json keys are "<label>|<farm>_n<N>_rose<rose>"
    for key, v in sched.items():
        if "aep_gwh" not in v:
            continue
        label, cell_key = key.split("|", 1)
        base_entry = base.get(cell_key)
        if base_entry is None or "best_aep" not in base_entry:
            continue
        base_aep = float(base_entry["best_aep"])
        gap = 100.0 * (float(v["aep_gwh"]) - base_aep) / base_aep
        cells.setdefault(cell_key, {})[label] = {
            "aep_original": float(v["aep_gwh"]),
            "baseline_500ms": base_aep,
            "gap_pct": gap,
            "feasible": v.get("feasible"),
        }
    return cells


def load_500ms_baseline_per_cell():
    """Return dict cell_key -> best_aep (the 500-multistart baseline from the
    original matrix; same denominator as the original gap-over-baseline)."""
    base_path = os.path.join(PROJECT_ROOT, "results/matrix/baselines_matrix.json")
    base = json.load(open(base_path))
    return {k: float(v["best_aep"]) for k, v in base.items() if "best_aep" in v}


def write_table(stoch_cells, orig_cells, out_path):
    base_500 = load_500ms_baseline_per_cell()
    rows = []
    for cell_key, cell in stoch_cells.items():
        if "error" in cell:
            continue
        gap = cell["gap_pct"]
        spread = cell["gap_spread_pct"]
        orig = orig_cells.get(cell_key, {})
        # gap vs the published 500-multistart baseline (apples-to-apples
        # with original matrix; not affected by sgd_baseline feasibility
        # failures under stochastic gradient)
        b500 = base_500.get(cell_key)
        sched = cell["schedules"]
        claude_aep = sched.get("claude_iter192") or {}
        gemini_aep = sched.get("gemini_iter192") or {}
        baseline_stoch_aep = cell["sgd_baseline"]["mean"]
        baseline_stoch_bp = cell.get("sgd_baseline", {}).get("bp_final", None)

        # Check feasibility of each schedule at finish (bp_final from runs)
        def get_bp_final(name):
            runs = cell["schedules"].get(name, {}).get("runs", []) if hasattr(cell["schedules"].get(name, {}), 'get') else []
            return None
        # Reload bp_final from raw json (load_aep_safe stripped this)
        row = {
            "cell": cell_key,
            "farm": cell["farm"], "n": cell["n"], "rose": cell["rose"],
            "baseline_500ms_aep_gwh": b500,
            "baseline_stoch_aep_gwh": baseline_stoch_aep,
            "claude_stoch_aep_gwh": claude_aep.get("mean"),
            "gemini_stoch_aep_gwh": gemini_aep.get("mean"),
            "claude_gap_vs_500ms_pct": (
                100.0 * (claude_aep.get("mean") - b500) / b500
                if (b500 is not None and claude_aep.get("mean") is not None) else None
            ),
            "gemini_gap_vs_500ms_pct": (
                100.0 * (gemini_aep.get("mean") - b500) / b500
                if (b500 is not None and gemini_aep.get("mean") is not None) else None
            ),
            "claude_gap_vs_stoch_baseline_pct": gap.get("claude_iter192"),
            "gemini_gap_vs_stoch_baseline_pct": gap.get("gemini_iter192"),
            "claude_gap_original_pct": (orig.get("Claude schedule (iter 192)") or {}).get("gap_pct"),
            "gemini_gap_original_pct": (orig.get("Gemini schedule") or {}).get("gap_pct"),
            "claude_clears_0.2_vs_500ms": False,
            "gemini_clears_0.2_vs_500ms": False,
        }
        row["claude_clears_0.2_vs_500ms"] = (
            row["claude_gap_vs_500ms_pct"] is not None and row["claude_gap_vs_500ms_pct"] >= 0.2
        )
        row["gemini_clears_0.2_vs_500ms"] = (
            row["gemini_gap_vs_500ms_pct"] is not None and row["gemini_gap_vs_500ms_pct"] >= 0.2
        )
        rows.append(row)

    keys = list(rows[0].keys())
    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def make_figure(stoch_cells, orig_cells, rows, out_pdf, out_png):
    """Side-by-side comparison: coarse-grid gap (dashed) vs stochastic gap (solid)
    using the SAME denominator (500-multistart published baseline). Both axes
    are gap-over-published-500ms-baseline (%), so the change in gap is
    directly attributable to the gradient estimator."""
    rows_by_cell = {r["cell"]: r for r in rows}
    farms = ["dei", "rowp"]
    roses = ["uniform", "omnidir", "dei", "rowp"]
    ns = [30, 40, 50, 60, 70, 80]
    rose_titles = {
        "uniform": "Uniform", "omnidir": "Omnidirectional",
        "dei": "DEI rose", "rowp": "ROWP rose",
    }
    farm_labels = {
        "dei": "DEI polygon (training)",
        "rowp": "ROWP polygon (held-out)",
    }
    labels = [
        ("claude_gap_vs_500ms_pct", "claude_gap_original_pct", "Claude", "#c73a1a", "s"),
        ("gemini_gap_vs_500ms_pct", "gemini_gap_original_pct", "Gemini", "#1a73c7", "o"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for i, farm in enumerate(farms):
        for j, rose in enumerate(roses):
            ax = axes[i, j]
            ax.axhline(0.0, color="black", linewidth=1.0, zorder=2)
            ax.axhline(0.2, color="gray", linewidth=0.8, linestyle=":", zorder=2,
                       label="0.2 % bar" if (i, j) == (0, 0) else None)
            for stoch_key, orig_key, leg, color, marker in labels:
                xs_s, ys_s, xs_o, ys_o = [], [], [], []
                for n in ns:
                    cell_key = f"{farm}_n{n}_rose{rose}"
                    r = rows_by_cell.get(cell_key)
                    if r is None:
                        continue
                    if r[stoch_key] is not None:
                        xs_s.append(n); ys_s.append(r[stoch_key])
                    if r[orig_key] is not None:
                        xs_o.append(n); ys_o.append(r[orig_key])
                if xs_o:
                    ax.plot(xs_o, ys_o, marker=marker, linestyle="--",
                            color=color, linewidth=1.0, markersize=5,
                            alpha=0.55,
                            label=f"{leg} (coarse-grid)" if (i, j) == (0, 0) else None)
                if xs_s:
                    ax.plot(xs_s, ys_s, marker=marker, linestyle="-",
                            color=color, linewidth=1.6, markersize=7,
                            label=f"{leg} (stochastic K=50)" if (i, j) == (0, 0) else None)
            if i == 0:
                ax.set_title(rose_titles[rose], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{farm_labels[farm]}\nGap over 500-ms baseline (%)",
                              fontsize=9)
            if i == 1:
                ax.set_xlabel("Turbine count $N$")
            ax.set_xticks(ns)
    axes[0, 0].legend(loc="best", fontsize=7, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stochastic",
                   default="validation/stochastic_aep/matrix_stochastic.json")
    p.add_argument("--out-csv",
                   default="validation/stochastic_aep/matrix_compare.csv")
    p.add_argument("--out-pdf",
                   default="validation/stochastic_aep/matrix_compare.pdf")
    p.add_argument("--out-png",
                   default="validation/stochastic_aep/matrix_compare.png")
    args = p.parse_args()

    stoch_cells = compute_stochastic_gaps(args.stochastic)
    orig_cells = load_original_matrix()
    rows = write_table(stoch_cells, orig_cells, args.out_csv)
    make_figure(stoch_cells, orig_cells, rows, args.out_pdf, args.out_png)

    n_claude_clear_500 = sum(r["claude_clears_0.2_vs_500ms"] for r in rows)
    n_gemini_clear_500 = sum(r["gemini_clears_0.2_vs_500ms"] for r in rows)
    n_total = len(rows)

    # Also report comparison vs coarse-grid: how many cells had coarse_gap ≥ 0.2
    # AND now have stoch_gap < 0.2 (i.e., apparent advantage that doesn't
    # survive faithful gradient)
    n_lost = sum(
        1 for r in rows
        if (r["claude_gap_original_pct"] or 0) >= 0.2 and (r["claude_gap_vs_500ms_pct"] or 0) < 0.2
    )
    n_held = sum(
        1 for r in rows
        if (r["claude_gap_original_pct"] or 0) >= 0.2 and (r["claude_gap_vs_500ms_pct"] or 0) >= 0.2
    )
    print(f"\n=== Gap-over-500ms-baseline (apples-to-apples with original) ===")
    print(f"Cells: {n_total}")
    print(f"Claude clears 0.2 %: {n_claude_clear_500}/{n_total}")
    print(f"Gemini clears 0.2 %: {n_gemini_clear_500}/{n_total}")
    print()
    print(f"Claude — original gap ≥ 0.2 % AND stoch gap ≥ 0.2 % (held): {n_held}")
    print(f"Claude — original gap ≥ 0.2 % BUT stoch gap < 0.2 % (lost): {n_lost}")
    print()
    print(f"Wrote {args.out_csv}, {args.out_pdf}, {args.out_png}")


if __name__ == "__main__":
    main()
