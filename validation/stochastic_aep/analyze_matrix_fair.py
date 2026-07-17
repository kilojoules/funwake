"""Analyse fair-baseline matrix (decay+ES SGD vs Claude vs Gemini, 3 seeds).

For each cell, compute per-cell:
  - aep_mean ± std across seeds, per schedule
  - gap_vs_decay_es_pct = (sched_mean - baseline_mean) / baseline_mean * 100
  - per-cell spread = sched_std / baseline_mean * 100
  - per-cell ES-trigger info (decay_es_baseline only; deterministic per cell)

Pre-registered threshold:
  cell shows real advantage iff gap_mean ≥ 0.2% AND gap > spread (the
  multi-seed spread, not the borrowed 0.022% floor).

Outputs:
  - matrix_fair.csv (per-cell table)
  - matrix_fair.{png,pdf} (figure)
  - REPORT_MATRIX_FAIR.md companion text
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = "/Users/julianquick/portfolio_copy/funwake"


def collect_per_cell(raw):
    """Index runs into nested dict[cell_key][schedule] = list-of-(aep, ...)."""
    cells = {}
    for r in raw["runs"]:
        if "error" in r:
            continue
        # cell_key from path: e.g. results/matrix/problem_dei_n30_roseuniform.json
        base = os.path.basename(r["cell_path"])
        parts = base.replace("problem_", "").replace(".json", "")
        # parts = "dei_n30_roseuniform"  ->  cell_key = "dei_n30_roseuniform"
        cell_key = parts
        # Parse farm/n/rose
        farm = parts.split("_")[0]
        n = int(parts.split("_n")[1].split("_")[0])
        rose = parts.split("rose")[1]
        if cell_key not in cells:
            cells[cell_key] = {
                "farm": farm, "n": n, "rose": rose,
                "schedules": {},
            }
        sch_name = r["schedule"]
        if sch_name not in cells[cell_key]["schedules"]:
            cells[cell_key]["schedules"][sch_name] = []
        cells[cell_key]["schedules"][sch_name].append(r)
    return cells


def summarize_cell(cell):
    """Per-schedule stats + gap-over-decay-es-baseline."""
    out = {"farm": cell["farm"], "n": cell["n"], "rose": cell["rose"], "schedules": {}}
    base_runs = cell["schedules"].get("decay_es_baseline", [])
    base_aeps = np.array([r["aep_det_gwh"] for r in base_runs])
    base_mean = float(base_aeps.mean()) if len(base_aeps) else None
    base_std = float(base_aeps.std(ddof=1)) if len(base_aeps) > 1 else 0.0
    out["baseline"] = {
        "mean": base_mean, "std": base_std,
        "n_seeds": len(base_aeps),
        "bp_final_max": max((r["bp_final"] for r in base_runs), default=None),
    }
    for sch_name, runs in cell["schedules"].items():
        aeps = np.array([r["aep_det_gwh"] for r in runs])
        if not len(aeps):
            continue
        mean = float(aeps.mean())
        std = float(aeps.std(ddof=1)) if len(aeps) > 1 else 0.0
        gap_pct = (mean - base_mean) / base_mean * 100 if base_mean else None
        spread_pct = std / base_mean * 100 if base_mean else None
        out["schedules"][sch_name] = {
            "aep_mean": mean, "aep_std": std,
            "n_seeds": len(aeps),
            "gap_pct": gap_pct,
            "spread_pct": spread_pct,
            "bp_final_max": float(max(r["bp_final"] for r in runs)),
        }
    return out


def make_figure(summary_by_cell, out_pdf, out_png):
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
    series = [
        ("claude_iter192", "Claude", "#c73a1a", "s"),
        ("gemini_iter192", "Gemini", "#1a73c7", "o"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for i, farm in enumerate(farms):
        for j, rose in enumerate(roses):
            ax = axes[i, j]
            ax.axhline(0.0, color="black", linewidth=1.0, zorder=2)
            ax.axhline(0.2, color="gray", linewidth=0.8, linestyle=":", zorder=2,
                       label="0.2 % bar" if (i, j) == (0, 0) else None)
            for sch_name, leg, color, marker in series:
                xs, gaps, spreads = [], [], []
                for n in ns:
                    cell_key = f"{farm}_n{n}_rose{rose}"
                    s = summary_by_cell.get(cell_key)
                    if not s:
                        continue
                    sch = s["schedules"].get(sch_name)
                    if not sch or sch["gap_pct"] is None:
                        continue
                    xs.append(n)
                    gaps.append(sch["gap_pct"])
                    spreads.append(sch["spread_pct"] or 0.0)
                if xs:
                    ax.errorbar(xs, gaps, yerr=spreads, marker=marker, linestyle="-",
                                color=color, linewidth=1.6, markersize=7, capsize=3,
                                label=f"{leg}" if (i, j) == (0, 0) else None)
            if i == 0:
                ax.set_title(rose_titles[rose], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{farm_labels[farm]}\nGap over decay+ES baseline (%)",
                              fontsize=9)
            if i == 1:
                ax.set_xlabel("Turbine count $N$")
            ax.set_xticks(ns)
    axes[0, 0].legend(loc="best", fontsize=8, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    plt.close()


def write_csv(summary_by_cell, out_csv):
    rows = []
    for cell_key in sorted(summary_by_cell.keys()):
        s = summary_by_cell[cell_key]
        sched_c = s["schedules"].get("claude_iter192", {})
        sched_g = s["schedules"].get("gemini_iter192", {})
        row = {
            "cell": cell_key,
            "farm": s["farm"], "n": s["n"], "rose": s["rose"],
            "baseline_mean": s["baseline"]["mean"],
            "baseline_std": s["baseline"]["std"],
            "baseline_n_seeds": s["baseline"]["n_seeds"],
            "baseline_bp_max": s["baseline"]["bp_final_max"],
            "claude_mean": sched_c.get("aep_mean"),
            "claude_std": sched_c.get("aep_std"),
            "claude_gap_pct": sched_c.get("gap_pct"),
            "claude_spread_pct": sched_c.get("spread_pct"),
            "gemini_mean": sched_g.get("aep_mean"),
            "gemini_std": sched_g.get("aep_std"),
            "gemini_gap_pct": sched_g.get("gap_pct"),
            "gemini_spread_pct": sched_g.get("spread_pct"),
        }
        # Real-advantage flag: gap ≥ 0.2 % AND gap > 1*spread
        for name in ("claude", "gemini"):
            gap = row[f"{name}_gap_pct"]
            spread = row[f"{name}_spread_pct"]
            row[f"{name}_clears_fair_bar"] = (
                gap is not None and spread is not None
                and gap >= 0.2 and gap > spread
            )
        rows.append(row)
    keys = list(rows[0].keys())
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="validation/stochastic_aep/matrix_fair.json")
    p.add_argument("--out-csv",
                   default="validation/stochastic_aep/matrix_fair.csv")
    p.add_argument("--out-pdf",
                   default="validation/stochastic_aep/matrix_fair.pdf")
    p.add_argument("--out-png",
                   default="validation/stochastic_aep/matrix_fair.png")
    args = p.parse_args()

    raw = json.load(open(args.raw))
    cells = collect_per_cell(raw)
    summary = {k: summarize_cell(v) for k, v in cells.items()}
    rows = write_csv(summary, args.out_csv)
    make_figure(summary, args.out_pdf, args.out_png)

    n_claude_clear = sum(r["claude_clears_fair_bar"] for r in rows)
    n_gemini_clear = sum(r["gemini_clears_fair_bar"] for r in rows)
    print(f"Cells: {len(rows)}")
    print(f"Claude clears fair bar (gap ≥ 0.2 AND > spread): {n_claude_clear}/{len(rows)}")
    print(f"Gemini clears fair bar (gap ≥ 0.2 AND > spread): {n_gemini_clear}/{len(rows)}")
    print(f"Wrote {args.out_csv} {args.out_pdf} {args.out_png}")


if __name__ == "__main__":
    main()
