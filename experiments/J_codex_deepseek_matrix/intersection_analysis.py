"""Cross-run intersection analysis for the codex+deepseek matrix.

Reads matrix_bestof.json and computes:
- Cells where ALL 12 runs produced >=1 feasible result (full intersection)
- Per-run / per-(provider, mode) mean AEP across the intersection
- Cell-level head-to-head winner counts (codex vs deepseek)

Usage:
    pixi run python experiments/J_codex_deepseek_matrix/intersection_analysis.py
"""
import collections
import json
import os
import statistics


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    d = json.load(open(os.path.join(HERE, "matrix_bestof.json")))

    by_cell = collections.defaultdict(dict)  # cell -> {label: best_of_n}
    for k, v in d.items():
        label, cell = k.split("|", 1)
        by_cell[cell][label] = v["best_of_n"]

    labels = sorted({label for k in d.keys() for label, _ in [k.split("|", 1)]})
    full_cov = [c for c, lbls in by_cell.items() if len(lbls) == len(labels)]

    out = {
        "n_runs": len(labels),
        "n_cells_any_data": len(by_cell),
        "n_cells_full_intersection": len(full_cov),
        "intersection_cells": sorted(full_cov),
    }

    # Per-run stats
    per_run = {}
    for label in labels:
        aeps = [by_cell[c][label] for c in full_cov]
        per_run[label] = {
            "mean": round(statistics.mean(aeps), 3),
            "std":  round(statistics.pstdev(aeps), 3) if len(aeps) > 1 else 0,
            "min":  round(min(aeps), 3),
            "max":  round(max(aeps), 3),
        }
    out["per_run"] = per_run

    # Per-(provider, mode)
    groups = collections.defaultdict(list)
    for label in labels:
        parts = label.split()
        if len(parts) >= 2:
            groups[f"{parts[0]} {parts[1]}"].extend([by_cell[c][label] for c in full_cov])
    out["per_group"] = {
        g: {"mean": round(statistics.mean(aeps), 3),
            "std":  round(statistics.pstdev(aeps), 3),
            "n_obs": len(aeps)}
        for g, aeps in groups.items()
    }

    # Cell-level head-to-head: codex_best vs deepseek_best (any mode)
    cells_winner = collections.Counter()
    cell_diffs = []
    for c in full_cov:
        cx = max(by_cell[c][f"codex {m} run{n}"] for m in ("sched", "fullopt") for n in (1,2,3))
        ds = max(by_cell[c][f"deepseek {m} run{n}"] for m in ("sched", "fullopt") for n in (1,2,3))
        diff = cx - ds
        cell_diffs.append(diff)
        if diff > 0: cells_winner["codex"] += 1
        elif diff < 0: cells_winner["deepseek"] += 1
        else: cells_winner["tie"] += 1
    out["head_to_head_codex_vs_deepseek"] = {
        "codex_wins": cells_winner["codex"],
        "deepseek_wins": cells_winner["deepseek"],
        "ties": cells_winner["tie"],
        "mean_codex_minus_deepseek": round(statistics.mean(cell_diffs), 3),
        "median_codex_minus_deepseek": round(statistics.median(cell_diffs), 3),
    }

    out_path = os.path.join(HERE, "intersection_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
