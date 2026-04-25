#!/usr/bin/env python
"""Collect results from parallel baseline jobs into a single JSON.

Usage:
    python tools/collect_baselines.py
"""
import glob
import json
import os


def main():
    log_dir = os.path.join(os.path.dirname(__file__), "..", "lumi", "logs", "baselines")
    results = {}

    for pattern_name in ["problem_dei_n30", "problem_dei_n40", "problem_dei_n50",
                          "problem_dei_n60", "problem_dei_n70", "problem_dei_n80",
                          "problem_rowp"]:
        files = glob.glob(os.path.join(log_dir, f"{pattern_name}_*.out"))
        aeps = []
        feasible_count = 0
        total_time = 0

        for f in files:
            try:
                with open(f) as fh:
                    content = fh.read().strip()
                    if not content:
                        continue
                    d = json.loads(content)
                    aeps.append(d["aep"])
                    if d["feasible"]:
                        feasible_count += 1
                    total_time += d["time"]
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                continue

        if not aeps:
            print(f"{pattern_name}: no results")
            continue

        best = max(aeps)
        feasible_aeps = [a for a, f in zip(aeps, [True]*len(aeps))]
        results[pattern_name] = {
            "n_starts": len(aeps),
            "best_aep": round(best, 2),
            "mean_aep": round(sum(aeps) / len(aeps), 2),
            "feasible_rate": round(feasible_count / len(aeps), 3),
            "total_time_s": round(total_time, 1),
        }
        print(f"{pattern_name}: {len(aeps)} starts, best={best:.1f}, "
              f"mean={sum(aeps)/len(aeps):.1f}, feas={100*feasible_count/len(aeps):.0f}%")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "baselines_500start.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
