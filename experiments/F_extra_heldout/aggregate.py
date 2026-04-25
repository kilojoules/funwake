"""Aggregate L-shape results vs DEI/ROWP for cross-polygon ranking check.

Usage:
    pixi run python experiments/F_extra_heldout/aggregate.py
"""
import glob
import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))


def load_baseline():
    p = os.path.join(PROJECT_ROOT, "results", "baseline_lshape.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    baseline = load_baseline()
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "*_lshape.json"))):
        d = json.load(open(path))
        rows.append({
            "schedule": os.path.basename(path).replace("_lshape.json", ""),
            **(d.get("summary") or {}),
        })

    summary = {
        "baseline": baseline.get("best") if baseline else None,
        "schedules": rows,
    }
    if baseline and rows:
        for r in rows:
            if r.get("mean_aep") is not None and baseline.get("best"):
                r["delta_vs_baseline_gwh"] = round(r["mean_aep"] - baseline["best"], 3)

    out = os.path.join(HERE, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
