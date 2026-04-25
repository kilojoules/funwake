"""Run every per-experiment aggregate.py and produce a single
results_summary.json + a printable comparison table.

This script does NOT auto-edit paper/short.tex. Inspect the summary,
update the paper by hand, regenerate figures with paper/make_figures.py.

Usage:
    pixi run python experiments/aggregate_all.py
"""
import json
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))


EXPERIMENTS = [
    ("A_multi_agent_seed",          "aggregate.py"),
    ("B_scaffold_cross",            "aggregate.py"),
    ("C_claude_fullopt_seeds",      "aggregate.py"),
    ("D_matrix_multi_seed",         "aggregate.py"),
    ("E_preregistered_random_search","aggregate.py"),
    ("F_extra_heldout",             "aggregate.py"),
    # G aggregate takes --run-dir; skip auto-call.
]


def run_one(name, script):
    path = os.path.join(HERE, name, script)
    if not os.path.exists(path):
        return {"status": "missing"}
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=600)
        if r.returncode != 0:
            return {"status": "error", "stderr": r.stderr[-1000:]}
    except Exception as e:
        return {"status": "exception", "error": str(e)}
    summary_path = os.path.join(HERE, name, "summary.json")
    if os.path.exists(summary_path):
        return {"status": "ok", "summary": json.load(open(summary_path))}
    return {"status": "ran_no_summary"}


def main():
    out = {}
    for name, script in EXPERIMENTS:
        print(f"[agg] {name}...")
        out[name] = run_one(name, script)
        print(f"      -> {out[name]['status']}")

    out_path = os.path.join(HERE, "results_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[agg] Wrote {out_path}")


if __name__ == "__main__":
    main()
