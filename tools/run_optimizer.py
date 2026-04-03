#!/usr/bin/env python
"""Run an optimizer on a farm and score it. Prints JSON to stdout.

Usage:
    python tools/run_optimizer.py <optimizer.py> [--problem path.json] [--timeout 60]

Output JSON:
    {"aep_gwh": 5540.72, "feasible": true, "time_s": 23.4, "baseline": 5540.72}
    or {"error": "..."}
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", help="Path to optimizer module")
    p.add_argument("--problem", default="results/problem_farm1.json")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--baselines", default="results/baselines.json")
    p.add_argument("--train-farm", default="1")
    args = p.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    harness = os.path.join(project_root, "playground", "harness.py")
    pixwake_src = os.path.join(project_root, "playground", "pixwake", "src")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH', '')}",
        "JAX_ENABLE_X64": "True",
        "FUNWAKE_PROBLEM": os.path.abspath(args.problem),
        "FUNWAKE_OUTPUT": output_path,
        "HOME": os.environ.get("HOME", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, harness, os.path.abspath(args.script)],
            capture_output=True, text=True, timeout=args.timeout,
            cwd=os.path.join(project_root, "playground"), env=env)
    except subprocess.TimeoutExpired:
        os.unlink(output_path) if os.path.exists(output_path) else None
        print(json.dumps({"error": f"Timeout after {args.timeout}s"}))
        return
    elapsed = time.time() - t0

    if result.returncode != 0:
        os.unlink(output_path) if os.path.exists(output_path) else None
        print(json.dumps({"error": result.stderr[-1000:]}))
        return

    if not os.path.exists(output_path):
        print(json.dumps({"error": "No output written"}))
        return

    # Score via ProblemBenchmark
    try:
        sys.path.insert(0, pixwake_src)
        sys.path.insert(0, os.path.join(project_root, "benchmarks"))
        from dei_layout import ProblemBenchmark

        with open(output_path) as f:
            layout = json.load(f)
        os.unlink(output_path)

        bm = ProblemBenchmark(os.path.abspath(args.problem))
        aep = bm.score(layout["x"], layout["y"])
        feas = bm.check_feasibility(layout["x"], layout["y"])
        feasible = feas["spacing_ok"] and feas["boundary_ok"]

        # Load baseline
        baseline = 0
        try:
            with open(os.path.join(project_root, args.baselines)) as f:
                baselines = json.load(f)
            baseline = baselines.get(args.train_farm, {}).get("aep_gwh", 0)
        except FileNotFoundError:
            pass

        print(json.dumps({
            "aep_gwh": round(aep, 2),
            "feasible": feasible,
            "time_s": round(elapsed, 1),
            "baseline": round(baseline, 2),
            "gap": round(aep - baseline, 2),
        }))
    except Exception as e:
        os.unlink(output_path) if os.path.exists(output_path) else None
        print(json.dumps({"error": str(e)[:500]}))


if __name__ == "__main__":
    main()
