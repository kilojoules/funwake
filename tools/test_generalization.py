#!/usr/bin/env python
"""Test optimizer on held-out farm. Reports PASS/FAIL only, no AEP.

Usage:
    python tools/test_generalization.py <optimizer.py> [--problem path.json] [--timeout 120]

Output JSON:
    {"passed": true, "feasible": true, "time_s": 45.2}
    or {"passed": false, "issues": ["SPACING VIOLATION", ...]}
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
    p.add_argument("--problem", default="results/problem_rowp.json")
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    harness = os.path.join(project_root, "playground", "harness.py")
    pixwake_src = os.path.join(project_root, "playground", "pixwake", "src")

    if not os.path.exists(args.problem):
        print(json.dumps({"passed": False, "issues": ["Held-out problem not found"]}))
        return

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
        print(json.dumps({"passed": False, "issues": ["Timeout"]}))
        return
    elapsed = time.time() - t0

    if result.returncode != 0:
        os.unlink(output_path) if os.path.exists(output_path) else None
        # Don't leak held-out details in error
        print(json.dumps({"passed": False, "issues": ["Script errored on held-out farm"],
                          "time_s": round(elapsed, 1)}))
        return

    if not os.path.exists(output_path):
        print(json.dumps({"passed": False, "issues": ["No output written"]}))
        return

    # Check feasibility — no AEP reported
    try:
        sys.path.insert(0, pixwake_src)
        sys.path.insert(0, os.path.join(project_root, "benchmarks"))
        from dei_layout import ProblemBenchmark

        with open(output_path) as f:
            layout = json.load(f)
        os.unlink(output_path)

        bm = ProblemBenchmark(os.path.abspath(args.problem))
        feas = bm.check_feasibility(layout["x"], layout["y"])

        issues = []
        if len(layout["x"]) != bm.n_target:
            issues.append("WRONG TURBINE COUNT")
        if not feas["spacing_ok"]:
            issues.append("SPACING VIOLATION")
        if not feas["boundary_ok"]:
            issues.append("BOUNDARY VIOLATION")

        passed = len(issues) == 0
        print(json.dumps({
            "passed": passed,
            "feasible": passed,
            "issues": issues if issues else None,
            "time_s": round(elapsed, 1),
        }))
    except Exception as e:
        os.unlink(output_path) if os.path.exists(output_path) else None
        print(json.dumps({"passed": False, "issues": [str(e)[:200]]}))


if __name__ == "__main__":
    main()
