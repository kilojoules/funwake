#!/usr/bin/env python
"""Test optimizer on held-out farm. Reports PASS/FAIL only, no AEP.
Also updates the last entry in attempt_log.json with ROWP feasibility.

Usage:
    python tools/test_generalization.py <optimizer.py> [--problem path.json] [--timeout 120] [--log path/attempt_log.json]

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


def _update_last_log_entry(log_path, rowp_data):
    """Update the last entry in attempt_log.json with ROWP results."""
    if not log_path or not os.path.exists(log_path):
        return
    try:
        with open(log_path) as f:
            entries = json.load(f)
        if entries:
            entries[-1]["rowp_feasible"] = rowp_data.get("feasible")
            entries[-1]["rowp_time"] = rowp_data.get("time_s")
            with open(log_path, "w") as f:
                json.dump(entries, f, indent=2)
    except (json.JSONDecodeError, IOError, IndexError):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", help="Path to optimizer module")
    p.add_argument("--problem", default="results/problem_rowp.json")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--log", default=None,
                   help="Path to attempt_log.json (auto-detected from script path)")
    p.add_argument("--schedule-only", action="store_true",
                   help="Require schedule_fn() only")
    args = p.parse_args()

    # Auto-detect log path
    log_path = args.log
    if not log_path:
        script_dir = os.path.dirname(os.path.abspath(args.script))
        candidate = os.path.join(script_dir, "attempt_log.json")
        if "results_agent" in script_dir:
            log_path = candidate

    project_root = os.path.join(os.path.dirname(__file__), "..")
    harness = os.path.join(project_root, "playground", "harness.py")
    pixwake_src = os.path.join(project_root, "playground", "pixwake", "src")

    if not os.path.exists(args.problem):
        result = {"passed": False, "issues": ["Held-out problem not found"]}
        print(json.dumps(result))
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
            [sys.executable, harness, os.path.abspath(args.script)]
            + (["--schedule-only"] if args.schedule_only else []),
            capture_output=True, text=True, timeout=args.timeout,
            cwd=os.path.join(project_root, "playground"), env=env)
    except subprocess.TimeoutExpired:
        os.unlink(output_path) if os.path.exists(output_path) else None
        out = {"passed": False, "issues": ["Timeout"]}
        _update_last_log_entry(log_path, {"feasible": False, "time_s": args.timeout})
        print(json.dumps(out))
        return
    elapsed = time.time() - t0

    if result.returncode != 0:
        os.unlink(output_path) if os.path.exists(output_path) else None
        out = {"passed": False, "issues": ["Script errored on held-out farm"],
               "time_s": round(elapsed, 1)}
        _update_last_log_entry(log_path, {"feasible": False, "time_s": round(elapsed, 1)})
        print(json.dumps(out))
        return

    if not os.path.exists(output_path):
        out = {"passed": False, "issues": ["No output written"]}
        print(json.dumps(out))
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
        out = {
            "passed": passed,
            "feasible": passed,
            "issues": issues if issues else None,
            "time_s": round(elapsed, 1),
        }

        _update_last_log_entry(log_path, {"feasible": passed, "time_s": round(elapsed, 1)})
        print(json.dumps(out))
    except Exception as e:
        os.unlink(output_path) if os.path.exists(output_path) else None
        out = {"passed": False, "issues": [str(e)[:200]]}
        print(json.dumps(out))


if __name__ == "__main__":
    main()
