#!/usr/bin/env python
"""Run an optimizer on a farm and score it. Prints JSON to stdout
AND appends to attempt_log.json for durable logging.

Usage:
    python tools/run_optimizer.py <optimizer.py> [--problem path.json] [--timeout 60] [--log path/attempt_log.json]

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


def _append_to_log(log_path, entry):
    """Append an entry to the attempt log JSON file."""
    if not log_path:
        return
    existing = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []
    existing.append(entry)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)


def _count_attempts(log_path):
    """Count existing attempts to determine next attempt number."""
    if not log_path or not os.path.exists(log_path):
        return 0
    try:
        with open(log_path) as f:
            return len(json.load(f))
    except (json.JSONDecodeError, IOError):
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", help="Path to optimizer module")
    p.add_argument("--problem", default="results/problem_farm1.json")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--baselines", default="results/baselines.json")
    p.add_argument("--train-farm", default="1")
    p.add_argument("--log", default=None,
                   help="Path to attempt_log.json (auto-detected from script path if not set)")
    p.add_argument("--schedule-only", action="store_true",
                   help="Require schedule_fn() only — reject optimize()")
    args = p.parse_args()

    # Auto-detect log path from script directory
    log_path = args.log
    if not log_path:
        script_dir = os.path.dirname(os.path.abspath(args.script))
        candidate = os.path.join(script_dir, "attempt_log.json")
        # Only auto-log if the script is in a results_agent_* directory
        if "results_agent" in script_dir:
            log_path = candidate

    attempt_num = _count_attempts(log_path) + 1

    project_root = os.path.join(os.path.dirname(__file__), "..")
    harness = os.path.join(project_root, "playground", "harness.py")
    pixwake_src = os.path.join(project_root, "playground", "pixwake", "src")

    # Safety check before execution
    try:
        sys.path.insert(0, project_root)
        from sandbox import check_code_safety
        with open(args.script) as f:
            code = f.read()
        safe, reason = check_code_safety(code)
        if not safe:
            entry = {"attempt": attempt_num, "timestamp": time.time(),
                     "error": f"Sandbox blocked: {reason}"}
            _append_to_log(log_path, entry)
            print(json.dumps({"error": f"Sandbox blocked: {reason}"}))
            return
    except ImportError:
        pass  # sandbox module not available

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
        entry = {"attempt": attempt_num, "timestamp": time.time(),
                 "error": f"Timeout after {args.timeout}s"}
        _append_to_log(log_path, entry)
        print(json.dumps({"error": f"Timeout after {args.timeout}s"}))
        return
    elapsed = time.time() - t0

    if result.returncode != 0:
        os.unlink(output_path) if os.path.exists(output_path) else None
        err = result.stderr[-1000:]
        entry = {"attempt": attempt_num, "timestamp": time.time(),
                 "error": err[:500]}
        _append_to_log(log_path, entry)
        print(json.dumps({"error": err}))
        return

    if not os.path.exists(output_path):
        entry = {"attempt": attempt_num, "timestamp": time.time(),
                 "error": "No output written"}
        _append_to_log(log_path, entry)
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

        # Classify strategy
        try:
            code = open(args.script).read()
            strategy = "sgd_solve" if "topfarm_sgd_solve" in code else "custom"
        except IOError:
            strategy = "unknown"

        output = {
            "aep_gwh": round(aep, 2),
            "feasible": feasible,
            "time_s": round(elapsed, 1),
            "baseline": round(baseline, 2),
            "gap": round(aep - baseline, 2),
        }

        # Log attempt
        entry = {
            "attempt": attempt_num,
            "timestamp": time.time(),
            "train_aep": round(aep, 2),
            "train_feasible": feasible,
            "train_time": round(elapsed, 1),
            "train_baseline": round(baseline, 2),
            "strategy": strategy,
        }
        _append_to_log(log_path, entry)

        print(json.dumps(output))
    except Exception as e:
        os.unlink(output_path) if os.path.exists(output_path) else None
        entry = {"attempt": attempt_num, "timestamp": time.time(),
                 "error": str(e)[:500]}
        _append_to_log(log_path, entry)
        print(json.dumps({"error": str(e)[:500]}))


if __name__ == "__main__":
    main()
