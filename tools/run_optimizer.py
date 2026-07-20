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


_META_PATTERNS = {
    "hypothesis": r"HYPOTHESIS\s*:\s*(.+)",
    "axis":       r"AXIS\s*:\s*(.+)",
    "lesson":     r"LESSON\s*:\s*(.+)",
}


def _parse_metadata(source: str):
    """Extract HYPOTHESIS / AXIS / LESSON lines from a script's docstrings.

    Returns (hypothesis, axis, lesson) — any of which may be None.
    Matches the first occurrence of each tag, case-sensitive, anywhere
    in the source (typically inside a top-level triple-quoted docstring).
    """
    import re
    out = {}
    for key, pat in _META_PATTERNS.items():
        m = re.search(pat, source)
        if m:
            value = m.group(1).strip().strip('"').strip("'")
            # Truncate at next blank line or another METAKEY line
            value = re.split(r"\n\s*\n|\n\s*[A-Z]+\s*:", value)[0].strip()
            if value:
                out[key] = value[:500]
    return out.get("hypothesis"), out.get("axis"), out.get("lesson")


def _classify_source(source: str, schedule_only: bool):
    """Classify the script into strategy taxonomy families, best-effort.

    Returns a sorted list of family names, or None if classification
    infrastructure isn't available. Loads the taxonomy module directly
    to avoid triggering runners/__init__.py (which imports heavy LLM
    SDK deps that may not be importable in the harness subprocess).
    """
    try:
        import importlib.util
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tax_path = os.path.join(project_root, "runners", "strategy_taxonomy.py")
        spec = importlib.util.spec_from_file_location("_funwake_taxonomy", tax_path)
        mod = importlib.util.module_from_spec(spec)
        # Register before exec_module — dataclass machinery looks up
        # cls.__module__ in sys.modules and errors with AttributeError
        # if the module isn't registered yet.
        sys.modules["_funwake_taxonomy"] = mod
        spec.loader.exec_module(mod)
        mode = "schedule" if schedule_only else "fullopt"
        return sorted(mod.classify(source, mode))
    except Exception as e:
        return [f"_classify_error:{type(e).__name__}:{str(e)[:80]}"]


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


def _run_harness(script_path, problem_path, schedule_only, timeout,
                 harness, pixwake_src, project_root, seed):
    """Run the harness on a single problem and return (layout_dict, elapsed_s)
    or raise RuntimeError with the failure reason."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH', '')}",
        "JAX_ENABLE_X64": "True",
        "FUNWAKE_PROBLEM": os.path.abspath(problem_path),
        "FUNWAKE_OUTPUT": output_path,
        "FUNWAKE_SEED": str(seed),
        "HOME": os.environ.get("HOME", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, harness, os.path.abspath(script_path)]
            + (["--schedule-only"] if schedule_only else []),
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.join(project_root, "playground"), env=env)
    except subprocess.TimeoutExpired:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError(f"Timeout after {timeout}s")
    elapsed = time.time() - t0

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError(result.stderr[-500:])

    if not os.path.exists(output_path):
        raise RuntimeError("No output written")

    with open(output_path) as f:
        layout = json.load(f)
    os.unlink(output_path)
    return layout, elapsed


def _score_layout(layout, problem_path):
    """Score a layout against a problem. Returns (aep_gwh, feasible)."""
    from dei_layout import ProblemBenchmark
    bm = ProblemBenchmark(os.path.abspath(problem_path))
    aep = bm.score(layout["x"], layout["y"])
    feas = bm.check_feasibility(layout["x"], layout["y"])
    feasible = feas["spacing_ok"] and feas["boundary_ok"]
    return aep, feasible


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", help="Path to optimizer module")
    p.add_argument("--problem", default="results/problem_farm1.json")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--baselines", default="results/baselines.json")
    p.add_argument("--train-farm", default="1")
    p.add_argument("--log", default=None,
                   help="Path to attempt_log.json (auto-detected from script path if not set)")
    p.add_argument("--schedule-only", action="store_true",
                   help="Require schedule_fn() only — reject optimize()")
    p.add_argument("--seed", type=int, default=0,
                   help="Initialization seed (grid subsampling in the skeleton)")
    p.add_argument("--stress-problem", default=None,
                   help="Second problem to score on (adversarial stress cell). "
                        "When set, scores on this problem after the main one "
                        "and reports stress_aep_gwh, stress_feasible, stress_gap, "
                        "and min_gap = min(gap, stress_gap) in stdout/log.")
    p.add_argument("--stress-baseline", type=float, default=None,
                   help="Baseline AEP for the stress problem (GWh). If unset, "
                        "stress_gap is reported as None.")
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
    pixwake_src = os.path.join(project_root, "dependencies", "pixwake", "src")

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

    sys.path.insert(0, pixwake_src)
    sys.path.insert(0, os.path.join(project_root, "benchmarks"))

    # Train evaluation
    try:
        layout, elapsed = _run_harness(
            args.script, args.problem, args.schedule_only,
            args.timeout, harness, pixwake_src, project_root, args.seed)
        aep, feasible = _score_layout(layout, args.problem)
    except RuntimeError as e:
        entry = {"attempt": attempt_num, "timestamp": time.time(),
                 "error": str(e)[:500]}
        _append_to_log(log_path, entry)
        print(json.dumps({"error": str(e)[:500]}))
        return
    except Exception as e:
        entry = {"attempt": attempt_num, "timestamp": time.time(),
                 "error": str(e)[:500]}
        _append_to_log(log_path, entry)
        print(json.dumps({"error": str(e)[:500]}))
        return

    # Load baseline
    baseline = 0
    try:
        with open(os.path.join(project_root, args.baselines)) as f:
            baselines = json.load(f)
        baseline = baselines.get(args.train_farm, {}).get("aep_gwh", 0)
    except FileNotFoundError:
        pass

    # Classify strategy + parse agent-authored metadata from the source
    hypothesis = axis = lesson = None
    families = None
    try:
        code = open(args.script).read()
        strategy = "sgd_solve" if "topfarm_sgd_solve" in code else "custom"
        hypothesis, axis, lesson = _parse_metadata(code)
        families = _classify_source(code, args.schedule_only)
    except IOError:
        strategy = "unknown"

    gap = round(aep - baseline, 2) if baseline else None
    output = {
        "aep_gwh": round(aep, 2),
        "feasible": feasible,
        "time_s": round(elapsed, 1),
        "baseline": round(baseline, 2),
        "gap": gap,
    }
    entry = {
        "attempt": attempt_num,
        "timestamp": time.time(),
        "train_aep": round(aep, 2),
        "train_feasible": feasible,
        "train_time": round(elapsed, 1),
        "train_baseline": round(baseline, 2),
        "strategy": strategy,
    }

    # Optional stress evaluation on a second problem
    if args.stress_problem:
        try:
            s_layout, s_elapsed = _run_harness(
                args.script, args.stress_problem, args.schedule_only,
                args.timeout, harness, pixwake_src, project_root, args.seed)
            s_aep, s_feasible = _score_layout(s_layout, args.stress_problem)
            s_gap = (round(s_aep - args.stress_baseline, 2)
                     if args.stress_baseline is not None else None)
            output["stress_aep_gwh"] = round(s_aep, 2)
            output["stress_feasible"] = s_feasible
            output["stress_time_s"] = round(s_elapsed, 1)
            output["stress_baseline"] = (round(args.stress_baseline, 2)
                                          if args.stress_baseline is not None else None)
            output["stress_gap"] = s_gap
            entry["stress_aep"] = round(s_aep, 2)
            entry["stress_feasible"] = s_feasible
            entry["stress_time"] = round(s_elapsed, 1)
            entry["stress_baseline"] = (round(args.stress_baseline, 2)
                                         if args.stress_baseline is not None else None)
            # Deployment criterion: worst of the two gaps (only meaningful if
            # both baselines and both feasible).
            if (gap is not None and s_gap is not None
                    and feasible and s_feasible):
                output["min_gap"] = min(gap, s_gap)
                entry["min_gap"] = min(gap, s_gap)
            else:
                output["min_gap"] = None
                entry["min_gap"] = None
        except RuntimeError as e:
            output["stress_error"] = str(e)[:500]
            entry["stress_error"] = str(e)[:500]
            output["min_gap"] = None
            entry["min_gap"] = None
        except Exception as e:
            output["stress_error"] = str(e)[:500]
            entry["stress_error"] = str(e)[:500]
            output["min_gap"] = None
            entry["min_gap"] = None

    if hypothesis:
        entry["hypothesis"] = hypothesis
    if axis:
        entry["axis"] = axis
    if lesson:
        entry["lesson"] = lesson
    if families:
        entry["families"] = families
    _append_to_log(log_path, entry)

    print(json.dumps(output))


if __name__ == "__main__":
    main()
