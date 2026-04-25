#!/usr/bin/env python
"""Evaluate optimizer scripts across DEI variants with different turbine counts.

Produces a generalization curve: AEP vs n_target for each script.

Usage:
    # Evaluate best scripts from each model
    python tools/eval_generalization.py results_agent_*/best_optimizer.py

    # Evaluate specific scripts on specific farms
    python tools/eval_generalization.py script1.py script2.py \
        --problems results/problem_dei_n30.json results/problem_dei_n50.json

    # Schedule-only mode
    python tools/eval_generalization.py iter_192.py --schedule-only
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description="Evaluate scripts across farm variants")
    p.add_argument("scripts", nargs="+", help="Optimizer scripts to evaluate")
    p.add_argument("--problems", nargs="*", default=None,
                   help="Problem JSONs (default: all results/problem_dei_n*.json)")
    p.add_argument("--timeout", type=int, default=180,
                   help="Timeout per evaluation (default: 180s)")
    p.add_argument("--schedule-only", action="store_true")
    p.add_argument("--output", default=None,
                   help="Output JSON file (default: print to stdout)")
    args = p.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    tools_dir = os.path.dirname(__file__)
    pixwake_src = os.path.join(project_root, "playground", "pixwake", "src")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH', '')}",
        "JAX_ENABLE_X64": "True",
        "HOME": os.environ.get("HOME", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }

    # Default: all DEI variants
    if args.problems is None:
        args.problems = sorted(glob.glob(
            os.path.join(project_root, "results", "problem_dei_n*.json")))
        # Also include ROWP for reference
        rowp = os.path.join(project_root, "results", "problem_rowp.json")
        if os.path.exists(rowp):
            args.problems.append(rowp)

    results = []

    for script in args.scripts:
        script_name = os.path.basename(script)
        script_dir = os.path.basename(os.path.dirname(script))
        label = f"{script_dir}/{script_name}"

        for problem in args.problems:
            prob_name = os.path.basename(problem).replace(".json", "")

            # Extract n_target from problem
            with open(problem) as f:
                prob_data = json.load(f)
            n_target = prob_data["n_target"]

            print(f"  {label} on {prob_name} (n={n_target})...",
                  end="", flush=True)

            t0 = time.time()
            try:
                cmd = [
                    sys.executable,
                    os.path.join(tools_dir, "run_optimizer.py"),
                    os.path.abspath(script),
                    "--problem", os.path.abspath(problem),
                    "--timeout", str(args.timeout),
                    "--log", "/dev/null",
                ]
                if args.schedule_only:
                    cmd.append("--schedule-only")

                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True,
                    timeout=args.timeout + 30,
                    env=env, cwd=project_root,
                )
                elapsed = time.time() - t0
                data = json.loads(result.stdout)

                entry = {
                    "script": label,
                    "problem": prob_name,
                    "n_target": n_target,
                    "aep_gwh": data.get("aep_gwh"),
                    "feasible": data.get("feasible"),
                    "time_s": round(elapsed, 1),
                }
                if "error" in data:
                    entry["error"] = data["error"][:200]
                    print(f" error: {data['error'][:50]}")
                else:
                    print(f" AEP={data['aep_gwh']:.1f} GWh, "
                          f"feasible={data.get('feasible')}, {elapsed:.0f}s")

            except subprocess.TimeoutExpired:
                entry = {
                    "script": label,
                    "problem": prob_name,
                    "n_target": n_target,
                    "error": f"Timeout after {args.timeout}s",
                }
                print(f" timeout")
            except Exception as e:
                entry = {
                    "script": label,
                    "problem": prob_name,
                    "n_target": n_target,
                    "error": str(e)[:200],
                }
                print(f" exception: {e}")

            results.append(entry)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(results)} results to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
