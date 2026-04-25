"""Evaluate a schedule_fn script across multiple initialization seeds.

Each seed controls the skeleton's grid-subsampling PRNG, so the
resulting AEP variance captures sensitivity to initialization only
(NOT agent-run variance — that requires re-running the full agent loop
and is out of scope here).

Usage:
    python tools/multi_seed_eval.py <schedule.py> \\
        --problem playground/problem.json \\
        --seeds 0 1 2 3 4 \\
        --output results_multi_seed/<name>.json

    # Shorthand: a range expands to --seeds 0 1 ... 9
    python tools/multi_seed_eval.py <schedule.py> \\
        --problem results/problem_rowp.json \\
        --n-seeds 5 \\
        --output results_multi_seed/<name>_rowp.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys


def score(script_path, problem_path, seed, timeout, schedule_only):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tools_dir = os.path.join(project_root, "tools")
    cmd = [
        sys.executable,
        os.path.join(tools_dir, "run_optimizer.py"),
        os.path.abspath(script_path),
        "--problem", os.path.abspath(problem_path),
        "--timeout", str(timeout),
        "--seed", str(seed),
        "--log", "/dev/null",
    ]
    if schedule_only:
        cmd.append("--schedule-only")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout + 30, cwd=project_root)
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)[:200]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", help="schedule_fn script to evaluate")
    p.add_argument("--problem", required=True,
                   help="problem JSON (training or held-out)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="explicit seed list (e.g. --seeds 0 1 2 3 4)")
    p.add_argument("--n-seeds", type=int, default=None,
                   help="run seeds 0..n-1 (ignored if --seeds given)")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--schedule-only", action="store_true", default=True)
    p.add_argument("--output", required=True,
                   help="output JSON with per-seed results")
    args = p.parse_args()

    if args.seeds is None:
        if args.n_seeds is None:
            args.n_seeds = 5
        args.seeds = list(range(args.n_seeds))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = {
        "script": os.path.abspath(args.script),
        "problem": os.path.abspath(args.problem),
        "seeds": args.seeds,
        "per_seed": {},
    }

    aeps = []
    feasibles = []
    for seed in args.seeds:
        print(f"[seed {seed}] running...", flush=True)
        out = score(args.script, args.problem, seed,
                    args.timeout, args.schedule_only)
        results["per_seed"][str(seed)] = out
        if "aep_gwh" in out:
            aeps.append(out["aep_gwh"])
            feasibles.append(bool(out.get("feasible")))
            print(f"[seed {seed}] AEP={out['aep_gwh']:.2f} "
                  f"feas={out.get('feasible')}", flush=True)
        else:
            print(f"[seed {seed}] ERROR: {out.get('error', 'unknown')[:120]}",
                  flush=True)

    if aeps:
        results["summary"] = {
            "n": len(aeps),
            "mean_aep": round(statistics.mean(aeps), 3),
            "std_aep": round(statistics.pstdev(aeps), 3) if len(aeps) > 1 else 0.0,
            "min_aep": round(min(aeps), 3),
            "max_aep": round(max(aeps), 3),
            "feasible_fraction": round(sum(feasibles) / len(feasibles), 3),
        }
        print()
        print(f"Summary: {results['summary']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
