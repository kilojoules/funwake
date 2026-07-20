"""Replay every iter_*.py in a results directory across the 64-cell
matrix and write per-(script, cell) AEP/feasibility rows to a JSON file.

Modeled on tools/eval_matrix_schedules.py but globs the directory
instead of hard-coding script paths. Used for Phase 2 of the adversary
analysis: after an adversarial-selection agent run, replay every saved
candidate through the full matrix and compare argmax-train vs
argmax-worst-cell selection.

Usage:
    pixi run python tools/replay_dir_matrix.py \\
        --scripts-dir results_agent_schedule_adversarial \\
        --output results/matrix/adversarial_replay.json
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score_one(script_rel, problem_rel, timeout=180):
    env = {
        **os.environ,
        "JAX_ENABLE_X64": "True",
        "PYTHONPATH": (
            os.path.join(PROJECT_ROOT, "dependencies", "pixwake", "src")
            + ":" + os.environ.get("PYTHONPATH", "")
        ),
    }
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
        os.path.join(PROJECT_ROOT, script_rel),
        "--problem", os.path.join(PROJECT_ROOT, problem_rel),
        "--timeout", str(timeout),
        "--log", "/dev/null",
        "--schedule-only",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 60, env=env,
                           cwd=PROJECT_ROOT)
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:200]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scripts-dir", required=True,
                   help="Directory containing iter_*.py schedule scripts")
    p.add_argument("--output", required=True,
                   help="Output JSON path (resumable)")
    p.add_argument("--manifest",
                   default="results/matrix/manifest.json",
                   help="Matrix manifest (default: results/matrix/manifest.json)")
    p.add_argument("--timeout", type=int, default=180,
                   help="Per-evaluation timeout in seconds")
    p.add_argument("--max-n", type=int, default=80,
                   help="Skip matrix cells with N > this (default: 80, since "
                        "N=200/300 are too expensive for batch replay)")
    args = p.parse_args()

    scripts = sorted(glob.glob(os.path.join(args.scripts_dir, "iter_*.py")),
                     key=lambda p: int(re.search(r"iter_(\d+)\.py$", p).group(1)))
    if not scripts:
        print(f"No iter_*.py scripts found in {args.scripts_dir}", file=sys.stderr)
        sys.exit(1)

    manifest = json.load(open(os.path.join(PROJECT_ROOT, args.manifest)))
    cells = [c for c in manifest["cells"] if c["n"] <= args.max_n]

    existing = {}
    if os.path.exists(args.output):
        existing = json.load(open(args.output))
    results = dict(existing)

    total = len(scripts) * len(cells)
    done = sum(1 for k in results if "aep_gwh" in results[k])
    print(f"Scripts: {len(scripts)}, cells: {len(cells)}, total: {total}, "
          f"already done: {done}")

    t_start = time.time()
    for script_abs in scripts:
        script_rel = os.path.relpath(script_abs, PROJECT_ROOT)
        m = re.search(r"iter_(\d+)\.py$", script_rel)
        label = f"iter_{m.group(1)}"
        for cell in cells:
            key = (f"{label}|{cell['farm']}_n{cell['n']}_"
                   f"rose{cell['rose']}")
            if key in results and "aep_gwh" in results[key]:
                continue
            t0 = time.time()
            r = score_one(script_rel, cell["path"], timeout=args.timeout)
            elapsed = time.time() - t0
            entry = {
                "label": label,
                "script": script_rel,
                "farm": cell["farm"],
                "n": cell["n"],
                "rose": cell["rose"],
                "wall_time_s": round(elapsed, 1),
            }
            if "aep_gwh" in r:
                entry["aep_gwh"] = r["aep_gwh"]
                entry["feasible"] = r["feasible"]
            else:
                entry["error"] = r.get("error", "unknown")
            results[key] = entry
            done += 1
            # Persist incrementally so a job timeout still leaves progress
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{done}/{total}] {key} -> "
                  f"{r.get('aep_gwh', r.get('error', '?'))} "
                  f"({elapsed:.1f}s, total {time.time()-t_start:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
