"""Aggregate frontier-breadth result.

Picks best feasible script in the run dir, scores on ROWP if missing,
classifies structure (reuses the regex set from Experiment A).

Usage:
    pixi run python experiments/G_frontier_breadth/aggregate.py \
        --run-dir results_agent_claude-opus-4-7_sched
"""
import argparse
import json
import os
import re
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))


STRUCTURE_RULES = [
    ("dual_gaussian_bumps",  r"exp\([^)]*-[^)]*\(t\s*-\s*0\.[57]"),
    ("cosine_restart",       r"cos\([^)]*restart|n_cycles|n_restarts"),
    ("cyclic_betas",         r"beta1\s*=.*cos|beta2\s*=.*cos"),
    ("alpha_squeeze",        r"alpha\s*=.*5e6|alpha.*\*\*\s*[3-9]"),
]


def score_rowp(script_path, timeout=180):
    pixwake_src = os.path.join(PROJECT_ROOT, "playground", "pixwake", "src")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH','')}",
        "JAX_ENABLE_X64": "True",
        "HOME": os.environ.get("HOME", ""),
    }
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
           os.path.abspath(script_path), "--problem",
           os.path.join(PROJECT_ROOT, "results", "problem_rowp.json"),
           "--timeout", str(timeout), "--log", "/dev/null", "--schedule-only"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 30, env=env, cwd=PROJECT_ROOT)
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": str(e)[:200]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    log = json.load(open(os.path.join(args.run_dir, "attempt_log.json")))
    feas = [e for e in log if e.get("train_feasible") and "train_aep" in e]
    if not feas:
        print("[G] No feasible attempts.")
        return
    best = max(feas, key=lambda e: e["train_aep"])
    script = os.path.join(args.run_dir, f"iter_{best['attempt']:03d}.py")

    rec = {
        "run_dir": args.run_dir,
        "best_attempt": best["attempt"],
        "best_train": best["train_aep"],
        "best_rowp": best.get("rowp_aep"),
    }
    if rec["best_rowp"] is None and os.path.exists(script):
        r = score_rowp(script)
        rec["best_rowp"] = r.get("aep_gwh")
        rec["best_rowp_feasible"] = r.get("feasible")

    if os.path.exists(script):
        src = open(script).read()
        rec["structures"] = [n for n, pat in STRUCTURE_RULES if re.search(pat, src)]

    out = os.path.join(HERE, f"summary_{os.path.basename(args.run_dir)}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
