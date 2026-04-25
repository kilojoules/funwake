"""Aggregate Experiment C: best ROWP per Claude full-opt agent run.

Reuses the same per-run analysis logic as Experiment A but for
results_agent_claude_fullopt_run{1..5}/ and adds a "wraps_slsqp" flag.

Usage:
    pixi run python experiments/C_claude_fullopt_seeds/aggregate.py
"""
import json
import os
import re
import statistics
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROWP_PROBLEM = os.path.join(PROJECT_ROOT, "results", "problem_rowp.json")

WRAPS_SLSQP = re.compile(r"method\s*=\s*['\"]SLSQP['\"]|scipy\.optimize\.minimize.*SLSQP", re.I)
WRAPS_SGD   = re.compile(r"topfarm_sgd_solve")


def score_rowp(script_path, timeout=180):
    pixwake_src = os.path.join(PROJECT_ROOT, "playground", "pixwake", "src")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH','')}",
        "JAX_ENABLE_X64": "True",
        "HOME": os.environ.get("HOME", ""),
    }
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
           os.path.abspath(script_path), "--problem", ROWP_PROBLEM,
           "--timeout", str(timeout), "--log", "/dev/null"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 30, env=env, cwd=PROJECT_ROOT)
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": str(e)[:200]}


def main():
    runs = []
    for n in range(1, 6):
        run_dir = (os.path.join(PROJECT_ROOT, "results_agent_claude_fullopt") if n == 1
                   else os.path.join(PROJECT_ROOT, f"results_agent_claude_fullopt_run{n}"))
        log_path = os.path.join(run_dir, "attempt_log.json")
        if not os.path.exists(log_path):
            continue
        log = json.load(open(log_path))
        feas = [e for e in log if e.get("train_feasible") and "train_aep" in e]
        if not feas:
            continue
        best = max(feas, key=lambda e: e["train_aep"])
        script = os.path.join(run_dir, f"iter_{best['attempt']:03d}.py")
        if not os.path.exists(script):
            continue
        src = open(script).read()
        rec = {
            "run": n,
            "run_dir": run_dir,
            "best_attempt": best["attempt"],
            "best_train": best["train_aep"],
            "best_rowp": best.get("rowp_aep"),
            "wraps_slsqp": bool(WRAPS_SLSQP.search(src)),
            "wraps_sgd":   bool(WRAPS_SGD.search(src)),
            "n_attempts":  len(log),
            "n_feasible":  len(feas),
        }
        if rec["best_rowp"] is None:
            r = score_rowp(script)
            if "aep_gwh" in r:
                rec["best_rowp"] = r["aep_gwh"]
                rec["best_rowp_feasible"] = r.get("feasible")
        runs.append(rec)

    rowp = sorted([r["best_rowp"] for r in runs if r.get("best_rowp")])
    summary = {"n_runs": len(runs), "runs": runs}
    if rowp:
        summary["rowp_best"]  = rowp[-1]
        summary["rowp_median"] = rowp[len(rowp) // 2]
        summary["rowp_worst"] = rowp[0]
        if len(rowp) > 1:
            summary["rowp_std"] = round(statistics.pstdev(rowp), 3)
    summary["fraction_slsqp"] = (sum(r["wraps_slsqp"] for r in runs) / len(runs)) if runs else None
    summary["fraction_sgd"]   = (sum(r["wraps_sgd"]   for r in runs) / len(runs)) if runs else None

    out = os.path.join(EXP_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
