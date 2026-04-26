"""Aggregate Experiment A: per-run best schedule + ROWP, structure classification.

Reads attempt_log.json from each results_agent_<agent>_sched_run<N>/, picks
the highest-train-AEP feasible script per run, scores it on ROWP if not
already scored, and writes summary.json.

Structure classification is rule-based string matching on the script text
(presence of "bump", "cosine_restart", "exp(" exponential decay etc.).
This is heuristic, not authoritative — useful for a quick reproducibility
look, not for automated paper claims.

Usage:
    pixi run python experiments/A_multi_agent_seed/aggregate.py
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


STRUCTURE_RULES = [
    ("dual_gaussian_bumps",  r"exp\([^)]*-[^)]*\(t\s*-\s*0\.[57]"),
    ("cosine_restart",       r"cos\([^)]*restart|n_cycles|n_restarts"),
    ("cyclic_betas",         r"beta1\s*=.*cos|beta2\s*=.*cos"),
    ("alpha_squeeze",        r"alpha\s*=.*5e6|alpha.*\*\*\s*[3-9]"),
    ("warmup",               r"warmup_frac|t\s*<\s*0\.0[0-9]"),
]


def classify(script_path):
    src = open(script_path).read()
    return [name for name, pat in STRUCTURE_RULES if re.search(pat, src)]


def score_rowp(script_path, timeout=180):
    pixwake_src = os.path.join(PROJECT_ROOT, "playground", "pixwake", "src")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH','')}",
        "JAX_ENABLE_X64": "True",
        "HOME": os.environ.get("HOME", ""),
    }
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
        os.path.abspath(script_path),
        "--problem", ROWP_PROBLEM,
        "--timeout", str(timeout),
        "--log", "/dev/null",
        "--schedule-only",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 30, env=env, cwd=PROJECT_ROOT)
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": str(e)[:200]}


def best_in_run(run_dir):
    log_path = os.path.join(run_dir, "attempt_log.json")
    if not os.path.exists(log_path):
        return None
    log = json.load(open(log_path))
    feas = [e for e in log if e.get("train_feasible") and "train_aep" in e]
    if not feas:
        return None
    feas.sort(key=lambda e: -e["train_aep"])
    best = feas[0]
    script = os.path.join(run_dir, f"iter_{best['attempt']:03d}.py")
    return {"attempt": best["attempt"], "train_aep": best["train_aep"],
            "rowp_aep": best.get("rowp_aep"), "script": script}


def main():
    summary = {"by_agent": {}}

    run1_dirs = {
        "claude": "results_agent_schedule_only_5hr",
        "gemini": "results_agent_gemini_cli_5hr",
        "codex":  "results_agent_codex_sched_run1",
    }

    for agent in ("claude", "gemini", "codex"):
        runs = []
        for n in range(1, 6):
            if n == 1:
                run_dir = os.path.join(PROJECT_ROOT, run1_dirs[agent])
            else:
                run_dir = os.path.join(PROJECT_ROOT, f"results_agent_{agent}_sched_run{n}")
            if not os.path.isdir(run_dir):
                continue
            b = best_in_run(run_dir)
            if not b:
                continue
            if b.get("rowp_aep") is None and os.path.exists(b["script"]):
                r = score_rowp(b["script"])
                if "aep_gwh" in r:
                    b["rowp_aep"] = r["aep_gwh"]
                    b["rowp_feasible"] = r.get("feasible")
            b["structures"] = classify(b["script"]) if os.path.exists(b["script"]) else []
            b["run"] = n
            b["run_dir"] = run_dir
            runs.append(b)

        if runs:
            rowp = [r["rowp_aep"] for r in runs if r.get("rowp_aep")]
            train = [r["train_aep"] for r in runs]
            summary["by_agent"][agent] = {
                "n_runs": len(runs),
                "runs": runs,
                "train_aep_mean": round(statistics.mean(train), 2) if train else None,
                "train_aep_std": round(statistics.pstdev(train), 2) if len(train) > 1 else None,
                "rowp_aep_mean": round(statistics.mean(rowp), 2) if rowp else None,
                "rowp_aep_std": round(statistics.pstdev(rowp), 2) if len(rowp) > 1 else None,
                "structures_per_run": [r["structures"] for r in runs],
            }

    out = os.path.join(EXP_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
