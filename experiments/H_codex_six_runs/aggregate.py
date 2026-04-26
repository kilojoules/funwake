"""Aggregate H: Codex 3-seed × 2-mode results.

Walks results_agent_codex_{sched,fullopt}_run{1,2,3}/, picks best
feasible script per run, scores ROWP if missing, classifies structure
heuristically, and emits a single summary.json that the codex-paper
aggregator (paper/build_short_codex.py) can consume.

Usage:
    pixi run python experiments/H_codex_six_runs/aggregate.py
"""
import json
import os
import re
import statistics
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
ROWP_PROBLEM = os.path.join(PROJECT_ROOT, "results", "problem_rowp.json")


SCHEDULE_RULES = [
    ("dual_gaussian_bumps",  r"exp\([^)]*-[^)]*\(t\s*-\s*0\.[57]"),
    ("cosine_restart",       r"cos\([^)]*restart|n_cycles|n_restarts"),
    ("cyclic_betas",         r"beta1\s*=.*cos|beta2\s*=.*cos"),
    ("alpha_squeeze",        r"alpha\s*=.*5e6|alpha.*\*\*\s*[3-9]"),
    ("warmup",               r"warmup_frac|t\s*<\s*0\.0[0-9]"),
]
FULLOPT_RULES = [
    ("wraps_slsqp",          r"method\s*=\s*['\"]SLSQP['\"]|scipy\.optimize\.minimize.*SLSQP"),
    ("wraps_topfarm_sgd",    r"topfarm_sgd_solve"),
    ("custom_adam",          r"jax.*grad|optax|jnp.*adam"),
    ("genetic_algorithm",    r"\b(genetic|GA|population|crossover|mutation)\b"),
    ("cma_es",               r"\b(CMA-?ES|cma\.|cmaes)\b", re.I),
]


def classify(script_path, rules):
    if not script_path or not os.path.exists(script_path):
        return []
    src = open(script_path).read()
    matches = []
    for rule in rules:
        name, pat = rule[0], rule[1]
        flags = rule[2] if len(rule) > 2 else 0
        if re.search(pat, src, flags):
            matches.append(name)
    return matches


def score_rowp(script_path, schedule_only, timeout=180):
    pixwake_src = os.path.join(PROJECT_ROOT, "playground", "pixwake", "src")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH','')}",
        "JAX_ENABLE_X64": "True",
        "HOME": os.environ.get("HOME", ""),
    }
    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
        os.path.abspath(script_path), "--problem", ROWP_PROBLEM,
        "--timeout", str(timeout), "--log", "/dev/null",
    ]
    if schedule_only:
        cmd.append("--schedule-only")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 30, env=env, cwd=PROJECT_ROOT)
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": str(e)[:200]}


def best_script(run_dir):
    log_path = os.path.join(run_dir, "attempt_log.json")
    if not os.path.exists(log_path):
        return None, None
    log = json.load(open(log_path))
    feas = [e for e in log if e.get("train_feasible") and "train_aep" in e]
    if not feas:
        return log, None
    best = max(feas, key=lambda e: e["train_aep"])
    script = os.path.join(run_dir, f"iter_{best['attempt']:03d}.py")
    return log, {"attempt": best["attempt"],
                 "train_aep": best["train_aep"],
                 "rowp_aep": best.get("rowp_aep"),
                 "rowp_feasible": best.get("rowp_feasible"),
                 "script": script}


def collect_mode(mode):
    rules = SCHEDULE_RULES if mode == "sched" else FULLOPT_RULES
    sched_only = (mode == "sched")
    runs = []
    for n in (1, 2, 3):
        run_dir = os.path.join(PROJECT_ROOT, f"results_agent_codex_{'sched' if mode == 'sched' else 'fullopt'}_run{n}")
        log, best = best_script(run_dir)
        if best is None:
            runs.append({"run": n, "run_dir": run_dir,
                         "status": "no_feasible" if log else "missing"})
            continue
        if best["rowp_aep"] is None and os.path.exists(best["script"]):
            r = score_rowp(best["script"], schedule_only=sched_only)
            if "aep_gwh" in r:
                best["rowp_aep"] = r["aep_gwh"]
                best["rowp_feasible"] = r.get("feasible")
        best["structures"] = classify(best["script"], rules)
        best["run"] = n
        best["run_dir"] = run_dir
        best["n_attempts"] = len(log) if log else 0
        runs.append(best)
    return runs


def summarize(runs):
    rowp = [r["rowp_aep"] for r in runs if r.get("rowp_aep") and r.get("rowp_feasible")]
    train = [r["train_aep"] for r in runs if r.get("train_aep")]
    return {
        "n_runs": len(runs),
        "n_with_rowp": len(rowp),
        "train_mean": round(statistics.mean(train), 3) if train else None,
        "train_std":  round(statistics.pstdev(train), 3) if len(train) > 1 else None,
        "train_max":  max(train) if train else None,
        "rowp_mean":  round(statistics.mean(rowp), 3) if rowp else None,
        "rowp_std":   round(statistics.pstdev(rowp), 3) if len(rowp) > 1 else None,
        "rowp_max":   max(rowp) if rowp else None,
        "rowp_min":   min(rowp) if rowp else None,
    }


def main():
    sched_runs   = collect_mode("sched")
    fullopt_runs = collect_mode("full")
    summary = {
        "sched":   {"runs": sched_runs,   "summary": summarize(sched_runs)},
        "fullopt": {"runs": fullopt_runs, "summary": summarize(fullopt_runs)},
        "comparators": {
            "claude_sched_iter192_rowp_5seed_mean": 4266.3,
            "gemini_sched_iter118_rowp_5seed_mean": 4262.2,
            "gemini_fullopt_slsqp_5seed_best":      4272.7,
            "claude_fullopt_run1_rowp":             4264.9,
            "baseline_500_start_rowp":              4246.7,
        },
    }
    out = os.path.join(HERE, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
