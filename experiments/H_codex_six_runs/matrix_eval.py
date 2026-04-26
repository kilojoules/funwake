"""Evaluate codex schedule-only top scripts across the 64-cell matrix.

Mirrors experiments/D_matrix_multi_seed/eval.py but pointed at the
3 codex sched-run best scripts (snapshotted to
results/codex_top_schedules/).

Resume-safe via a single results.json keyed on (label|farm_n|rose|seed).

Usage:
    pixi run python experiments/H_codex_six_runs/matrix_eval.py
    # or sharded for parallelism:
    pixi run python experiments/H_codex_six_runs/matrix_eval.py \\
        --shard 0 --num-shards 4
"""
import argparse
import json
import os
import subprocess
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))


SCHEDULES = [
    ("Codex run 1", "results/codex_top_schedules/codex_sched_run1_best.py"),
    ("Codex run 2", "results/codex_top_schedules/codex_sched_run2_best.py"),
    ("Codex run 3", "results/codex_top_schedules/codex_sched_run3_best.py"),
]


def score_one(script_rel, problem_rel, seed, timeout):
    env = {
        **os.environ,
        "JAX_ENABLE_X64": "True",
        "PYTHONPATH": (
            os.path.join(PROJECT_ROOT, "playground", "pixwake", "src")
            + ":" + os.environ.get("PYTHONPATH", "")
        ),
    }
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
        os.path.join(PROJECT_ROOT, script_rel),
        "--problem", os.path.join(PROJECT_ROOT, problem_rel),
        "--timeout", str(timeout),
        "--seed", str(seed),
        "--log", "/dev/null",
        "--schedule-only",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 60, env=env, cwd=PROJECT_ROOT)
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:200]}


def cell_timeout(n):
    if n <= 80:  return 240
    if n <= 200: return 600
    return 1500


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=1,
                   help="init seeds per cell (default 1; bump to 3 for variance)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--output", default=os.path.join(HERE, "matrix_results.json"))
    args = p.parse_args()

    manifest_path = os.path.join(PROJECT_ROOT, "results", "matrix", "manifest.json")
    manifest = json.load(open(manifest_path))

    existing = json.load(open(args.output)) if os.path.exists(args.output) else {}
    results = dict(existing)

    seeds = list(range(args.n_seeds))
    plan = []
    for label, script in SCHEDULES:
        if not os.path.exists(os.path.join(PROJECT_ROOT, script)):
            print(f"[H-matrix] missing {script}; skipping {label}")
            continue
        for cell in manifest["cells"]:
            for seed in seeds:
                key = f"{label}|{cell['farm']}_n{cell['n']}_rose{cell['rose']}|seed{seed}"
                if hash(key) % args.num_shards != args.shard:
                    continue
                if key in results and "aep_gwh" in results[key]:
                    continue
                plan.append((key, label, script, cell, seed))

    print(f"[H-matrix] shard {args.shard}/{args.num_shards}: {len(plan)} evals queued.")

    t_start = time.time()
    for i, (key, label, script, cell, seed) in enumerate(plan):
        timeout = cell_timeout(cell["n"])
        t0 = time.time()
        r = score_one(script, cell["path"], seed, timeout)
        elapsed = time.time() - t0
        results[key] = {
            "label": label, "script": script,
            "farm": cell["farm"], "n": cell["n"], "rose": cell["rose"],
            "seed": seed, "time_s": round(elapsed, 1),
            **r,
        }
        if (i + 1) % 5 == 0 or i + 1 == len(plan):
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            wall = time.time() - t_start
            print(f"[H-matrix] {i+1}/{len(plan)}  last={r.get('aep_gwh','?')}  feas={r.get('feasible','?')}  {elapsed:.0f}s  wall={wall:.0f}s")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[H-matrix] wrote {args.output}")


if __name__ == "__main__":
    main()
