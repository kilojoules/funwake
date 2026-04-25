"""Per-seed 64-cell matrix evaluation.

Generalizes tools/eval_matrix_schedules.py with --n-seeds. Resume-safe:
existing per-(script, cell, seed) results are skipped.

Usage:
    pixi run python experiments/D_matrix_multi_seed/eval.py --n-seeds 3
    pixi run python experiments/D_matrix_multi_seed/eval.py --n-seeds 5 \
        --shard 0 --num-shards 4    # for parallel sharding (LUMI)
"""
import argparse
import json
import os
import subprocess
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))


SCHEDULES = [
    ("Claude (iter_192)", "results_agent_schedule_only_5hr/iter_192.py"),
    ("Gemini (iter_118)", "results_agent_gemini_cli_5hr/iter_118.py"),
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
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--output", default=os.path.join(EXP_DIR, "results.json"))
    args = p.parse_args()

    manifest_path = os.path.join(PROJECT_ROOT, "results", "matrix", "manifest.json")
    manifest = json.load(open(manifest_path))

    existing = {}
    if os.path.exists(args.output):
        existing = json.load(open(args.output))

    results = dict(existing)
    seeds = list(range(args.n_seeds))
    plan = []
    for si, (label, script) in enumerate(SCHEDULES):
        for ci, cell in enumerate(manifest["cells"]):
            for seed in seeds:
                key = f"{label}|{cell['farm']}_n{cell['n']}_rose{cell['rose']}|seed{seed}"
                # shard distribution: round-robin by hash of key
                slot = hash(key) % args.num_shards
                if slot != args.shard:
                    continue
                if key in results and "aep_gwh" in results[key]:
                    continue
                plan.append((key, label, script, cell, seed))

    print(f"[D] Shard {args.shard}/{args.num_shards}: {len(plan)} evals queued.")

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
        # checkpoint every 10 evals
        if (i + 1) % 10 == 0 or i + 1 == len(plan):
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            done = i + 1
            wall = time.time() - t_start
            print(f"[D] {done}/{len(plan)}  last={r.get('aep_gwh','?')} feas={r.get('feasible','?')}  {elapsed:.0f}s  wall={wall:.0f}s")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[D] Wrote {args.output}")


if __name__ == "__main__":
    main()
