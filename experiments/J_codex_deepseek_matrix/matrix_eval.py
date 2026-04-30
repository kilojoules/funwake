"""Matrix eval for codex + deepseek best scripts (3 seeds x 2 modes x 2 providers).

12 scripts in results/agent_top_schedules/, 8 sched + 4 fullopt:
    {provider}_{mode}_run{n}.py for provider in {codex, deepseek}, mode in {sched, fullopt}

Each script is scored across the 64-cell matrix at 3 init seeds (default).
Sched scripts pass --schedule-only; fullopt scripts run their own optimize().

Usage (local serial):
    pixi run python experiments/J_codex_deepseek_matrix/matrix_eval.py --n-seeds 3

Sharded (LUMI array):
    pixi run python experiments/J_codex_deepseek_matrix/matrix_eval.py \\
        --n-seeds 3 --shard 0 --num-shards 100
"""
import argparse
import json
import os
import subprocess
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))


# (label, script_path, mode)  mode in {"sched", "fullopt"}
SCHEDULES = []
for prov in ("codex", "deepseek"):
    for mode in ("sched", "fullopt"):
        for n in (1, 2, 3):
            label = f"{prov} {mode} run{n}"
            script = f"results/agent_top_schedules/{prov}_{mode}_run{n}.py"
            SCHEDULES.append((label, script, mode))


def score_one(script_rel, problem_rel, seed, timeout, schedule_only):
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
    ]
    if schedule_only:
        cmd.append("--schedule-only")
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
    p.add_argument("--output", default=os.path.join(HERE, "matrix_results.json"))
    args = p.parse_args()

    manifest = json.load(open(os.path.join(PROJECT_ROOT, "results", "matrix", "manifest.json")))
    existing = json.load(open(args.output)) if os.path.exists(args.output) else {}
    results = dict(existing)

    seeds = list(range(args.n_seeds))
    plan = []
    for label, script, mode in SCHEDULES:
        if not os.path.exists(os.path.join(PROJECT_ROOT, script)):
            print(f"[J] missing {script}; skipping {label}")
            continue
        for cell in manifest["cells"]:
            for seed in seeds:
                key = f"{label}|{cell['farm']}_n{cell['n']}_rose{cell['rose']}|seed{seed}"
                if hash(key) % args.num_shards != args.shard:
                    continue
                if key in results and "aep_gwh" in results[key]:
                    continue
                plan.append((key, label, script, mode, cell, seed))

    print(f"[J] shard {args.shard}/{args.num_shards}: {len(plan)} evals queued.")

    t_start = time.time()
    for i, (key, label, script, mode, cell, seed) in enumerate(plan):
        timeout = cell_timeout(cell["n"])
        t0 = time.time()
        r = score_one(script, cell["path"], seed, timeout,
                      schedule_only=(mode == "sched"))
        elapsed = time.time() - t0
        results[key] = {
            "label": label, "script": script, "mode": mode,
            "farm": cell["farm"], "n": cell["n"], "rose": cell["rose"],
            "seed": seed, "time_s": round(elapsed, 1),
            **r,
        }
        if (i + 1) % 5 == 0 or i + 1 == len(plan):
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            wall = time.time() - t_start
            print(f"[J] {i+1}/{len(plan)}  last={r.get('aep_gwh','?')}  feas={r.get('feasible','?')}  {elapsed:.0f}s  wall={wall:.0f}s")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[J] wrote {args.output}")


if __name__ == "__main__":
    main()
