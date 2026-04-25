"""Run every (problem, seed) pair in the 48-cell matrix as a
separate subprocess, parallelized with multiprocessing.Pool.

The per-problem batched approach (run_baseline_batch.py in-process
over 500 seeds) was unreliable on LUMI CPU because topfarm_sgd_solve
accumulates JAX XLA state across repeated calls and OOMs after
~50-100 seeds. One process per seed avoids the accumulation at
the cost of ~10s Python/JAX startup per seed, which is manageable
in parallel.

Usage:
    pixi run python tools/run_matrix_baselines_parallel.py \\
        --workers 64 --n-seeds 500 \\
        --out-dir lumi/logs/matrix_baselines
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one_seed(args):
    problem_rel, out_key, seed, out_dir = args
    out_path = os.path.join(out_dir, f"{out_key}_seed{seed}.out")
    # Skip if already done
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return (out_key, seed, "skipped")

    env = {
        **os.environ,
        "JAX_ENABLE_X64": "True",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    }
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "tools", "run_single_baseline.py"),
        os.path.join(PROJECT_ROOT, problem_rel),
        "--seed", str(seed),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env=env, cwd=PROJECT_ROOT, timeout=300)
        if r.returncode == 0 and r.stdout.strip():
            with open(out_path, "w") as f:
                f.write(r.stdout)
            return (out_key, seed, "ok")
        return (out_key, seed, f"rc={r.returncode}: {r.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        return (out_key, seed, "timeout")
    except Exception as e:
        return (out_key, seed, f"exc: {str(e)[:200]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="results/matrix/manifest.json")
    p.add_argument("--out-dir", default="lumi/logs/matrix_baselines")
    p.add_argument("--n-seeds", type=int, default=500)
    p.add_argument("--workers", type=int, default=64)
    args = p.parse_args()

    manifest = json.load(open(os.path.join(PROJECT_ROOT, args.manifest)))
    os.makedirs(os.path.join(PROJECT_ROOT, args.out_dir), exist_ok=True)

    # Build (problem, seed) job list. Skip already-done seeds.
    jobs = []
    for cell in manifest["cells"]:
        key = f"{cell['farm']}_n{cell['n']}_rose{cell['rose']}"
        for seed in range(args.n_seeds):
            out_path = os.path.join(
                PROJECT_ROOT, args.out_dir, f"{key}_seed{seed}.out")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                continue
            jobs.append((cell["path"], key, seed, args.out_dir))

    total_cells = len(manifest["cells"])
    total_seeds = total_cells * args.n_seeds
    print(f"Matrix baselines: {total_cells} cells × {args.n_seeds} seeds "
          f"= {total_seeds} total ({len(jobs)} to run, "
          f"{total_seeds - len(jobs)} already done)")
    print(f"Workers: {args.workers}, cores available: {mp.cpu_count()}")
    print(f"Out: {args.out_dir}")
    if not jobs:
        print("Nothing to do.")
        return

    t0 = time.time()
    done = 0
    errors = 0
    status_count = {"ok": 0, "skipped": 0, "err": 0}

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for (key, seed, status) in pool.imap_unordered(run_one_seed, jobs):
            done += 1
            if status == "ok":
                status_count["ok"] += 1
            elif status == "skipped":
                status_count["skipped"] += 1
            else:
                status_count["err"] += 1
                errors += 1
                if errors < 10:
                    print(f"  ERR {key} seed{seed}: {status[:120]}",
                          flush=True)

            if done % 200 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (len(jobs) - done) / rate if rate > 0 else 0
                print(f"[{done}/{len(jobs)}] ok={status_count['ok']} "
                      f"err={status_count['err']} "
                      f"rate={rate:.1f}/s "
                      f"elapsed={elapsed/60:.1f}m eta={eta_s/60:.1f}m",
                      flush=True)

    print(f"\nDone. ok={status_count['ok']} err={status_count['err']} "
          f"skipped={status_count['skipped']} "
          f"total elapsed={(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
