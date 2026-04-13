#!/usr/bin/env python
"""Compute multi-start SGD baselines using parallel workers.

Each worker runs a batch of starts in a fresh subprocess to avoid
JAX memory leaks from running hundreds of JIT compilations in one
process.

Usage:
    # 500 starts across 10 workers (50 each)
    python tools/compute_baselines_parallel.py --n-starts 500 --workers 10

    # Specific farm
    python tools/compute_baselines_parallel.py --n-starts 500 --workers 10 \
        --problems results/problem_dei_n50.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_batch(args_tuple):
    """Run a batch of starts in a fresh subprocess."""
    problem_path, start_offset, batch_size, timeout_per_start = args_tuple

    project_root = os.path.join(os.path.dirname(__file__), "..")
    pixwake_src = os.path.join(project_root, "playground", "pixwake", "src")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH', '')}",
        "JAX_ENABLE_X64": "True",
        "HOME": os.environ.get("HOME", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }

    worker_script = f'''
import json, sys, time
sys.path.insert(0, "{pixwake_src}")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty

info = json.load(open("{problem_path}"))
D = info["rotor_diameter"]
hub_height = info.get("hub_height", 150.0)
t = info["turbine"]
turb = Turbine(
    rotor_diameter=D, hub_height=hub_height,
    power_curve=Curve(ws=jnp.array(t["power_curve_ws"], dtype=float),
                      values=jnp.array(t["power_curve_kw"], dtype=float)),
    ct_curve=Curve(ws=jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float),
                   values=jnp.array(t["ct_curve_ct"], dtype=float)))
sim = WakeSimulation(turb, BastankhahGaussianDeficit(k=0.04))

wd = jnp.array(info["wind_rose"]["directions_deg"])
ws = jnp.array(info["wind_rose"]["speeds_ms"])
weights = jnp.array(info["wind_rose"]["weights"])
boundary = jnp.array(info["boundary_vertices"])
n_target = info["n_target"]
min_spacing = info["min_spacing_m"]

def objective(x, y):
    r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
    p = r.power()[:, :len(x)]
    return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

settings = SGDSettings(learning_rate=50.0, max_iter=4000,
                       additional_constant_lr_iterations=2000,
                       beta1=0.1, beta2=0.2)

x_min, y_min = jnp.min(boundary, axis=0)
x_max, y_max = jnp.max(boundary, axis=0)
nx = int(jnp.ceil((x_max - x_min) / min_spacing))
ny = int(jnp.ceil((y_max - y_min) / min_spacing))
gx, gy = jnp.meshgrid(
    jnp.linspace(x_min + min_spacing/2, x_max - min_spacing/2, nx),
    jnp.linspace(y_min + min_spacing/2, y_max - min_spacing/2, ny))
cand_x, cand_y = gx.flatten(), gy.flatten()
n_verts = boundary.shape[0]
def edge_dist(i):
    x1, y1 = boundary[i]
    x2, y2 = boundary[(i + 1) % n_verts]
    ex, ey = x2 - x1, y2 - y1
    el = jnp.sqrt(ex**2 + ey**2) + 1e-10
    return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)
inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
inside_x, inside_y = cand_x[inside], cand_y[inside]

results = []
for i in range({start_offset}, {start_offset + batch_size}):
    key = jax.random.PRNGKey(i)
    if len(inside_x) >= n_target:
        idx = jax.random.choice(key, len(inside_x), (n_target,), replace=False)
        init_x, init_y = inside_x[idx], inside_y[idx]
    else:
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))
    t0 = time.time()
    opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y, boundary, min_spacing, settings)
    elapsed = time.time() - t0
    aep = float(-objective(opt_x, opt_y))
    bnd_pen = float(boundary_penalty(opt_x, opt_y, boundary))
    dx = opt_x[:, None] - opt_x[None, :]
    dy = opt_y[:, None] - opt_y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(n_target) * 1e10)
    min_dist = float(jnp.min(dist))
    feasible = (bnd_pen < 1e-3) and (min_dist >= min_spacing * 0.99)
    results.append({{"seed": i, "aep": aep, "feasible": feasible, "time": round(elapsed, 1)}})

print(json.dumps(results))
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", worker_script],
            capture_output=True, text=True,
            timeout=timeout_per_start * batch_size + 120,
            env=env, cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        if result.returncode != 0:
            return {"error": result.stderr[-500:], "offset": start_offset}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "offset": start_offset}
    except Exception as e:
        return {"error": str(e), "offset": start_offset}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problems", nargs="*", default=None)
    p.add_argument("--n-starts", type=int, default=500)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--timeout-per-start", type=int, default=120)
    p.add_argument("--output", default="results/baselines_500start.json")
    args = p.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")

    if args.problems is None:
        args.problems = sorted(glob.glob(
            os.path.join(project_root, "results", "problem_dei_n*.json")))
        rowp = os.path.join(project_root, "results", "problem_rowp.json")
        if os.path.exists(rowp):
            args.problems.append(rowp)
        # Skip n=100 (too slow)
        args.problems = [p for p in args.problems if "n100" not in p]

    batch_size = max(1, args.n_starts // args.workers)
    all_results = {}

    for prob_path in args.problems:
        prob_name = os.path.basename(prob_path).replace(".json", "")
        info = json.load(open(prob_path))
        n_target = info["n_target"]

        print("=== %s (n=%d) === %d starts across %d workers (%d each)" % (
            prob_name, n_target, args.n_starts, args.workers, batch_size))

        batches = []
        for w in range(args.workers):
            offset = w * batch_size
            sz = batch_size if w < args.workers - 1 else args.n_starts - offset
            batches.append((os.path.abspath(prob_path), offset, sz, args.timeout_per_start))

        all_aeps = []
        feasible_count = 0
        total_time = 0

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_batch, b): b for b in batches}
            for future in as_completed(futures):
                result = future.result()
                if isinstance(result, dict) and "error" in result:
                    print("  Worker error: %s" % result["error"][:100])
                    continue
                for r in result:
                    all_aeps.append(r["aep"])
                    if r["feasible"]:
                        feasible_count += 1
                    total_time += r["time"]
                print("  Batch done: %d results, best so far %.1f" % (
                    len(result), max(all_aeps) if all_aeps else 0))

        if not all_aeps:
            print("  NO RESULTS")
            continue

        feasible_aeps = [a for a, r in zip(all_aeps, [True]*len(all_aeps))
                         if True]  # all tracked above

        best_aep = max(all_aeps)
        entry = {
            "problem": prob_name,
            "n_target": n_target,
            "n_starts": len(all_aeps),
            "total_time_s": round(total_time, 1),
            "best_aep": round(best_aep, 2),
            "feasible_rate": round(feasible_count / len(all_aeps), 3),
            "aep_gwh": round(best_aep, 2),
        }
        all_results[prob_name] = entry
        print("  DONE: %d starts, best=%.1f, feas_rate=%.1f%%" % (
            len(all_aeps), best_aep, 100 * feasible_count / len(all_aeps)))

    out_path = os.path.join(project_root, args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nWrote %s" % out_path)


if __name__ == "__main__":
    main()
