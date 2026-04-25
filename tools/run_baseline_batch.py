"""Run N multi-start SGD baselines on ONE problem in a single process.

Loads the problem and JIT-compiles the objective ONCE, then iterates
over seeds reusing the compiled graph. This avoids the ~15-30s
Python-startup + JAX-JIT overhead that `run_single_baseline.py`
incurs per invocation.

Used by the matrix_baselines sbatch for the 48-cell farm × N × rose
grid, where single-run would cost 24000 × 16s (≈14h on 8 GCDs) but
batched costs 48 × (30s JIT + 500 × 3s) ≈ 25min per problem
(≈2.5h on 8 GCDs).

Writes one output file per seed so the aggregator is unchanged:
    <out_dir>/<key>_seed<K>.out  (one JSON per seed)

Usage:
    python tools/run_baseline_batch.py \
        --problem results/matrix/problem_dei_n50_roseuniform.json \
        --seeds 0-499 \
        --out-dir lumi/logs/matrix_baselines \
        --out-key dei_n50_roseuniform
"""
import argparse
import json
import os
import sys
import time


def parse_seed_range(s: str):
    """Parse '0-499' or '0,1,2' or single '42'."""
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True)
    p.add_argument("--seeds", required=True,
                   help="seed range: '0-499' or '0,1,2'")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-key", required=True,
                   help="file prefix; outputs <out-dir>/<out-key>_seed<K>.out")
    p.add_argument("--skip-existing", action="store_true", default=True)
    args = p.parse_args()

    seeds = parse_seed_range(args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)

    # Filter out already-completed seeds
    pending = []
    for s in seeds:
        out_path = os.path.join(args.out_dir, f"{args.out_key}_seed{s}.out")
        if args.skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue
        pending.append((s, out_path))

    print(f"[{args.out_key}] {len(pending)}/{len(seeds)} seeds to run",
          flush=True)
    if not pending:
        return

    # Lazy JAX import inside main so --help is fast
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "playground", "pixwake", "src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from pixwake import Curve, Turbine, WakeSimulation
    from pixwake.deficit import BastankhahGaussianDeficit
    from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty

    t_load = time.time()
    info = json.load(open(args.problem))
    D = info["rotor_diameter"]
    t = info["turbine"]
    turb = Turbine(
        rotor_diameter=D, hub_height=info.get("hub_height", 150.0),
        power_curve=Curve(
            ws=jnp.array(t["power_curve_ws"], dtype=float),
            values=jnp.array(t["power_curve_kw"], dtype=float),
        ),
        ct_curve=Curve(
            ws=jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float),
            values=jnp.array(t["ct_curve_ct"], dtype=float),
        ),
    )
    sim = WakeSimulation(turb, BastankhahGaussianDeficit(k=0.04))

    wd = jnp.array(info["wind_rose"]["directions_deg"])
    ws = jnp.array(info["wind_rose"]["speeds_ms"])
    weights = jnp.array(info["wind_rose"]["weights"])
    boundary = jnp.array(info["boundary_vertices"])
    n_target = int(info["n_target"])
    min_spacing = float(info["min_spacing_m"])

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        pw = r.power()[:, :len(x)]
        return -jnp.sum(pw * weights[:, None]) * 8760 / 1e6

    # Grid candidates (shared across seeds)
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    nx = int(jnp.ceil((x_max - x_min) / min_spacing))
    ny = int(jnp.ceil((y_max - y_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(x_min + min_spacing / 2, x_max - min_spacing / 2, nx),
        jnp.linspace(y_min + min_spacing / 2, y_max - min_spacing / 2, ny),
    )
    cand_x, cand_y = gx.flatten(), gy.flatten()
    n_verts = boundary.shape[0]

    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex ** 2 + ey ** 2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)

    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
    inside_x, inside_y = cand_x[inside], cand_y[inside]

    settings = SGDSettings(
        learning_rate=50.0, max_iter=4000,
        additional_constant_lr_iterations=2000,
        beta1=0.1, beta2=0.2,
    )

    print(f"[{args.out_key}] problem loaded in {time.time()-t_load:.1f}s, "
          f"inside candidates: {int(inside_x.shape[0])}, n_target={n_target}",
          flush=True)

    # Iterate over seeds; first iteration triggers JIT compile for the
    # objective and topfarm_sgd_solve. All subsequent seeds reuse the
    # compiled graph — only the init positions change.
    t_loop = time.time()
    for i, (seed, out_path) in enumerate(pending):
        key = jax.random.PRNGKey(seed)
        if int(inside_x.shape[0]) >= n_target:
            idx = jax.random.choice(key, inside_x.shape[0], (n_target,), replace=False)
            init_x, init_y = inside_x[idx], inside_y[idx]
        else:
            init_x = jax.random.uniform(key, (n_target,),
                                         minval=float(x_min), maxval=float(x_max))
            k2, _ = jax.random.split(key)
            init_y = jax.random.uniform(k2, (n_target,),
                                         minval=float(y_min), maxval=float(y_max))

        t0 = time.time()
        try:
            opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y,
                                             boundary, min_spacing, settings)
            aep = float(-objective(opt_x, opt_y))
            bnd_pen = float(boundary_penalty(opt_x, opt_y, boundary))
            dx = opt_x[:, None] - opt_x[None, :]
            dy = opt_y[:, None] - opt_y[None, :]
            dist = jnp.sqrt(dx ** 2 + dy ** 2 + jnp.eye(n_target) * 1e10)
            min_dist = float(jnp.min(dist))
            feasible = (bnd_pen < 1e-3) and (min_dist >= min_spacing * 0.99)
            elapsed = time.time() - t0
            out = {
                "seed": int(seed),
                "aep": round(aep, 2),
                "feasible": bool(feasible),
                "time": round(elapsed, 1),
            }
        except Exception as e:
            out = {"seed": int(seed), "error": str(e)[:200]}

        with open(out_path, "w") as f:
            json.dump(out, f)

        if (i + 1) % 50 == 0:
            elapsed_total = time.time() - t_loop
            rate = (i + 1) / elapsed_total
            rem = (len(pending) - i - 1) / rate if rate > 0 else 0
            print(f"[{args.out_key}] {i+1}/{len(pending)} done, "
                  f"rate={rate:.2f}/s, eta={rem:.0f}s", flush=True)

    total = time.time() - t_loop
    print(f"[{args.out_key}] done: {len(pending)} seeds in {total:.0f}s "
          f"(avg {total/len(pending):.2f}s/seed)", flush=True)


if __name__ == "__main__":
    main()
