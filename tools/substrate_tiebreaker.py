"""Substrate-independent tiebreaker for the FunWake ceiling claim.

Per the idea-critic review, every method currently in the ~4273 GWh
cluster shares the continuous-relaxation substrate: pixwake + JAX +
Adam + boundary/spacing penalty schedule. Whether 4273 is a real
Pareto frontier or a shared-substrate fixed point cannot be
disambiguated from inside that family.

This script runs **CMA-ES directly on the 2N-dim raw position vector**
starting from the same feasible wind-aware grid init the skeleton
uses, with no Adam, no LR schedule, no penalty ramp, no JAX gradient,
no SGD polish. Feasibility is enforced as a hard objective penalty.
The forward AEP evaluation still uses pixwake (physics is fixed) but
the search algorithm is substrate-independent.

Three possible outcomes:
  1. Final feasible AEP > 4278 GWh ----> ceiling is a shared-basin
     artifact of the Adam+penalty substrate. Paper pivots.
  2. 4268 <= final <= 4278 --------------> ceiling looks real and is now
     backed by a genuinely independent method. Pareto claim defensible.
  3. Final < 4268 ----------------------> raw-position CMA-ES
     underperforms, suggesting the encoding (Adam+penalty) is
     near-optimal given constraint geometry. Ceiling lives in the
     encoding, still publishable.

Usage:
    pixi run python tools/substrate_tiebreaker.py \
        --problem results/problem_rowp.json \
        --sigma 200 --maxiter 120 --popsize 30 \
        --output results_substrate_tiebreaker/rowp.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def wind_aware_grid_init(boundary_np: np.ndarray,
                          wd_deg: np.ndarray,
                          ws: np.ndarray,
                          weights: np.ndarray,
                          min_spacing: float,
                          n_target: int,
                          seed: int,
                          margin: float = 1.20) -> tuple[np.ndarray, np.ndarray]:
    """Wind-aware feasible grid init.

    Uses a grid cell of margin*min_spacing so subsampled layouts are
    guaranteed feasible (pairwise distance >= margin*min_spacing >
    min_spacing). The skeleton uses margin=1.0 and relies on Adam to
    drive constraint violations to zero; here we need a starting point
    that's feasible WITHOUT running Adam.
    """
    grid_spacing = margin * min_spacing

    x_min, y_min = boundary_np[:, 0].min(), boundary_np[:, 1].min()
    x_max, y_max = boundary_np[:, 0].max(), boundary_np[:, 1].max()

    wd_rad = np.deg2rad(wd_deg)
    dominant = np.arctan2(
        np.sum(weights * np.sin(wd_rad)),
        np.sum(weights * np.cos(wd_rad)))
    angle = dominant + np.pi / 2

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = boundary_np[:, 0].mean(), boundary_np[:, 1].mean()
    translated = boundary_np - np.array([cx, cy])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = rot_bnd[:, 0].min(), rot_bnd[:, 1].min()
    rx_max, ry_max = rot_bnd[:, 0].max(), rot_bnd[:, 1].max()
    nx = max(int(np.floor((rx_max - rx_min) / grid_spacing)), 1)
    ny = max(int(np.floor((ry_max - ry_min) / grid_spacing)), 1)
    xs = np.linspace(rx_min + grid_spacing / 2, rx_max - grid_spacing / 2, nx)
    ys = np.linspace(ry_min + grid_spacing / 2, ry_max - grid_spacing / 2, ny)
    gx, gy = np.meshgrid(xs, ys)
    rot_pts = np.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + np.array([cx, cy])
    cand_x, cand_y = orig_pts[:, 0], orig_pts[:, 1]

    # Filter inside boundary (signed distance via edge normals)
    n_verts = boundary_np.shape[0]
    dists = np.full(len(cand_x), np.inf)
    for i in range(n_verts):
        x1, y1 = boundary_np[i]
        x2, y2 = boundary_np[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = np.sqrt(ex ** 2 + ey ** 2) + 1e-10
        edge = (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)
        dists = np.minimum(dists, edge)
    inside = dists > 0
    ix, iy = cand_x[inside], cand_y[inside]

    rng = np.random.default_rng(seed)
    if len(ix) >= n_target:
        idx = rng.choice(len(ix), n_target, replace=False)
        return ix[idx], iy[idx]

    # Not enough inside-polygon grid cells at this margin. Retry with
    # progressively smaller margin (down to 1.02) and break once enough
    # candidates fit inside the polygon.
    for retry_margin in (1.15, 1.10, 1.05, 1.02):
        return wind_aware_grid_init(boundary_np, wd_deg, ws, weights,
                                     min_spacing, n_target, seed,
                                     margin=retry_margin)
    # Still not enough — fall back to bbox random
    xs = rng.uniform(x_min, x_max, n_target)
    ys = rng.uniform(y_min, y_max, n_target)
    return xs, ys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True,
                   help="problem JSON (e.g. results/problem_rowp.json)")
    p.add_argument("--sigma", type=float, default=200.0,
                   help="initial CMA-ES sigma in meters (search radius scale)")
    p.add_argument("--maxiter", type=int, default=120,
                   help="CMA-ES max iterations (generations)")
    p.add_argument("--popsize", type=int, default=30,
                   help="CMA-ES population size (lambda)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--infeas-penalty", type=float, default=1e6)
    p.add_argument("--warm-start-script", default=None,
                   help="Path to a schedule_fn / optimize() script. If set, "
                        "the script is run once via the harness on the same "
                        "problem and its output layout is used as CMA-ES x0 "
                        "instead of the wind-aware grid init.")
    args = p.parse_args()

    import cma

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, "playground", "pixwake", "src"))
    sys.path.insert(0, os.path.join(project_root, "benchmarks"))
    from dei_layout import ProblemBenchmark

    problem_path = os.path.abspath(args.problem)
    bm = ProblemBenchmark(problem_path)

    with open(problem_path) as f:
        info = json.load(f)
    n = int(info["n_target"])
    boundary = np.array(info["boundary_vertices"])
    wd = np.array(info["wind_rose"]["directions_deg"])
    ws = np.array(info["wind_rose"]["speeds_ms"])
    weights = np.array(info["wind_rose"]["weights"])
    min_spacing = float(info["min_spacing_m"])

    # Get the starting layout. Two modes:
    #   --warm-start-script: run an existing schedule/optimizer script on
    #     the problem via the harness, capture the layout, use as x0
    #   default: feasible wind-aware grid init (no Adam, no schedule)
    if args.warm_start_script:
        print(f"Warm-starting from {args.warm_start_script} on {problem_path}...",
              flush=True)
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            warm_out = f.name
        try:
            harness = os.path.join(project_root, "playground", "harness.py")
            env = {
                **os.environ,
                "PYTHONPATH": (
                    os.path.join(project_root, "playground", "pixwake", "src")
                    + ":" + os.environ.get("PYTHONPATH", "")
                ),
                "JAX_ENABLE_X64": "True",
                "FUNWAKE_PROBLEM": problem_path,
                "FUNWAKE_OUTPUT": warm_out,
            }
            cmd = [sys.executable, harness,
                   os.path.abspath(args.warm_start_script)]
            # Detect whether the script defines schedule_fn or optimize() —
            # try schedule-only first since that's the more common case
            try:
                with open(args.warm_start_script) as fsrc:
                    src = fsrc.read()
                if "def schedule_fn" in src:
                    cmd.append("--schedule-only")
            except OSError:
                pass
            r = subprocess.run(cmd, env=env,
                               cwd=os.path.join(project_root, "playground"),
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                print(f"  warm-start script failed: {r.stderr[-400:]}",
                      file=sys.stderr)
                sys.exit(1)
            with open(warm_out) as f:
                layout = json.load(f)
            init_x = np.asarray(layout["x"], dtype=float)
            init_y = np.asarray(layout["y"], dtype=float)
            assert len(init_x) == n, f"layout has {len(init_x)} turbines, expected {n}"
        finally:
            if os.path.exists(warm_out):
                os.unlink(warm_out)
    else:
        print("Initializing from wind-aware grid init (same as skeleton)...",
              flush=True)
        init_x, init_y = wind_aware_grid_init(
            boundary, wd, ws, weights, min_spacing, n, args.seed)

    init_aep = float(bm.score(init_x, init_y))
    init_feas = bm.check_feasibility(init_x, init_y)
    init_feasible = bool(init_feas["spacing_ok"] and init_feas["boundary_ok"])
    print(f"  init AEP: {init_aep:.1f} GWh, feasible: {init_feasible}", flush=True)
    if not init_feasible:
        print(f"  init feasibility details: {init_feas}", flush=True)

    x0 = np.concatenate([init_x, init_y])
    n_dim = x0.size

    history = {
        "problem": problem_path,
        "n_target": n,
        "n_dim": n_dim,
        "sigma": args.sigma,
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "seed": args.seed,
        "init_aep": round(init_aep, 2),
        "init_feasible": init_feasible,
        "evals": [],
        "best_feasible_aep": init_aep if init_feasible else -float("inf"),
        "best_any_aep": init_aep,
        "eval_count": 0,
        "start_time": time.time(),
    }

    def objective(flat):
        """CMA-ES minimizes. Soft penalty for infeasibility gives CMA a
        gradient signal back toward the feasible manifold rather than a
        flat infeasible region.

        Feasible:    return -AEP
        Infeasible:  return -AEP + soft_penalty(violation_magnitude)
                     where soft_penalty grows continuously with how much
                     the layout violates boundary + spacing constraints.
        """
        history["eval_count"] += 1
        x = np.asarray(flat[:n], dtype=float)
        y = np.asarray(flat[n:], dtype=float)
        try:
            aep = float(bm.score(x, y))
            feas = bm.check_feasibility(x, y)
            feasible = bool(feas["spacing_ok"] and feas["boundary_ok"])
            # Continuous violation magnitudes
            min_d = float(feas.get("min_turbine_distance_m", 0.0))
            spacing_viol = max(0.0, min_spacing - min_d)  # meters short
            bpen = float(feas.get("boundary_penalty", 0.0))
        except Exception as ex:
            history["evals"].append({
                "i": history["eval_count"], "error": str(ex)[:200]})
            return args.infeas_penalty

        # Soft penalty: if infeasible, add a continuous amount proportional
        # to how close we are to feasibility. Scale chosen so that a
        # spacing violation of ~100m or a boundary penalty of ~1 is worth
        # ~500 GWh (comparable to the whole AEP range), guiding CMA-ES
        # back to the feasible set but not flattening the AEP signal once
        # feasibility is achieved.
        penalty = 0.0
        if not feasible:
            penalty = 5.0 * spacing_viol + 500.0 * bpen + 50.0  # base term prevents rank-tie

        entry = {"i": history["eval_count"], "aep": round(aep, 2),
                 "feasible": feasible,
                 "min_dist": round(min_d, 1),
                 "bpen": round(bpen, 4),
                 "penalty": round(penalty, 2)}
        if aep > history["best_any_aep"]:
            history["best_any_aep"] = aep
        if feasible and aep > history["best_feasible_aep"]:
            history["best_feasible_aep"] = aep
            entry["new_feasible_best"] = True
            print(f"  [eval {history['eval_count']}] new feasible best: "
                  f"{aep:.2f} GWh", flush=True)
        history["evals"].append(entry)

        if history["eval_count"] % 100 == 0:
            elapsed = time.time() - history["start_time"]
            n_feas = sum(1 for e in history["evals"] if e.get("feasible"))
            print(f"  [eval {history['eval_count']}] "
                  f"best_feas={history['best_feasible_aep']:.1f} "
                  f"feas_frac={n_feas/history['eval_count']:.2f} "
                  f"({elapsed/60:.1f} min)", flush=True)

        return -aep + penalty

    print(f"CMA-ES on raw {n_dim}-D positions")
    print(f"  seeding from feasible grid init (AEP={init_aep:.1f})")
    print(f"  sigma={args.sigma}m, popsize={args.popsize}, maxiter={args.maxiter}")
    print(f"  target budget: ~{args.popsize * args.maxiter} evals")
    print("", flush=True)

    t0 = time.time()
    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        args.sigma,
        {
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "seed": args.seed,
            "verbose": 0,
            "verb_disp": 10,
            # Disable early-stop on flat objective — we expect many
            # near-identical infeasible penalty values early on
            "tolfun": 0.0,
            "tolfunhist": 0.0,
            "tolx": 1e-8,
            "tolstagnation": 1000,
        },
    )
    while not es.stop():
        solutions = es.ask()
        values = [objective(np.array(s)) for s in solutions]
        es.tell(solutions, values)
        es.logger.add()

    elapsed = time.time() - t0
    history["elapsed_s"] = round(elapsed, 1)
    history["cma_stop"] = str(es.stop())

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(history, f, indent=2)

    print()
    print("=== DONE ===")
    print(f"  total evals: {history['eval_count']}")
    print(f"  wall time: {elapsed/60:.1f} min")
    print(f"  init feasible AEP: {init_aep:.1f}")
    print(f"  best feasible AEP: {history['best_feasible_aep']:.1f}")
    print(f"  best any AEP (incl infeasible): {history['best_any_aep']:.1f}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
