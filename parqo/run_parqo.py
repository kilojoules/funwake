"""Run the deployed dual-bump schedule (iter_192) on the ParqueFicticio
inclusion-zone case study (Criado Risco et al. 2024): 12 x V80 in five
disconnected inclusion polygons, via the multi-zone variant of the fixed
schedule-only skeleton.

Usage:
    pixi run python parqo/run_parqo.py [--seed N]

Writes: parqo/layout_parqo.json, parqo/results_parqo.json,
        parqo/parqo_layout.png (via plot_layout.py)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "playground"))
sys.path.insert(0, os.path.join(ROOT, "dependencies/pixwake/src"))
sys.path.insert(0, HERE)

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

PROBLEM = os.path.join(HERE, "problem_parqo.json")
LAYOUT_OUT = os.path.join(HERE, "layout_parqo.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=None,
                    help="override n_target (default: problem JSON)")
    ap.add_argument("--init", choices=["zones", "box"], default="zones",
                    help="box = grid over bounding box, may start outside zones")
    args = ap.parse_args()

    from harness import build_sim
    from schedule_dual_bump import schedule_fn
    from skeleton_multizone import (run_with_schedule, multizone_sdf)

    problem = json.load(open(PROBLEM))
    sim = build_sim(problem)
    wd = jnp.array(problem["wind_rose"]["directions_deg"])
    ws = jnp.array(problem["wind_rose"]["speeds_ms"])
    weights = jnp.array(problem["wind_rose"]["weights"])
    zones = problem["inclusion_zones"]
    n_target = args.n or problem["n_target"]
    min_spacing = problem["min_spacing_m"]

    global LAYOUT_OUT
    suffix = "" if n_target == problem["n_target"] else f"_n{n_target}"
    LAYOUT_OUT = os.path.join(HERE, f"layout_parqo{suffix}.json")
    results_out = os.path.join(HERE, f"results_parqo{suffix}.json")

    print(f"running multi-zone skeleton, n={n_target}, init={args.init}, "
          "iter_192 dual-bump schedule...", flush=True)
    x, y = run_with_schedule(schedule_fn, sim, n_target, zones,
                             min_spacing, wd, ws, weights, seed=args.seed,
                             init=args.init)

    with open(LAYOUT_OUT, "w") as f:
        json.dump({"x": [float(v) for v in x],
                   "y": [float(v) for v in y]}, f)

    # Evaluation
    def aep(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return float(jnp.sum(p * weights[:, None]) * 8760 / 1e6)

    sdf = multizone_sdf(x, y, [jnp.asarray(z) for z in zones])
    d = np.sqrt((np.asarray(x)[:, None] - np.asarray(x)[None, :]) ** 2
                + (np.asarray(y)[:, None] - np.asarray(y)[None, :]) ** 2)
    np.fill_diagonal(d, np.inf)
    max_sdf = float(sdf.max())      # <=0 means every turbine inside a zone
    min_dist = float(d.min())
    zone_pen = float(jnp.sum(jnp.maximum(0.0, sdf) ** 2))
    feasible = zone_pen < 1e-3 and min_dist >= min_spacing * 0.99

    n_inside = int((sdf <= 1.0).sum())
    n_pairs_viol = int((d < min_spacing * 0.99).sum() // 2)
    results = {
        "seed": args.seed,
        "n_target": n_target,
        "init": args.init,
        "optimized": {
            "aep_gwh": aep(x, y),
            "max_zone_sdf_m": max_sdf,
            "zone_penalty": zone_pen,
            "min_spacing_m": min_dist,
            "n_inside_zones": n_inside,
            "n_spacing_violating_pairs": n_pairs_viol,
            "feasible": feasible,
        },
    }
    with open(results_out, "w") as f:
        json.dump(results, f, indent=2)

    r = results["optimized"]
    print(f"optimized: AEP {r['aep_gwh']:8.3f} GWh | "
          f"max zone SDF {r['max_zone_sdf_m']:7.1f} m | "
          f"min spacing {r['min_spacing_m']:6.1f} m | "
          f"inside zones {n_inside}/{n_target} | "
          f"spacing-violating pairs {n_pairs_viol} | "
          f"{'FEASIBLE' if r['feasible'] else 'INFEASIBLE'}")


if __name__ == "__main__":
    main()
