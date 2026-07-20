"""Step 3 — 740-10 ROWP four-cell feasibility-confound + generalization test.

Cells (per the user's spec):
  topfarm_default_decay  OFF
  topfarm_default_decay  ON   ← pivotal cell
  funwake_iter192        OFF
  funwake_iter192        ON

Setup:
  - 740-10 ROWP irregular: N=74, NOJ k=0.05 + SquaredSum (Part 1 engine).
  - Stochastic K=50 MC AEP gradient (Part 2 unbiased, z=-0.58 on this farm).
  - 8000 SGD iters per restart, 20 restarts per cell, shared seeds across cells.
  - Threshold = 0.1.

Pre-flight gate: confirm topfarm_default_decay schedule produces
lr_ratio ≤ threshold somewhere in [0, total_steps). If not, ES is inert and
the test is void.

Per-cell metrics: strict feasibility /20, practical feasibility /20,
boundary-penalty range, min-pair-distance range, AEP mean ± std.

For ES-ON cells: ES-trigger statistics (iter at which lr_ratio first crosses,
fraction of restarts that triggered).

Usage:
    PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \\
        validation/stochastic_aep/run_step3_rowp.py \\
        --restarts 20 --total-steps 8000 --K 50 \\
        --out validation/stochastic_aep/step3_rowp.json
"""
import argparse
import json
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from pixwake.optim.sgd import boundary_penalty, spacing_penalty

from stochastic_aep import build_sim, deterministic_fine_grid_aep, stochastic_aep_factory
from run_part3 import funwake_iter192, wind_aware_init, check_feasibility
from run_step3 import run_with_stochastic_schedule_es, summarize


# --- TopFarm-default decaying schedule via Quick 2023 / pixwake bisection ---

def _bisect_mid(lr_init: float, gamma_min: float, T: int, n_iter: int = 100) -> float:
    """Bisect mid such that prod_{t=1..T} 1/(1+mid*t) == gamma_min / lr_init.

    Mirrors pixwake.optim.sgd._compute_mid_bisection (and matches TopFarm's
    convention up to an off-by-one index — but Step 2's lr-decay off-by-one
    cancels in this run because all four cells run pixwake's convention).
    """
    target = float(gamma_min) / float(lr_init)
    lo, hi = 0.0, 0.1
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        # compute final lr factor
        prod = 1.0
        for t in range(1, T + 1):
            prod *= 1.0 / (1.0 + mid * t)
        if prod < target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def topfarm_default_decay(lr_init: float = 50.0, gamma_min_factor: float = 0.01,
                          total_steps: int = 8000):
    """TopFarm / Quick 2023 default SGD schedule: bisected inverse-time LR decay
    from lr_init to lr_init * gamma_min_factor over `total_steps` steps.

    Adam: beta1=0.1, beta2=0.2. Alpha rescaling: alpha = alpha_0 * lr_0 / lr (so
    constraint enforcement grows as lr shrinks, matching TopFarm's behavior)."""
    lr0_val = float(lr_init)
    gamma_min = lr0_val * float(gamma_min_factor)
    mid = _bisect_mid(lr0_val, gamma_min, int(total_steps))
    # Pre-tabulate lr trajectory (deterministic in `step`, no closed form for
    # the running product).
    ts = np.arange(1, total_steps + 1, dtype=np.float64)
    factors = 1.0 / (1.0 + mid * ts)
    lr_array = lr0_val * np.cumprod(factors)
    lr_array_j = jnp.asarray(lr_array, dtype=jnp.float64)

    def apply(step, total_steps_arg, lr0_arg, alpha0_arg):
        lr = lr_array_j[step]
        # alpha grows as lr shrinks; matches TopFarm's per-step rescale
        alpha = alpha0_arg * lr0_val / jnp.maximum(lr, 1e-10)
        beta1 = jnp.float64(0.1)
        beta2 = jnp.float64(0.2)
        return lr, alpha, beta1, beta2
    # Expose the pre-tabulated trajectory for pre-flight introspection
    apply.lr_trajectory = lr_array
    apply.mid = mid
    apply.lr_init = lr0_val
    apply.gamma_min = gamma_min
    return apply


def preflight_es(schedule_apply, total_steps: int, threshold: float) -> dict:
    """Confirm lr_ratio drops to or below threshold somewhere in [0, T)."""
    lr_traj = schedule_apply.lr_trajectory
    lr0 = schedule_apply.lr_init
    lr_ratio = lr_traj / lr0
    first_cross = int(np.argmax(lr_ratio <= threshold)) if (lr_ratio <= threshold).any() else None
    final_ratio = float(lr_ratio[-1])
    return {
        "lr_init": float(lr0),
        "gamma_min": float(schedule_apply.gamma_min),
        "mid": float(schedule_apply.mid),
        "total_steps": int(total_steps),
        "threshold": float(threshold),
        "lr_ratio_at_25pct": float(lr_ratio[total_steps // 4]),
        "lr_ratio_at_50pct": float(lr_ratio[total_steps // 2]),
        "lr_ratio_at_75pct": float(lr_ratio[3 * total_steps // 4]),
        "lr_ratio_final": final_ratio,
        "iter_first_cross_threshold": first_cross,
        "crosses_threshold": first_cross is not None,
    }


def _translate_to_local(boundary_utm):
    """Translate UTM-scale boundary to origin-centred coords AND enforce
    CCW winding. Pixwake's boundary_penalty + wind_aware_init's
    inside-polygon filter assume CCW (positive signed area). The 740-10
    Borssele polygon ships CW, so we reverse here.
    Returns (boundary_local_ccw, offset_x, offset_y)."""
    arr = np.array(boundary_utm)
    offset_x = float(arr[:, 0].mean())
    offset_y = float(arr[:, 1].mean())
    local = arr - np.array([offset_x, offset_y])
    # Shoelace signed area — positive if CCW
    n = len(local)
    signed_area = 0.5 * sum(
        local[i, 0] * local[(i + 1) % n, 1] - local[(i + 1) % n, 0] * local[i, 1]
        for i in range(n)
    )
    if signed_area < 0:
        local = local[::-1]
    return [[float(v[0]), float(v[1])] for v in local], offset_x, offset_y


def run_cell(label, schedule_fn, sim, aep_stoch_fn, problem, resource, args,
             early_stopping: bool, es_threshold: float):
    # Shift UTM-scale boundary to local origin-centred coords so the
    # constraint-penalty scale matches DEI (penalty ~ distance², so
    # UTM millions → 1e10× penalty inflation, which destabilizes Adam).
    # The wake simulation is translation-invariant, so this only affects
    # the SGD geometry, not AEP.
    boundary_utm = problem["boundary_vertices"]
    boundary_local, ox, oy = _translate_to_local(boundary_utm)
    boundary = boundary_local
    min_spacing = float(problem["min_spacing_m"])
    n_target = int(problem["n_target"])
    weights = resource["sector_probability"]
    wd = resource["sector_centers_deg"]

    print(f"\n=== cell: {label}  ES={early_stopping} threshold={es_threshold} ===", flush=True)
    per_restart = []
    for r in range(args.restarts):
        t0 = time.time()
        x_opt, y_opt = run_with_stochastic_schedule_es(
            schedule_fn, sim, aep_stoch_fn, args.K,
            n_target, boundary, min_spacing, weights, wd,
            total_steps=args.total_steps,
            init_seed=r, sample_seed=r + 200_000,
            early_stopping=early_stopping, es_threshold=es_threshold,
        )
        elapsed = time.time() - t0
        feas = check_feasibility(x_opt, y_opt, boundary, min_spacing)
        try:
            aep_det = deterministic_fine_grid_aep(
                sim, x_opt, y_opt, resource,
                ws_min=4.0, ws_max=25.0, ws_step=1.0, wd_step=1.0,
            )
        except Exception:
            aep_det = None
        entry = {
            "restart": r,
            "elapsed_s": round(elapsed, 1),
            "aep_gwh_det_weibull": aep_det,
            **feas,
        }
        per_restart.append(entry)
        tag = "✓" if feas["feasible"] else "✗"
        aep_str = f"{aep_det:.2f}" if aep_det is not None else "ERR"
        print(
            f"  r={r:2d} {elapsed:.0f}s {tag} AEP={aep_str}  "
            f"bp={feas['boundary_violation']:.2e} sp={feas['spacing_penalty']:.2e} "
            f"min_d={feas['min_pair_dist_m']:.0f}m",
            flush=True,
        )
    return per_restart


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", default="validation/stochastic_aep/problem_740.json")
    p.add_argument("--resource", default="validation/stochastic_aep/rowp_weibull_12.json")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    p.add_argument("--restarts", type=int, default=20)
    p.add_argument("--es-threshold", type=float, default=0.1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.problem) as f:
        problem = json.load(f)
    with open(args.resource) as f:
        resource = json.load(f)
    sim, D = build_sim(problem, wake_model="noj_0.05")
    aep_stoch_fn = stochastic_aep_factory(sim, resource)

    baseline_sched = topfarm_default_decay(
        lr_init=50.0, gamma_min_factor=0.01, total_steps=args.total_steps,
    )
    iter192_sched = funwake_iter192()

    # ===== PRE-FLIGHT GATE =====
    pf_baseline = preflight_es(baseline_sched, args.total_steps, args.es_threshold)
    print("\n=== PRE-FLIGHT: topfarm_default_decay schedule ===")
    print(json.dumps(pf_baseline, indent=2))
    if not pf_baseline["crosses_threshold"]:
        print("\n*** GATE FAIL: baseline lr_ratio never crosses threshold. ***", flush=True)
        out = {
            "config": vars(args),
            "preflight": {"topfarm_default_decay": pf_baseline},
            "verdict": "VOID: baseline never crosses threshold",
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return

    schedules = {
        "topfarm_default_decay": baseline_sched,
        "funwake_iter192": iter192_sched,
    }
    cells = {}
    t_start = time.time()
    for sch_name, sched in schedules.items():
        for es_on, es_label in [(False, "OFF"), (True, "ON")]:
            key = f"{sch_name}__ES_{es_label}"
            t_cell = time.time()
            pr = run_cell(
                key, sched, sim, aep_stoch_fn, problem, resource, args,
                early_stopping=es_on, es_threshold=args.es_threshold,
            )
            cells[key] = {
                "schedule": sch_name,
                "early_stopping": es_on,
                "es_threshold": args.es_threshold,
                "cell_elapsed_s": round(time.time() - t_cell, 1),
                "per_restart": pr,
                "summary": summarize(pr),
            }
            print(f"   ↳ summary: {cells[key]['summary']}", flush=True)

    # Per ES-ON cell: trigger statistics. Schedule is deterministic, so all
    # restarts trigger at the same iter (or none). Report iter of first cross.
    trigger_info = {}
    for key, cell in cells.items():
        if not cell["early_stopping"]:
            continue
        sch_name = cell["schedule"]
        sched = schedules[sch_name]
        if hasattr(sched, "lr_trajectory"):
            lr_traj = sched.lr_trajectory
            lr0 = sched.lr_init
            lr_ratio = lr_traj / lr0
            first_cross = int(np.argmax(lr_ratio <= args.es_threshold))
            crosses = bool((lr_ratio <= args.es_threshold).any())
        else:
            # iter_192 schedule has no .lr_trajectory; compute it
            lr_at = []
            from run_part3 import funwake_iter192 as _f192
            apply_fn = _f192()
            for i in range(args.total_steps):
                lr_v, *_ = apply_fn(i, args.total_steps, 50.0, 1.0)
                lr_at.append(float(lr_v))
            lr_traj = np.asarray(lr_at)
            lr_init_est = float(lr_traj.max())
            lr_ratio = lr_traj / lr_init_est
            crosses = bool((lr_ratio <= args.es_threshold).any())
            first_cross = int(np.argmax(lr_ratio <= args.es_threshold)) if crosses else None
        trigger_info[key] = {
            "trigger_fraction": 1.0 if crosses else 0.0,  # deterministic schedule
            "iter_first_cross": first_cross,
            "n_restarts_triggered": args.restarts if crosses else 0,
            "lr_ratio_max": float(lr_ratio.max()),
            "lr_ratio_min": float(lr_ratio.min()),
        }

    out = {
        "config": vars(args),
        "preflight": {
            "topfarm_default_decay": pf_baseline,
        },
        "cells": cells,
        "es_trigger_info": trigger_info,
        "elapsed_total_s": round(time.time() - t_start, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== SUMMARIES (740-10 ROWP) ===")
    for k, c in cells.items():
        print(f"{k}: {c['summary']}")
    print("\n=== ES trigger info ===")
    print(json.dumps(trigger_info, indent=2))
    print(f"\nWrote {args.out}  ({out['elapsed_total_s']:.1f}s)")


if __name__ == "__main__":
    main()
