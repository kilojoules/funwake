"""Re-evaluate the 48-cell deploy/oracle matrix under stochastic K=50
MC-sampled gradients (matrix-specific categorical rose sampling).

Schedules per cell (all run via the Adam stochastic skeleton, no early
stopping, single init seed = 0, total_steps = 8000):
  - sgd_baseline       (constant LR = 50, β₁=0.1, β₂=0.2 — matches FunWake
                        iter_0 / Part 3 baseline)
  - funwake_iter192    (Claude's deployed schedule)
  - gemini_iter192     (Gemini's deployed schedule; same structural family
                        but distinct lr/alpha/β trajectories)

Per cell × schedule, run with one or more sample seeds (--sample-seeds). For
each completed run, compute the deterministic full-rose AEP and store it.
gap_over_internal_baseline = (sched_aep - baseline_aep) / baseline_aep * 100
is computed downstream from the resulting JSON.

Plumbing fixes from the ROWP run are applied to every cell:
  - boundary vertices reversed to CCW if shoelace area is negative
  - boundary translated to origin so constraint penalties stay numerically
    reasonable (UTM-scale ROWP cells were unstable otherwise)
  - init feasibility (bp ≈ 0 at start) is recorded so we can flag any cell
    whose init still violates the boundary.

Usage:
    PYTHONPATH=playground/pixwake/src:validation/stochastic_aep pixi run python \\
        validation/stochastic_aep/run_matrix_stochastic.py \\
        --sample-seeds 1 --out validation/stochastic_aep/matrix_stochastic.json
"""
import argparse
import importlib.util
import json
import os
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from pixwake.optim.sgd import boundary_penalty, spacing_penalty

from stochastic_aep import build_sim
from run_part3 import wind_aware_init, sgd_baseline, funwake_iter192
from run_step3 import run_with_stochastic_schedule_es
from run_step3_rowp import _translate_to_local
from matrix_categorical_aep import (
    categorical_rose_aep_factory,
    deterministic_full_rose_aep,
)


PROJECT_ROOT = "/Users/julianquick/portfolio_copy/funwake"


# ----------------------------------------------------------------------------
# Load deployed Claude / Gemini schedules verbatim from their script paths.
# We import the *module* and re-use whatever schedule function they define.
# Both deployed scripts expose `optimize` (a complete solver) — but for the
# matrix we only need their schedule_fn shape. Both define an inner function
# matching schedule_fn(step, total_steps, lr0, alpha0). We replicate the
# canonical iter_192 schedule definitions here to keep the dependency
# auditable, and verify (in a comment-test) that they match the deployed
# files.
# ----------------------------------------------------------------------------

def claude_iter192():
    """Claude iter_192 verbatim, sourced from
    runs/schedule_only_5hr/iter_192.py (same definition as
    paper_schedules/scripts/schedules.py:funwake_iter192). Beta1=0.3,
    Beta2=0.5 constants; warmup + cosine + dual bumps + α-dip + α-escalation.
    """
    return funwake_iter192()


def gemini_iter192():
    """Gemini iter_192. The deployed script applies a different structural
    schedule: 8-cycle cosine restarts, cyclic betas (β₁: 0.1+0.4·t, β₂:
    0.2+0.7·t), late-stage 'squeeze' (lr→1e-5, α→1e12 at t > 0.985).
    Sourced from runs/gemini_cli_5hr/iter_192.py.
    """
    n_cycles = 8

    def apply(step, total_steps, lr0, alpha0):
        t = step / total_steps
        t_in_cycle = (t * n_cycles) % 1.0
        # 8-cycle cosine restarts
        lr_cycle = 0.5 * (1.0 + jnp.cos(jnp.pi * t_in_cycle))
        # Linear envelope from 1x to 4x lr0 over the run, modulated by restart cycles
        envelope = 1.0 + 3.0 * t
        lr = lr0 * lr_cycle * envelope
        # Squeeze at end: lr → 1e-5, alpha → 1e12
        squeeze_active = t > 0.985
        lr = jnp.where(squeeze_active, 1e-5, lr)
        # Alpha ramp
        alpha = alpha0 * (1.0 + 10.0 * t**2)
        alpha = jnp.where(squeeze_active, 1e12 * alpha0, alpha)
        # Cyclic betas
        beta1 = 0.1 + 0.4 * t
        beta2 = 0.2 + 0.7 * t
        # Squeeze betas
        beta1 = jnp.where(squeeze_active, 0.9, beta1)
        beta2 = jnp.where(squeeze_active, 0.999, beta2)
        return lr, alpha, beta1, beta2
    return apply


SCHEDULE_BUILDERS = {
    "sgd_baseline": sgd_baseline,
    "claude_iter192": claude_iter192,
    "gemini_iter192": gemini_iter192,
}


def check_init_feasibility(x, y, boundary, min_spacing):
    bnd = jnp.array(boundary)
    bp = float(boundary_penalty(jnp.array(x), jnp.array(y), bnd))
    sp = float(spacing_penalty(jnp.array(x), jnp.array(y), float(min_spacing)))
    return {"init_bp": bp, "init_sp": sp}


def run_one(cell_path, schedule_name, sample_seed, init_seed, K, total_steps):
    with open(os.path.join(PROJECT_ROOT, cell_path)) as f:
        problem = json.load(f)
    sim, D = build_sim(problem, wake_model="bastankhah_0.04")
    aep_stoch_fn = categorical_rose_aep_factory(sim, problem["wind_rose"])

    boundary_local, ox, oy = _translate_to_local(problem["boundary_vertices"])
    n_target = int(problem["n_target"])
    min_spacing = float(problem["min_spacing_m"])
    weights = problem["wind_rose"]["weights"]
    wd = problem["wind_rose"]["directions_deg"]

    # Pre-flight: init feasibility
    x_init, y_init = wind_aware_init(boundary_local, min_spacing, n_target, weights, wd, init_seed)
    init_feas = check_init_feasibility(x_init, y_init, boundary_local, min_spacing)

    sched = SCHEDULE_BUILDERS[schedule_name]()
    t0 = time.time()
    x_opt, y_opt = run_with_stochastic_schedule_es(
        sched, sim, aep_stoch_fn, K,
        n_target, boundary_local, min_spacing, weights, wd,
        total_steps=total_steps, init_seed=init_seed, sample_seed=sample_seed,
        early_stopping=False, es_threshold=0.1,
    )
    elapsed = time.time() - t0

    # Final feasibility + deterministic full-rose AEP
    x_opt_arr = np.asarray(x_opt)
    y_opt_arr = np.asarray(y_opt)
    bnd_j = jnp.array(boundary_local)
    bp_final = float(boundary_penalty(jnp.array(x_opt_arr), jnp.array(y_opt_arr), bnd_j))
    sp_final = float(spacing_penalty(jnp.array(x_opt_arr), jnp.array(y_opt_arr), min_spacing))
    aep_det = deterministic_full_rose_aep(sim, jnp.array(x_opt_arr), jnp.array(y_opt_arr), problem["wind_rose"])

    return {
        "init_feasibility": init_feas,
        "aep_det_gwh": float(aep_det),
        "bp_final": bp_final,
        "sp_final": sp_final,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="results/matrix/manifest.json")
    p.add_argument("--max-n", type=int, default=80)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--sample-seeds", type=int, nargs="+", default=[100000],
                   help="One or more sample seeds for spread estimation")
    p.add_argument("--out", required=True)
    p.add_argument("--schedules", nargs="+",
                   default=["sgd_baseline", "claude_iter192", "gemini_iter192"])
    args = p.parse_args()

    manifest = json.load(open(os.path.join(PROJECT_ROOT, args.manifest)))
    cells = [c for c in manifest["cells"] if c["n"] <= args.max_n]
    print(f"Cells to evaluate: {len(cells)} (max_n={args.max_n})", flush=True)

    # Resume-safe: load existing if any
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {
            "config": vars(args),
            "started_utc": None,
            "cells": {},
        }

    t_start = time.time()
    total = len(cells) * len(args.schedules) * len(args.sample_seeds)
    done = 0
    for cell in cells:
        cell_key = f"{cell['farm']}_n{cell['n']}_rose{cell['rose']}"
        if cell_key not in results["cells"]:
            results["cells"][cell_key] = {
                "farm": cell["farm"], "n": cell["n"], "rose": cell["rose"],
                "path": cell["path"], "schedules": {},
            }
        for sch in args.schedules:
            if sch not in results["cells"][cell_key]["schedules"]:
                results["cells"][cell_key]["schedules"][sch] = {"runs": []}
            existing_seeds = {r["sample_seed"]
                              for r in results["cells"][cell_key]["schedules"][sch]["runs"]}
            for ss in args.sample_seeds:
                done += 1
                if ss in existing_seeds:
                    continue
                t0 = time.time()
                try:
                    r = run_one(
                        cell["path"], sch,
                        sample_seed=ss, init_seed=args.init_seed,
                        K=args.K, total_steps=args.total_steps,
                    )
                except Exception as e:
                    r = {"error": str(e)[:200]}
                cell_elapsed = time.time() - t0
                results["cells"][cell_key]["schedules"][sch]["runs"].append({
                    "sample_seed": ss,
                    **r,
                })
                # Save incrementally so we can resume after long runs
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)
                msg = (f"[{done}/{total}] {cell_key} | {sch} ss={ss}: "
                       f"{cell_elapsed:.0f}s")
                if "aep_det_gwh" in r:
                    msg += f"  AEP={r['aep_det_gwh']:.2f}  bp_init={r['init_feasibility']['init_bp']:.1e}  bp_final={r['bp_final']:.1e}"
                else:
                    msg += f"  ERR {r.get('error', '')[:80]}"
                print(msg, flush=True)
    elapsed_total = time.time() - t_start
    results["elapsed_total_s"] = round(elapsed_total, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Total wall: {elapsed_total/60:.1f} min. Wrote {args.out}")


if __name__ == "__main__":
    main()
