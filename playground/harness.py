#!/usr/bin/env python
"""Harness that loads the problem, builds the simulation, and calls
the LLM's optimize() function.

The LLM writes ONLY the optimize() function. This harness handles:
- Loading the problem JSON
- Building the Turbine and WakeSimulation objects
- Providing the wind rose and boundary
- Writing the output

Usage:
    python harness.py <optimizer_module.py>

Environment:
    FUNWAKE_PROBLEM  — path to problem JSON
    FUNWAKE_OUTPUT   — path to write output layout JSON
"""

import jax
jax.config.update("jax_enable_x64", True)

import importlib.util
import json
import os
import sys

import jax.numpy as jnp
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit


def load_problem(path):
    with open(path) as f:
        return json.load(f)


def build_sim(info):
    """Build Turbine and WakeSimulation from problem JSON."""
    D = info["rotor_diameter"]
    hub_height = info.get("hub_height", 150.0)
    t = info["turbine"]
    ws_arr = jnp.array(t["power_curve_ws"], dtype=float)
    power = jnp.array(t["power_curve_kw"], dtype=float)
    ct_ws = jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float)
    ct = jnp.array(t["ct_curve_ct"], dtype=float)
    turbine = Turbine(
        rotor_diameter=D, hub_height=hub_height,
        power_curve=Curve(ws=ws_arr, values=power),
        ct_curve=Curve(ws=ct_ws, values=ct))
    sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))
    return sim


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <optimizer_module.py>")
        sys.exit(1)

    optimizer_path = sys.argv[1]

    # Safety check before loading
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from sandbox import check_code_safety
        with open(optimizer_path) as f:
            code = f.read()
        safe, reason = check_code_safety(code)
        if not safe:
            print(f"SANDBOX BLOCKED: {reason}", file=sys.stderr)
            sys.exit(1)
    except ImportError:
        pass  # sandbox module not available — skip check

    # Load the optimizer module
    spec = importlib.util.spec_from_file_location("optimizer", optimizer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Load problem
    info = load_problem(os.environ["FUNWAKE_PROBLEM"])

    # Build simulation (LLM cannot modify this)
    sim = build_sim(info)

    # Extract problem parameters
    wd = jnp.array(info["wind_rose"]["directions_deg"])
    ws = jnp.array(info["wind_rose"]["speeds_ms"])
    weights = jnp.array(info["wind_rose"]["weights"])
    boundary = jnp.array(info["boundary_vertices"])
    n_target = info["n_target"]
    min_spacing = info["min_spacing_m"]

    # Check for mode flag: --schedule-only forces schedule_fn mode
    schedule_only = "--schedule-only" in sys.argv

    # Initialization seed (for grid subsampling in the skeleton).
    # Defaults to 0 to preserve prior behavior.
    init_seed = int(os.environ.get("FUNWAKE_SEED", "0"))

    if schedule_only:
        if not hasattr(mod, "schedule_fn"):
            print("ERROR: --schedule-only mode requires schedule_fn(), "
                  "not optimize(). Write ONLY:\n"
                  "  def schedule_fn(step, total_steps, lr0, alpha0):\n"
                  "      return (lr, alpha, beta1, beta2)",
                  file=sys.stderr)
            sys.exit(1)
        from skeleton import run_with_schedule
        opt_x, opt_y = run_with_schedule(
            mod.schedule_fn, sim, n_target, boundary,
            min_spacing, wd, ws, weights, seed=init_seed,
        )
    elif hasattr(mod, "optimize"):
        opt_x, opt_y = mod.optimize(
            sim=sim,
            n_target=n_target,
            boundary=boundary,
            min_spacing=min_spacing,
            wd=wd,
            ws=ws,
            weights=weights,
        )
    elif hasattr(mod, "schedule_fn"):
        from skeleton import run_with_schedule
        opt_x, opt_y = run_with_schedule(
            mod.schedule_fn, sim, n_target, boundary,
            min_spacing, wd, ws, weights, seed=init_seed,
        )
    else:
        print("ERROR: module must define optimize() or schedule_fn()",
              file=sys.stderr)
        sys.exit(1)

    # Write output
    with open(os.environ["FUNWAKE_OUTPUT"], "w") as f:
        json.dump({
            "x": [float(v) for v in opt_x],
            "y": [float(v) for v in opt_y],
        }, f)


if __name__ == "__main__":
    main()
