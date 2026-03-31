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

    # Load the optimizer module
    spec = importlib.util.spec_from_file_location("optimizer", optimizer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "optimize"):
        print("ERROR: optimizer module must define an optimize() function",
              file=sys.stderr)
        sys.exit(1)

    # Load problem
    info = load_problem(os.environ["FUNWAKE_PROBLEM"])

    # Build simulation (LLM cannot modify this)
    sim = build_sim(info)

    # Extract problem parameters
    wd = jnp.array(info["wind_rose"]["directions_deg"])
    ws = jnp.array(info["wind_rose"]["speeds_ms"])
    weights = jnp.array(info["wind_rose"]["weights"])
    boundary = jnp.array(info["boundary_vertices"])
    init_x = jnp.array(info["init_x"])
    init_y = jnp.array(info["init_y"])
    min_spacing = info["min_spacing_m"]

    # Call the LLM's optimize function
    opt_x, opt_y = mod.optimize(
        sim=sim,
        init_x=init_x,
        init_y=init_y,
        boundary=boundary,
        min_spacing=min_spacing,
        wd=wd,
        ws=ws,
        weights=weights,
    )

    # Write output
    with open(os.environ["FUNWAKE_OUTPUT"], "w") as f:
        json.dump({
            "x": [float(v) for v in opt_x],
            "y": [float(v) for v in opt_y],
        }, f)


if __name__ == "__main__":
    main()
