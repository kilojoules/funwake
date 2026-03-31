#!/usr/bin/env python
"""Unit tests for optimizer modules.

Tests that an optimizer module:
  - Defines an optimize() function
  - Returns the correct number of turbines
  - Satisfies boundary and spacing constraints
  - Produces non-degenerate AEP

Usage (from playground/):
    python test_optimizer.py <optimizer_module.py> <problem.json> [timeout]

The module is loaded via the harness (harness.py), which builds the
WakeSimulation and calls optimize().
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def load_problem(problem_path):
    with open(problem_path) as f:
        return json.load(f)


def run_via_harness(optimizer_path, problem_path, timeout=120):
    """Run an optimizer module via the harness and return the output layout."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    harness_path = os.path.join(os.path.dirname(__file__), "harness.py")
    pixwake_src = os.path.join(os.path.dirname(__file__), "pixwake", "src")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": pixwake_src,
        "JAX_ENABLE_X64": "True",
        "FUNWAKE_PROBLEM": os.path.abspath(problem_path),
        "FUNWAKE_OUTPUT": output_path,
    }

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, harness_path, os.path.abspath(optimizer_path)],
        capture_output=True, text=True, timeout=timeout,
        cwd=os.path.dirname(__file__), env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        os.unlink(output_path) if os.path.exists(output_path) else None
        return None, result.stderr[-2000:], elapsed

    if not os.path.exists(output_path):
        return None, "No output file written", elapsed

    with open(output_path) as f:
        layout = json.load(f)
    os.unlink(output_path)
    return layout, None, elapsed


def check_layout(layout, info):
    """Validate a layout against the problem definition."""
    results = []

    # 1. Correct number of turbines
    n_expected = info["n_target"]
    n_got = len(layout.get("x", []))
    results.append((
        "turbine_count",
        n_got == n_expected,
        f"expected {n_expected}, got {n_got}",
    ))

    if n_got == 0:
        results.append(("boundary", False, "no turbines to check"))
        results.append(("spacing", False, "no turbines to check"))
        results.append(("aep_positive", False, "no turbines to check"))
        return results

    x = jnp.array(layout["x"])
    y = jnp.array(layout["y"])
    boundary = jnp.array(info["boundary_vertices"])
    min_spacing = info["min_spacing_m"]

    # 2. Boundary constraint
    bnd_pen = float(boundary_penalty(x, y, boundary))
    results.append((
        "boundary",
        bnd_pen < 1e-3,
        f"penalty={bnd_pen:.6f} (need < 1e-3)",
    ))

    # 3. Spacing constraint
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(len(x)) * 1e10)
    min_dist = float(jnp.min(dist))
    threshold = min_spacing * 0.99
    results.append((
        "spacing",
        min_dist >= threshold,
        f"min_dist={min_dist:.1f}m (need >= {threshold:.1f}m)",
    ))

    # 4. Non-degenerate AEP
    D = info["rotor_diameter"]
    hub_height = info.get("hub_height", 150.0)
    t = info["turbine"]
    ws_arr = jnp.array(t["power_curve_ws"], dtype=float)
    power_arr = jnp.array(t["power_curve_kw"], dtype=float)
    ct_ws = jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float)
    ct_arr = jnp.array(t["ct_curve_ct"], dtype=float)
    turb = Turbine(rotor_diameter=D, hub_height=hub_height,
                   power_curve=Curve(ws=ws_arr, values=power_arr),
                   ct_curve=Curve(ws=ct_ws, values=ct_arr))
    sim = WakeSimulation(turb, BastankhahGaussianDeficit(k=0.04))

    wd = jnp.array(info["wind_rose"]["directions_deg"])
    ws = jnp.array(info["wind_rose"]["speeds_ms"])
    weights = jnp.array(info["wind_rose"]["weights"])
    r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
    p = r.power()[:, :len(x)]
    aep = float(jnp.sum(p * weights[:, None]) * 8760 / 1e6)

    max_rated = float(jnp.max(power_arr))
    theoretical_max = n_got * max_rated * 8760 / 1e6
    results.append((
        "aep_positive",
        aep > theoretical_max * 0.05,
        f"AEP={aep:.2f} GWh (theoretical max ~{theoretical_max:.0f})",
    ))

    return results


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <optimizer_module.py> <problem.json> [timeout]")
        sys.exit(1)

    optimizer_path = sys.argv[1]
    problem_path = sys.argv[2]
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 120

    info = load_problem(problem_path)
    print(f"Problem: {info.get('farm_name', problem_path)}")
    print(f"  {info['n_target']} turbines, D={info['rotor_diameter']}m, "
          f"spacing={info['min_spacing_m']}m")
    print(f"  {len(info['wind_rose']['directions_deg'])} wind sectors")
    print()

    print(f"Running {optimizer_path} via harness...")
    layout, error, elapsed = run_via_harness(optimizer_path, problem_path, timeout)

    if error:
        print(f"SCRIPT FAILED ({elapsed:.1f}s):")
        print(f"  {error[:500]}")
        sys.exit(1)

    print(f"Completed in {elapsed:.1f}s")
    print()

    results = check_layout(layout, info)
    all_passed = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}: {detail}")

    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
