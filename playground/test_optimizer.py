#!/usr/bin/env python
"""Unit tests for optimizer modules.

Tests that an optimizer module:
  - Defines optimize() with the correct signature
  - Returns the correct number of turbines
  - Returns finite, non-NaN positions
  - Satisfies boundary and spacing constraints
  - Produces non-degenerate AEP
  - Works on multiple problem sizes (quick check)

Usage:
    # Full test (runs optimizer, ~20-30s)
    python test_optimizer.py <optimizer_module.py> <problem.json>

    # Quick test (signature + import check only, <1s)
    python test_optimizer.py <optimizer_module.py> --quick
"""

import importlib.util
import inspect
import json
import math
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


def load_module(path):
    """Import an optimizer module and return it."""
    spec = importlib.util.spec_from_file_location("optimizer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_signature(mod):
    """Check that optimize() exists with the right parameters."""
    results = []

    if not hasattr(mod, "optimize"):
        results.append(("has_optimize", False, "module has no optimize() function"))
        return results
    results.append(("has_optimize", True, "optimize() found"))

    sig = inspect.signature(mod.optimize)
    params = list(sig.parameters.keys())
    expected = ["sim", "n_target", "boundary", "min_spacing", "wd", "ws", "weights"]

    if params == expected:
        results.append(("signature", True, f"params: {params}"))
    else:
        results.append(("signature", False,
                        f"expected {expected}, got {params}"))

    return results


def check_quick_run(mod):
    """Run optimize() on a tiny problem to catch crashes fast."""
    results = []

    # Tiny 3-turbine problem
    D = 100.0
    ws_arr = jnp.array([0, 5, 10, 15, 20, 25.0])
    power = jnp.array([0, 100, 500, 800, 800, 800.0])
    ct = jnp.array([0.8, 0.8, 0.7, 0.5, 0.3, 0.2])
    turb = Turbine(rotor_diameter=D, hub_height=80.0,
                   power_curve=Curve(ws=ws_arr, values=power),
                   ct_curve=Curve(ws=ws_arr, values=ct))
    sim = WakeSimulation(turb, BastankhahGaussianDeficit(k=0.04))

    boundary = jnp.array([[-2000, -2000], [2000, -2000],
                           [2000, 2000], [-2000, 2000.0]])
    try:
        opt_x, opt_y = mod.optimize(
            sim=sim, n_target=3, boundary=boundary,
            min_spacing=400.0,
            wd=jnp.array([0, 90, 180, 270.0]),
            ws=jnp.array([8, 9, 7, 10.0]),
            weights=jnp.array([0.25, 0.25, 0.25, 0.25]),
        )
    except Exception as e:
        results.append(("quick_run", False, f"crashed: {e}"))
        return results

    results.append(("quick_run", True, "ran without error"))

    # Check output types
    n = len(opt_x)
    results.append(("quick_count", n == 3, f"returned {n} turbines, expected 3"))

    # Check finite
    x_finite = bool(jnp.all(jnp.isfinite(jnp.array(opt_x))))
    y_finite = bool(jnp.all(jnp.isfinite(jnp.array(opt_y))))
    results.append(("quick_finite", x_finite and y_finite,
                    f"x_finite={x_finite}, y_finite={y_finite}"))

    return results


def check_stressed_polygon(mod):
    """Run on a tight, elongated polygon to stress-test constraint handling.

    This catches optimizers that work on spacious polygons (DEI) but fail
    when packing density is high and the boundary is narrow. A thin rhombus
    with 25 turbines at 600m spacing leaves very little margin.
    """
    results = []

    D = 150.0
    ws_arr = jnp.array([0, 5, 10, 15, 20, 25.0])
    power = jnp.array([0, 50, 300, 600, 600, 600.0])
    ct = jnp.array([0.8, 0.8, 0.7, 0.5, 0.3, 0.2])
    turb = Turbine(rotor_diameter=D, hub_height=100.0,
                   power_curve=Curve(ws=ws_arr, values=power),
                   ct_curve=Curve(ws=ws_arr, values=ct))
    sim = WakeSimulation(turb, BastankhahGaussianDeficit(k=0.04))

    # Thin rhombus: 16km long, 4km wide — tight packing
    boundary = jnp.array([
        [0.0, 0.0], [8000.0, -2000.0],
        [16000.0, 0.0], [8000.0, 2000.0],
    ])
    n_target = 25
    min_spacing = 600.0

    try:
        opt_x, opt_y = mod.optimize(
            sim=sim, n_target=n_target, boundary=boundary,
            min_spacing=min_spacing,
            wd=jnp.array([0, 90, 180, 270.0]),
            ws=jnp.array([9, 8, 9, 8.0]),
            weights=jnp.array([0.25, 0.25, 0.25, 0.25]),
        )
    except Exception as e:
        results.append(("stressed_run", False, f"crashed: {e}"))
        return results

    results.append(("stressed_run", True, "ran without error"))

    n = len(opt_x)
    results.append(("stressed_count", n == n_target,
                    f"returned {n}, expected {n_target}"))

    if n == 0:
        results.append(("stressed_boundary", False, "no turbines"))
        results.append(("stressed_spacing", False, "no turbines"))
        return results

    x = jnp.array(opt_x)
    y = jnp.array(opt_y)

    # Check finite
    finite = bool(jnp.all(jnp.isfinite(x)) and jnp.all(jnp.isfinite(y)))
    if not finite:
        results.append(("stressed_finite", False, "NaN or Inf in output"))
        results.append(("stressed_boundary", False, "non-finite positions"))
        results.append(("stressed_spacing", False, "non-finite positions"))
        return results

    # Boundary feasibility
    bnd_pen = float(boundary_penalty(x, y, boundary))
    results.append(("stressed_boundary", bnd_pen < 1e-3,
                    f"penalty={bnd_pen:.6f} (need < 1e-3)"))

    # Spacing feasibility
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(n) * 1e10)
    min_dist = float(jnp.min(dist))
    threshold = min_spacing * 0.99
    results.append(("stressed_spacing", min_dist >= threshold,
                    f"min_dist={min_dist:.1f}m (need >= {threshold:.1f}m)"))

    return results


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

    n_expected = info["n_target"]
    n_got = len(layout.get("x", []))
    results.append(("turbine_count", n_got == n_expected,
                    f"expected {n_expected}, got {n_got}"))

    if n_got == 0:
        for name in ["finite", "no_duplicates", "boundary", "spacing", "aep_positive"]:
            results.append((name, False, "no turbines"))
        return results

    x = jnp.array(layout["x"])
    y = jnp.array(layout["y"])

    # Finite values
    finite = bool(jnp.all(jnp.isfinite(x)) and jnp.all(jnp.isfinite(y)))
    results.append(("finite", finite,
                    "all positions finite" if finite else "NaN or Inf detected"))

    if not finite:
        for name in ["no_duplicates", "boundary", "spacing", "aep_positive"]:
            results.append((name, False, "non-finite positions"))
        return results

    # No duplicate positions (the spacing=0 bug)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(len(x)) * 1e10)
    min_dist = float(jnp.min(dist))
    results.append(("no_duplicates", min_dist > 1.0,
                    f"min_dist={min_dist:.1f}m" if min_dist > 1.0
                    else f"DUPLICATE POSITIONS: min_dist={min_dist:.1f}m"))

    boundary = jnp.array(info["boundary_vertices"])
    min_spacing = info["min_spacing_m"]

    # Boundary constraint
    bnd_pen = float(boundary_penalty(x, y, boundary))
    results.append(("boundary", bnd_pen < 1e-3,
                    f"penalty={bnd_pen:.6f} (need < 1e-3)"))

    # Spacing constraint
    threshold = min_spacing * 0.99
    results.append(("spacing", min_dist >= threshold,
                    f"min_dist={min_dist:.1f}m (need >= {threshold:.1f}m)"))

    # Non-degenerate AEP
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
    results.append(("aep_positive", aep > theoretical_max * 0.05,
                    f"AEP={aep:.2f} GWh (theoretical max ~{theoretical_max:.0f})"))

    return results


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <optimizer.py> [problem.json] [timeout]")
        print(f"       python {sys.argv[0]} <optimizer.py> --quick")
        sys.exit(1)

    optimizer_path = sys.argv[1]
    quick_mode = "--quick" in sys.argv

    # Always run signature and quick checks
    print(f"Loading {optimizer_path}...")
    try:
        mod = load_module(os.path.abspath(optimizer_path))
    except Exception as e:
        print(f"IMPORT FAILED: {e}")
        sys.exit(1)

    print("\n=== Signature Check ===")
    sig_results = check_signature(mod)
    all_passed = True
    for name, passed, detail in sig_results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}: {detail}")

    if not all_passed:
        print("\nSignature check failed — fix before running full tests.")
        sys.exit(1)

    print("\n=== Quick Run (3 turbines, tiny problem) ===")
    quick_results = check_quick_run(mod)
    for name, passed, detail in quick_results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}: {detail}")

    print("\n=== Stressed Polygon (25 turbines, thin rhombus) ===")
    stressed_results = check_stressed_polygon(mod)
    for name, passed, detail in stressed_results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}: {detail}")

    if quick_mode:
        print()
        if all_passed:
            print("QUICK TESTS PASSED")
        else:
            print("QUICK TESTS FAILED")
            sys.exit(1)
        return

    # Full test requires a problem JSON
    if len(sys.argv) < 3 or sys.argv[2] == "--quick":
        print("\nQuick tests done. Provide a problem.json for full tests.")
        return

    problem_path = sys.argv[2]
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "--quick" else 120

    info = load_problem(problem_path)
    print(f"\n=== Full Test: {info.get('farm_name', problem_path)} ===")
    print(f"  {info['n_target']} turbines, D={info['rotor_diameter']}m, "
          f"spacing={info['min_spacing_m']}m")

    print(f"\nRunning via harness...")
    layout, error, elapsed = run_via_harness(optimizer_path, problem_path, timeout)

    if error:
        print(f"SCRIPT FAILED ({elapsed:.1f}s):")
        print(f"  {error[:500]}")
        sys.exit(1)

    print(f"Completed in {elapsed:.1f}s\n")

    layout_results = check_layout(layout, info)
    for name, passed, detail in layout_results:
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
