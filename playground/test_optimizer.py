#!/usr/bin/env python
"""Unit tests for optimizer modules (open-source py_wake backend).

Tests that an optimizer module:
  - Defines optimize() with the correct signature
  - Returns the correct number of turbines
  - Returns finite, non-NaN positions
  - Satisfies boundary and spacing constraints
  - Produces non-degenerate AEP
  - Works on multiple problem sizes (quick check)

Wake model: py_wake PropagateDownwind + Bastankhah-Gaussian deficit
(k=0.04, ambient-WS reference, 2-sigma radius mask, SquaredSum
superposition) via pywake_adapter; constraint penalties are the plain
numpy TopFarm formulations in penalties_np.  AEP, penalty, and gradient
values match the reference scoring model to <1e-6 relative.

Step count: schedule-mode runs use TEST_TOTAL_STEPS = 5000 (the scoring
stack uses 8000).  py_wake gradients cost ~63 ms/step on the 25-turbine
stressed fixture, so 8000 steps would take ~9 min; 5000 keeps the suite
near 5 min while preserving the reference behavior of the seed schedule
(stressed_boundary FAIL with penalty ~0.1; verified against the reference
model at the same step count — note 4000 anomalously passes and 4500 is
marginal, so do not lower this further).

Usage:
    # Full test (runs optimizer via harness)
    python test_optimizer.py <optimizer_module.py> <problem.json>

    # Quick test (signature + fixture runs, ~5 min for schedule modules)
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

from penalties_np import boundary_penalty, spacing_penalty
from pywake_adapter import Curve, Turbine, WakeSimulation, BastankhahGaussianDeficit

# Step count for schedule_fn-based modules (see header).
TEST_TOTAL_STEPS = 5000


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

    # Support two modes: optimize() or schedule_fn()
    if hasattr(mod, "optimize"):
        results.append(("has_function", True, "optimize() found"))
        sig = inspect.signature(mod.optimize)
        params = list(sig.parameters.keys())
        expected = ["sim", "n_target", "boundary", "min_spacing", "wd", "ws", "weights"]
        if params == expected:
            results.append(("signature", True, f"params: {params}"))
        else:
            results.append(("signature", False,
                            f"expected {expected}, got {params}"))
    elif hasattr(mod, "schedule_fn"):
        results.append(("has_function", True, "schedule_fn() found"))
        sig = inspect.signature(mod.schedule_fn)
        params = list(sig.parameters.keys())
        expected = ["step", "total_steps", "lr0", "alpha0"]
        if params == expected:
            results.append(("signature", True, f"params: {params}"))
        else:
            results.append(("signature", False,
                            f"expected {expected}, got {params}"))
    else:
        results.append(("has_function", False,
                        "module must define optimize() or schedule_fn()"))
        return results

    return results


def _call_optimizer(mod, sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Call the module's optimizer — either optimize() or schedule_fn via skeleton."""
    if hasattr(mod, "optimize"):
        return mod.optimize(sim=sim, n_target=n_target, boundary=boundary,
                           min_spacing=min_spacing, wd=wd, ws=ws, weights=weights)
    elif hasattr(mod, "schedule_fn"):
        from skeleton_pywake import run_with_schedule
        return run_with_schedule(mod.schedule_fn, sim, n_target, boundary,
                                min_spacing, wd, ws, weights,
                                total_steps=TEST_TOTAL_STEPS)
    else:
        raise ValueError("Module must define optimize() or schedule_fn()")


def check_quick_run(mod):
    """Run optimizer on a tiny problem to catch crashes fast."""
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
        opt_x, opt_y = _call_optimizer(
            mod, sim, 3, boundary, 400.0,
            jnp.array([0, 90, 180, 270.0]),
            jnp.array([8, 9, 7, 10.0]),
            jnp.array([0.25, 0.25, 0.25, 0.25]),
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
        opt_x, opt_y = _call_optimizer(
            mod, sim, n_target, boundary, min_spacing,
            jnp.array([0, 90, 180, 270.0]),
            jnp.array([9, 8, 9, 8.0]),
            jnp.array([0.25, 0.25, 0.25, 0.25]),
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
    # harness.py runs on the closed-source scoring stack; the caller must
    # provide a PYTHONPATH where its simulation library is importable.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
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


# Feasibility checks (boundary + spacing). With --expect-infeasible these are
# treated as expected-fail (XFAIL) instead of hard failures, so the suite exits
# 0 for a script that is infeasible *by design* — notably the seed schedule
# (results/seed_schedule.py). The baseline is the best-FEASIBLE of 500
# multistarts; the seed trades feasibility for AEP on purpose. See the README
# ("Quick start" / seed infeasibility note).
FEAS_CHECKS = {"stressed_boundary", "stressed_spacing", "boundary", "spacing"}


def _report(results, expect_infeasible):
    """Print a result block; return True unless a hard (non-expected) check failed.

    A failing feasibility check becomes [XFAIL] when expect_infeasible is set,
    so it does not fail the suite.
    """
    ok = True
    for name, passed, detail in results:
        if passed:
            status = "PASS"
        elif expect_infeasible and name in FEAS_CHECKS:
            status = "XFAIL"  # expected: infeasible by design (see README)
        else:
            status = "FAIL"
            ok = False
        print(f"  [{status}] {name}: {detail}")
    return ok


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <optimizer.py> [problem.json] [timeout]")
        print(f"       python {sys.argv[0]} <optimizer.py> --quick [--expect-infeasible]")
        sys.exit(1)

    optimizer_path = sys.argv[1]
    quick_mode = "--quick" in sys.argv
    expect_infeasible = "--expect-infeasible" in sys.argv

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
    all_passed = _report(check_quick_run(mod), expect_infeasible) and all_passed

    print("\n=== Stressed Polygon (25 turbines, thin rhombus) ===")
    all_passed = _report(check_stressed_polygon(mod), expect_infeasible) and all_passed

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

    all_passed = _report(check_layout(layout, info), expect_infeasible) and all_passed

    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
