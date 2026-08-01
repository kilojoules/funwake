#!/usr/bin/env python
"""G4: alpha0 normalization gate.

Confirm the skeleton computes alpha0 = mean|grad J| / D, and show numerically
that the v1 driver default mean|grad J| / (0.833 D) = 1.2x that value (so
shipping the /lr form would put the penalty scale 1.2x off). Confirm the native
port consumes the /D alpha0.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FW2 = os.path.dirname(HERE)
ROOT = os.path.dirname(FW2)
sys.path.insert(0, FW2)
sys.path.insert(0, os.path.join(ROOT, "dependencies", "pixwake", "src"))
sys.path.insert(0, os.path.join(ROOT, "playground"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import skeleton_v2
from harness import build_sim  # noqa: E402

C = 200.0 / 240.0   # diameter-rule c


def main():
    prob = json.load(open(os.path.join(ROOT, "results", "problem_dei_n50.json")))
    D = float(prob["rotor_diameter"])
    min_spacing = float(prob["min_spacing_m"])
    n = int(prob["n_target"])
    sim = build_sim(prob)
    wr = prob["wind_rose"]
    wd = jnp.array(wr["directions_deg"], dtype=jnp.float64)
    ws = jnp.array(wr["speeds_ms"], dtype=jnp.float64)
    wt = jnp.array(wr["weights"], dtype=jnp.float64)
    wt = wt / jnp.sum(wt)
    boundary = jnp.array(prob["boundary_vertices"], dtype=jnp.float64)

    gnorm, alpha0_D = skeleton_v2.compute_alpha0_and_gradnorm(
        sim, n, boundary, min_spacing, wd, ws, wt, D, seed=0, zones=None)

    lr0 = C * D                       # 0.833 D = 200
    alpha0_driver = gnorm / lr0       # v1 / clean-room driver default (/lr0)
    ratio = alpha0_driver / alpha0_D

    print("=== G4 alpha0 gate (DEI n50, seed 0) ===")
    print(f"  D = {D}")
    print(f"  mean|grad J|                 = {gnorm:.6e}")
    print(f"  alpha0 (skeleton, /D)        = {alpha0_D:.6e}")
    print(f"  alpha0 (driver default, /lr0={lr0:.3f}) = {alpha0_driver:.6e}")
    print(f"  ratio driver/skeleton        = {ratio:.6f}  (expected 1/0.833 = 1.2)")
    ok = abs(ratio - 1.2) < 1e-6
    print("  native port uses alpha = alpha0*D/lr = mean|grad J|/lr "
          "(consumes the /D alpha0; never forms /lr alpha0)")
    print(f"G4: {'PASS' if ok else 'FAIL'} "
          f"(driver /lr form is 1.2x the shipped /D form)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
