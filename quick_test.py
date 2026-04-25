#!/usr/bin/env python
"""Quick test script to score an optimizer without pixi."""
import sys
import os

# Setup environment
os.environ['JAX_ENABLE_X64'] = 'True'
sys.path.insert(0, 'playground/pixwake/src')

import json
import time
import importlib.util

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit

def load_module(path):
    """Import an optimizer module."""
    spec = importlib.util.spec_from_file_location("optimizer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_problem(path):
    """Load problem JSON."""
    with open(path) as f:
        return json.load(f)

def score_optimizer(optimizer_path, problem_path='playground/problem.json', timeout=60):
    """Score an optimizer on a problem."""
    try:
        # Load problem
        prob = load_problem(problem_path)

        # Setup simulation
        t = prob['turbine']
        ws_arr = jnp.array(t['power_curve_ws'], dtype=float)
        power_arr = jnp.array(t['power_curve_kw'], dtype=float)
        ct_ws = jnp.array(t.get('ct_curve_ws', t['power_curve_ws']), dtype=float)
        ct_arr = jnp.array(t['ct_curve_ct'], dtype=float)

        turb = Turbine(
            rotor_diameter=prob['rotor_diameter'],
            hub_height=prob.get('hub_height', 150.0),
            power_curve=Curve(ws=ws_arr, values=power_arr),
            ct_curve=Curve(ws=ct_ws, values=ct_arr)
        )
        sim = WakeSimulation(turb, BastankhahGaussianDeficit(k=0.04))

        # Load optimizer
        mod = load_module(optimizer_path)

        # Run optimizer
        boundary = jnp.array(prob['boundary_vertices'])
        wd = jnp.array(prob['wind_rose']['directions_deg'])
        ws = jnp.array(prob['wind_rose']['speeds_ms'])
        weights = jnp.array(prob['wind_rose']['weights'])

        print(f"Running optimizer on {prob['farm_name']} ({prob['n_target']} turbines)...")
        t0 = time.time()
        opt_x, opt_y = mod.optimize(
            sim=sim,
            n_target=prob['n_target'],
            boundary=boundary,
            min_spacing=prob['min_spacing_m'],
            wd=wd,
            ws=ws,
            weights=weights
        )
        elapsed = time.time() - t0

        # Compute AEP
        r = sim(opt_x, opt_y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(opt_x)]
        aep = float(jnp.sum(p * weights[:, None]) * 8760 / 1e6)

        # Check feasibility
        from pixwake.optim.sgd import boundary_penalty, spacing_penalty
        bnd_pen = float(boundary_penalty(opt_x, opt_y, boundary))
        spc_pen = float(spacing_penalty(opt_x, opt_y, prob['min_spacing_m']))
        feasible = (bnd_pen < 1e-3 and spc_pen < 1e-3)

        result = {
            'aep_gwh': round(aep, 2),
            'feasible': feasible,
            'time_s': round(elapsed, 1),
            'boundary_penalty': round(bnd_pen, 6),
            'spacing_penalty': round(spc_pen, 6),
            'baseline': 5544.09,
            'improvement': round((aep - 5544.09) / 5544.09 * 100, 2)
        }

        print(json.dumps(result, indent=2))
        return result

    except Exception as e:
        result = {'error': str(e)}
        print(json.dumps(result, indent=2))
        return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <optimizer.py>")
        sys.exit(1)

    score_optimizer(sys.argv[1])
