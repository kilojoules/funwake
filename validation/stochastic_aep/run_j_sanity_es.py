"""Sanity: factor=1.0 ROWP with ES-on running-max trigger.
Confirms whether ROWP needs ES-on to hit ~4271 (paper claim)."""
import os, sys, json, time, traceback
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dependencies/pixwake/src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "validation/stochastic_aep"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pixwake.optim.sgd import boundary_penalty, spacing_penalty
from stochastic_aep import build_sim
from matrix_categorical_aep import categorical_rose_aep_factory, deterministic_full_rose_aep
from run_step3 import run_with_stochastic_schedule_es
from run_step3_rowp import _translate_to_local
from schedules_ablation import funwake_iter192_alpha_scaled

with open(os.path.join(PROJECT_ROOT, "results/problem_rowp.json")) as f:
    problem = json.load(f)
sim, D = build_sim(problem, wake_model="noj_0.05")
aep_stoch_fn = categorical_rose_aep_factory(sim, problem["wind_rose"])
boundary_local, _, _ = _translate_to_local(problem["boundary_vertices"])
n_target = int(problem["n_target"])
min_spacing = float(problem["min_spacing_m"])
weights = problem["wind_rose"]["weights"]
wd = problem["wind_rose"]["directions_deg"]

for ss in (100000, 200000, 300000):
    for es_on in (False, True):
        sched = funwake_iter192_alpha_scaled(lr_init=50.0, factor=1.0)
        t0 = time.time()
        x_opt, y_opt = run_with_stochastic_schedule_es(
            sched, sim, aep_stoch_fn, 50,
            n_target, boundary_local, min_spacing, weights, wd,
            total_steps=8000, init_seed=0, sample_seed=ss,
            early_stopping=es_on, es_threshold=0.1,
        )
        elapsed = time.time() - t0
        bnd_j = jnp.array(boundary_local)
        bp = float(boundary_penalty(jnp.array(np.asarray(x_opt)), jnp.array(np.asarray(y_opt)), bnd_j))
        sp = float(spacing_penalty(jnp.array(np.asarray(x_opt)), jnp.array(np.asarray(y_opt)), min_spacing))
        aep = deterministic_full_rose_aep(sim, jnp.array(np.asarray(x_opt)), jnp.array(np.asarray(y_opt)), problem["wind_rose"])
        feas = (bp < 1e-2) and (sp < 1e-2)
        print(f"ss={ss} ES={'on' if es_on else 'off'}  AEP={float(aep):.2f}  bp={bp:.2e}  sp={sp:.2e}  feas={feas}  {elapsed:.1f}s", flush=True)
