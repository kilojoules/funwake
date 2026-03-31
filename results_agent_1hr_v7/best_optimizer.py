
import jax
jax.config.update("jax_enable_x64", True)
import os, json
import jax.numpy as jnp
import numpy as np
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

with open(os.environ["FUNWAKE_PROBLEM"]) as f:
    info = json.load(f)

# Read ALL config from the problem JSON (must generalize to other farms)
D = info["rotor_diameter"]
hub_height = info.get("hub_height", 150.0)
t = info["turbine"]
ws_arr = jnp.array(t["power_curve_ws"], dtype=float)
power = jnp.array(t["power_curve_kw"], dtype=float)
ct_ws = jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float)
ct = jnp.array(t["ct_curve_ct"], dtype=float)
turbine = Turbine(rotor_diameter=D, hub_height=hub_height,
                  power_curve=Curve(ws=ws_arr, values=power),
                  ct_curve=Curve(ws=ct_ws, values=ct))
sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

wd = jnp.array(info["wind_rose"]["directions_deg"])
ws = jnp.array(info["wind_rose"]["speeds_ms"])
weights = jnp.array(info["wind_rose"]["weights"])
boundary = jnp.array(info["boundary_vertices"])
init_x_original = jnp.array(info["init_x"])
init_y_original = jnp.array(info["init_y"])
min_spacing = info["min_spacing_m"]
n_target = info["n_target"]

def objective(x, y):
    r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
    p = r.power()[:, :len(x)]
    return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

# Stage 1: Feasibility-focused optimization
# Aim for a quick, feasible solution
settings_stage1 = SGDSettings(learning_rate=100.0, max_iter=1000,
                              additional_constant_lr_iterations=500, tol=1e-5,
                              spacing_weight=10.0, boundary_weight=10.0)

key = jax.random.PRNGKey(0)

opt_x_stage1, opt_y_stage1 = topfarm_sgd_solve(objective, init_x_original, init_y_original,
                                                boundary, min_spacing, settings_stage1)

# Stage 2: AEP-focused optimization with multi-start from feasible layout
best_aep = -jnp.inf
best_x, best_y = None, None

num_starts_stage2 = 10  # Number of multi-starts for AEP refinement
perturbation_scale_stage2 = D * 0.05 # Smaller perturbation for refinement

settings_stage2 = SGDSettings(learning_rate=30.0, max_iter=4000,
                              additional_constant_lr_iterations=2000, tol=1e-6)

for i in range(num_starts_stage2):
    key, subkey = jax.random.split(key)
    perturb_x = jax.random.uniform(subkey, (n_target,), minval=-1, maxval=1) * perturbation_scale_stage2
    key, subkey = jax.random.split(key)
    perturb_y = jax.random.uniform(subkey, (n_target,), minval=-1, maxval=1) * perturbation_scale_stage2

    current_init_x_stage2 = opt_x_stage1 + perturb_x
    current_init_y_stage2 = opt_y_stage1 + perturb_y

    opt_x, opt_y = topfarm_sgd_solve(objective, current_init_x_stage2, current_init_y_stage2,
                                      boundary, min_spacing, settings_stage2)

    current_aep = -objective(opt_x, opt_y)

    if current_aep > best_aep:
        best_aep = current_aep
        best_x, best_y = opt_x, opt_y

with open(os.environ["FUNWAKE_OUTPUT"], "w") as f:
    json.dump({"x": [float(v) for v in best_x],
               "y": [float(v) for v in best_y]}, f)
