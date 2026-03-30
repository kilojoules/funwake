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
init_x = jnp.array(info["init_x"])
init_y = jnp.array(info["init_y"])
min_spacing = info["min_spacing_m"]

def objective(x, y):
    r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
    p = r.power()[:, :len(x)]
    return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

settings = SGDSettings(learning_rate=50.0, max_iter=4000,
                       additional_constant_lr_iterations=2000, tol=1e-6)
opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y,
                                  boundary, min_spacing, settings)

with open(os.environ["FUNWAKE_OUTPUT"], "w") as f:
    json.dump({"x": [float(v) for v in opt_x],
               "y": [float(v) for v in opt_y]}, f)
