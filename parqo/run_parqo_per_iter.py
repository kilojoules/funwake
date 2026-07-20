"""Per-iteration AEP trace of the deployed dual-bump schedule on the parqo
site (native ParqueFicticio case: N=12 V80, five disconnected zones).

Chunked Adam loop (mirrors parqo/skeleton_multizone.run_with_schedule) that
deterministically evaluates full-rose AEP every `probe_every` steps, so the
convergence trajectory can be plotted alongside the DEI/ROWP traces.

Output schema matches validation/stochastic_aep/per_iter_*.json:
  {config, runs:[{iter_trace, aep_trace_gwh, final_aep_gwh, feasible}]}
Written to parqo/per_iter_parqo.json.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "dependencies/pixwake/src"))
sys.path.insert(0, os.path.join(ROOT, "playground"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from harness import build_sim
from skeleton_multizone import (
    multizone_penalty, multizone_sdf, _init_positions,
)
from pixwake.optim.sgd import spacing_penalty

TOTAL = 8000
PROBE = 200
OUT = os.path.join(HERE, "per_iter_parqo.json")

spec = importlib.util.spec_from_file_location(
    "claude", os.path.join(ROOT, "runs/schedule_only_5hr/iter_192.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
schedule_fn = mod.schedule_fn


def main():
    problem = json.load(open(os.path.join(HERE, "problem_parqo.json")))
    zones = [jnp.asarray(z) for z in problem["inclusion_zones"]]
    n = int(problem["n_target"])            # 12
    min_spacing = float(problem["min_spacing_m"])
    sim = build_sim(problem)
    wr = problem["wind_rose"]
    wd = jnp.array(wr["directions_deg"], dtype=jnp.float64)
    ws = jnp.array(wr["speeds_ms"], dtype=jnp.float64)
    wts = jnp.array(wr["weights"], dtype=jnp.float64); wts = wts / jnp.sum(wts)

    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * wts[:, None]) * 8760 / 1e6

    def con_penalty(x, y):
        return multizone_penalty(x, y, zones) + spacing_penalty(x, y, min_spacing)

    def det_aep(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return float(jnp.sum(jnp.sum(p, axis=1) * wts) * 8760 / 1e6)

    grad_obj = jax.grad(aep_objective, argnums=(0, 1))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    x, y = _init_positions(zones, n, min_spacing, 0, init="zones")
    gox, goy = grad_obj(x, y)
    lr0 = 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0

    @jax.jit
    def run_chunk(x, y, mx, my, vx, vy, start):
        eps = 1e-12
        def step(k, carry):
            x, y, mx, my, vx, vy = carry
            i = start + k
            lr, alpha, b1, b2 = schedule_fn(i, TOTAL, lr0, alpha0)
            gx, gy = grad_obj(x, y)
            cx, cy = grad_con(x, y)
            jx = gx + alpha * cx; jy = gy + alpha * cy
            it = (i + 1).astype(float)
            mx = b1 * mx + (1 - b1) * jx; my = b1 * my + (1 - b1) * jy
            vx = b2 * vx + (1 - b2) * jx**2; vy = b2 * vy + (1 - b2) * jy**2
            mxh = mx / (1 - b1**it); myh = my / (1 - b1**it)
            vxh = vx / (1 - b2**it); vyh = vy / (1 - b2**it)
            x = x - lr * mxh / (jnp.sqrt(vxh) + eps)
            y = y - lr * myh / (jnp.sqrt(vyh) + eps)
            return (x, y, mx, my, vx, vy)
        return jax.lax.fori_loop(0, PROBE, step, (x, y, mx, my, vx, vy))

    mx = jnp.zeros_like(x); my = jnp.zeros_like(y)
    vx = jnp.zeros_like(x); vy = jnp.zeros_like(y)
    iters = [0]; aeps = [det_aep(x, y)]
    for start in range(0, TOTAL, PROBE):
        x, y, mx, my, vx, vy = run_chunk(x, y, mx, my, vx, vy, jnp.int32(start))
        iters.append(start + PROBE); aeps.append(det_aep(x, y))
        print(f"  iter {start+PROBE}: AEP={aeps[-1]:.2f}", flush=True)

    max_sdf = float(np.max(np.asarray(multizone_sdf(x, y, zones))))
    xa = np.asarray(x); ya = np.asarray(y)
    dx = xa[:, None]-xa[None, :]; dy = ya[:, None]-ya[None, :]
    md = float((np.sqrt(dx**2+dy**2)+np.eye(n)*1e9).min())
    out = {
        "config": {"site": "parqo", "schedule": "claude_iter192", "n": n,
                   "total_steps": TOTAL, "probe_every": PROBE},
        "runs": [{
            "iter_trace": iters, "aep_trace_gwh": aeps,
            "final_aep_gwh": aeps[-1],
            "max_sdf_m": round(max_sdf, 3), "min_dist_m": round(md, 2),
            "feasible": bool(max_sdf <= 0.1 and md >= min_spacing - 0.1),
        }],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"final AEP {aeps[-1]:.2f} GWh, feasible={out['runs'][0]['feasible']} -> {OUT}")


if __name__ == "__main__":
    main()
