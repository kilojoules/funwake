#!/usr/bin/env python
"""G6-ROWP: per-cell evaluation-texture floor on the ROWP holdout/test farm.

Same method as the Parque G6 floor: hold an OPTIMIZED feasible ROWP layout fixed
and jitter turbine positions at mm scale (the wake-cone-mask discretization
texture); the scatter in re-scored AEP is the absolute-GWh texture floor. Feeds
the pre-registration per-cell deployment floors (DEI ~0.3, Parque ~0.1, ROWP =
measured here). Measured on one ROWP cell (rowp_n74), as G6 measured one Parque
cell.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FW2 = os.path.dirname(HERE)
sys.path.insert(0, FW2)
sys.path.insert(0, os.path.join(FW2, "seeds"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import evaluator
import skeleton_v2
from native import schedule_fn as native


def main():
    cell = evaluator.CELLS["rowp_n74"]
    prob = json.load(open(os.path.join(evaluator.ROOT, cell["problem"])))
    D = float(prob["rotor_diameter"])
    min_spacing = float(prob["min_spacing_m"])
    n = int(cell["n"])
    sim = evaluator.build_sim(prob)
    wd, ws, wt = evaluator._load_wind(prob)          # ROWP own rose
    boundary = jnp.array(prob["boundary_vertices"], dtype=jnp.float64)

    # optimize one feasible ROWP layout (native, seed 0, 8000 steps)
    x, y = skeleton_v2.run_with_schedule(
        native, sim, n, boundary, min_spacing, wd, ws, wt, D, 0.01,
        total_steps=8000, seed=0, zones=None)
    xa, ya = np.asarray(x), np.asarray(y)

    def aep(xx, yy):
        r = sim(jnp.array(xx), jnp.array(yy), ws_amb=ws, wd_amb=wd, ti_amb=None)
        pw = r.power()[:, :len(xx)]
        return float(jnp.sum(jnp.sum(pw, axis=1) * wt) * 8760 / 1e6)

    base = aep(xa, ya)
    print("=== G6-ROWP texture floor (rowp_n74) ===")
    print(f"  optimized ROWP native layout AEP = {base:.4f} GWh")
    rng = np.random.default_rng(0)
    out = {"cell": "rowp_n74", "base_aep_gwh": round(base, 4), "jitter": {}}
    for jit_m in (0.001, 0.01, 0.1):     # 1 mm, 1 cm, 10 cm
        vals = []
        for _ in range(40):
            dx = rng.normal(0, jit_m, xa.shape)
            dy = rng.normal(0, jit_m, ya.shape)
            vals.append(aep(xa + dx, ya + dy))
        vals = np.array(vals)
        std = float(vals.std())
        mad = float(np.max(np.abs(vals - base)))
        print(f"  jitter sigma={jit_m*1000:.0f} mm: AEP mean={vals.mean():.4f} "
              f"std={std:.4e} GWh  max|dev|={mad:.4e} GWh")
        out["jitter"][f"{jit_m*1000:.0f}mm"] = {"std_gwh": std, "max_dev_gwh": mad}
    floor = out["jitter"]["10mm"]["std_gwh"]
    out["rowp_texture_floor_gwh"] = round(floor, 4)
    print(f"  -> ROWP per-cell texture floor ~ {floor:.4f} GWh (10 mm-scale std)")
    with open(os.path.join(FW2, "state", "g6_rowp_floor.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("G6-ROWP: DONE")


if __name__ == "__main__":
    main()
