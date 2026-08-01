#!/usr/bin/env python
"""G6: per-cell evaluation-texture floor on the Parque cell.

The 0.3 GWh deployment floor is DEI-calibrated (~0.005% of ~5500 GWh); at
Parque's ~230 GWh scale it is ~25x stricter in relative terms. Measure the
absolute-GWh evaluation-texture floor at Parque's scale by holding an OPTIMIZED
feasible Parque layout fixed and jittering turbine positions at mm scale (the
wake-cone-mask texture): the scatter in re-scored AEP is the floor.

Reports the AEP std at 1 mm and 10 mm jitter (absolute GWh) -> the per-cell
texture floor that feeds the prereg's per-cell floors (DEI's is ~0.3 GWh).
"""
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
from native import schedule_fn as native


def main():
    cell = evaluator.CELLS["parque_n20"]
    prob = __import__("json").load(open(os.path.join(evaluator.ROOT, cell["problem"])))
    D = float(prob["rotor_diameter"])
    min_spacing = float(prob["min_spacing_m"])
    n = int(cell["n"])
    sim = evaluator.build_sim(prob)
    rose = __import__("json").load(open(os.path.join(evaluator.ROOT, cell["rose"])))
    wd, ws, wt = evaluator._load_wind(rose)
    zones = prob["inclusion_zones"]

    # optimize one feasible Parque layout (native, seed 0, 8000 steps)
    import skeleton_v2
    x, y = skeleton_v2.run_with_schedule(
        native, sim, n, None, min_spacing, wd, ws, wt, D, 0.01,
        total_steps=8000, seed=0, zones=zones)
    xa, ya = np.asarray(x), np.asarray(y)

    def aep(xx, yy):
        r = sim(jnp.array(xx), jnp.array(yy), ws_amb=ws, wd_amb=wd, ti_amb=None)
        pw = r.power()[:, :len(xx)]
        return float(jnp.sum(jnp.sum(pw, axis=1) * wt) * 8760 / 1e6)

    base = aep(xa, ya)
    print("=== G6 Parque texture floor ===")
    print(f"  optimized Parque native layout AEP = {base:.4f} GWh")
    rng = np.random.default_rng(0)
    for jit_m in (0.001, 0.01, 0.1):     # 1 mm, 1 cm, 10 cm
        vals = []
        for _ in range(40):
            dx = rng.normal(0, jit_m, xa.shape)
            dy = rng.normal(0, jit_m, ya.shape)
            vals.append(aep(xa + dx, ya + dy))
        vals = np.array(vals)
        print(f"  jitter sigma={jit_m*1000:.0f} mm: AEP mean={vals.mean():.4f} "
              f"std={vals.std():.4e} GWh  max|dev|={np.max(np.abs(vals-base)):.4e} GWh")
    print("  -> Parque per-cell texture floor ~ the 10 mm-scale std above "
          "(absolute GWh at ~230 GWh scale; contrast DEI ~0.3 GWh at ~5500).")
    print("G6: DONE (texture floor measured)")


if __name__ == "__main__":
    main()
