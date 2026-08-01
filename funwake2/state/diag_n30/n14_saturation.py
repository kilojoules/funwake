"""Round-2 item 1: n14 saturation check. Compute 14 x single-turbine free-stream
AEP under the uniform rose and compare to the optimized n14 baseline (184.812).
If they match within the Parque texture floor (0.1 GWh), the objective is
saturated (all-escape layouts score exactly free-stream -> std=0, no texture) and
parque_n14_uniform should be a FEASIBILITY-ONLY stage-B gate. If the optimized
AEP is below 14x free-stream by > the floor, there ARE wake losses -> keep it in
the mean-% aggregate.
"""
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "funwake2")
sys.path.insert(0, "parqo")
import evaluator as E  # noqa: E402

prob = json.load(open(os.path.join(E.ROOT, "parqo/problem_parqo.json")))
sim = E.build_sim(prob)
rose = json.load(open(os.path.join(E.ROOT, E._ROSE_UNIFORM)))
wd, ws, wt = E._load_wind(rose)

# single isolated turbine at each zone centroid (homogeneous site -> position-
# independent free-stream power; average over zones as a sanity check)
zones = prob["inclusion_zones"]
singles = []
for z in zones:
    za = np.array(z)
    cx, cy = float(za[:, 0].mean()), float(za[:, 1].mean())
    x = jnp.array([cx]); y = jnp.array([cy])
    r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
    p = r.power()[:, :1]
    aep1 = float(jnp.sum(p * wt[:, None]) * 8760 / 1e6)
    singles.append(aep1)

aep_single = float(np.mean(singles))
free14 = 14.0 * aep_single
opt = 184.812
deficit = free14 - opt
floor = 0.1  # Parque texture floor (GWh)

print("=== n14 saturation check (uniform rose) ===")
print(f"  single-turbine free-stream AEP: {aep_single:.4f} GWh "
      f"(per-zone spread {np.round(singles,4)})")
print(f"  14 x single (no-wake ceiling):  {free14:.4f} GWh")
print(f"  optimized n14 baseline:         {opt:.4f} GWh")
print(f"  deficit (14xfree - opt):        {deficit:.4f} GWh   (Parque floor {floor})")
print(f"  SATURATED (deficit <= floor): {abs(deficit) <= floor}")
json.dump({"aep_single": aep_single, "free14": free14, "opt": opt,
           "deficit_gwh": deficit, "parque_floor_gwh": floor,
           "saturated": bool(abs(deficit) <= floor)},
          open("funwake2/state/diag_n30/n14_saturation.json", "w"), indent=2)
