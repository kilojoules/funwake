"""Build the HETEROGENEOUS ParqueFicticio problem JSON — the pre-test artifact
for the FunWake-2 confirmatory TEST set ("Parque real heterogeneous wind
resource"). Unlike build_problem.py (which averages the WAsP maps into a single
homogeneous climate), this PRESERVES the per-cell, per-sector spatial maps at hub
height so a deployment-time heterogeneous evaluator (py_wake ParqueFicticioSite,
which is natively heterogeneous, or an interpolating wrapper) can give each
turbine its local wind climate.

Output: parqo/problem_parqo_hetero.json  (SOURCE tree; firewalled from mutators —
it is a one-touch TEST cell, never evaluated during the search.)

Grid: 20x20 (x,y) x 12 sectors, hub height 70 m (V80). Preserved maps:
Weibull_A, Weibull_k, Sector_frequency, Speedup, Turning.
"""
import json
import os
import warnings

import numpy as np
from scipy.special import gamma

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
HUB_HEIGHT = 70.0
ROTOR_D = 80.0
N_TARGET = 12
MIN_SPACING = 2.0 * ROTOR_D


def main():
    from py_wake.examples.data.ParqueFicticio._parque_ficticio import (
        ParqueFicticioSite)
    from py_wake.examples.data.hornsrev1 import V80

    site = ParqueFicticioSite()
    ds = site.ds.interp(h=HUB_HEIGHT)
    ds = ds.sel(wd=ds.wd[:-1])          # drop duplicate wd=360 -> 12 sectors
    wd = ds.wd.values
    gx = ds.x.values
    gy = ds.y.values

    def grid(varname):
        # (x, y, wd) -> nested [wd][y][x] lists of floats
        arr = ds[varname].transpose("wd", "y", "x").values
        return [[[float(v) for v in row] for row in sector] for sector in arr]

    # site-averaged representative sector speeds (for a quick homogeneous
    # fallback / cross-check; the heterogeneous maps are the payload)
    A_avg = ds.Weibull_A.mean(("x", "y")).values
    k_avg = ds.Weibull_k.mean(("x", "y")).values
    f_avg = ds.Sector_frequency.mean(("x", "y")).values
    f_avg = f_avg / f_avg.sum()
    speeds_avg = A_avg * gamma(1.0 + 1.0 / k_avg)

    zones = json.load(open(os.path.join(HERE, "inclusion_zones.json")))
    v80 = V80()
    ws_curve = np.arange(3.0, 26.0, 1.0)

    problem = {
        "farm_id": "parqo_hetero",
        "farm_name": ("ParqueFicticio inclusion-zone case study "
                      "(Criado Risco et al. 2024) — HETEROGENEOUS WAsP resource"),
        "n_target": N_TARGET,
        "rotor_diameter": ROTOR_D,
        "hub_height": HUB_HEIGHT,
        "min_spacing_m": MIN_SPACING,
        "inclusion_zones": zones["zones"],
        "inclusion_zones_source": zones["source"],
        "heterogeneous": True,
        "flow_grid": {
            "x": [float(v) for v in gx],
            "y": [float(v) for v in gy],
            "directions_deg": [float(v) for v in wd],
            "note": ("per-cell per-sector maps at hub height; index [wd][y][x]. "
                     "Deployment-time evaluator interpolates at turbine (x,y)."),
            "Weibull_A": grid("Weibull_A"),
            "Weibull_k": grid("Weibull_k"),
            "Sector_frequency": grid("Sector_frequency"),
            "Speedup": grid("Speedup"),
            "Turning": grid("Turning"),
        },
        # homogeneous fallback rose (site-avg) — NOT the test payload, a cross-check
        "wind_rose_siteavg": {
            "directions_deg": [float(v) for v in wd],
            "speeds_ms": [float(v) for v in speeds_avg],
            "weights": [float(v) for v in (f_avg / f_avg.sum())],
        },
        "turbine": {
            "power_curve_ws": [float(v) for v in ws_curve],
            "power_curve_kw": [float(v) for v in (v80.power(ws_curve) / 1e3)],
            "ct_curve_ws": [float(v) for v in ws_curve],
            "ct_curve_ct": [float(v) for v in v80.ct(ws_curve)],
        },
    }

    out = os.path.join(HERE, "problem_parqo_hetero.json")
    with open(out, "w") as fjson:
        json.dump(problem, fjson, indent=1)
    sz = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({sz:.0f} KB)")
    print(f"grid: {len(gx)}x{len(gy)} cells x {len(wd)} sectors, hub {HUB_HEIGHT} m")
    print(f"Weibull_A range: {ds.Weibull_A.min().values:.2f}..{ds.Weibull_A.max().values:.2f} m/s")
    print(f"Speedup range: {ds.Speedup.min().values:.3f}..{ds.Speedup.max().values:.3f}")
    print(f"heterogeneous=True; homogeneous cross-check speeds {np.round(speeds_avg,2)}")


if __name__ == "__main__":
    main()
