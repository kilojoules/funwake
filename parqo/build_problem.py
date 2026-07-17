"""Build a pixwake/funwake problem JSON for the ParqueFicticio case study
of Criado Risco et al. (2024), WES 9, 585-600 (doi:10.5194/wes-9-585-2024):
12 x Vestas V80-2.0, min spacing 2 D = 160 m, five irregular disconnected
inclusion-zone polygons (digitized from Fig. 5, parqo/inclusion_zones.json).

pixwake's WakeSimulation takes a single ambient wind per flow case, so the
heterogeneous WAsP maps (per-cell Weibull A/k, speedup, turning) are
reduced to a homogeneous, site-averaged wind climate at hub height:
  - interpolate the dataset to h = 70 m (V80 hub height)
  - average Weibull A, k and sector frequency over the valid-mode grid
  - representative sector speed = A * gamma(1 + 1/k)  (Weibull mean)
AEP is therefore NOT comparable to the paper's heterogeneous-flow values.

Output: parqo/problem_parqo.json
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
MIN_SPACING = 2.0 * ROTOR_D  # 160 m, as in the paper


def main():
    from py_wake.examples.data.ParqueFicticio._parque_ficticio import (
        ParqueFicticioSite)
    from py_wake.examples.data.hornsrev1 import V80

    site = ParqueFicticioSite()
    ds = site.ds.interp(h=HUB_HEIGHT)
    ds = ds.sel(wd=ds.wd[:-1])  # drop duplicate wd=360, keep 12 sectors
    wd = ds.wd.values

    A = ds.Weibull_A.mean(("x", "y")).values
    k = ds.Weibull_k.mean(("x", "y")).values
    f = ds.Sector_frequency.mean(("x", "y")).values
    f = f / f.sum()
    speeds = A * gamma(1.0 + 1.0 / k)

    zones = json.load(open(os.path.join(HERE, "inclusion_zones.json")))

    v80 = V80()
    ws_curve = np.arange(3.0, 26.0, 1.0)
    power_kw = v80.power(ws_curve) / 1e3
    ct = v80.ct(ws_curve)

    problem = {
        "farm_id": "parqo",
        "farm_name": ("ParqueFicticio inclusion-zone case study "
                      "(Criado Risco et al. 2024, site-avg climate)"),
        "n_target": N_TARGET,
        "rotor_diameter": ROTOR_D,
        "hub_height": HUB_HEIGHT,
        "min_spacing_m": MIN_SPACING,
        "inclusion_zones": zones["zones"],
        "inclusion_zones_source": zones["source"],
        "wind_rose": {
            "directions_deg": [float(v) for v in wd],
            "speeds_ms": [float(v) for v in speeds],
            "weights": [float(v) for v in f],
        },
        "turbine": {
            "power_curve_ws": [float(v) for v in ws_curve],
            "power_curve_kw": [float(v) for v in power_kw],
            "ct_curve_ws": [float(v) for v in ws_curve],
            "ct_curve_ct": [float(v) for v in ct],
        },
    }

    out = os.path.join(HERE, "problem_parqo.json")
    with open(out, "w") as fjson:
        json.dump(problem, fjson, indent=2)

    from shapely.geometry import Polygon
    areas = [Polygon(z).area / 1e6 for z in zones["zones"]]
    print(f"wrote {out}")
    print(f"sectors: {wd}")
    print(f"speeds:  {np.round(speeds, 2)}")
    print(f"weights: {np.round(f, 3)}")
    print(f"{len(areas)} inclusion zones, areas (km2): "
          f"{np.round(areas, 3)}, total {sum(areas):.3f}")


if __name__ == "__main__":
    main()
