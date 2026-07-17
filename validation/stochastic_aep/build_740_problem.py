"""Build a problem-style JSON for the IEA 740-10 ROWP (Jensen NOJ k=0.05)
from the official yaml, for Part-2 stochastic-cost measurements at N=74.

Reuses the Weibull/sector_probability from Wind_Resource.yaml, the Borssele
boundary from Site.yaml, the turbine curves from ROWP_Irregular.yaml.

Wind resource uses Weibull(A, k) per sector as published (no fitting needed).

Usage:
    pixi run python validation/stochastic_aep/build_740_problem.py
"""
import json
import os

import numpy as np
from windIO import load_yaml


ROWP = "/Users/julianquick/portfolio_copy/IEA-Wind-740-10-ROWP"


def main():
    system = load_yaml(os.path.join(ROWP, "ROWP_Irregular_System.yaml"))
    farm = system["wind_farm"]
    resource = system["site"]["energy_resource"]["wind_resource"]
    site_yaml = load_yaml(os.path.join(ROWP, "Site.yaml"))

    # Turbine curves
    t = farm["turbines"]
    perf = t["performance"]
    p_ws = list(map(float, perf["power_curve"]["power_wind_speeds"]))
    p_kw = [v / 1e3 for v in perf["power_curve"]["power_values"]]  # W → kW
    ct_ws = list(map(float, perf["Ct_curve"]["Ct_wind_speeds"]))
    ct = list(map(float, perf["Ct_curve"]["Ct_values"]))

    # Boundary polygon: site_yaml has 'boundaries' with 'polygons'
    # Falls back to ROWP_Irregular layout extent if missing.
    boundary_vertices = None
    try:
        b = site_yaml["boundaries"]
        # boundary geometry uses polygons[0]['x'], polygons[0]['y']
        poly = b["polygons"][0]
        bx = poly["x"]
        by = poly["y"]
        boundary_vertices = [[float(xx), float(yy)] for xx, yy in zip(bx, by)]
    except Exception as e:
        print(f"WARN: could not extract boundary polygon ({e}); falling back to bbox.")
        coords = farm["layouts"]["initial_layout"]["coordinates"]
        x_arr = np.array(coords["x"])
        y_arr = np.array(coords["y"])
        x0, x1 = float(x_arr.min()), float(x_arr.max())
        y0, y1 = float(y_arr.min()), float(y_arr.max())
        boundary_vertices = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    # Weibull resource per sector
    wd_centers = list(map(float, resource["wind_direction"]))
    A = list(map(float, resource["weibull_a"]["data"]))
    k = list(map(float, resource["weibull_k"]["data"]))
    freq = list(map(float, resource["sector_probability"]["data"]))
    n_sectors = len(wd_centers)
    sector_width = 360.0 / n_sectors

    problem = {
        "farm_id": "rowp_740_10",
        "farm_name": "IEA Wind 740-10-MW ROWP irregular",
        "n_target": int(t.get("number_of_turbines", 74)),
        "rotor_diameter": float(t["rotor_diameter"]),
        "hub_height": float(t["hub_height"]),
        # 2D minimum spacing per published study
        "min_spacing_m": float(t["rotor_diameter"]) * 2.0,
        "boundary_vertices": boundary_vertices,
        "turbine": {
            "power_curve_ws": p_ws,
            "power_curve_kw": p_kw,
            "ct_curve_ws": ct_ws,
            "ct_curve_ct": ct,
        },
    }
    # Add Weibull resource side-by-side
    resource_out = {
        "n_sectors": n_sectors,
        "sector_centers_deg": wd_centers,
        "sector_width_deg": sector_width,
        "sector_probability": freq,
        "weibull_A": A,
        "weibull_k": k,
    }
    out_problem = "/Users/julianquick/portfolio_copy/funwake/validation/stochastic_aep/problem_740.json"
    out_resource = "/Users/julianquick/portfolio_copy/funwake/validation/stochastic_aep/dei_weibull_740.json"  # convenient alias name
    out_resource_renamed = "/Users/julianquick/portfolio_copy/funwake/validation/stochastic_aep/rowp_weibull_12.json"
    with open(out_problem, "w") as f:
        json.dump(problem, f, indent=2)
    with open(out_resource_renamed, "w") as f:
        json.dump(resource_out, f, indent=2)
    print(f"Wrote {out_problem}\nWrote {out_resource_renamed}")
    print(f"n_turbines target: {problem['n_target']}; D={problem['rotor_diameter']} m; "
          f"min_spacing={problem['min_spacing_m']} m (2D); "
          f"n_boundary_verts={len(boundary_vertices)}; "
          f"resource: {n_sectors} sectors x Weibull(A,k); sum(freq)={sum(freq):.4f}")


if __name__ == "__main__":
    main()
