#!/usr/bin/env python
"""Build a problem_farm JSON for the IEA Wind 740-10 ROWP irregular layout case.

This creates a test case with a different turbine (IEA 10MW, D=198m),
different polygon boundary, and Weibull-based wind resource — distinct
from the DEI training case.

Source: https://github.com/IEAWindSystems/IEA-Wind-740-10-ROWP
"""

import json
import numpy as np
from pathlib import Path
from scipy.special import gamma as gamma_func

# ── Site boundary (from Site.yaml) ──
boundary_x = [484178.55, 500129.9, 497318.1, 503163.37, 501266.5, 488951.0]
boundary_y = [5732482.8, 5737534.4, 5731880.24, 5729155.3, 5715990.05, 5.72794e+06]

# Center the boundary (translation-invariant for wake sim)
cx = np.mean(boundary_x)
cy = np.mean(boundary_y)
centered = np.array([[x - cx, y - cy] for x, y in zip(boundary_x, boundary_y)])

# Convex hull in CCW order — boundary_penalty assumes convex CCW polygon
from scipy.spatial import ConvexHull
hull = ConvexHull(centered)
boundary_vertices = centered[hull.vertices].tolist()

# ── Initial turbine layout (from ROWP_Irregular.yaml, 74 turbines) ──
init_x_raw = [
    501266.54317574, 493922.24954271, 502907.19014959, 502095.12557912,
    496804.65625979, 485102.77027462, 490588.37581976, 502784.83623495,
    503032.60103984, 484578.18744373, 486707.11173729, 492087.72419315,
    502245.04987847, 486993.72883541, 486339.8895555,  501937.23428743,
    490866.23972975, 489583.97960791, 493244.9730643,  484184.85247367,
    495837.28295403, 501763.67866768, 497805.65064534, 502520.00847477,
    488196.97665934, 502261.66468538, 487660.64327449, 496326.9935416,
    495560.72737215, 492442.84358768, 489811.54071237, 488384.83348118,
    485719.30135799, 499693.18157517, 494872.64094661, 491373.54080591,
    492179.32192035, 497223.78998917, 497331.96509787, 496443.58939824,
    499696.09772372, 497791.49735685, 503157.65666188, 502656.8554572,
    499013.99779544, 502389.74235056, 499568.71665005, 498412.30882307,
    500126.55758186, 493699.48183891, 500599.36370223, 497855.63257599,
    495278.26879751, 493700.84819603, 492224.36209737, 498554.39258276,
    489085.25534532, 500588.51971806, 499488.70435074, 494285.31394068,
    495302.17515336, 498899.58556002, 493038.70804906, 494794.71182856,
    498335.26945098, 499450.50531967, 501551.75352904, 499153.2151173,
    489443.95885503, 495667.21147171, 499425.06290419, 497613.73273252,
    501417.31149458, 490809.4125091,
]
init_y_raw = [
    5715995.89368689, 5723126.80050036, 5727399.9805506,
    5721783.86274859, 5720326.91432108, 5731616.65356541,
    5726358.98043443, 5726554.75877663, 5728282.55126408,
    5732113.07451875, 5733271.99003168, 5734978.77939308,
    5722819.59436817, 5729811.23565774, 5730437.7600403,
    5720684.82136608, 5734593.97542309, 5734183.81996312,
    5735342.15989407, 5732482.14974234, 5721263.84408409,
    5719468.98024809, 5719362.35078842, 5724757.97736015,
    5733753.39470066, 5729567.10859905, 5729175.80279442,
    5736318.33484618, 5730365.5374425,  5732709.48377899,
    5727115.20095509, 5728488.33293297, 5731024.64807625,
    5726888.85185297, 5722211.95206308, 5725600.69245181,
    5724815.96578192, 5723028.55637662, 5736641.50567781,
    5729036.22360224, 5730763.14987881, 5727794.82805826,
    5729153.34310612, 5725664.84960664, 5718720.22921093,
    5723814.40501551, 5717641.93281177, 5736984.27770828,
    5737528.57233063, 5728053.24974494, 5716646.9421783,
    5725266.2923445,  5735992.35112698, 5731256.8927141,
    5729156.69983083, 5731292.34250883, 5727820.39671998,
    5730350.50472903, 5736258.41812358, 5735679.58364592,
    5733057.09366068, 5735074.06082225, 5723984.10312561,
    5726569.83357447, 5733940.1526999,  5737313.57272465,
    5718018.11318477, 5721395.33171942, 5731966.4812102,
    5725151.81249156, 5723990.57679841, 5732484.17798327,
    5729964.36557568, 5730586.99083753,
]

init_x = [x - cx for x in init_x_raw]
init_y = [y - cy for y in init_y_raw]

# ── Turbine: IEA 10MW (D=198m, hub=119m) ──
D = 198.0
hub_height = 119.0
min_spacing = 4.0 * D  # 792m

# Power curve (W -> kW)
power_ws = [4, 4.515, 5.001, 5.457, 5.883, 6.278, 6.640, 6.968, 7.263,
            7.523, 7.748, 7.938, 8.091, 8.208, 8.288, 8.331, 8.337,
            8.368, 8.436, 8.540, 8.681, 8.859, 9.072, 9.320, 9.604,
            9.921, 10.272, 10.656, 10.758, 11.518, 11.994, 12.499,
            13.032, 13.592, 14.177, 14.786, 15.418, 16.070, 16.743,
            17.434, 18.142, 18.865, 19.602, 20.351, 21.110, 21.877,
            22.652, 23.432, 24.215, 25.0]
power_kw = [387.5, 644.7, 938.0, 1257.9, 1604.1, 1969.5, 2340.1, 2708.6,
            3067.1, 3408.7, 3723.7, 4003.4, 4239.6, 4425.8, 4556.6,
            4628.1, 4638.4, 4690.0, 4804.9, 4985.7, 5237.0, 5564.4,
            5975.8, 6480.5, 7089.7, 7816.4, 8690.6, 9716.7, 10000.0,
            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
            10000.0, 10000.0, 10000.0]
ct_values = [0.770, 0.776, 0.782, 0.782, 0.780, 0.777, 0.772, 0.777,
             0.777, 0.777, 0.777, 0.777, 0.777, 0.777, 0.777, 0.777,
             0.777, 0.777, 0.777, 0.777, 0.777, 0.777, 0.777, 0.777,
             0.777, 0.777, 0.768, 0.765, 0.759, 0.506, 0.431, 0.371,
             0.321, 0.279, 0.243, 0.213, 0.187, 0.165, 0.145, 0.129,
             0.115, 0.102, 0.092, 0.083, 0.074, 0.067, 0.061, 0.056,
             0.051, 0.047]

# Resample to integer wind speeds 0-25 for pixwake compatibility
ws_out = list(range(0, 26))
power_out = list(np.interp(ws_out, power_ws, power_kw, left=0, right=0))
ct_out = list(np.interp(ws_out, power_ws, ct_values, left=ct_values[0], right=ct_values[-1]))
# Below cut-in (4 m/s): zero power, keep Ct for interpolation
for i in range(4):
    power_out[i] = 0.0

# ── Wind resource: Weibull -> discrete (direction, mean_speed, weight) ──
directions = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
sector_prob = [0.06692, 0.07626, 0.07374, 0.06463, 0.04696, 0.04643,
               0.07672, 0.12233, 0.19147, 0.10080, 0.06927, 0.06447]
weibull_a = [9.08, 9.30, 9.18, 8.89, 8.13, 8.76, 11.38, 12.58, 12.74,
             10.80, 9.76, 9.63]
weibull_k = [2.22, 2.26, 2.28, 2.28, 2.15, 2.11, 2.13, 2.29, 2.43,
             2.09, 2.01, 2.01]

# Mean wind speed per sector: A * Gamma(1 + 1/k)
mean_speeds = [a * gamma_func(1 + 1/k) for a, k in zip(weibull_a, weibull_k)]

# ── Build problem JSON ──
problem = {
    "farm_id": "rowp",
    "farm_name": "IEA Wind 740-10 ROWP (irregular)",
    "n_target": len(init_x),
    "rotor_diameter": D,
    "hub_height": hub_height,
    "min_spacing_m": min_spacing,
    "boundary_vertices": boundary_vertices,
    "init_x": init_x,
    "init_y": init_y,
    "wind_rose": {
        "directions_deg": directions,
        "speeds_ms": mean_speeds,
        "weights": sector_prob,
    },
    "turbine": {
        "power_curve_ws": ws_out,
        "power_curve_kw": power_out,
        "ct_curve_ws": ws_out,
        "ct_curve_ct": ct_out,
    },
    "source": "https://github.com/IEAWindSystems/IEA-Wind-740-10-ROWP",
}

out = Path(__file__).parent.parent / "results" / "problem_rowp.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(problem, f, indent=2)
print(f"Wrote {out}")
print(f"  {len(init_x)} turbines, D={D}m, min_spacing={min_spacing}m")
print(f"  {len(directions)} wind sectors")
print(f"  boundary: {len(boundary_vertices)} vertices")
print(f"  mean wind speeds: {[f'{s:.1f}' for s in mean_speeds]}")
