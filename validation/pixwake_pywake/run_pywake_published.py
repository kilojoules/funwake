"""Reproduce the published IEA 740-10 ROWP AEP under py_wake.

Patch of upstream ../IEA-Wind-740-10-ROWP/Examples/pywake_ex.py for windIO 2.x
(load_yaml moved from windIO.utils.yml_utils → windIO.load_yaml). Plotting
stripped. Output written as JSON for downstream comparison with pixwake.

Usage:
    pixi run python validation/pixwake_pywake/run_pywake_published.py \\
        ../IEA-Wind-740-10-ROWP/ROWP_Irregular_System.yaml \\
        validation/pixwake_pywake/pywake_irregular.json
"""
import json
import sys

import numpy as np
import xarray as xr
from py_wake import NOJ
from py_wake.rotor_avg_models import RotorCenter
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from windIO import load_yaml


def main():
    file_path = sys.argv[1]
    out_path = sys.argv[2]

    ws_sw = 1.0
    wd_sw = 1.0

    system_dat = load_yaml(file_path)
    farm_dat = system_dat["wind_farm"]
    resource_dat = system_dat["site"]["energy_resource"]

    A = resource_dat["wind_resource"]["weibull_a"]
    k = resource_dat["wind_resource"]["weibull_k"]
    freq = resource_dat["wind_resource"]["sector_probability"]
    wd = resource_dat["wind_resource"]["wind_direction"]
    ws = resource_dat["wind_resource"]["wind_speed"]
    TI = resource_dat["wind_resource"]["turbulence_intensity"]["data"]

    x = farm_dat["layouts"]["initial_layout"]["coordinates"]["x"]
    y = farm_dat["layouts"]["initial_layout"]["coordinates"]["y"]

    hh = farm_dat["turbines"]["hub_height"]
    rd = farm_dat["turbines"]["rotor_diameter"]
    p = farm_dat["turbines"]["performance"]["power_curve"]["power_values"]
    p_ws = farm_dat["turbines"]["performance"]["power_curve"]["power_wind_speeds"]
    ct = farm_dat["turbines"]["performance"]["Ct_curve"]["Ct_values"]
    ct_ws = farm_dat["turbines"]["performance"]["Ct_curve"]["Ct_wind_speeds"]
    cut_in = farm_dat["turbines"]["performance"]["cutin_wind_speed"]
    cut_out = farm_dat["turbines"]["performance"]["cutout_wind_speed"]

    int_speeds = np.linspace(
        np.min(np.min([p_ws, ct_ws])), np.max(np.max([p_ws, ct_ws])), 10000
    )
    ps_int = np.interp(int_speeds, p_ws, p)
    cts_int = np.interp(int_speeds, ct_ws, ct)

    windTurbines = WindTurbine(
        name=farm_dat["turbines"]["name"],
        diameter=rd,
        hub_height=hh,
        powerCtFunction=PowerCtTabular(int_speeds, ps_int, power_unit="W", ct=cts_int),
    )
    site = XRSite(
        ds=xr.Dataset(
            data_vars={
                "Sector_frequency": ("wd", freq["data"]),
                "Weibull_A": ("wd", A["data"]),
                "Weibull_k": ("wd", k["data"]),
                "TI": (
                    resource_dat["wind_resource"]["turbulence_intensity"]["dims"][0],
                    TI,
                ),
            },
            coords={"wd": wd, "ws": ws},
        )
    )
    site.interp_method = "linear"

    ws_py = np.arange(cut_in, cut_out + ws_sw, ws_sw)
    wd_py = np.arange(0, 360, wd_sw)
    TI_int = np.interp(ws_py, ws, TI)

    noj = NOJ(site, windTurbines, turbulenceModel=None, k=0.05, rotorAvgModel=RotorCenter())
    sim_res = noj(x, y, time=False, ws=ws_py, wd=wd_py, TI=TI_int)
    aep = float(sim_res.aep(normalize_probabilities=False).sum())

    # Dump joint probabilities and per-cell power so pixwake can mirror with
    # identical wind-rose discretization (eliminates Weibull-interp drift as
    # a source of cross-engine difference).
    P = sim_res.P.values  # shape: (n_wd, n_ws) — joint probability per case
    # Per-turbine power per case: shape (n_wt, n_wd, n_ws)
    power_per_case = sim_res.Power.values

    # Also dump full setup for pixwake to ingest
    setup_path = out_path.replace(".json", "_setup.json")
    setup = {
        "x": list(map(float, x)),
        "y": list(map(float, y)),
        "rotor_diameter": float(rd),
        "hub_height": float(hh),
        "power_curve_ws": list(map(float, int_speeds)),
        "power_curve_w": list(map(float, ps_int)),
        "ct_curve_ws": list(map(float, int_speeds)),
        "ct_curve_ct": list(map(float, cts_int)),
        "ws_grid": list(map(float, ws_py)),
        "wd_grid": list(map(float, wd_py)),
        "joint_probabilities": [list(map(float, row)) for row in P],
        "ti_per_ws": list(map(float, TI_int)),
        "k_noj": 0.05,
    }
    with open(setup_path, "w") as f:
        json.dump(setup, f)

    result = {
        "engine": "py_wake",
        "py_wake_version": __import__("py_wake").__version__,
        "config": {
            "wake_model": "NOJ",
            "k": 0.05,
            "rotor_avg_model": "RotorCenter",
            "turbulence_model": None,
            "superposition": "SquaredSum (NOJ default)",
            "ws_step_m_s": ws_sw,
            "wd_step_deg": wd_sw,
            "ws_grid": [float(ws_py.min()), float(ws_py.max())],
            "wd_grid": [float(wd_py.min()), float(wd_py.max())],
        },
        "system_file": file_path,
        "n_turbines": len(x),
        "aep_gwh": aep,
        "capacity_factor": float(
            aep / (len(x) * windTurbines.power(10000) * 8760 / 1e9)
        ),
        "published_aep_gwh": 3429.63,
        "delta_vs_published_gwh": aep - 3429.63,
        "delta_vs_published_pct": (aep - 3429.63) / 3429.63 * 100,
        "setup_dump_path": setup_path,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
