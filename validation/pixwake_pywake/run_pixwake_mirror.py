"""Mirror py_wake's 740-10 published AEP under pixwake with identical
NOJDeficit(k=0.05) + SquaredSum + same fine-grid joint probabilities.

The py_wake setup is dumped to validation/pixwake_pywake/pywake_irregular_setup.json
by run_pywake_published.py. We load it, build a pixwake WakeSimulation, and
compute AEP over the exact same (wd, ws) grid weighted by the exact same
joint probabilities. This eliminates Weibull-interpolation drift as a
confound — any remaining discrepancy is purely the wake-model + power
computation pipeline.

Usage:
    pixi run python validation/pixwake_pywake/run_pixwake_mirror.py \\
        validation/pixwake_pywake/pywake_irregular_setup.json \\
        validation/pixwake_pywake/pixwake_irregular.json
"""
import json
import sys

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit.noj import NOJDeficit
from pixwake.superposition import SquaredSum


def main():
    setup_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(setup_path) as f:
        s = json.load(f)

    x = jnp.array(s["x"])
    y = jnp.array(s["y"])
    rd = float(s["rotor_diameter"])
    hh = float(s["hub_height"])

    # Power curve in W → kW for pixwake's Curve API (Turbine expects kW)
    p_w = jnp.array(s["power_curve_w"])
    p_kw = p_w / 1e3
    p_ws = jnp.array(s["power_curve_ws"])
    ct = jnp.array(s["ct_curve_ct"])
    ct_ws = jnp.array(s["ct_curve_ws"])

    turbine = Turbine(
        rotor_diameter=rd,
        hub_height=hh,
        power_curve=Curve(ws=p_ws, values=p_kw),
        ct_curve=Curve(ws=ct_ws, values=ct),
    )
    sim = WakeSimulation(turbine, NOJDeficit(k=float(s["k_noj"]), superposition=SquaredSum()))

    ws_grid = np.array(s["ws_grid"])  # shape (n_ws,)
    wd_grid = np.array(s["wd_grid"])  # shape (n_wd,)
    P = np.array(s["joint_probabilities"])  # shape (n_wd, n_ws)
    n_wd = len(wd_grid)
    n_ws = len(ws_grid)
    assert P.shape == (n_wd, n_ws), f"P shape {P.shape} != ({n_wd}, {n_ws})"

    # Flatten the (wd, ws) grid to 1D cases as pixwake expects
    ws_flat = jnp.array(np.tile(ws_grid, n_wd))
    wd_flat = jnp.array(np.repeat(wd_grid, n_ws))
    prob_flat = jnp.array(P.flatten())
    ti_flat = jnp.full_like(ws_flat, 0.06)  # TI unused since no turbulence model

    result = sim(x, y, ws_amb=ws_flat, wd_amb=wd_flat, ti_amb=ti_flat)
    # AEP weighted by probabilities, in GWh
    aep_gwh = float(result.aep(probabilities=prob_flat))

    pywake_aep_published = 3429.63
    # Read pywake result for direct comparison
    pywake_result_path = setup_path.replace("_setup.json", ".json")
    try:
        with open(pywake_result_path) as f:
            pywake_dat = json.load(f)
        pywake_aep = float(pywake_dat["aep_gwh"])
    except Exception:
        pywake_aep = None

    out = {
        "engine": "pixwake",
        "config": {
            "wake_model": "NOJDeficit",
            "k": float(s["k_noj"]),
            "superposition": "SquaredSum",
            "n_wd": int(n_wd),
            "n_ws": int(n_ws),
            "weighting": "joint_probabilities imported from py_wake setup",
        },
        "n_turbines": int(x.shape[0]),
        "aep_gwh": aep_gwh,
        "vs_pywake": {
            "pywake_aep_gwh": pywake_aep,
            "delta_gwh": (aep_gwh - pywake_aep) if pywake_aep is not None else None,
            "delta_pct": (
                (aep_gwh - pywake_aep) / pywake_aep * 100 if pywake_aep is not None else None
            ),
        },
        "vs_published": {
            "published_aep_gwh": pywake_aep_published,
            "delta_gwh": aep_gwh - pywake_aep_published,
            "delta_pct": (aep_gwh - pywake_aep_published) / pywake_aep_published * 100,
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
