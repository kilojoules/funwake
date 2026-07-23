#!/usr/bin/env python
"""ES-mechanism experiment 5: characterize the AEP evaluation "texture".

Establishes that the raw AEP objective is locally piecewise-constant with
discrete ~0.1 GWh jumps under millimeter-to-centimeter layout changes:

  1. Single-turbine line scans (2 mm resolution over +-5 cm): AEP is flat
     to <1e-3 GWh except for isolated finite steps of ~0.1 GWh.
  2. Perturbation response: random +-1 cm / 10 cm / 1 m coordinate
     perturbations all produce dAEP of the same ~+-0.2 GWh magnitude
     (non-scaling => discontinuity web, not gradient response).
  3. Fixed-point tolerance sweep: identical results at fpi_tol=1e-6 and
     1e-12 => the jumps are NOT solver truncation error.
  4. Source: pixwake/deficit/base.py applies a hard wake-cone mask
     (dw > 0) & (|cw| < wake_radius = 2*sigma); the Gaussian deficit at the
     cone edge is exp(-2) ~ 13.5% of centerline, a finite deficit switched
     on/off discontinuously when a (source, receiver, direction) triple
     crosses the cone edge.

Usage:
    pixi run python tools/es_mechanism_texture.py \
        --out results/equiv_cost_sgd/es_mechanism/texture_probe.json
"""
import argparse
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from probe_es_truncation import log

import numpy as np
import jax
import jax.numpy as jnp
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit


def build_objective(problem_path, fpi_tol):
    info = json.load(open(problem_path))
    t = info["turbine"]
    turb = Turbine(
        rotor_diameter=info["rotor_diameter"],
        hub_height=info.get("hub_height", 150.0),
        power_curve=Curve(ws=jnp.array(t["power_curve_ws"], dtype=float),
                          values=jnp.array(t["power_curve_kw"], dtype=float)),
        ct_curve=Curve(ws=jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]),
                                    dtype=float),
                       values=jnp.array(t["ct_curve_ct"], dtype=float)))
    sim = WakeSimulation(turb, BastankhahGaussianDeficit(k=0.04),
                         fpi_tol=fpi_tol)
    wd = jnp.array(info["wind_rose"]["directions_deg"])
    ws = jnp.array(info["wind_rose"]["speeds_ms"])
    weights = jnp.array(info["wind_rose"]["weights"])

    def aep(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        pw = r.power()[:, :len(x)]
        return jnp.sum(pw * weights[:, None]) * 8760 / 1e6

    return jax.jit(aep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?",
                   default=os.path.join(REPO_ROOT, "results",
                                        "problem_dei_n50.json"))
    p.add_argument("--paired",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism",
                                        "paired_10seeds.json"))
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism", "texture_probe.json"))
    args = p.parse_args()

    paired = json.load(open(args.paired))
    x0 = np.array(paired["seeds"][0]["full"]["x"])
    y0 = np.array(paired["seeds"][0]["full"]["y"])

    f6 = build_objective(args.problem, 1e-6)
    f12 = build_objective(args.problem, 1e-12)
    a0 = float(f6(jnp.asarray(x0), jnp.asarray(y0)))

    # 1. line scans
    scans = []
    for ti in [0, 12, 25, 37, 49]:
        offs = np.arange(-0.05, 0.0501, 0.002)
        aeps = np.array([float(f6(jnp.asarray(
            np.where(np.arange(len(x0)) == ti, x0 + o, x0)), jnp.asarray(y0)))
            for o in offs])
        jumps = np.abs(np.diff(aeps))
        scans.append({
            "turbine": ti,
            "n_jumps_over_0.01": int((jumps > 0.01).sum()),
            "max_jump_gwh": round(float(jumps.max()), 4),
            "smooth_variation_gwh": round(
                float(jumps[jumps <= 0.01].sum()), 5),
        })
        log(f"[scan] turbine {ti}: {scans[-1]}")

    # 2. perturbation response (non-scaling check)
    rng = np.random.default_rng(0)
    resp = {}
    for scale in [0.01, 0.1, 1.0]:
        ds = []
        for _ in range(16):
            px = x0 + scale * rng.choice([-1, 1], size=x0.shape)
            py = y0 + scale * rng.choice([-1, 1], size=y0.shape)
            ds.append(float(f6(jnp.asarray(px), jnp.asarray(py))) - a0)
        resp[str(scale)] = {"std": round(float(np.std(ds)), 4),
                            "max_abs": round(float(np.max(np.abs(ds))), 4)}
        log(f"[perturb] scale {scale} m: std {resp[str(scale)]['std']} "
            f"max {resp[str(scale)]['max_abs']}")

    # 3. tolerance independence
    tol_diffs = []
    for rec in paired["seeds"][:5]:
        for which in ["full", "es"]:
            x = jnp.asarray(np.array(rec[which]["x"]))
            y = jnp.asarray(np.array(rec[which]["y"]))
            tol_diffs.append(abs(float(f6(x, y)) - float(f12(x, y))))
    log(f"[tol] max |AEP(tol=1e-6) - AEP(tol=1e-12)| over 10 layouts: "
        f"{max(tol_diffs):.2e}")

    out = {
        "problem": args.problem,
        "line_scans_pm5cm_2mm": scans,
        "perturbation_response_gwh": resp,
        "max_tol_sensitivity_gwh": max(tol_diffs),
        "jump_source": ("pixwake/deficit/base.py: in_wake_mask = (dw > 0) & "
                        "(|cw| < wake_radius=2*sigma); e^-2 edge deficit "
                        "switched discontinuously"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fjson:
        json.dump(out, fjson, indent=1)
    log(f"[done] wrote {args.out}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
