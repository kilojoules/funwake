"""Item-2 probe: largest uniform-rose Parque N with a FEASIBLE (all-seed) c*D
native reference, to replace parque_n30_uniform (whose c*D ref is 0/10) in the
stage-B uniform-Parque slot. Replicates evaluator's multizone path verbatim
(same build_sim / rose / zones / feasibility gate) for arbitrary N. DIAGNOSTIC.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "funwake2")
sys.path.insert(0, "parqo")
import evaluator as E                       # noqa: E402
import skeleton_v2                          # noqa: E402
from skeleton_multizone import multizone_sdf  # noqa: E402


def _load_native():
    spec = importlib.util.spec_from_file_location(
        "native_sched", "funwake2/seeds/native.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.schedule_fn


def probe(n, seed, sched, steps, gamma=0.01):
    prob = json.load(open(os.path.join(E.ROOT, "parqo/problem_parqo.json")))
    D = float(prob["rotor_diameter"])
    ms = float(prob["min_spacing_m"])
    sim = E.build_sim(prob)
    rose = json.load(open(os.path.join(E.ROOT, E._ROSE_UNIFORM)))
    wd, ws, wt = E._load_wind(rose)
    zones = prob["inclusion_zones"]
    zj = [jnp.asarray(z) for z in zones]
    x, y = skeleton_v2.run_with_schedule(
        sched, sim, n, None, ms, wd, ws, wt, D, gamma,
        total_steps=steps, seed=seed, zones=zones)
    xa, ya = np.asarray(x), np.asarray(y)
    sdf = float(np.max(np.asarray(multizone_sdf(jnp.array(xa), jnp.array(ya), zj))))
    dx = xa[:, None] - xa[None, :]
    dy = ya[:, None] - ya[None, :]
    md = float((np.sqrt(dx**2 + dy**2) + np.eye(n) * 1e9).min())
    r = sim(jnp.array(xa), jnp.array(ya), ws_amb=ws, wd_amb=wd, ti_amb=None)
    aep = float(jnp.sum(jnp.sum(r.power()[:, :len(xa)], axis=1) * wt) * 8760 / 1e6)
    feasible = bool(sdf <= 0.1 and md >= ms - 0.1)
    return {"n": n, "seed": seed, "max_sdf": round(sdf, 4),
            "min_dist": round(md, 2), "feasible": feasible, "aep": round(aep, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="funwake2/state/diag_n30/probe_results.json")
    args = ap.parse_args()
    sched = _load_native()
    allrows = []
    if os.path.exists(args.out):
        allrows = json.load(open(args.out))
    for n in args.ns:
        rows = []
        for s in args.seeds:
            r = probe(n, s, sched, args.steps)
            rows.append(r)
            print(f"  N={n} seed{s}: sdf={r['max_sdf']:>8} min_dist={r['min_dist']} "
                  f"feasible={r['feasible']} aep={r['aep']}", flush=True)
        nf = sum(1 for r in rows if r["feasible"])
        print(f"  === N={n}: {nf}/{len(rows)} feasible (c*D native, T={args.steps}) ===",
              flush=True)
        allrows += rows
        json.dump(allrows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
