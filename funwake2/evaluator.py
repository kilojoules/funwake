#!/usr/bin/env python
"""FunWake-2 Phase-1 minimal evaluator.

Wraps the scale-aware v2 skeleton to run ONE schedule on a (cell, seed,
gamma_min) and return AEP + feasibility. This is the smoke-test scorer used by
the Phase-1 gates. It is NOT the Phase-3 cascade/controller (deliberately: no
search, no islands, no LLM mutators).

A "cell" is a (geometry, wind-rose, N, D, spacing) tuple:
  * single-polygon (DEI/ROWP): geometry + wind rose come from one problem JSON;
    constraint = boundary + spacing; scored by benchmarks ProblemBenchmark.
  * multizone (Parque): geometry/turbine from parqo/problem_parqo.json but the
    wind rose from a SEPARATE rose problem (the diameter-rule Parque cell uses
    the DEI rose); constraint = nearest-zone SDF + spacing; scored directly with
    a max-zone-SDF feasibility gate.

Usage (CLI, checkpoints one JSON per (cell,seed) into --outdir):
    pixi run python funwake2/evaluator.py \
        --cell dei_n50 --schedule funwake2/seeds/native.py \
        --steps 6000 --gamma-min 0.01 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --outdir funwake2/state/g1_native_dei
"""
import argparse
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dependencies", "pixwake", "src"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "playground"))
sys.path.insert(0, os.path.join(ROOT, "parqo"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # funwake2/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import skeleton_v2
from harness import build_sim                                   # noqa: E402
from skeleton_multizone import multizone_sdf                    # noqa: E402


# ── Cell registry ─────────────────────────────────────────────────────
# Fields consumed by evaluate(): problem, rose, n, multizone, feas_tol.
# Metadata for the Phase-3 controller (ignored by evaluate):
#   role     : "train" | "holdout" | "test" | "deferred"
#   stage_a  : eligible for the cheap fast-reject stage (train cells only)
#   stage_b  : part of the full frozen training matrix (train cells only)
# ROWP appears ONLY as holdout/test — it is in NO training cell (farm-level
# holdout). n200 DEI is DEFERRED (infeasible 0/50 in the 500-multistart
# baseline + >5 min/eval) → needs author decision, NOT in stage-B.
#
# For DEI cells the wind rose is baked into the problem JSON (rose=None).
# For Parque (multizone) the geometry/turbine come from problem_parqo.json and
# the rose comes from a SEPARATE rose problem (only its wind_rose is read).
_ROSE_DEI = "results/matrix/problem_dei_n50_rosedei.json"
_ROSE_UNIFORM = "results/matrix/problem_dei_n50_roseuniform.json"     # unidirectional (dir=0)
_ROSE_OMNIDIR = "results/matrix/problem_dei_n50_roseomnidir.json"

CELLS = {
    # ── FROZEN TRAINING CELLS (stage-B set; D-4) ──────────────────────
    "dei_n50": {                                    # DEI, N=50, DEI rose
        "problem": "results/problem_dei_n50.json",
        "rose": None, "n": 50, "multizone": False,
        "role": "train", "stage_a": True, "stage_b": True,
        "baseline_0833D_6000": 5561.34, "sem": 0.65,
    },
    "dei_n80_omnidir": {                            # DEI, N=80, omnidir rose
        "problem": "results/matrix/problem_dei_n80_roseomnidir.json",
        "rose": None, "n": 80, "multizone": False,
        "role": "train", "stage_a": False, "stage_b": True,
    },
    "dei_n120_rosedei": {                           # DEI, N=120, DEI rose — HIGH-N (frozen)
        # Chosen by the n80/n100/n120/n150 timing curve: largest N with eval
        # <= ~3 min AND a usable feasible native reference. Replaces n200 in
        # stage-B (n200 -> stage-B+ gbar-only). native@c*D 1-seed: 159 s,
        # feasible, AEP 13016.6, min_dist 1037 > 960.
        "problem": "funwake2/problems/problem_dei_n120_rosedei.json",
        "rose": None, "n": 120, "multizone": False,
        "role": "train", "stage_a": False, "stage_b": True, "high_n": True,
    },
    "dei_n50_uniform": {                            # DEI, N=50, unidirectional (dir=0)
        "problem": "results/matrix/problem_dei_n50_roseuniform.json",
        "rose": None, "n": 50, "multizone": False,
        "role": "train", "stage_a": False, "stage_b": True,
    },
    "parque_n20": {                                 # Parque multizone, N=20, DEI rose
        "problem": "parqo/problem_parqo.json",
        "rose": _ROSE_DEI, "n": 20, "multizone": True, "feas_tol": 0.1,
        "role": "train", "stage_a": True, "stage_b": True,
        "baseline_0833D": 231.06, "sem": 0.32,
    },
    "parque_n30_uniform": {                         # Parque multizone, N=30, unidirectional
        "problem": "parqo/problem_parqo.json",
        "rose": _ROSE_UNIFORM, "n": 30, "multizone": True, "feas_tol": 0.1,
        "role": "train", "stage_a": False, "stage_b": True,
    },
    "parque_n10_omnidir": {                         # Parque multizone, N=10, omnidir
        "problem": "parqo/problem_parqo.json",
        "rose": _ROSE_OMNIDIR, "n": 10, "multizone": True, "feas_tol": 0.1,
        "role": "train", "stage_a": False, "stage_b": True,
    },

    # ── DEFERRED (author decision; NOT in stage-B) ────────────────────
    "dei_n200_rosedei": {                           # high-N; gbar-only stage-B+ cell
        "problem": "results/matrix/problem_dei_n200_rosedei.json",
        "rose": None, "n": 200, "multizone": False,
        # stage-B+ (elite-tier, gbar only). NOT in per-generation stage-B.
        # Classification DEFERRED to gbar (native@c*D, 5 seeds): feasible ->
        # ordinary stage-B+ elite cell; infeasible -> capability-frontier cell
        # (goes in the confirmatory test set). Mac seed0 probe was infeasible
        # (1 seed only) + ~357 s/eval — NOT a formal classification.
        "role": "stage_b_plus", "stage_a": False, "stage_b": False,
        "gbar_only": True,
    },
    "dei_n100": {                                   # proposed tractable high-N substitute
        "problem": "results/problem_dei_n100.json",
        "rose": None, "n": 100, "multizone": False,
        "role": "candidate", "stage_a": False, "stage_b": False,
    },

    # ── HOLDOUT (selection only; AEP firewalled) ──────────────────────
    "rowp_n74": {                                   # ROWP farm-level holdout
        "problem": "results/problem_rowp.json",
        "rose": None, "n": 74, "multizone": False,
        "role": "holdout", "stage_a": False, "stage_b": False,
        "baseline_0833D_6000": 4261.72, "sem": 0.83,
    },

    # ── FROZEN TEST SET (touched once, at deployment; AEP firewalled) ─
    "rowp_n200_roserowp": {                         # ROWP high-N (test)
        "problem": "results/matrix/problem_rowp_n200_roserowp.json",
        "rose": None, "n": 200, "multizone": False,
        "role": "test", "stage_a": False, "stage_b": False,
    },
    "rowp_n300_roserowp": {                         # ROWP high-N (test)
        "problem": "results/matrix/problem_rowp_n300_roserowp.json",
        "rose": None, "n": 300, "multizone": False,
        "role": "test", "stage_a": False, "stage_b": False,
    },
    "rowp_n74_uniform": {                           # unidirectional extreme on unseen farm (test)
        "problem": "results/matrix/problem_rowp_n200_roseuniform.json",
        "rose": None, "n": 74, "multizone": False,
        "role": "test", "stage_a": False, "stage_b": False,
    },
    # NOTE: "Parque real heterogeneous wind resource" test cell is frozen in
    # the pre-registration by COMPOSITION; its problem JSON (per-cell WAsP
    # Weibull A/k + speedup/turning, currently unused) is a pre-test build step
    # from parqo/build_problem.py — see PHASE2_REPORT.
}



def load_schedule(path):
    spec = importlib.util.spec_from_file_location("sched_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "schedule_fn"):
        raise RuntimeError("module does not define schedule_fn")
    return mod.schedule_fn


def _load_wind(problem_dict):
    wr = problem_dict["wind_rose"]
    wd = jnp.array(wr["directions_deg"], dtype=jnp.float64)
    ws = jnp.array(wr["speeds_ms"], dtype=jnp.float64)
    wt = jnp.array(wr["weights"], dtype=jnp.float64)
    wt = wt / jnp.sum(wt)
    return wd, ws, wt


def evaluate(cell_name, schedule_fn, seed=0, total_steps=8000, gamma_min=0.01):
    """Run one schedule on one cell/seed; return a result dict."""
    cell = CELLS[cell_name]
    prob = json.load(open(os.path.join(ROOT, cell["problem"])))
    D = float(prob["rotor_diameter"])
    min_spacing = float(prob["min_spacing_m"])
    n = int(cell["n"])
    sim = build_sim(prob)

    # wind rose: own (single-poly) or a separate rose problem (Parque)
    if cell["rose"] is None:
        wd, ws, wt = _load_wind(prob)
    else:
        rose = json.load(open(os.path.join(ROOT, cell["rose"])))
        wd, ws, wt = _load_wind(rose)

    t0 = time.time()
    if cell["multizone"]:
        zones = prob["inclusion_zones"]
        zones_j = [jnp.asarray(z) for z in zones]
        x, y = skeleton_v2.run_with_schedule(
            schedule_fn, sim, n, None, min_spacing, wd, ws, wt, D, gamma_min,
            total_steps=total_steps, seed=seed, zones=zones)
        xa, ya = np.asarray(x), np.asarray(y)
        r = sim(jnp.array(xa), jnp.array(ya), ws_amb=ws, wd_amb=wd, ti_amb=None)
        pw = r.power()[:, :len(xa)]
        aep = float(jnp.sum(jnp.sum(pw, axis=1) * wt) * 8760 / 1e6)
        sdf = float(np.max(np.asarray(multizone_sdf(
            jnp.array(xa), jnp.array(ya), zones_j))))
        dx = xa[:, None] - xa[None, :]
        dy = ya[:, None] - ya[None, :]
        md = float((np.sqrt(dx**2 + dy**2) + np.eye(n) * 1e9).min())
        tol = cell.get("feas_tol", 0.1)
        feasible = bool(sdf <= tol and md >= min_spacing - 0.1)
        extra = {"max_sdf_m": round(sdf, 4), "min_dist_m": round(md, 2),
                 "feas_tol_m": tol}
    else:
        boundary = jnp.array(prob["boundary_vertices"], dtype=jnp.float64)
        x, y = skeleton_v2.run_with_schedule(
            schedule_fn, sim, n, boundary, min_spacing, wd, ws, wt, D, gamma_min,
            total_steps=total_steps, seed=seed, zones=None)
        from dei_layout import ProblemBenchmark
        bm = ProblemBenchmark(os.path.join(ROOT, cell["problem"]))
        aep = bm.score(np.asarray(x), np.asarray(y))
        feas = bm.check_feasibility(np.asarray(x), np.asarray(y))
        feasible = bool(feas["spacing_ok"] and feas["boundary_ok"])
        extra = {"boundary_penalty": round(feas["boundary_penalty"], 6),
                 "min_dist_m": round(feas["min_turbine_distance_m"], 2)}

    return {
        "cell": cell_name, "seed": seed, "steps": total_steps,
        "gamma_min": gamma_min, "D": D, "n": n, "min_spacing": min_spacing,
        "aep_gwh": round(aep, 4), "feasible": feasible,
        "time_s": round(time.time() - t0, 1), **extra,
    }


import contextlib
import signal


@contextlib.contextmanager
def _watchdog(seconds):
    """Raise TimeoutError if a single eval exceeds `seconds` (Unix SIGALRM).
    Raised/configurable so long high-N evals (n80/n100-class) don't trip it;
    n200-class runs are gbar-only. seconds<=0 disables."""
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"eval exceeded watchdog of {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, choices=sorted(CELLS))
    p.add_argument("--schedule", required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--gamma-min", type=float, default=0.01)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--watchdog-seconds", type=int,
                   default=int(os.environ.get("FUNWAKE_WATCHDOG_S", "900")),
                   help="per-seed watchdog (raised for long high-N cells); 0 disables")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    schedule_fn = load_schedule(args.schedule)
    tag = args.tag or os.path.splitext(os.path.basename(args.schedule))[0]

    for seed in args.seeds:
        out_path = os.path.join(
            args.outdir,
            f"{tag}_{args.cell}_g{args.gamma_min:g}_s{args.steps}_seed{seed}.json")
        if os.path.exists(out_path):
            prev = json.load(open(out_path))
            print(f"SKIP {os.path.basename(out_path)} "
                  f"(aep={prev.get('aep_gwh')}, feas={prev.get('feasible')})",
                  flush=True)
            continue
        # per-seed heartbeat (start) — visibility on long high-N evals
        t_start = time.time()
        print(f"HB   start {tag} {args.cell} seed={seed} "
              f"steps={args.steps} at {time.strftime('%H:%M:%S')}", flush=True)
        try:
            with _watchdog(args.watchdog_seconds):
                rec = evaluate(args.cell, schedule_fn, seed=seed,
                               total_steps=args.steps, gamma_min=args.gamma_min)
            rec["tag"] = tag
            rec["schedule_file"] = os.path.basename(args.schedule)
        except Exception as e:
            rec = {"cell": args.cell, "seed": seed, "steps": args.steps,
                   "gamma_min": args.gamma_min, "error": str(e)[:400],
                   "tag": tag}
        with open(out_path, "w") as f:
            json.dump(rec, f)
        if "error" in rec:
            print(f"ERR  {args.cell} seed={seed}: {rec['error'][:150]} "
                  f"(after {time.time()-t_start:.0f}s)", flush=True)
        else:
            print(f"OK   {tag} {args.cell} seed={seed} aep={rec['aep_gwh']} "
                  f"feas={rec['feasible']} ({rec['time_s']}s)", flush=True)


if __name__ == "__main__":
    main()
