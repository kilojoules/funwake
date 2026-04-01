#!/usr/bin/env python
"""DEI layout optimization benchmark suite.

10 real wind farm polygons from the Danish Energy Island cluster.
All share the same turbine spec (IEA 15 MW, D=240m) and wind resource
(10-year timeseries binned to 24 sectors).

Train set (LLM sees these): farms 1-9 (the neighbor polygons).
Test set (held out):         farm 0 (the target/center farm).

Each benchmark case: optimize N turbine positions inside a polygon to
maximize AEP, with no neighbors. Constraints: boundary + 4D spacing.

The LLM develops its optimizer on the training farms. We evaluate on
the held-out target farm to measure generalization.

Usage:
    # Run baseline on all training farms
    python dei_layout.py baseline-train --wind-csv wind.csv --output results/

    # Run baseline on held-out test farm
    python dei_layout.py baseline-test --wind-csv wind.csv --output results/

    # Score a layout
    python dei_layout.py score --farm-id 0 --layout layout.json --wind-csv wind.csv

    # Run an LLM-generated optimizer on a training farm
    python dei_layout.py run --farm-id 3 --script optimizer.py --wind-csv wind.csv
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

# ── Turbine ────────────────────────────────────────────────────────────
D = 240.0
MIN_SPACING = 4.0 * D

_WS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25.0]
_POWER = [0.0,0.0,2.399,209.258,689.198,1480.608,2661.238,4308.929,
          6501.057,9260.516,12081.404,13937.297,14705.016,14931.039,
          14985.209,14996.906,14999.343,14999.855,14999.966,14999.992,
          14999.998,14999.999,15000.0,15000.0,15000.0,15000.0]
_CT = [0.889,0.889,0.889,0.800,0.800,0.800,0.800,0.800,0.800,0.793,
       0.735,0.610,0.476,0.370,0.292,0.234,0.191,0.158,0.132,0.112,
       0.096,0.083,0.072,0.063,0.055,0.049]

# ── Farm boundaries (UTM coordinates) ──────────────────────────────────
FARM_BOUNDARIES = {
    0: np.array([  # Target farm (dk0w_tender_3) — HELD OUT
        706694.3923283464, 6224158.532895836,
        703972.0844905999, 6226906.597455995,
        702624.6334635273, 6253853.5386425415,
        712771.6248419734, 6257704.934445341,
        715639.3355871611, 6260664.6846508905,
        721593.2420745814, 6257906.998015941,
    ]).reshape((-1, 2)),
    1: np.array([  # dk1d_tender_9 (SW)
        679052.6969498377, 6227229.556685498,
        676330.3891120912, 6229977.621245657,
        674982.9380850186, 6256924.562429089,
        685129.9294634647, 6260775.958231889,
        687997.6402086524, 6263735.708437439,
        693951.5466960727, 6260978.021802489,
    ]).reshape((-1, 2)),
    2: np.array([  # dk0z_tender_5 (W)
        685185.3969498377, 6238229.556685498,
        682463.0891120912, 6240977.621245657,
        681115.6380850186, 6267924.562429089,
        691262.6294634647, 6271775.958231889,
        694130.3402086524, 6274735.708437439,
        700084.2466960727, 6271978.021802489,
    ]).reshape((-1, 2)),
    3: np.array([  # dk0v_tender_1 (NW)
        695585.3969498377, 6268229.556685498,
        692863.0891120912, 6270977.621245657,
        691515.6380850186, 6297924.562429089,
        701662.6294634647, 6301775.958231889,
        704530.3402086524, 6304735.708437439,
        710484.2466960727, 6301978.021802489,
    ]).reshape((-1, 2)),
    4: np.array([  # dk0Y_tender_4 (N)
        715585.3969498377, 6288229.556685498,
        712863.0891120912, 6290977.621245657,
        711515.6380850186, 6317924.562429089,
        721662.6294634647, 6321775.958231889,
        724530.3402086524, 6324735.708437439,
        730484.2466960727, 6321978.021802489,
    ]).reshape((-1, 2)),
    5: np.array([  # dk0x_tender_2 (NE)
        725585.3969498377, 6268229.556685498,
        722863.0891120912, 6270977.621245657,
        721515.6380850186, 6297924.562429089,
        731662.6294634647, 6301775.958231889,
        734530.3402086524, 6304735.708437439,
        740484.2466960727, 6301978.021802489,
    ]).reshape((-1, 2)),
    6: np.array([  # dk1a_tender_6 (E)
        745585.3969498377, 6258229.556685498,
        742863.0891120912, 6260977.621245657,
        741515.6380850186, 6287924.562429089,
        751662.6294634647, 6291775.958231889,
        754530.3402086524, 6294735.708437439,
        760484.2466960727, 6291978.021802489,
    ]).reshape((-1, 2)),
    7: np.array([  # dk1b_tender_7 (SE)
        735585.3969498377, 6228229.556685498,
        732863.0891120912, 6230977.621245657,
        731515.6380850186, 6257924.562429089,
        741662.6294634647, 6261775.958231889,
        744530.3402086524, 6264735.708437439,
        750484.2466960727, 6261978.021802489,
    ]).reshape((-1, 2)),
    8: np.array([  # dk1c_tender_8 (S)
        715585.3969498377, 6178229.556685498,
        712863.0891120912, 6180977.621245657,
        711515.6380850186, 6207924.562429089,
        721662.6294634647, 6211775.958231889,
        724530.3402086524, 6214735.708437439,
        730484.2466960727, 6211978.021802489,
    ]).reshape((-1, 2)),
    9: np.array([  # dk1e_tender_10 (SSW)
        695585.3969498377, 6168229.556685498,
        692863.0891120912, 6170977.621245657,
        691515.6380850186, 6197924.562429089,
        701662.6294634647, 6201775.958231889,
        704530.3402086524, 6204735.708437439,
        710484.2466960727, 6201978.021802489,
    ]).reshape((-1, 2)),
}

FARM_NAMES = {
    0: "dk0w_tender_3 (target)",
    1: "dk1d_tender_9 (SW)",
    2: "dk0z_tender_5 (W)",
    3: "dk0v_tender_1 (NW)",
    4: "dk0Y_tender_4 (N)",
    5: "dk0x_tender_2 (NE)",
    6: "dk1a_tender_6 (E)",
    7: "dk1b_tender_7 (SE)",
    8: "dk1c_tender_8 (S)",
    9: "dk1e_tender_10 (SSW)",
}

TRAIN_FARMS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TEST_FARM = 0


def create_turbine():
    ws = jnp.array(_WS)
    return Turbine(rotor_diameter=D, hub_height=150.0,
                   power_curve=Curve(ws=ws, values=jnp.array(_POWER)),
                   ct_curve=Curve(ws=ws, values=jnp.array(_CT)))


def load_wind(csv_path, n_bins=24):
    df = pd.read_csv(csv_path, sep=";")
    wd_ts, ws_ts = df["WD_150"].values, df["WS_150"].values
    edges = np.linspace(0, 360, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.zeros(n_bins)
    speeds = np.zeros(n_bins)
    for i in range(n_bins):
        mask = ((wd_ts >= edges[i]) & (wd_ts < edges[i+1])) if i < n_bins-1 \
               else ((wd_ts >= edges[i]) | (wd_ts < edges[0]))
        w[i] = mask.sum()
        speeds[i] = ws_ts[mask].mean() if mask.sum() > 0 else ws_ts.mean()
    w /= w.sum()
    return jnp.array(centers), jnp.array(speeds), jnp.array(w)


def make_boundary(farm_id):
    """Return centered boundary + convex hull for a farm."""
    raw = FARM_BOUNDARIES[farm_id]
    cx, cy = raw[:, 0].mean(), raw[:, 1].mean()
    centered = raw - np.array([cx, cy])
    hull = ConvexHull(centered)
    boundary_np = centered[hull.vertices]
    return jnp.array(boundary_np), boundary_np, (cx, cy)


def init_grid(boundary_np, n_target):
    from matplotlib.path import Path as MplPath
    poly = MplPath(boundary_np)
    spacing = 4 * D
    x_lo, x_hi = boundary_np[:, 0].min(), boundary_np[:, 0].max()
    y_lo, y_hi = boundary_np[:, 1].min(), boundary_np[:, 1].max()
    gx, gy = np.meshgrid(
        np.arange(x_lo + 2*D, x_hi - 2*D, spacing),
        np.arange(y_lo + 2*D, y_hi - 2*D, spacing))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts[poly.contains_points(pts)]
    if len(pts) < n_target:
        gx, gy = np.meshgrid(
            np.arange(x_lo + D, x_hi - D, spacing * 0.7),
            np.arange(y_lo + D, y_hi - D, spacing * 0.7))
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        pts = pts[poly.contains_points(pts)]
    n = min(n_target, len(pts))
    idx = np.round(np.linspace(0, len(pts)-1, n)).astype(int)
    return jnp.array(pts[idx, 0]), jnp.array(pts[idx, 1])


class FarmBenchmark:
    """Single-farm layout optimization benchmark."""

    def __init__(self, farm_id, wind_csv, n_target=50):
        self.farm_id = farm_id
        self.farm_name = FARM_NAMES[farm_id]
        self.n_target = n_target
        self.turbine = create_turbine()
        self.sim = WakeSimulation(self.turbine, BastankhahGaussianDeficit(k=0.04))
        self.wd, self.ws, self.weights = load_wind(wind_csv)
        self.boundary, self.boundary_np, self.centroid = make_boundary(farm_id)
        self.min_spacing = MIN_SPACING
        self.init_x, self.init_y = init_grid(self.boundary_np, n_target)
        # Actual count may be less if boundary is small
        self.n_target = len(self.init_x)

    def objective(self, x, y):
        """Negative AEP (minimize). No neighbors — pure layout optimization."""
        r = self.sim(x, y, ws_amb=self.ws, wd_amb=self.wd, ti_amb=None)
        power = r.power()[:, :len(x)]
        return -jnp.sum(power * self.weights[:, None]) * 8760 / 1e6

    def score(self, x, y):
        """AEP in GWh (maximize)."""
        return -float(self.objective(jnp.array(x), jnp.array(y)))

    def check_feasibility(self, x, y):
        from pixwake.optim.sgd import boundary_penalty
        x, y = jnp.array(x), jnp.array(y)
        bnd_pen = float(boundary_penalty(x, y, self.boundary))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(len(x)) * 1e10)
        min_dist = float(jnp.min(dist))
        return {
            "boundary_penalty": bnd_pen,
            "min_turbine_distance_m": min_dist,
            "min_spacing_m": float(self.min_spacing),
            "spacing_ok": min_dist >= float(self.min_spacing) * 0.99,
            "boundary_ok": bnd_pen < 1e-3,
        }

    def info(self):
        """Problem description for the LLM."""
        return {
            "farm_id": self.farm_id,
            "farm_name": self.farm_name,
            "n_target": self.n_target,
            "rotor_diameter": float(D),
            "min_spacing_m": float(self.min_spacing),
            "boundary_vertices": self.boundary_np.tolist(),
            "init_x": [float(v) for v in self.init_x],
            "init_y": [float(v) for v in self.init_y],
            "wind_rose": {
                "directions_deg": [float(v) for v in self.wd],
                "speeds_ms": [float(v) for v in self.ws],
                "weights": [float(v) for v in self.weights],
            },
        }

    def random_layout(self, rng):
        """Generate a random feasible layout inside the polygon via rejection sampling."""
        from matplotlib.path import Path as MplPath
        poly = MplPath(self.boundary_np)
        x_lo, x_hi = self.boundary_np[:, 0].min(), self.boundary_np[:, 0].max()
        y_lo, y_hi = self.boundary_np[:, 1].min(), self.boundary_np[:, 1].max()
        pts = []
        while len(pts) < self.n_target:
            cx = rng.uniform(x_lo, x_hi, size=self.n_target * 20)
            cy = rng.uniform(y_lo, y_hi, size=self.n_target * 20)
            candidates = np.column_stack([cx, cy])
            inside = candidates[poly.contains_points(candidates)]
            for p in inside:
                if len(pts) >= self.n_target:
                    break
                pts.append(p)
        pts = np.array(pts[:self.n_target])
        return jnp.array(pts[:, 0]), jnp.array(pts[:, 1])

    def run_baseline(self, max_iter=4000, lr=50.0,
                     additional_constant_lr_iterations=2000,
                     n_starts=1, verbose=False):
        """Run baseline optimizer, optionally with multi-start.

        Returns (best_x, best_y, best_aep, total_elapsed).
        """
        settings = SGDSettings(
            learning_rate=lr, max_iter=max_iter,
            additional_constant_lr_iterations=additional_constant_lr_iterations,
            tol=1e-6)
        best_aep = -np.inf
        best_x, best_y = None, None
        t0 = time.time()

        for s in range(n_starts):
            if s == 0:
                ix, iy = self.init_x, self.init_y
            else:
                rng = np.random.default_rng(s)
                ix, iy = self.random_layout(rng)

            opt_x, opt_y = topfarm_sgd_solve(
                self.objective, ix, iy,
                self.boundary, self.min_spacing, settings)
            aep = self.score(opt_x, opt_y)

            if aep > best_aep:
                best_aep = aep
                best_x, best_y = opt_x, opt_y

            if verbose and (s % 50 == 0 or s == n_starts - 1):
                print(f"    start {s+1}/{n_starts}: AEP={aep:.2f}, "
                      f"best={best_aep:.2f} GWh, "
                      f"{time.time()-t0:.0f}s elapsed")

        elapsed = time.time() - t0
        return best_x, best_y, best_aep, elapsed


class ProblemBenchmark:
    """Benchmark loaded from a problem JSON file (any turbine/boundary/wind)."""

    def __init__(self, problem_path):
        with open(problem_path) as f:
            info = json.load(f)
        self.info_data = info
        self.farm_id = info.get("farm_id", "unknown")
        self.farm_name = info.get("farm_name", str(self.farm_id))
        self.n_target = info["n_target"]

        rotor_d = info["rotor_diameter"]
        hub_h = info.get("hub_height", 150.0)
        self.min_spacing = info["min_spacing_m"]

        # Build turbine from embedded curves or fallback to DEI defaults
        if "turbine" in info:
            t = info["turbine"]
            ws_arr = jnp.array(t["power_curve_ws"], dtype=float)
            power_arr = jnp.array(t["power_curve_kw"], dtype=float)
            ct_ws = jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float)
            ct_arr = jnp.array(t["ct_curve_ct"], dtype=float)
        else:
            ws_arr = jnp.array(_WS)
            power_arr = jnp.array(_POWER)
            ct_ws = ws_arr
            ct_arr = jnp.array(_CT)

        self.turbine = Turbine(
            rotor_diameter=rotor_d, hub_height=hub_h,
            power_curve=Curve(ws=ws_arr, values=power_arr),
            ct_curve=Curve(ws=ct_ws, values=ct_arr))
        self.sim = WakeSimulation(self.turbine, BastankhahGaussianDeficit(k=0.04))

        wr = info["wind_rose"]
        self.wd = jnp.array(wr["directions_deg"])
        self.ws = jnp.array(wr["speeds_ms"])
        self.weights = jnp.array(wr["weights"])

        bv = np.array(info["boundary_vertices"])
        from scipy.spatial import ConvexHull
        hull = ConvexHull(bv)
        self.boundary_np = bv[hull.vertices]
        self.boundary = jnp.array(self.boundary_np)

        if "init_x" in info:
            self.init_x = jnp.array(info["init_x"])
            self.init_y = jnp.array(info["init_y"])
        else:
            # Generate grid layout when no init positions provided
            self.init_x, self.init_y = init_grid(self.boundary_np, self.n_target)

    def objective(self, x, y):
        r = self.sim(x, y, ws_amb=self.ws, wd_amb=self.wd, ti_amb=None)
        power = r.power()[:, :len(x)]
        return -jnp.sum(power * self.weights[:, None]) * 8760 / 1e6

    def score(self, x, y):
        return -float(self.objective(jnp.array(x), jnp.array(y)))

    def check_feasibility(self, x, y):
        from pixwake.optim.sgd import boundary_penalty
        x, y = jnp.array(x), jnp.array(y)
        bnd_pen = float(boundary_penalty(x, y, self.boundary))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(len(x)) * 1e10)
        min_dist = float(jnp.min(dist))
        return {
            "boundary_penalty": bnd_pen,
            "min_turbine_distance_m": min_dist,
            "min_spacing_m": float(self.min_spacing),
            "spacing_ok": min_dist >= float(self.min_spacing) * 0.99,
            "boundary_ok": bnd_pen < 1e-3,
        }

    def random_layout(self, rng):
        from matplotlib.path import Path as MplPath
        poly = MplPath(self.boundary_np)
        x_lo, x_hi = self.boundary_np[:, 0].min(), self.boundary_np[:, 0].max()
        y_lo, y_hi = self.boundary_np[:, 1].min(), self.boundary_np[:, 1].max()
        pts = []
        while len(pts) < self.n_target:
            cx = rng.uniform(x_lo, x_hi, size=self.n_target * 20)
            cy = rng.uniform(y_lo, y_hi, size=self.n_target * 20)
            candidates = np.column_stack([cx, cy])
            inside = candidates[poly.contains_points(candidates)]
            for p in inside:
                if len(pts) >= self.n_target:
                    break
                pts.append(p)
        pts = np.array(pts[:self.n_target])
        return jnp.array(pts[:, 0]), jnp.array(pts[:, 1])

    def run_baseline(self, max_iter=4000, lr=50.0,
                     additional_constant_lr_iterations=2000,
                     n_starts=1, verbose=False):
        settings = SGDSettings(
            learning_rate=lr, max_iter=max_iter,
            additional_constant_lr_iterations=additional_constant_lr_iterations,
            tol=1e-6)
        best_aep = -np.inf
        best_x, best_y = None, None
        t0 = time.time()

        for s in range(n_starts):
            if s == 0:
                ix, iy = self.init_x, self.init_y
            else:
                rng = np.random.default_rng(s)
                ix, iy = self.random_layout(rng)

            opt_x, opt_y = topfarm_sgd_solve(
                self.objective, ix, iy,
                self.boundary, self.min_spacing, settings)
            aep = self.score(opt_x, opt_y)

            if aep > best_aep:
                best_aep = aep
                best_x, best_y = opt_x, opt_y

            if verbose and (s % 50 == 0 or s == n_starts - 1):
                print(f"    start {s+1}/{n_starts}: AEP={aep:.2f}, "
                      f"best={best_aep:.2f} GWh, "
                      f"{time.time()-t0:.0f}s elapsed")

        elapsed = time.time() - t0
        return best_x, best_y, best_aep, elapsed


def _run_baselines(farm_ids, wind_csv, n_target, max_iter, lr, output_dir,
                   additional_constant_lr_iterations=2000, n_starts=1):
    """Run baseline on multiple farms, save results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    for fid in farm_ids:
        bm = FarmBenchmark(fid, wind_csv, n_target)
        print(f"\nFarm {fid} ({bm.farm_name}): {bm.n_target} turbines")
        verbose = n_starts > 1
        if n_starts > 1:
            print(f"  Running {n_starts} multi-starts "
                  f"(max_iter={max_iter}, const_lr={additional_constant_lr_iterations})...")
        opt_x, opt_y, aep, elapsed = bm.run_baseline(
            max_iter, lr, additional_constant_lr_iterations, n_starts,
            verbose=verbose)
        feas = bm.check_feasibility(opt_x, opt_y)
        results[fid] = {
            "farm_id": fid, "farm_name": bm.farm_name,
            "n_target": bm.n_target,
            "aep_gwh": aep, "elapsed_s": elapsed,
            "n_starts": n_starts,
            "x": [float(v) for v in opt_x],
            "y": [float(v) for v in opt_y],
            **feas,
        }
        print(f"  AEP: {aep:.2f} GWh, time: {elapsed:.1f}s, "
              f"feasible: {feas['spacing_ok'] and feas['boundary_ok']}")

        # Also write problem info per farm
        info_path = out / f"problem_farm{fid}.json"
        with open(info_path, "w") as f:
            json.dump(bm.info(), f, indent=2)

    # Save all baselines
    with open(out / "baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaselines saved to {out / 'baselines.json'}")
    return results


def main():
    p = argparse.ArgumentParser(description="DEI farm benchmark suite")
    p.add_argument("--wind-csv", required=True)
    p.add_argument("--n-target", type=int, default=50)

    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared baseline args
    for sp in [
        sub.add_parser("baseline-train", help="Run baseline on training farms (1-9)"),
        sub.add_parser("baseline-test", help="Run baseline on held-out test farm (0)"),
        sub.add_parser("baseline-all", help="Run baseline on all farms"),
    ]:
        sp.add_argument("--max-iter", type=int, default=4000)
        sp.add_argument("--const-lr-iters", type=int, default=2000,
                        help="additional_constant_lr_iterations (default: 2000)")
        sp.add_argument("--lr", type=float, default=50.0)
        sp.add_argument("--n-starts", type=int, default=500,
                        help="Number of multi-start restarts (default: 500)")
        sp.add_argument("--output", default="results")

    bp = sub.add_parser("baseline-problem",
                        help="Run baseline on a problem JSON file")
    bp.add_argument("--problem", required=True, help="Path to problem JSON")
    bp.add_argument("--max-iter", type=int, default=4000)
    bp.add_argument("--const-lr-iters", type=int, default=2000)
    bp.add_argument("--lr", type=float, default=50.0)
    bp.add_argument("--n-starts", type=int, default=500)
    bp.add_argument("--output", default="results")

    sc = sub.add_parser("score", help="Score a layout")
    sc.add_argument("--farm-id", type=int, required=True)
    sc.add_argument("--layout", required=True)

    sp_score = sub.add_parser("score-problem",
                              help="Score a layout against a problem JSON")
    sp_score.add_argument("--problem", required=True)
    sp_score.add_argument("--layout", required=True)

    args = p.parse_args()

    baseline_kwargs = {}
    if hasattr(args, "max_iter"):
        baseline_kwargs = dict(
            max_iter=args.max_iter, lr=args.lr, output_dir=args.output,
            additional_constant_lr_iterations=args.const_lr_iters,
            n_starts=args.n_starts,
        )

    if args.cmd == "baseline-train":
        _run_baselines(TRAIN_FARMS, args.wind_csv, args.n_target,
                       **baseline_kwargs)
    elif args.cmd == "baseline-test":
        _run_baselines([TEST_FARM], args.wind_csv, args.n_target,
                       **baseline_kwargs)
    elif args.cmd == "baseline-all":
        _run_baselines(list(range(10)), args.wind_csv, args.n_target,
                       **baseline_kwargs)
    elif args.cmd == "baseline-problem":
        bm = ProblemBenchmark(args.problem)
        print(f"\n{bm.farm_name}: {bm.n_target} turbines")
        verbose = args.n_starts > 1
        if args.n_starts > 1:
            print(f"  Running {args.n_starts} multi-starts "
                  f"(max_iter={args.max_iter}, const_lr={args.const_lr_iters})...")
        opt_x, opt_y, aep, elapsed = bm.run_baseline(
            args.max_iter, args.lr, args.const_lr_iters, args.n_starts,
            verbose=verbose)
        feas = bm.check_feasibility(opt_x, opt_y)
        result = {
            "farm_id": bm.farm_id, "farm_name": bm.farm_name,
            "n_target": bm.n_target, "aep_gwh": aep, "elapsed_s": elapsed,
            "n_starts": args.n_starts,
            "x": [float(v) for v in opt_x],
            "y": [float(v) for v in opt_y],
            **feas,
        }
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / f"baseline_{bm.farm_id}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  AEP: {aep:.2f} GWh, time: {elapsed:.1f}s, "
              f"feasible: {feas['spacing_ok'] and feas['boundary_ok']}")
        print(f"  Saved to {out_file}")
    elif args.cmd == "score":
        bm = FarmBenchmark(args.farm_id, args.wind_csv, args.n_target)
        with open(args.layout) as f:
            layout = json.load(f)
        aep = bm.score(layout["x"], layout["y"])
        feas = bm.check_feasibility(layout["x"], layout["y"])
        print(f"Farm {args.farm_id}: AEP={aep:.2f} GWh, "
              f"feasible={feas['spacing_ok'] and feas['boundary_ok']}")
    elif args.cmd == "score-problem":
        bm = ProblemBenchmark(args.problem)
        with open(args.layout) as f:
            layout = json.load(f)
        aep = bm.score(layout["x"], layout["y"])
        feas = bm.check_feasibility(layout["x"], layout["y"])
        print(f"{bm.farm_name}: AEP={aep:.2f} GWh, "
              f"feasible={feas['spacing_ok'] and feas['boundary_ok']}")


if __name__ == "__main__":
    main()
