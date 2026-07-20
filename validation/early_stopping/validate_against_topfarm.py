"""Validate pixwake's new early_stopping path against TopFarm2's EasySGDDriver
(speedupSGD=True, sgd_thresh=0.1) on a 9-turbine fixture.

Both engines wired to identical wake model (NOJ k=0.05 + RSS, established
equivalent in Part 1 to float64 roundoff) and identical SGD hyperparameters
(learning_rate, beta1, beta2, gamma_min_factor, max_iter). Same random init
per seed. Differences should be attributable to the early-stopping
implementation alone.

Standard fixture (synthetic 720x720 m square, HornsrevV80, single wind dir
mirrors TopFarm's docs/notebooks/sgd_slsqp_comparison.ipynb in miniature.

Outputs:
- validation/early_stopping/topfarm_vs_pixwake_es.json — per-seed results
- validation/early_stopping/REPORT_STEP2.md — summary

Usage:
    PYTHONPATH=dependencies/pixwake/src pixi run python \\
        validation/early_stopping/validate_against_topfarm.py \\
        --seeds 50 --max-iter 500
"""
import argparse
import json
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# --- TopFarm imports ---
import warnings
warnings.filterwarnings("ignore")

from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.site import UniformSite
from py_wake import NOJ
from py_wake.utils.gradients import autograd
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation
import openmdao  # ensure backend available

# --- pixwake imports ---
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit.noj import NOJDeficit
from pixwake.superposition import SquaredSum
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


# ---------- Fixture: synthetic 720x720m, V80, NOJ k=0.05, single-dir wind ----------

WT = HornsrevV80()
D = float(WT.diameter())                 # 80.0
HH = float(WT.hub_height())              # 70.0
XU = float(25.0 * D)                     # 2000 — huge box (no constraint activation)
YU = float(25.0 * D)                     # 2000
N_WT = 9
MIN_SPACING = float(2 * D)               # 160
BOUNDARY = np.array([(0.0, 0.0), (XU, 0.0), (XU, YU), (0.0, YU)])
WD_GRID = np.array([270.0])              # single direction for clean reproduction
WS_GRID = np.array([9.0])                # single speed
SITE = UniformSite(p_wd=[1.0], ti=0.06, ws=WS_GRID.tolist())


def init_layout(seed: int):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.0, XU, N_WT)
    y = rng.uniform(0.0, YU, N_WT)
    return x, y


# ---------- TopFarm side ----------

def topfarm_solve(seed: int, max_iter: int, lr: float, gamma_min: float,
                  beta1: float, beta2: float, speedupSGD: bool, sgd_thresh: float):
    """Solve via TopFarm2 EasySGDDriver."""
    x0, y0 = init_layout(seed)

    # py_wake wake model
    wake_model = NOJ(SITE, WT, k=0.05)  # SquaredSum by default

    def aep_func(x, y, **kwargs):
        return float(wake_model(x, y, wd=WD_GRID, ws=WS_GRID, time=False).aep(
            normalize_probabilities=False).sum().values * 1e6)

    def aep_jac(x, y, **kwargs):
        jx, jy = wake_model.aep_gradients(
            gradient_method=autograd, wrt_arg=['x', 'y'],
            x=x, y=y, ws=WS_GRID, wd=WD_GRID, time=False,
        )
        return np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6

    aep_comp = CostModelComponent(
        input_keys=['x', 'y'], n_wt=N_WT, cost_function=aep_func, objective=True,
        cost_gradient_function=aep_jac, maximize=True,
    )
    boundary_constraint = XYBoundaryConstraint(BOUNDARY, 'polygon')
    constraints = DistanceConstraintAggregation(
        boundary_constraint, N_WT, MIN_SPACING, WT,
    )
    driver = EasySGDDriver(
        maxiter=max_iter,
        learning_rate=lr,
        gamma_min_factor=gamma_min,
        beta1=beta1, beta2=beta2,
        speedupSGD=speedupSGD,
        sgd_thresh=sgd_thresh,
    )
    tf = TopFarmProblem(
        design_vars={'x': x0, 'y': y0},
        cost_comp=aep_comp,
        constraints=constraints,
        driver=driver,
        plot_comp=None,
        expected_cost=1.0,
    )
    t0 = time.time()
    cost, state, recorder = tf.optimize()
    elapsed = time.time() - t0
    x_opt = np.asarray(state['x'])
    y_opt = np.asarray(state['y'])
    return {
        'x_opt': x_opt.tolist(),
        'y_opt': y_opt.tolist(),
        'aep_at_finish_w': aep_func(x_opt, y_opt),
        'elapsed_s': elapsed,
    }


# ---------- pixwake side ----------

def _pixwake_sim():
    """Build pixwake WakeSimulation with NOJ k=0.05 + SquaredSum."""
    # Build a HornsrevV80 turbine in pixwake — extract power/Ct curve from py_wake
    ws_grid = jnp.linspace(3.0, 25.0, 200)
    p_w = WT.power(np.asarray(ws_grid))   # W
    ct = WT.ct(np.asarray(ws_grid))        # dimensionless
    p_kw = jnp.array(p_w) / 1e3
    ct = jnp.array(ct)
    turbine = Turbine(
        rotor_diameter=D, hub_height=HH,
        power_curve=Curve(ws=ws_grid, values=p_kw),
        ct_curve=Curve(ws=ws_grid, values=ct),
    )
    return WakeSimulation(turbine, NOJDeficit(k=0.05, superposition=SquaredSum()))


def pixwake_solve(seed: int, max_iter: int, lr: float, gamma_min: float,
                  beta1: float, beta2: float, early_stopping: bool, threshold: float):
    """Solve via pixwake topfarm_sgd_solve."""
    sim = _pixwake_sim()
    x0, y0 = init_layout(seed)
    x0_j = jnp.array(x0)
    y0_j = jnp.array(y0)
    boundary = jnp.array(BOUNDARY)

    ws_j = jnp.array(WS_GRID)
    wd_j = jnp.array(WD_GRID)

    def neg_aep(x, y):
        r = sim(x, y, ws_amb=ws_j, wd_amb=wd_j, ti_amb=None)
        # TopFarm's aep_func returns py_wake's .aep().sum() * 1e6 = GWh * 1e6
        # = kWh (annual energy in kWh). Match that scale here so the SGD
        # gradient is on the SAME numerical scale as TopFarm's — otherwise the
        # same learning_rate produces vastly different step sizes.
        p_total_kw = jnp.sum(r.power())
        return -p_total_kw * 8760.0  # negative kWh per year

    settings = SGDSettings(
        learning_rate=lr,
        gamma_min_factor=gamma_min,
        beta1=beta1, beta2=beta2,
        max_iter=max_iter,
        tol=1e-10,            # effectively disabled — let max_iter / ES decide
        early_stopping=early_stopping,
        early_stop_threshold=threshold,
    )
    t0 = time.time()
    x_opt, y_opt = topfarm_sgd_solve(neg_aep, x0_j, y0_j, boundary, MIN_SPACING, settings)
    x_opt = np.asarray(x_opt)
    y_opt = np.asarray(y_opt)
    elapsed = time.time() - t0
    # Deterministic AEP eval matching TopFarm's aep_func (kWh/year)
    p_total_kw = float(jnp.sum(sim(jnp.array(x_opt), jnp.array(y_opt),
                                    ws_amb=ws_j, wd_amb=wd_j, ti_amb=None).power()))
    aep_at_finish_w = p_total_kw * 8760.0  # kWh/year, matches TF aep_func
    return {
        'x_opt': x_opt.tolist(),
        'y_opt': y_opt.tolist(),
        'aep_at_finish_w': aep_at_finish_w,
        'elapsed_s': elapsed,
    }


def check_feasibility(x, y):
    """Boundary + 2D spacing satisfied? Returns max in/out boundary distance and
    min pair distance, plus practical-feasibility verdict."""
    x_j, y_j = jnp.array(x), jnp.array(y)
    boundary = jnp.array(BOUNDARY)
    bp = float(boundary_penalty(x_j, y_j, boundary))
    sp = float(spacing_penalty(x_j, y_j, MIN_SPACING))
    n = len(x)
    dx = np.subtract.outer(x, x)
    dy = np.subtract.outer(y, y)
    d = np.sqrt(dx**2 + dy**2)
    d[np.diag_indices(n)] = np.inf
    min_pair = float(d.min())
    # Inside boundary: each turbine's distances to each polygon edge (signed).
    # Use boundary_penalty as the smoothed constraint; raw check via point-in-polygon.
    from shapely.geometry import Point, Polygon
    poly = Polygon([(p[0], p[1]) for p in BOUNDARY])
    n_inside = sum(int(poly.contains(Point(xi, yi))) for xi, yi in zip(x, y))
    n_on_or_outside = n - n_inside
    return {
        'boundary_penalty': bp,
        'spacing_penalty': sp,
        'min_pair_dist_m': min_pair,
        'n_inside_polygon': n_inside,
        'n_on_or_outside': n_on_or_outside,
        'practical_feasible': (bp < 1e-2) and (sp < 1e-2) and (min_pair >= MIN_SPACING - 1e-3),
    }


def run_seed(seed, args, es_on: bool):
    """Run one seed in BOTH impls with the same ES setting."""
    tf_result = topfarm_solve(
        seed, args.max_iter, args.lr, args.gamma_min,
        args.beta1, args.beta2,
        speedupSGD=es_on, sgd_thresh=args.threshold,
    )
    px_result = pixwake_solve(
        seed, args.max_iter, args.lr, args.gamma_min,
        args.beta1, args.beta2,
        early_stopping=es_on, threshold=args.threshold,
    )
    tf_feas = check_feasibility(tf_result['x_opt'], tf_result['y_opt'])
    px_feas = check_feasibility(px_result['x_opt'], px_result['y_opt'])

    # Layout difference (positions in meters; sort both so identical turbines align)
    tf_xy = np.array(list(zip(tf_result['x_opt'], tf_result['y_opt'])))
    px_xy = np.array(list(zip(px_result['x_opt'], px_result['y_opt'])))
    # Order may differ between impls; use Hungarian assignment for max-min pairing
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(tf_xy[:, None, :] - px_xy[None, :, :], axis=-1)
    row_idx, col_idx = linear_sum_assignment(cost)
    matched_diff = cost[row_idx, col_idx]
    max_pos_diff = float(matched_diff.max())
    mean_pos_diff = float(matched_diff.mean())
    aep_delta_pct = (
        (px_result['aep_at_finish_w'] - tf_result['aep_at_finish_w'])
        / abs(tf_result['aep_at_finish_w']) * 100.0
        if tf_result['aep_at_finish_w'] else None
    )

    return {
        'seed': seed,
        'es_on': es_on,
        'topfarm': {**tf_result, **tf_feas},
        'pixwake': {**px_result, **px_feas},
        'agreement': {
            'max_position_diff_m': max_pos_diff,
            'mean_position_diff_m': mean_pos_diff,
            'aep_delta_pct': aep_delta_pct,
            'aep_delta_w': px_result['aep_at_finish_w'] - tf_result['aep_at_finish_w'],
            'both_feasible': tf_feas['practical_feasible'] and px_feas['practical_feasible'],
            'feasibility_agreement': tf_feas['practical_feasible'] == px_feas['practical_feasible'],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=50)
    p.add_argument('--max-iter', type=int, default=500)
    p.add_argument('--lr', type=float, default=D / 5)           # 16
    p.add_argument('--gamma-min', type=float, default=0.1)
    p.add_argument('--beta1', type=float, default=0.1)
    p.add_argument('--beta2', type=float, default=0.2)
    p.add_argument('--threshold', type=float, default=0.1)
    p.add_argument('--out', default='validation/early_stopping/topfarm_vs_pixwake_es.json')
    p.add_argument('--es-only', action='store_true', help='Skip ES-OFF runs')
    args = p.parse_args()

    results = {'config': vars(args), 'fixture': {
        'n_wt': N_WT, 'D': D, 'min_spacing_m': MIN_SPACING,
        'boundary': BOUNDARY.tolist(),
        'wake': 'NOJ k=0.05 + SquaredSum',
        'wd': WD_GRID.tolist(), 'ws': WS_GRID.tolist(),
        'site': 'UniformSite (single dir, single speed)',
    }, 'es_on': [], 'es_off': []}

    t_start = time.time()
    print('\n=== ES = True (both impls) ===')
    for s in range(args.seeds):
        r = run_seed(s, args, es_on=True)
        results['es_on'].append(r)
        agr = r['agreement']
        print(
            f"seed={s} max_pos_Δ={agr['max_position_diff_m']:.3f} m  "
            f"AEP_Δ={agr['aep_delta_pct']:+.4f}%  "
            f"feas_agree={agr['feasibility_agreement']}  "
            f"both_feas={agr['both_feasible']}  "
            f"TF[bp={r['topfarm']['boundary_penalty']:.2e}, sp={r['topfarm']['spacing_penalty']:.2e}]  "
            f"PX[bp={r['pixwake']['boundary_penalty']:.2e}, sp={r['pixwake']['spacing_penalty']:.2e}]",
            flush=True,
        )

    if not args.es_only:
        print('\n=== ES = False (positive control prep) ===')
        for s in range(args.seeds):
            r = run_seed(s, args, es_on=False)
            results['es_off'].append(r)
            agr = r['agreement']
            print(
                f"seed={s} max_pos_Δ={agr['max_position_diff_m']:.3f} m  "
                f"feas_agree={agr['feasibility_agreement']}  "
                f"TF_feas={r['topfarm']['practical_feasible']}  PX_feas={r['pixwake']['practical_feasible']}",
                flush=True,
            )

    # ============ Aggregate ============
    es_on_pos_diff = np.array([r['agreement']['max_position_diff_m'] for r in results['es_on']])
    es_on_aep_pct = np.array([r['agreement']['aep_delta_pct'] for r in results['es_on']
                              if r['agreement']['aep_delta_pct'] is not None])
    feas_tf_es = sum(r['topfarm']['practical_feasible'] for r in results['es_on'])
    feas_px_es = sum(r['pixwake']['practical_feasible'] for r in results['es_on'])
    feas_agree_es = sum(r['agreement']['feasibility_agreement'] for r in results['es_on'])

    summary = {
        'es_on': {
            'n_seeds': len(results['es_on']),
            'max_position_diff_p50_m': float(np.median(es_on_pos_diff)),
            'max_position_diff_p95_m': float(np.percentile(es_on_pos_diff, 95)),
            'max_position_diff_max_m': float(np.max(es_on_pos_diff)),
            'aep_delta_pct_p50': float(np.median(es_on_aep_pct)) if len(es_on_aep_pct) else None,
            'aep_delta_pct_max_abs': float(np.max(np.abs(es_on_aep_pct))) if len(es_on_aep_pct) else None,
            'topfarm_feasibility_rate': feas_tf_es / len(results['es_on']),
            'pixwake_feasibility_rate': feas_px_es / len(results['es_on']),
            'feasibility_agreement_rate': feas_agree_es / len(results['es_on']),
        },
    }

    if results['es_off']:
        # Positive control: find seeds where ES=OFF -> infeasible in BOTH but ES=ON -> feasible in BOTH.
        es_on_lookup = {r['seed']: r for r in results['es_on']}
        positive_control = []
        for r_off in results['es_off']:
            s = r_off['seed']
            r_on = es_on_lookup.get(s)
            if r_on is None:
                continue
            cond_off = (not r_off['topfarm']['practical_feasible']) or (not r_off['pixwake']['practical_feasible'])
            cond_on = r_on['agreement']['both_feasible']
            if cond_off and cond_on:
                positive_control.append({
                    'seed': s,
                    'off_tf_feas': r_off['topfarm']['practical_feasible'],
                    'off_px_feas': r_off['pixwake']['practical_feasible'],
                    'on_both_feas': True,
                })
        feas_tf_off = sum(r['topfarm']['practical_feasible'] for r in results['es_off'])
        feas_px_off = sum(r['pixwake']['practical_feasible'] for r in results['es_off'])
        summary['es_off'] = {
            'n_seeds': len(results['es_off']),
            'topfarm_feasibility_rate': feas_tf_off / len(results['es_off']),
            'pixwake_feasibility_rate': feas_px_off / len(results['es_off']),
        }
        summary['positive_control_seeds_off_infeasible_on_feasible'] = positive_control

    results['summary'] = summary
    results['elapsed_total_s'] = time.time() - t_start

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}  ({results['elapsed_total_s']:.1f}s)")


if __name__ == '__main__':
    main()
