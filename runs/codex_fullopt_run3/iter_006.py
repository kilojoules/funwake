"""L-BFGS-B penalty refinement from wind-oriented lattice basins.

HYPOTHESIS: Direct L-BFGS-B with smooth spacing/boundary penalties can move
quickly inside the same good lattice basins as earlier custom optimizers while
avoiding the hand-rolled Adam dynamics from the previous best attempt.
AXIS: scipy_lbfgs penalty optimization with deterministic wind-oriented lattice starts.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def _edge_clearance(x, y, boundary):
    n_verts = boundary.shape[0]

    def one_edge(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex = x2 - x1
        ey = y2 - y1
        el = jnp.sqrt(ex * ex + ey * ey) + 1e-12
        return (x - x1) * (-ey / el) + (y - y1) * (ex / el)

    return jax.vmap(one_edge)(jnp.arange(n_verts))


def _min_distance(x, y):
    n = x.shape[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n) * 1e18)
    return jnp.min(d)


def _feasible(x, y, boundary, min_spacing):
    return jnp.logical_and(
        jnp.all(jnp.min(_edge_clearance(x, y, boundary), axis=0) >= -1e-5),
        _min_distance(x, y) >= min_spacing * 0.9995,
    )


def _select_inside(rx, ry, boundary, n_target, min_spacing):
    clearance = jnp.min(_edge_clearance(rx, ry, boundary), axis=0)
    score = jnp.where(clearance > min_spacing * 0.025, clearance, -1e15)
    idx = jnp.argsort(score)[-n_target:]
    return rx[idx], ry[idx]


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    @jax.jit
    def neg_aep(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    center = jnp.mean(boundary, axis=0)
    span_x = x_max - x_min
    span_y = y_max - y_min
    diag = jnp.sqrt(span_x * span_x + span_y * span_y)

    wd_rad = jnp.deg2rad(wd)
    energy = weights * ws**3
    dominant = jnp.arctan2(
        jnp.sum(jnp.sin(wd_rad) * energy),
        jnp.sum(jnp.cos(wd_rad) * energy),
    )

    def lattice(params):
        sx, sy, theta_off, ox_raw, oy_raw, shear, aspect = params
        theta = dominant + theta_off
        row_step = sy * aspect * min_spacing * 0.8660254037844386
        n_side = int(np.ceil(float(diag / (min_spacing * 0.78)))) + 12
        ii, jj = jnp.meshgrid(
            jnp.arange(n_side) - n_side // 2,
            jnp.arange(n_side) - n_side // 2,
        )
        ix = ii.ravel()
        iy = jj.ravel()
        hx = (ix + 0.5 * (iy % 2)) * sx * min_spacing
        hy = iy * row_step
        hx = hx + shear * hy
        ox = x_min + ox_raw * span_x
        oy = y_min + oy_raw * span_y
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        return _select_inside(rx, ry, boundary, n_target, min_spacing)

    low = jnp.array([1.025, 1.03, -jnp.pi / 3.0, 0.05, 0.05, -0.18, 0.92])
    high = jnp.array([3.6, 3.8, jnp.pi / 3.0, 0.95, 0.95, 0.18, 1.22])

    def scale(raw):
        return low + raw * (high - low)

    seed_raw = jnp.array(
        [
            [0.02, 0.03, 0.50, 0.50, 0.50, 0.50, 0.20],
            [0.04, 0.06, 0.32, 0.43, 0.46, 0.43, 0.24],
            [0.05, 0.02, 0.69, 0.56, 0.52, 0.57, 0.23],
            [0.14, 0.10, 0.18, 0.45, 0.58, 0.50, 0.30],
            [0.10, 0.16, 0.84, 0.53, 0.42, 0.58, 0.36],
        ],
        dtype=jnp.float64,
    )

    key = jax.random.PRNGKey(906)
    key, sub = jax.random.split(key)
    n_rand = 8 if n_target <= 55 else 3
    raw = jnp.vstack([seed_raw, jax.random.uniform(sub, (n_rand, 7))])

    def start_score(params):
        sx, sy = lattice(scale(params))
        coords = jnp.concatenate([sx, sy])
        feas_pen = jnp.where(_feasible(sx, sy, boundary, min_spacing), 0.0, 5e5)
        return neg_aep(coords) + feas_pen

    scored = [(float(start_score(raw[i])), raw[i]) for i in range(raw.shape[0])]
    scored = sorted(scored, key=lambda item: item[0])
    starts = [lattice(scale(item[1])) for item in scored[:4]]

    best_x, best_y = starts[0]
    best_obj = jnp.where(
        _feasible(best_x, best_y, boundary, min_spacing),
        neg_aep(jnp.concatenate([best_x, best_y])),
        jnp.inf,
    )
    for sx, sy in starts:
        val = neg_aep(jnp.concatenate([sx, sy]))
        if _feasible(sx, sy, boundary, min_spacing) & (val < best_obj):
            best_x, best_y, best_obj = sx, sy, val

    upper = jnp.triu(jnp.ones((n_target, n_target)), k=1)

    @jax.jit
    def penalty_objective(coords, penalty_weight, spacing_margin, boundary_margin):
        x = coords[:n_target]
        y = coords[n_target:]
        obj = neg_aep(coords)

        clearance = _edge_clearance(x, y, boundary)
        b_violation = jnp.maximum(0.0, boundary_margin - clearance)
        b_pen = jnp.sum(b_violation * b_violation)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n_target) * 1e18)
        s_violation = jnp.maximum(0.0, min_spacing * spacing_margin - dist)
        s_pen = jnp.sum(upper * s_violation * s_violation)

        return obj + penalty_weight * (b_pen + s_pen)

    value_and_grad = jax.jit(jax.value_and_grad(penalty_objective))

    def scipy_fun(coords, penalty_weight, spacing_margin, boundary_margin):
        val, grad = value_and_grad(
            jnp.asarray(coords),
            penalty_weight,
            spacing_margin,
            boundary_margin,
        )
        return float(val), np.asarray(grad, dtype=np.float64)

    bounds = [(float(x_min), float(x_max))] * n_target + [(float(y_min), float(y_max))] * n_target
    n_starts = 2 if n_target <= 55 else 1
    maxiter = 52 if n_target <= 55 else 24
    phases = (
        (2.5, 1.0003, 0.0, maxiter),
        (35.0, 1.0008, 0.0, maxiter),
        (240.0, 1.0010, 0.0, max(14, maxiter // 2)),
    )

    for sx, sy in starts[:n_starts]:
        cur = np.asarray(jnp.concatenate([sx, sy]), dtype=np.float64)
        for weight, spacing_margin, boundary_margin, iters in phases:
            res = minimize(
                lambda z, w=weight, sm=spacing_margin, bm=boundary_margin: scipy_fun(z, w, sm, bm),
                cur,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": iters,
                    "ftol": 1e-8,
                    "gtol": 1e-4,
                    "maxls": 18,
                    "disp": False,
                },
            )
            cur = np.asarray(res.x, dtype=np.float64)

        rx = jnp.asarray(cur[:n_target])
        ry = jnp.asarray(cur[n_target:])
        robj = neg_aep(jnp.asarray(cur))
        if _feasible(rx, ry, boundary, min_spacing) & (robj < best_obj):
            best_x, best_y, best_obj = rx, ry, robj

    return jnp.asarray(best_x), jnp.asarray(best_y)
