"""Sparse trust-constr with active-pair post-search.

HYPOTHESIS: The expensive BO/AL-NAdam family has plateaued near 5584 GWh and
is too close to the timeout in this environment. A sparse trust-constr solve
from Latin-hypercube lattice starts is much faster, and a small active-pair
post-search can target tight spacing contacts without returning to the last
wake-weighted relocation strategy.
AXIS: scipy_trust_constr with sparse exact constraints and init_latin_hypercube
lattice screening, followed by active nearest-neighbor pair shuffles.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
from scipy.sparse import coo_matrix, lil_matrix
from scipy.stats import qmc


def _edge_clearance(x, y, boundary):
    n_verts = boundary.shape[0]

    def edge(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex = x2 - x1
        ey = y2 - y1
        el = jnp.sqrt(ex * ex + ey * ey) + 1e-12
        return (x - x1) * (-ey / el) + (y - y1) * (ex / el)

    return jax.vmap(edge)(jnp.arange(n_verts))


def _min_distance(x, y):
    n = x.shape[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n) * 1e18)
    return jnp.min(dist)


def _feasible(x, y, boundary, min_spacing):
    return (
        (jnp.min(_edge_clearance(x, y, boundary)) >= -1e-5)
        & (_min_distance(x, y) >= min_spacing * 0.9995)
    )


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    @jax.jit
    def neg_aep(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    grad_neg_aep = jax.jit(jax.grad(neg_aep))

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

    low = jnp.array([1.03, 1.04, -jnp.pi / 3.0, 0.03, 0.03, -0.20, 0.90])
    high = jnp.array([3.9, 3.9, jnp.pi / 3.0, 0.97, 0.97, 0.20, 1.25])

    def scale(raw):
        return low + jnp.asarray(raw, dtype=jnp.float64) * (high - low)

    def lattice(raw):
        sx, sy, theta_off, ox_raw, oy_raw, shear, aspect = scale(raw)
        theta = dominant + theta_off
        row_step = sy * aspect * min_spacing * 0.8660254037844386
        n_side = int(np.ceil(float(diag / (min_spacing * 0.74)))) + 15
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
        clearance = jnp.min(_edge_clearance(rx, ry, boundary), axis=0)
        edge_pull = clearance / diag
        spread = ((rx - center[0]) ** 2 + (ry - center[1]) ** 2) / (diag * diag)
        score = jnp.where(clearance > min_spacing * 0.02, edge_pull + 0.02 * spread, -1e12)
        idx = jnp.argsort(score)[-n_target:]
        return rx[idx], ry[idx]

    def start_score(raw):
        sx, sy = lattice(raw)
        coords = jnp.concatenate([sx, sy])
        penalty = jnp.where(_feasible(sx, sy, boundary, min_spacing), 0.0, 1e6)
        return neg_aep(coords) + penalty

    seeds = np.array(
        [
            [0.02, 0.03, 0.50, 0.50, 0.50, 0.50, 0.20],
            [0.04, 0.06, 0.32, 0.43, 0.46, 0.43, 0.24],
            [0.05, 0.02, 0.69, 0.56, 0.52, 0.57, 0.23],
            [0.14, 0.10, 0.18, 0.45, 0.58, 0.50, 0.30],
            [0.10, 0.16, 0.84, 0.53, 0.42, 0.58, 0.36],
        ],
        dtype=np.float64,
    )
    sampler = qmc.LatinHypercube(d=7, seed=909)
    n_lhs = 18 if n_target <= 55 else 6
    raw = np.vstack([seeds, sampler.random(n_lhs)])
    scored = sorted(((float(start_score(row)), row) for row in raw), key=lambda item: item[0])
    starts = [lattice(row) for _, row in scored[:3]]

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

    bnd = np.asarray(boundary, dtype=np.float64)
    n_edges = bnd.shape[0]
    linear = lil_matrix((n_target * n_edges, 2 * n_target), dtype=np.float64)
    lb = np.zeros(n_target * n_edges, dtype=np.float64)
    for e in range(n_edges):
        x1, y1 = bnd[e]
        x2, y2 = bnd[(e + 1) % n_edges]
        ex = x2 - x1
        ey = y2 - y1
        el = np.sqrt(ex * ex + ey * ey) + 1e-12
        nx = -ey / el
        ny = ex / el
        c = -(x1 * nx + y1 * ny)
        for i in range(n_target):
            row = i * n_edges + e
            linear[row, i] = nx
            linear[row, n_target + i] = ny
            lb[row] = -c
    boundary_constraint = LinearConstraint(linear.tocsr(), lb, np.inf)

    pair_i, pair_j = np.triu_indices(n_target, k=1)
    n_pairs = pair_i.size
    min_spacing2 = float(min_spacing * min_spacing)

    def spacing_fun(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        return dx * dx + dy * dy - min_spacing2

    def spacing_jac(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        rows = np.repeat(np.arange(n_pairs), 4)
        cols = np.empty(n_pairs * 4, dtype=np.int64)
        data = np.empty(n_pairs * 4, dtype=np.float64)
        cols[0::4] = pair_i
        cols[1::4] = pair_j
        cols[2::4] = n_target + pair_i
        cols[3::4] = n_target + pair_j
        data[0::4] = 2.0 * dx
        data[1::4] = -2.0 * dx
        data[2::4] = 2.0 * dy
        data[3::4] = -2.0 * dy
        return coo_matrix((data, (rows, cols)), shape=(n_pairs, 2 * n_target)).tocsr()

    spacing_constraint = NonlinearConstraint(
        spacing_fun,
        np.zeros(n_pairs, dtype=np.float64),
        np.inf,
        jac=spacing_jac,
    )

    def scipy_obj(coords):
        return float(neg_aep(jnp.asarray(coords)))

    def scipy_grad(coords):
        return np.asarray(grad_neg_aep(jnp.asarray(coords)), dtype=np.float64)

    bounds = Bounds(
        np.r_[np.full(n_target, float(x_min)), np.full(n_target, float(y_min))],
        np.r_[np.full(n_target, float(x_max)), np.full(n_target, float(y_max))],
    )

    n_polish = 2 if n_target <= 55 else 1
    maxiter = 58 if n_target <= 55 else 24
    for sx, sy in starts[:n_polish]:
        init = np.asarray(jnp.concatenate([sx, sy]), dtype=np.float64)
        res = minimize(
            scipy_obj,
            init,
            method="trust-constr",
            jac=scipy_grad,
            bounds=bounds,
            constraints=(boundary_constraint, spacing_constraint),
            options={
                "maxiter": maxiter,
                "gtol": 2e-4,
                "xtol": 1e-6,
                "barrier_tol": 2e-4,
                "initial_tr_radius": float(min_spacing) * 0.9,
                "initial_constr_penalty": 0.8,
                "initial_barrier_parameter": 0.08,
                "initial_barrier_tolerance": 0.08,
                "sparse_jacobian": True,
                "verbose": 0,
            },
        )
        rx = jnp.asarray(res.x[:n_target])
        ry = jnp.asarray(res.x[n_target:])
        robj = neg_aep(jnp.asarray(res.x))
        if _feasible(rx, ry, boundary, min_spacing) & (robj < best_obj):
            best_x, best_y, best_obj = rx, ry, robj

    def repair(x, y, n_steps=16):
        for _ in range(n_steps):
            for e in range(n_edges):
                x1, y1 = boundary[e]
                x2, y2 = boundary[(e + 1) % n_edges]
                ex = x2 - x1
                ey = y2 - y1
                el = jnp.sqrt(ex * ex + ey * ey) + 1e-12
                nx = -ey / el
                ny = ex / el
                clearance = (x - x1) * nx + (y - y1) * ny
                push = jnp.maximum(0.0, 0.04 - clearance)
                x = x + push * nx
                y = y + push * ny
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n_target) * 1e18)
            force = jnp.maximum(0.0, min_spacing * 1.001 - dist)
            x = x + jnp.sum(force * dx / dist, axis=1) * 0.42
            y = y + jnp.sum(force * dy / dist, axis=1) * 0.42
        return x, y

    def active_pair_finish(seed_x, seed_y):
        ux = jnp.cos(dominant)
        uy = jnp.sin(dominant)
        vx = -uy
        vy = ux
        dx = seed_x[pair_i] - seed_x[pair_j]
        dy = seed_y[pair_i] - seed_y[pair_j]
        dist = jnp.sqrt(dx * dx + dy * dy + 1e-6)
        n_active = min(6, n_pairs)
        active = jnp.argsort(jnp.abs(dist - min_spacing))[:n_active]
        active_i = jnp.asarray(pair_i)[active]
        active_j = jnp.asarray(pair_j)[active]
        moves = min_spacing * jnp.array(
            [
                [0.10, 0.00, 0.10, 0.00],
                [-0.10, 0.00, -0.10, 0.00],
                [0.00, 0.10, 0.00, 0.10],
                [0.00, -0.10, 0.00, -0.10],
                [0.08, 0.00, -0.08, 0.00],
                [0.00, 0.08, 0.00, -0.08],
            ]
        )
        idx_i = jnp.repeat(active_i, moves.shape[0])
        idx_j = jnp.repeat(active_j, moves.shape[0])
        shifts = jnp.tile(moves, (n_active, 1))

        def perturb(i, j, shift):
            du_i, dv_i, du_j, dv_j = shift
            mask_i = (jnp.arange(n_target) == i).astype(seed_x.dtype)
            mask_j = (jnp.arange(n_target) == j).astype(seed_x.dtype)
            x = (
                seed_x
                + mask_i * (du_i * ux + dv_i * vx)
                + mask_j * (du_j * ux + dv_j * vx)
            )
            y = (
                seed_y
                + mask_i * (du_i * uy + dv_i * vy)
                + mask_j * (du_j * uy + dv_j * vy)
            )
            return repair(x, y, 10)

        px, py = jax.vmap(perturb)(idx_i, idx_j, shifts)
        cand_x = jnp.concatenate([seed_x[None, :], px], axis=0)
        cand_y = jnp.concatenate([seed_y[None, :], py], axis=0)
        vals = jax.vmap(lambda x, y: neg_aep(jnp.concatenate([x, y])))(cand_x, cand_y)
        feas = jax.vmap(lambda x, y: _feasible(x, y, boundary, min_spacing))(cand_x, cand_y)
        idx = jnp.argmin(jnp.where(feas, vals, vals + 1e9))
        return cand_x[idx], cand_y[idx]

    best_x, best_y = active_pair_finish(jnp.asarray(best_x), jnp.asarray(best_y))
    return jnp.asarray(best_x), jnp.asarray(best_y)
