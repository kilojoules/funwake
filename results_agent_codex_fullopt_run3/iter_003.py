"""SLSQP with exact convex constraints from deterministic lattice starts.

HYPOTHESIS: The custom Adam run found a good basin but still used soft
constraints. SLSQP with exact boundary half-planes and pairwise squared
spacing Jacobians can polish feasible lattice starts into a higher-AEP basin.
AXIS: scipy_slsqp with explicit JAX objective gradients and constraint Jacobians.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def _edge_clearance(x, y, boundary):
    n_verts = boundary.shape[0]

    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex = x2 - x1
        ey = y2 - y1
        el = jnp.sqrt(ex * ex + ey * ey) + 1e-12
        return (x - x1) * (-ey / el) + (y - y1) * (ex / el)

    return jax.vmap(edge_dist)(jnp.arange(n_verts))


def _inside(x, y, boundary, margin):
    return jnp.min(_edge_clearance(x, y, boundary), axis=0) >= margin


def _min_pair_distance(x, y):
    n = x.shape[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n) * 1e18)
    return jnp.min(d)


def _feasible(x, y, boundary, min_spacing):
    return jnp.logical_and(
        jnp.all(_inside(x, y, boundary, -1e-6)),
        _min_pair_distance(x, y) >= min_spacing * 0.999,
    )


def _farthest_subset(cand_x, cand_y, n_target, center_x, center_y, mode):
    radial = (cand_x - center_x) ** 2 + (cand_y - center_y) ** 2
    first = jnp.where(mode == 0, jnp.argmax(radial), jnp.argmin(cand_y))

    out_x = jnp.zeros(n_target, dtype=cand_x.dtype)
    out_y = jnp.zeros(n_target, dtype=cand_y.dtype)
    out_x = out_x.at[0].set(cand_x[first])
    out_y = out_y.at[0].set(cand_y[first])
    best_d2 = (cand_x - out_x[0]) ** 2 + (cand_y - out_y[0]) ** 2

    for i in range(1, n_target):
        nxt = jnp.argmax(best_d2)
        out_x = out_x.at[i].set(cand_x[nxt])
        out_y = out_y.at[i].set(cand_y[nxt])
        d2 = (cand_x - out_x[i]) ** 2 + (cand_y - out_y[i]) ** 2
        best_d2 = jnp.minimum(best_d2, d2)

    return out_x, out_y


def _lattice_start(boundary, min_spacing, n_target, angle, phase_x, phase_y, mode):
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    center = jnp.mean(boundary, axis=0)
    span_x = x_max - x_min
    span_y = y_max - y_min
    diag = jnp.sqrt(span_x * span_x + span_y * span_y)

    step = min_spacing * 1.035
    row_step = step * 0.8660254037844386
    n_side = int(np.ceil(float(diag / row_step))) + 7
    coords = jnp.arange(n_side) - n_side // 2
    ii, jj = jnp.meshgrid(coords, coords)
    raw_x = (ii + 0.5 * (jj % 2) + phase_x) * step
    raw_y = (jj + phase_y) * row_step
    ca = jnp.cos(angle)
    sa = jnp.sin(angle)
    cand_x = center[0] + raw_x.ravel() * ca - raw_y.ravel() * sa
    cand_y = center[1] + raw_x.ravel() * sa + raw_y.ravel() * ca

    mask = _inside(cand_x, cand_y, boundary, min_spacing * 0.06)
    cand_x = cand_x[mask]
    cand_y = cand_y[mask]

    if len(cand_x) >= n_target:
        if mode < 2:
            return _farthest_subset(cand_x, cand_y, n_target, center[0], center[1], mode)

        order = jnp.lexsort((cand_x, cand_y))
        idx = jnp.round(jnp.linspace(0, len(cand_x) - 1, n_target)).astype(int)
        return cand_x[order][idx], cand_y[order][idx]

    # Dense fallback for very tight polygons; SLSQP will repair small violations.
    step = min_spacing * 0.72
    row_step = step * 0.8660254037844386
    n_side = int(np.ceil(float(diag / row_step))) + 9
    coords = jnp.arange(n_side) - n_side // 2
    ii, jj = jnp.meshgrid(coords, coords)
    raw_x = (ii + 0.5 * (jj % 2) + phase_x) * step
    raw_y = (jj + phase_y) * row_step
    cand_x = center[0] + raw_x.ravel() * ca - raw_y.ravel() * sa
    cand_y = center[1] + raw_x.ravel() * sa + raw_y.ravel() * ca
    mask = _inside(cand_x, cand_y, boundary, min_spacing * 0.03)
    cand_x = cand_x[mask]
    cand_y = cand_y[mask]
    if len(cand_x) >= n_target:
        return _farthest_subset(cand_x, cand_y, n_target, center[0], center[1], mode)

    idx = jnp.arange(n_target, dtype=jnp.float64)
    fx = (0.6180339887498949 * (idx + phase_x + 0.5)) % 1.0
    fy = (0.7548776662466927 * (idx + phase_y + 0.5)) % 1.0
    return x_min + fx * span_x, y_min + fy * span_y


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    @jax.jit
    def objective(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    grad_objective = jax.jit(jax.grad(objective))

    @jax.jit
    def boundary_constraints(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        return _edge_clearance(x, y, boundary).T.ravel()

    boundary_jac = jax.jit(jax.jacobian(boundary_constraints))

    @jax.jit
    def spacing_constraints(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        iu, ju = jnp.triu_indices(n_target, k=1)
        return d2[iu, ju] - min_spacing * min_spacing

    spacing_jac = jax.jit(jax.jacobian(spacing_constraints))

    def scipy_obj(coords):
        return float(objective(jnp.asarray(coords)))

    def scipy_grad(coords):
        return np.asarray(grad_objective(jnp.asarray(coords)), dtype=np.float64)

    def scipy_boundary(coords):
        return np.asarray(boundary_constraints(jnp.asarray(coords)), dtype=np.float64)

    def scipy_boundary_jac(coords):
        return np.asarray(boundary_jac(jnp.asarray(coords)), dtype=np.float64)

    def scipy_spacing(coords):
        return np.asarray(spacing_constraints(jnp.asarray(coords)), dtype=np.float64)

    def scipy_spacing_jac(coords):
        return np.asarray(spacing_jac(jnp.asarray(coords)), dtype=np.float64)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    wd_rad = jnp.deg2rad(wd)
    energy = weights * ws**3
    dom = jnp.arctan2(jnp.sum(jnp.sin(wd_rad) * energy), jnp.sum(jnp.cos(wd_rad) * energy))

    candidates = [
        _lattice_start(boundary, min_spacing, n_target, 0.0, 0.0, 0.0, 2),
        _lattice_start(boundary, min_spacing, n_target, dom, 0.15, 0.25, 0),
        _lattice_start(boundary, min_spacing, n_target, dom + jnp.pi / 6.0, 0.45, 0.1, 1),
        _lattice_start(boundary, min_spacing, n_target, dom + jnp.pi / 3.0, 0.25, 0.45, 2),
    ]

    scored = []
    for sx, sy in candidates:
        coords = jnp.concatenate([sx, sy])
        score = jnp.where(_feasible(sx, sy, boundary, min_spacing), objective(coords), jnp.inf)
        scored.append((score, sx, sy))
    scored = sorted(scored, key=lambda item: float(item[0]))

    n_starts = 3 if n_target <= 55 else 1
    maxiter = 120 if n_target <= 55 else 45
    ftol = 5e-7 if n_target <= 55 else 2e-6

    best_x = scored[0][1]
    best_y = scored[0][2]
    best_obj = scored[0][0]

    cons = (
        {"type": "ineq", "fun": scipy_boundary, "jac": scipy_boundary_jac},
        {"type": "ineq", "fun": scipy_spacing, "jac": scipy_spacing_jac},
    )
    bounds = [(float(x_min), float(x_max))] * n_target + [(float(y_min), float(y_max))] * n_target

    for _, sx, sy in scored[:n_starts]:
        init = np.asarray(jnp.concatenate([sx, sy]), dtype=np.float64)
        res = minimize(
            scipy_obj,
            init,
            jac=scipy_grad,
            method="SLSQP",
            constraints=cons,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": ftol, "disp": False},
        )
        rx = jnp.asarray(res.x[:n_target])
        ry = jnp.asarray(res.x[n_target:])
        robj = objective(jnp.asarray(res.x))
        if _feasible(rx, ry, boundary, min_spacing) & (robj < best_obj):
            best_obj = robj
            best_x = rx
            best_y = ry

    return jnp.asarray(best_x), jnp.asarray(best_y)
