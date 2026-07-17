"""Bayesian lattice start selection followed by exact SLSQP polishing.

HYPOTHESIS: The prior Adam run's strongest component was its parametric
wind-aligned lattice search, while its soft penalties left room at the final
constraint boundary. Replacing the Adam stages with SLSQP should keep the good
basin but enforce exact boundary and spacing constraints directly.
AXIS: scipy_slsqp polishing of BO-selected wind-aligned hex lattice starts.
LESSON: Feasible and fast at 5568.09 GWh, improving the baseline but not the
current custom-Adam best; SLSQP should be kept as a polish/init tool, not the
main search direction for this farm.
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
    return jnp.min(jnp.sqrt(dx * dx + dy * dy + jnp.eye(n) * 1e18))


def _feasible(x, y, boundary, min_spacing):
    return jnp.logical_and(
        jnp.all(jnp.min(_edge_clearance(x, y, boundary), axis=0) >= -1e-6),
        _min_distance(x, y) >= min_spacing * 0.999,
    )


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

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    center = jnp.mean(boundary, axis=0)
    wd_rad = jnp.deg2rad(wd)
    energy = weights * ws**3
    dominant = jnp.arctan2(jnp.sum(jnp.sin(wd_rad) * energy), jnp.sum(jnp.cos(wd_rad) * energy))

    def generate_grid(params):
        sx, sy, theta_off, ox_raw, oy_raw, shear, aspect = params
        theta = dominant + theta_off
        row_step = sy * aspect * min_spacing * 0.8660254037844386
        n_side = int(np.sqrt(n_target)) + 17
        ii, jj = jnp.meshgrid(
            jnp.arange(n_side) - n_side // 2,
            jnp.arange(n_side) - n_side // 2,
        )
        ix = ii.ravel()
        iy = jj.ravel()
        hx = (ix + 0.5 * (iy % 2)) * sx * min_spacing
        hy = iy * row_step
        hx = hx + shear * hy
        ox = x_min + ox_raw * (x_max - x_min)
        oy = y_min + oy_raw * (y_max - y_min)
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        clearance = jnp.min(_edge_clearance(rx, ry, boundary), axis=0)
        inside_score = jnp.where(clearance > min_spacing * 0.02, clearance, -1e12)
        idx = jnp.argsort(inside_score)[-n_target:]
        return rx[idx], ry[idx]

    def grid_score(raw):
        gx, gy = generate_grid(scale_params(raw))
        coords = jnp.concatenate([gx, gy])
        penalty = jnp.where(_feasible(gx, gy, boundary, min_spacing), 0.0, 1e6)
        return -objective(coords) - penalty

    low = jnp.array([1.04, 1.18, -jnp.pi / 3.0, 0.0, 0.0, -0.14, 0.95])
    high = jnp.array([3.4, 3.4, jnp.pi / 3.0, 1.0, 1.0, 0.14, 1.18])

    def scale_params(raw):
        return low + raw * (high - low)

    seed_raw = jnp.array(
        [
            [0.04, 0.04, 0.50, 0.50, 0.50, 0.50, 0.20],
            [0.02, 0.08, 0.30, 0.42, 0.45, 0.45, 0.20],
            [0.08, 0.02, 0.70, 0.55, 0.52, 0.55, 0.25],
            [0.18, 0.12, 0.15, 0.46, 0.58, 0.50, 0.30],
        ],
        dtype=jnp.float64,
    )
    key = jax.random.PRNGKey(314)
    key, sub = jax.random.split(key)
    x_raw = jnp.vstack([seed_raw, jax.random.uniform(sub, (10, 7))])
    y_raw = jnp.array([grid_score(p) for p in x_raw])

    @jax.jit
    def gp_predict(x_test, x_train, y_train, length=0.32):
        def kernel(a, b):
            d = jnp.sqrt(jnp.sum((a - b) ** 2) + 1e-9)
            z = jnp.sqrt(5.0) * d / length
            return (1.0 + z + z * z / 3.0) * jnp.exp(-z)

        k = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(x_train))(x_train)
        k = k + jnp.eye(x_train.shape[0]) * 1e-5
        ks = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(x_train))(x_test)
        chol = jnp.linalg.cholesky(k)
        alpha = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, y_train))
        mu = ks @ alpha
        v = jnp.linalg.solve(chol, ks.T)
        var = 1.0 - jnp.sum(v * v, axis=0)
        return mu, jnp.sqrt(jnp.maximum(var, 1e-9))

    bo_steps = 14 if n_target <= 55 else 5
    pool_size = 700 if n_target <= 55 else 220
    for _ in range(bo_steps):
        key, sub = jax.random.split(key)
        cand = jax.random.uniform(sub, (pool_size, 7))
        mu, sig = gp_predict(cand, x_raw, y_raw)
        incumbent = jnp.max(y_raw)
        z = (mu - incumbent) / sig
        cdf = 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))
        pdf = jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)
        ei = (mu - incumbent) * cdf + sig * pdf
        nxt = cand[jnp.argmax(ei)]
        x_raw = jnp.vstack([x_raw, nxt])
        y_raw = jnp.append(y_raw, grid_score(nxt))

    order = jnp.argsort(y_raw)[::-1]
    starts = [generate_grid(scale_params(x_raw[i])) for i in order[:4]]

    best_x, best_y = starts[0]
    best_obj = objective(jnp.concatenate([best_x, best_y]))
    if not _feasible(best_x, best_y, boundary, min_spacing):
        best_obj = jnp.inf
        for sx, sy in starts:
            sobj = objective(jnp.concatenate([sx, sy]))
            if _feasible(sx, sy, boundary, min_spacing) & (sobj < best_obj):
                best_x = sx
                best_y = sy
                best_obj = sobj

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

    constraints = (
        {"type": "ineq", "fun": scipy_boundary, "jac": scipy_boundary_jac},
        {"type": "ineq", "fun": scipy_spacing, "jac": scipy_spacing_jac},
    )
    bounds = [(float(x_min), float(x_max))] * n_target + [(float(y_min), float(y_max))] * n_target
    n_starts = 2 if n_target <= 55 else 1
    maxiter = 140 if n_target <= 55 else 35

    for sx, sy in starts[:n_starts]:
        init = np.asarray(jnp.concatenate([sx, sy]), dtype=np.float64)
        res = minimize(
            scipy_obj,
            init,
            jac=scipy_grad,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 4e-7, "disp": False},
        )
        rx = jnp.asarray(res.x[:n_target])
        ry = jnp.asarray(res.x[n_target:])
        robj = objective(jnp.asarray(res.x))
        if _feasible(rx, ry, boundary, min_spacing) & (robj < best_obj):
            best_x = rx
            best_y = ry
            best_obj = robj

    return jnp.asarray(best_x), jnp.asarray(best_y)
