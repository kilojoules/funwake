"""CMA-ES search over compact lattice basin parameters.

HYPOTHESIS: The best known layouts are sensitive to the low-dimensional
lattice basin. CMA-ES can adapt covariance across scale, phase, orientation,
and shear more efficiently than independent random/DE mutations, then a short
coordinate penalty polish can improve wakes without changing strategy family.
AXIS: cmaes lattice-parameter search with brief L-BFGS-B penalty polish.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


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


def _feasible(x, y, boundary, min_spacing):
    n = x.shape[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n) * 1e18)
    return (
        (jnp.min(_edge_clearance(x, y, boundary)) >= -1e-5)
        & (jnp.min(dist) >= min_spacing * 0.9995)
    )


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

    low = jnp.array([1.02, 1.03, -jnp.pi / 3.0, 0.02, 0.02, -0.22, 0.88])
    high = jnp.array([3.95, 3.95, jnp.pi / 3.0, 0.98, 0.98, 0.22, 1.26])

    def scale(raw):
        return low + jnp.asarray(raw) * (high - low)

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
        score = jnp.where(clearance > min_spacing * 0.018, edge_pull + 0.025 * spread, -1e12)
        idx = jnp.argsort(score)[-n_target:]
        return rx[idx], ry[idx]

    def score_raw(raw):
        raw = jnp.clip(jnp.asarray(raw, dtype=jnp.float64), 0.0, 1.0)
        x, y = lattice(raw)
        coords = jnp.concatenate([x, y])
        penalty = jnp.where(_feasible(x, y, boundary, min_spacing), 0.0, 1e6)
        return neg_aep(coords) + penalty

    seed_raw = np.array(
        [
            [0.02, 0.03, 0.50, 0.50, 0.50, 0.50, 0.20],
            [0.04, 0.06, 0.32, 0.43, 0.46, 0.43, 0.24],
            [0.05, 0.02, 0.69, 0.56, 0.52, 0.57, 0.23],
            [0.14, 0.10, 0.18, 0.45, 0.58, 0.50, 0.30],
            [0.10, 0.16, 0.84, 0.53, 0.42, 0.58, 0.36],
            [0.20, 0.05, 0.60, 0.48, 0.50, 0.62, 0.18],
            [0.06, 0.20, 0.40, 0.58, 0.48, 0.38, 0.35],
        ],
        dtype=np.float64,
    )

    scored = [(float(score_raw(row)), row.copy()) for row in seed_raw]
    centers = [scored[0][1]]
    if n_target <= 55:
        centers.append(scored[3][1])

    generations = 14 if n_target <= 55 else 5
    popsize = 18 if n_target <= 55 else 10
    mu = popsize // 2
    weights_cma = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights_cma = weights_cma / np.sum(weights_cma)
    c_cov = 0.28
    rng = np.random.default_rng(808)
    for i, center0 in enumerate(centers):
        mean = center0.copy()
        sigma = 0.18
        cov = np.eye(7)
        best_seen = float(score_raw(mean))
        for _ in range(generations):
            chol = np.linalg.cholesky(cov + np.eye(7) * 1e-8)
            z = rng.normal(size=(popsize, 7))
            xs = np.clip(mean[None, :] + sigma * z @ chol.T, 0.0, 1.0)
            vals = np.array([float(score_raw(x)) for x in xs])
            order = np.argsort(vals)
            elite = xs[order[:mu]]
            old_mean = mean
            mean = np.sum(weights_cma[:, None] * elite, axis=0)
            y = (elite - old_mean) / max(sigma, 1e-9)
            cov_update = np.einsum("i,ij,ik->jk", weights_cma, y, y)
            cov = (1.0 - c_cov) * cov + c_cov * cov_update + np.eye(7) * 1e-6
            if vals[order[0]] < best_seen:
                best_seen = vals[order[0]]
                sigma *= 0.97
            else:
                sigma *= 0.90
            sigma = float(np.clip(sigma, 0.035, 0.24))
            scored.append((float(vals[order[0]]), xs[order[0]].copy()))
        scored.append((float(score_raw(mean)), np.clip(mean, 0.0, 1.0)))

    scored = sorted(scored, key=lambda item: item[0])
    starts = [lattice(jnp.asarray(item[1], dtype=jnp.float64)) for item in scored[:4]]

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
    def penalty_objective(coords, penalty_weight, spacing_margin):
        x = coords[:n_target]
        y = coords[n_target:]
        clearance = _edge_clearance(x, y, boundary)
        b_violation = jnp.maximum(0.0, -clearance)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n_target) * 1e18)
        s_violation = jnp.maximum(0.0, min_spacing * spacing_margin - dist)
        penalty = jnp.sum(b_violation * b_violation) + jnp.sum(upper * s_violation * s_violation)
        return neg_aep(coords) + penalty_weight * penalty

    value_and_grad = jax.jit(jax.value_and_grad(penalty_objective))

    def scipy_penalty(coords, penalty_weight, spacing_margin):
        val, grad = value_and_grad(jnp.asarray(coords), penalty_weight, spacing_margin)
        return float(val), np.asarray(grad, dtype=np.float64)

    bounds = [(float(x_min), float(x_max))] * n_target + [(float(y_min), float(y_max))] * n_target
    n_polish = 2 if n_target <= 55 else 1
    maxiter = 44 if n_target <= 55 else 20
    for sx, sy in starts[:n_polish]:
        cur = np.asarray(jnp.concatenate([sx, sy]), dtype=np.float64)
        for weight, spacing_margin in ((8.0, 1.0002), (120.0, 1.0009), (360.0, 1.0011)):
            res = minimize(
                lambda z, w=weight, sm=spacing_margin: scipy_penalty(z, w, sm),
                cur,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-4, "maxls": 16, "disp": False},
            )
            cur = np.asarray(res.x, dtype=np.float64)
        rx = jnp.asarray(cur[:n_target])
        ry = jnp.asarray(cur[n_target:])
        robj = neg_aep(jnp.asarray(cur))
        if _feasible(rx, ry, boundary, min_spacing) & (robj < best_obj):
            best_x, best_y, best_obj = rx, ry, robj

    return jnp.asarray(best_x), jnp.asarray(best_y)
