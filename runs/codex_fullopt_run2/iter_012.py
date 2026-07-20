"""Trust-constr interior-point polish from feasible rotated lattice starts.

HYPOTHESIS: Exact nonlinear spacing constraints plus linear polygon half-planes
can polish a good feasible lattice into a better active-constraint layout than
penalty-only Adam, without projection bias.
AXIS: scipy_trust_constr with analytic spacing Jacobian and feasible tracking.
LESSON: Pending score.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    start_time = time.time()

    bnd = np.asarray(boundary, dtype=float)
    signed_area = 0.5 * np.sum(
        bnd[:, 0] * np.roll(bnd[:, 1], -1) - np.roll(bnd[:, 0], -1) * bnd[:, 1]
    )
    if signed_area < 0.0:
        bnd = bnd[::-1].copy()
        boundary = boundary[::-1]

    x_min, y_min = np.min(bnd, axis=0)
    x_max, y_max = np.max(bnd, axis=0)
    n_verts = bnd.shape[0]

    def objective_xy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, : len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def objective_vec(z):
        x = z[:n_target]
        y = z[n_target:]
        return objective_xy(x, y) / 1000.0

    value_grad = jax.jit(jax.value_and_grad(objective_vec))

    def edge_clearance_np(px, py):
        vals = []
        for i in range(n_verts):
            x1, y1 = bnd[i]
            x2, y2 = bnd[(i + 1) % n_verts]
            ex = x2 - x1
            ey = y2 - y1
            el = np.hypot(ex, ey) + 1e-12
            vals.append((px - x1) * (-ey / el) + (py - y1) * (ex / el))
        return np.min(np.vstack(vals), axis=0)

    def candidate_cloud(step_x, step_y, margin, angle, stagger):
        cx = float(np.mean(bnd[:, 0]))
        cy = float(np.mean(bnd[:, 1]))
        ca = np.cos(angle)
        sa = np.sin(angle)
        rot = np.array([[ca, -sa], [sa, ca]])
        inv = np.array([[ca, sa], [-sa, ca]])
        rb = (rot @ (bnd - np.array([cx, cy])).T).T
        rx_min, ry_min = np.min(rb, axis=0)
        rx_max, ry_max = np.max(rb, axis=0)

        nx = max(3, int(np.ceil((rx_max - rx_min - 2.0 * margin) / step_x)) + 1)
        ny = max(3, int(np.ceil((ry_max - ry_min - 2.0 * margin) / step_y)) + 1)
        gx, gy = np.meshgrid(
            np.linspace(rx_min + margin, rx_max - margin, nx),
            np.linspace(ry_min + margin, ry_max - margin, ny),
        )
        gx = gx + ((np.arange(ny)[:, None] % 2) * stagger * step_x)
        pts = (inv @ np.stack([gx.ravel(), gy.ravel()], axis=0)).T + np.array([cx, cy])
        clear = edge_clearance_np(pts[:, 0], pts[:, 1])
        keep = clear > max(1.0, margin * 0.03)
        return pts[keep, 0], pts[keep, 1]

    def farthest_init(cand_x, cand_y, mode):
        if len(cand_x) < n_target:
            gx, gy = candidate_cloud(
                min_spacing * 1.04,
                min_spacing * 0.91,
                min_spacing * 0.18,
                0.0,
                0.5,
            )
            cand_x = np.concatenate([cand_x, gx])
            cand_y = np.concatenate([cand_y, gy])

        if len(cand_x) == 0:
            return (
                jnp.linspace(x_min + min_spacing, x_max - min_spacing, n_target),
                jnp.linspace(y_min + min_spacing, y_max - min_spacing, n_target),
            )

        cx = np.mean(bnd[:, 0])
        cy = np.mean(bnd[:, 1])
        wd_np = np.asarray(wd, dtype=float)
        ws_np = np.asarray(ws, dtype=float)
        wt_np = np.asarray(weights, dtype=float)
        energy = wt_np * ws_np**3
        theta = np.deg2rad(wd_np[int(np.argmax(energy))])
        down_x = np.sin(theta)
        down_y = np.cos(theta)
        proj = (cand_x - cx) * down_x + (cand_y - cy) * down_y
        radial = (cand_x - cx) ** 2 + (cand_y - cy) ** 2
        first_idx = int(np.argmin(proj) if mode == 0 else np.argmax(radial))

        chosen_x = [cand_x[first_idx]]
        chosen_y = [cand_y[first_idx]]
        best_dist2 = (cand_x - chosen_x[0]) ** 2 + (cand_y - chosen_y[0]) ** 2
        used = np.zeros(len(cand_x), dtype=bool)
        used[first_idx] = True

        for _ in range(1, n_target):
            score = np.where(
                (~used) & (best_dist2 >= (min_spacing * 1.002) ** 2),
                best_dist2,
                -1.0,
            )
            if np.max(score) <= 0.0:
                score = np.where(~used, best_dist2, -1.0)
            idx = int(np.argmax(score))
            used[idx] = True
            chosen_x.append(cand_x[idx])
            chosen_y.append(cand_y[idx])
            dist2 = (cand_x - cand_x[idx]) ** 2 + (cand_y - cand_y[idx]) ** 2
            best_dist2 = np.minimum(best_dist2, dist2)

        return jnp.asarray(chosen_x), jnp.asarray(chosen_y)

    def min_distance_np(z):
        x = z[:n_target]
        y = z[n_target:]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy + np.eye(n_target) * 1e20)
        return float(np.min(dist))

    def feasible_vec(z):
        x = jnp.asarray(z[:n_target])
        y = jnp.asarray(z[n_target:])
        return (
            float(boundary_penalty(x, y, boundary)) < 1e-3
            and float(spacing_penalty(x, y, min_spacing)) < 1e-3
            and min_distance_np(np.asarray(z)) >= float(min_spacing) * 0.99
        )

    def aep_vec(z):
        x = jnp.asarray(z[:n_target])
        y = jnp.asarray(z[n_target:])
        return float(-objective_xy(x, y))

    wd_np = np.asarray(wd, dtype=float)
    ws_np = np.asarray(ws, dtype=float)
    wt_np = np.asarray(weights, dtype=float)
    theta = np.deg2rad(wd_np[int(np.argmax(wt_np * ws_np**3))])
    angles = (theta + np.pi / 2.0, theta, 0.0)
    clouds = (
        candidate_cloud(
            min_spacing * 1.02,
            min_spacing * 0.89,
            min_spacing * 0.20,
            angles[0],
            0.5,
        ),
        candidate_cloud(
            min_spacing * 1.05,
            min_spacing * 0.92,
            min_spacing * 0.12,
            angles[1],
            0.35,
        ),
        candidate_cloud(
            min_spacing * 1.01,
            min_spacing * 0.88,
            min_spacing * 0.18,
            angles[2],
            0.5,
        ),
    )
    starts = (
        farthest_init(clouds[0][0], clouds[0][1], 0),
        farthest_init(clouds[1][0], clouds[1][1], 1),
        farthest_init(clouds[2][0], clouds[2][1], 0),
    )

    best_x, best_y = starts[0]
    best_z = np.concatenate([np.asarray(best_x), np.asarray(best_y)])
    best_aep = -np.inf
    for sx, sy in starts:
        z0 = np.concatenate([np.asarray(sx), np.asarray(sy)])
        if feasible_vec(z0):
            val = aep_vec(z0)
            if val > best_aep:
                best_aep = val
                best_z = z0.copy()

    if n_target > 60:
        return jnp.asarray(best_z[:n_target]), jnp.asarray(best_z[n_target:])

    rows = []
    for i in range(n_verts):
        x1, y1 = bnd[i]
        x2, y2 = bnd[(i + 1) % n_verts]
        ex = x2 - x1
        ey = y2 - y1
        el = np.hypot(ex, ey) + 1e-12
        nx = -ey / el
        ny = ex / el
        for t_idx in range(n_target):
            row = np.zeros(2 * n_target)
            row[t_idx] = nx / float(min_spacing)
            row[n_target + t_idx] = ny / float(min_spacing)
            rows.append((row, (nx * x1 + ny * y1) / float(min_spacing)))
    a_mat = np.vstack([r[0] for r in rows])
    b_vec = np.asarray([r[1] for r in rows])
    boundary_constraint = LinearConstraint(a_mat, b_vec, np.full_like(b_vec, np.inf))

    pair_i, pair_j = np.triu_indices(n_target, k=1)
    spacing_sq = float(min_spacing) ** 2

    def spacing_fun(z):
        x = z[:n_target]
        y = z[n_target:]
        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        return (dx * dx + dy * dy) / spacing_sq - 1.0

    def spacing_jac(z):
        x = z[:n_target]
        y = z[n_target:]
        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        jac = np.zeros((len(pair_i), 2 * n_target))
        vals_x = 2.0 * dx / spacing_sq
        vals_y = 2.0 * dy / spacing_sq
        rows_idx = np.arange(len(pair_i))
        jac[rows_idx, pair_i] = vals_x
        jac[rows_idx, pair_j] = -vals_x
        jac[rows_idx, n_target + pair_i] = vals_y
        jac[rows_idx, n_target + pair_j] = -vals_y
        return jac

    spacing_constraint = NonlinearConstraint(
        spacing_fun, np.zeros(len(pair_i)), np.full(len(pair_i), np.inf), jac=spacing_jac
    )
    bounds = Bounds(
        np.concatenate([
            np.full(n_target, x_min),
            np.full(n_target, y_min),
        ]),
        np.concatenate([
            np.full(n_target, x_max),
            np.full(n_target, y_max),
        ]),
    )

    def scipy_fun(z):
        value, grad = value_grad(jnp.asarray(z))
        value_f = float(value)
        grad_np = np.asarray(grad, dtype=float)
        if not np.isfinite(value_f) or not np.all(np.isfinite(grad_np)):
            return 1e20, np.zeros_like(z)
        return value_f, grad_np

    max_starts = 1 if n_target >= 45 else 2
    for sx, sy in starts[:max_starts]:
        if time.time() - start_time > 105.0:
            break
        z0 = np.concatenate([np.asarray(sx), np.asarray(sy)])
        if not feasible_vec(z0):
            z0 = best_z.copy()

        local = {"aep": best_aep, "z": best_z.copy()}

        def callback(z, state=None):
            if feasible_vec(z):
                val = aep_vec(z)
                if val > local["aep"]:
                    local["aep"] = val
                    local["z"] = np.asarray(z).copy()
            return time.time() - start_time > 155.0

        try:
            res = minimize(
                scipy_fun,
                z0,
                method="trust-constr",
                jac=True,
                bounds=bounds,
                constraints=(boundary_constraint, spacing_constraint),
                callback=callback,
                options={
                    "maxiter": 45 if n_target >= 45 else 60,
                    "gtol": 1e-5,
                    "xtol": 1e-6,
                    "barrier_tol": 1e-4,
                    "initial_tr_radius": float(min_spacing) * 0.6,
                    "verbose": 0,
                },
            )
            if feasible_vec(res.x):
                val = aep_vec(res.x)
                if val > local["aep"]:
                    local["aep"] = val
                    local["z"] = np.asarray(res.x).copy()
        except Exception:
            pass

        if local["aep"] > best_aep:
            best_aep = local["aep"]
            best_z = local["z"].copy()

    return jnp.asarray(best_z[:n_target]), jnp.asarray(best_z[n_target:])
