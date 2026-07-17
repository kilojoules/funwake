"""SLSQP with analytic objective and constraint Jacobians.

HYPOTHESIS: Direct constrained SQP can improve AEP without the penalty-weight
tradeoff that made the custom Adam attempt conservative, as long as the
constraint vector and Jacobian are scaled and initialized from a feasible
wind-aware grid.

AXIS: scipy_slsqp with explicit polygon and spacing inequalities.

LESSON: Pending score.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    n_verts = boundary.shape[0]
    x_min = float(jnp.min(boundary[:, 0]))
    x_max = float(jnp.max(boundary[:, 0]))
    y_min = float(jnp.min(boundary[:, 1]))
    y_max = float(jnp.max(boundary[:, 1]))
    center = jnp.mean(boundary, axis=0)
    ms2 = min_spacing * min_spacing

    def aep_gwh(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def objective_xy(x, y):
        return -aep_gwh(x, y)

    def edge_distances_xy(x, y):
        def one_edge(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex = x2 - x1
            ey = y2 - y1
            el = jnp.sqrt(ex * ex + ey * ey) + 1e-10
            nx = -ey / el
            ny = ex / el
            return (x - x1) * nx + (y - y1) * ny

        return jax.vmap(one_edge)(jnp.arange(n_verts))

    def inside_mask(x, y):
        return jnp.min(edge_distances_xy(x, y), axis=0) > 0.0

    def rotated_grid(angle, spacing_mult):
        ca = jnp.cos(angle)
        sa = jnp.sin(angle)
        rot = jnp.array([[ca, -sa], [sa, ca]])
        inv_rot = jnp.array([[ca, sa], [-sa, ca]])
        rb = (rot @ (boundary - center).T).T
        rx_min, ry_min = jnp.min(rb, axis=0)
        rx_max, ry_max = jnp.max(rb, axis=0)
        sp = min_spacing * spacing_mult
        nx = max(4, int(jnp.ceil((rx_max - rx_min) / sp)) + 2)
        ny = max(4, int(jnp.ceil((ry_max - ry_min) / sp)) + 2)
        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min + 0.35 * sp, rx_max - 0.35 * sp, nx),
            jnp.linspace(ry_min + 0.35 * sp, ry_max - 0.35 * sp, ny),
        )
        pts_r = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
        pts = (inv_rot @ pts_r.T).T + center
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        if len(px) >= n_target:
            idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
            return px[idx], py[idx]

        nx2 = max(4, int(jnp.sqrt(n_target * 2.0)))
        ny2 = max(4, int(jnp.ceil(n_target * 2.0 / nx2)))
        gx2, gy2 = jnp.meshgrid(
            jnp.linspace(x_min + 0.08 * (x_max - x_min), x_max - 0.08 * (x_max - x_min), nx2),
            jnp.linspace(y_min + 0.08 * (y_max - y_min), y_max - 0.08 * (y_max - y_min), ny2),
        )
        pts2 = jnp.stack([gx2.ravel(), gy2.ravel()], axis=1)
        mask2 = inside_mask(pts2[:, 0], pts2[:, 1])
        px2 = pts2[:, 0][mask2]
        py2 = pts2[:, 1][mask2]
        idx2 = jnp.round(jnp.linspace(0, len(px2) - 1, n_target)).astype(int)
        return px2[idx2], py2[idx2]

    def project_boundary(x, y, margin):
        for i in range(n_verts):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex = x2 - x1
            ey = y2 - y1
            el = jnp.sqrt(ex * ex + ey * ey) + 1e-10
            nx = -ey / el
            ny = ex / el
            d = (x - x1) * nx + (y - y1) * ny
            push = jnp.maximum(0.0, margin - d)
            x = x + push * nx
            y = y + push * ny
        return x, y

    def repair(x, y):
        x, y = project_boundary(x, y, 1.0)
        for _ in range(8):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n_target) * 1e12)
            target = min_spacing * 1.002
            viol = jnp.maximum(0.0, target - dist)
            x = x + 0.5 * jnp.sum(viol * dx / (dist + 1e-9), axis=1)
            y = y + 0.5 * jnp.sum(viol * dy / (dist + 1e-9), axis=1)
            x, y = project_boundary(x, y, 1.0)
        return x, y

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    init_x, init_y = rotated_grid(mean_wind + 0.5 * jnp.pi, 1.0)
    init_x, init_y = repair(init_x, init_y)
    z0 = jnp.concatenate([init_x, init_y])

    def obj_z(z):
        return objective_xy(z[:n_target], z[n_target:])

    obj_value_grad = jax.jit(jax.value_and_grad(obj_z))

    def constraints_z(z):
        x = z[:n_target]
        y = z[n_target:]
        bnd = (edge_distances_xy(x, y).T).ravel() / min_spacing
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx * dx + dy * dy
        iu = jnp.triu_indices(n_target, k=1)
        spacing = dist_sq[iu] / ms2 - 0.992 * 0.992
        return jnp.concatenate([bnd, spacing])

    con_value = jax.jit(constraints_z)
    con_jacobian = jax.jit(jax.jacfwd(constraints_z))

    def fun(z_np):
        value, _ = obj_value_grad(jnp.asarray(z_np))
        return float(value)

    def jac(z_np):
        _, grad = obj_value_grad(jnp.asarray(z_np))
        return np.asarray(grad, dtype=float)

    def con_fun(z_np):
        return np.asarray(con_value(jnp.asarray(z_np)), dtype=float)

    def con_jac(z_np):
        return np.asarray(con_jacobian(jnp.asarray(z_np)), dtype=float)

    best_x, best_y = init_x, init_y
    best_obj = objective_xy(best_x, best_y)

    bounds = [(x_min, x_max)] * n_target + [(y_min, y_max)] * n_target
    maxiter = 28 if n_target > 60 else 40
    try:
        res = minimize(
            fun,
            np.asarray(z0, dtype=float),
            method="SLSQP",
            jac=jac,
            bounds=bounds,
            constraints=({"type": "ineq", "fun": con_fun, "jac": con_jac},),
            options={"maxiter": maxiter, "ftol": 1e-4, "disp": False},
        )
        rz = jnp.asarray(res.x)
        sx, sy = repair(rz[:n_target], rz[n_target:])
        feasible_pen = boundary_penalty(sx, sy, boundary) + spacing_penalty(
            sx, sy, min_spacing * 0.99
        )
        obj = objective_xy(sx, sy)
        if feasible_pen < 1e-3 and obj < best_obj:
            best_x, best_y = sx, sy
    except Exception:
        best_x, best_y = init_x, init_y

    return best_x, best_y
