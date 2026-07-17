"""SciPy L-BFGS-B penalty continuation from wind-rotated staggered starts.

HYPOTHESIS: L-BFGS-B on normalized turbine coordinates can use JAX
value-and-gradient evaluations to make larger, curvature-informed local moves
than the Adam-style runs, while a staged penalty keeps the layout feasible.

AXIS: scipy_lbfgs with wind-aware staggered-grid initialization and pure penalty
constraints; no TopFarm solver or explicit constraint Jacobian optimizer.

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
    span_x = x_max - x_min
    span_y = y_max - y_min
    scale = min_spacing * min_spacing
    center = jnp.mean(boundary, axis=0)

    def objective_xy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def unpack(z):
        x = x_min + span_x * z[:n_target]
        y = y_min + span_y * z[n_target:]
        return x, y

    def edge_distances(x, y):
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
        return jnp.min(edge_distances(x, y), axis=0) > 0.0

    def staggered_start(angle, spacing_mult, phase):
        ca = jnp.cos(angle)
        sa = jnp.sin(angle)
        rot = jnp.array([[ca, -sa], [sa, ca]])
        inv_rot = jnp.array([[ca, sa], [-sa, ca]])
        rb = (rot @ (boundary - center).T).T
        rx_min, ry_min = jnp.min(rb, axis=0)
        rx_max, ry_max = jnp.max(rb, axis=0)
        spacing = min_spacing * spacing_mult
        row_step = spacing * 0.8660254037844386
        nx = max(4, int(jnp.ceil((rx_max - rx_min) / spacing)) + 4)
        ny = max(4, int(jnp.ceil((ry_max - ry_min) / row_step)) + 4)

        gx = rx_min - spacing + spacing * phase + spacing * jnp.arange(nx)
        gy = ry_min - row_step + row_step * (0.35 + phase) + row_step * jnp.arange(ny)
        xx, yy = jnp.meshgrid(gx, gy)
        row_shift = (jnp.arange(ny) % 2)[:, None] * 0.5 * spacing
        xx = xx + row_shift

        pts_r = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        pts = (inv_rot @ pts_r.T).T + center
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        if len(px) >= n_target:
            rank = (
                (px - center[0]) * jnp.cos(angle + 0.37)
                + (py - center[1]) * jnp.sin(angle + 0.37)
                + 0.15
                * min_spacing
                * jnp.sin((px + 0.7 * py) / (3.0 * min_spacing))
            )
            order = jnp.argsort(rank)
            px = px[order]
            py = py[order]
            idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
            return px[idx], py[idx]

        gx, gy = jnp.meshgrid(
            jnp.linspace(x_min + 0.05 * span_x, x_max - 0.05 * span_x, n_target),
            jnp.linspace(y_min + 0.05 * span_y, y_max - 0.05 * span_y, n_target),
        )
        pts = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
        return px[idx], py[idx]

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )

    def feasible(x, y):
        bnd = boundary_penalty(x, y, boundary)
        spc = spacing_penalty(x, y, min_spacing * 0.99)
        return float(bnd) < 1e-3 and float(spc) < 1e-3

    def obj_value(x, y):
        return float(objective_xy(x, y))

    def make_penalized(spacing_factor, penalty_weight):
        def penalized(z):
            x, y = unpack(z)
            obj = objective_xy(x, y)
            bnd = boundary_penalty(x, y, boundary) / scale
            spc = spacing_penalty(x, y, min_spacing * spacing_factor) / scale
            return obj + penalty_weight * (bnd + spc)

        return jax.jit(jax.value_and_grad(penalized))

    compiled = [
        make_penalized(0.9905, 350.0),
        make_penalized(0.9930, 1800.0),
        make_penalized(0.9950, 8500.0),
    ]

    def pack(x, y):
        zx = (x - x_min) / span_x
        zy = (y - y_min) / span_y
        return jnp.clip(jnp.concatenate([zx, zy]), 0.0, 1.0)

    def run_lbfgs(x0, y0, maxiters):
        z_np = np.asarray(pack(x0, y0), dtype=float)
        for value_grad, maxiter in zip(compiled, maxiters):
            def fun_and_jac(z):
                value, grad = value_grad(jnp.asarray(z))
                return float(value), np.asarray(grad, dtype=float)

            res = minimize(
                fun_and_jac,
                z_np,
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, 1.0)] * (2 * n_target),
                options={
                    "maxiter": maxiter,
                    "maxls": 12,
                    "ftol": 2e-7,
                    "gtol": 1e-5,
                    "disp": False,
                },
            )
            z_np = np.asarray(res.x, dtype=float)
        rz = jnp.asarray(z_np)
        return unpack(rz)

    base_angle = mean_wind + 0.5 * jnp.pi
    starts = [
        staggered_start(base_angle, 1.000, 0.18),
        staggered_start(base_angle + 0.24, 1.004, 0.46),
    ]

    best_x, best_y = starts[0]
    best_obj = obj_value(best_x, best_y) if feasible(best_x, best_y) else 1.0e30

    for i, (init_x, init_y) in enumerate(starts):
        if feasible(init_x, init_y):
            init_obj = obj_value(init_x, init_y)
            if init_obj < best_obj:
                best_x, best_y, best_obj = init_x, init_y, init_obj

        maxiters = (28, 24, 14) if i == 0 else (18, 16, 10)
        try:
            cand_x, cand_y = run_lbfgs(init_x, init_y, maxiters)
            if feasible(cand_x, cand_y):
                cand_obj = obj_value(cand_x, cand_y)
                if cand_obj < best_obj:
                    best_x, best_y, best_obj = cand_x, cand_y, cand_obj
        except Exception:
            pass

    return best_x, best_y
