"""Interior-point polish after the best custom wake basin.

HYPOTHESIS: The strongest Nesterov/hexagonal layout is close to a spacing-
active constrained optimum, so a short trust-constr pass with explicit polygon
and pairwise spacing inequalities can move along the feasible boundary better
than another penalty-gradient polish.

AXIS: scipy_trust_constr interior-point polish after nesterov_momentum and
init_hexagonal starts.

LESSON: Pending score.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import BFGS, Bounds, NonlinearConstraint, minimize
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    n_verts = boundary.shape[0]
    x_min = float(jnp.min(boundary[:, 0]))
    x_max = float(jnp.max(boundary[:, 0]))
    y_min = float(jnp.min(boundary[:, 1]))
    y_max = float(jnp.max(boundary[:, 1]))
    span_x = x_max - x_min
    span_y = y_max - y_min
    center = jnp.mean(boundary, axis=0)
    ms2 = min_spacing * min_spacing

    def objective_xy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def objective_z(z):
        return objective_xy(z[:n_target], z[n_target:])

    def penalty_z(z, spacing_scale):
        x = z[:n_target]
        y = z[n_target:]
        return 100.0 * boundary_penalty(x, y, boundary) + 100.0 * spacing_penalty(
            x, y, min_spacing * spacing_scale
        )

    obj_vg = jax.jit(jax.value_and_grad(objective_z))
    pen_vg = jax.jit(jax.value_and_grad(penalty_z))

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

    def hexagonal_lattice(angle, spacing_mult, phase, skew):
        ca = jnp.cos(angle)
        sa = jnp.sin(angle)
        rot = jnp.array([[ca, -sa], [sa, ca]])
        inv_rot = jnp.array([[ca, sa], [-sa, ca]])
        rb = (rot @ (boundary - center).T).T
        rx_min, ry_min = jnp.min(rb, axis=0)
        rx_max, ry_max = jnp.max(rb, axis=0)
        spacing = min_spacing * spacing_mult
        row_step = spacing * jnp.sqrt(3.0) * 0.5
        nx = max(4, int(jnp.ceil((rx_max - rx_min) / spacing)) + 5)
        ny = max(4, int(jnp.ceil((ry_max - ry_min) / row_step)) + 5)
        gx = rx_min - 1.5 * spacing + spacing * phase + spacing * jnp.arange(nx)
        gy = ry_min - 1.5 * row_step + row_step * (0.41 + phase) + row_step * jnp.arange(ny)
        xx, yy = jnp.meshgrid(gx, gy)
        row = jnp.arange(ny)[:, None]
        xx = xx + (row % 2) * 0.5 * spacing + skew * (yy - center[1])
        pts_r = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        pts = (inv_rot @ pts_r.T).T + center
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        if len(px) >= n_target:
            rank = (
                (px - center[0]) * jnp.cos(angle + 0.31)
                + (py - center[1]) * jnp.sin(angle + 0.31)
                + 0.08 * min_spacing * jnp.sin(px / (2.7 * min_spacing))
            )
            order = jnp.argsort(rank)
            px = px[order]
            py = py[order]
            idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
            return px[idx], py[idx]

        gx, gy = jnp.meshgrid(
            jnp.linspace(x_min + 0.06 * span_x, x_max - 0.06 * span_x, n_target),
            jnp.linspace(y_min + 0.06 * span_y, y_max - 0.06 * span_y, n_target),
        )
        pts = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
        return px[idx], py[idx]

    def compute_mid(lr0, gamma_min, n_steps):
        lo = 0.0
        hi = 0.1
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lr = lr0
            for t in range(1, n_steps + 1):
                lr *= 1.0 / (1.0 + mid * t)
            if lr < gamma_min:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def make_runner(lr0, gamma_min, const_steps, decay_steps, momentum):
        total_steps = const_steps + decay_steps
        mid = compute_mid(lr0, gamma_min, decay_steps)

        @jax.jit
        def run(z0, spacing_scale):
            _, g0 = obj_vg(z0)
            alpha0 = jnp.mean(jnp.abs(g0)) / lr0
            v0 = jnp.zeros_like(z0)

            def body(carry, t):
                z, v, lr, alpha = carry
                look = z + momentum * v
                _, go = obj_vg(look)
                _, gp = pen_vg(look, spacing_scale)
                g = go + alpha * gp
                rms = jnp.sqrt(jnp.mean(g * g)) + 1e-12
                v = momentum * v - lr * g / rms
                z = z + v
                z = jnp.concatenate(
                    [
                        jnp.clip(z[:n_target], x_min - 0.08 * span_x, x_max + 0.08 * span_x),
                        jnp.clip(z[n_target:], y_min - 0.08 * span_y, y_max + 0.08 * span_y),
                    ]
                )
                step = t + 1.0
                decaying = step > const_steps
                decay_step = jnp.where(decaying, step - const_steps, 0.0)
                new_lr = jnp.where(decaying, lr / (1.0 + mid * decay_step), lr)
                new_alpha = jnp.where(decaying, alpha0 * lr0 / new_lr, alpha)
                return (z, v, new_lr, new_alpha), None

            (z, _, _, _), _ = jax.lax.scan(
                body, (z0, v0, lr0, alpha0), jnp.arange(float(total_steps))
            )
            return z

        return run

    main_run = make_runner(95.0, 0.0035, 1400, 3600, 0.38)
    polish_run = make_runner(18.0, 0.10, 80, 520, 0.18)

    def feasible(x, y):
        bnd = boundary_penalty(x, y, boundary)
        spc = spacing_penalty(x, y, min_spacing * 0.99)
        return float(bnd) < 1e-3 and float(spc) < 1e-3

    def consider(best_x, best_y, best_obj, cand_z):
        x = cand_z[:n_target]
        y = cand_z[n_target:]
        if feasible(x, y):
            obj = float(objective_xy(x, y))
            if obj < best_obj:
                return x, y, obj
        return best_x, best_y, best_obj

    def trust_polish(best_x, best_y, best_obj):
        def constraints_z(z):
            x = z[:n_target]
            y = z[n_target:]
            bnd = (edge_distances(x, y).T).ravel() / min_spacing
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist_sq = dx * dx + dy * dy
            iu = jnp.triu_indices(n_target, k=1)
            spacing = dist_sq[iu] / ms2 - 0.99 * 0.99
            return jnp.concatenate([bnd, spacing])

        con_value = jax.jit(constraints_z)
        con_jacobian = jax.jit(jax.jacfwd(constraints_z))

        def fun(z_np):
            value, _ = obj_vg(jnp.asarray(z_np))
            return float(value)

        def jac(z_np):
            _, grad = obj_vg(jnp.asarray(z_np))
            return np.asarray(grad, dtype=float)

        def con_fun(z_np):
            return np.asarray(con_value(jnp.asarray(z_np)), dtype=float)

        def con_jac(z_np):
            return np.asarray(con_jacobian(jnp.asarray(z_np)), dtype=float)

        z0 = np.asarray(jnp.concatenate([best_x, best_y]), dtype=float)
        n_cons = n_target * n_verts + n_target * (n_target - 1) // 2
        try:
            res = minimize(
                fun,
                z0,
                method="trust-constr",
                jac=jac,
                hess=BFGS(),
                bounds=Bounds(
                    [x_min] * n_target + [y_min] * n_target,
                    [x_max] * n_target + [y_max] * n_target,
                ),
                constraints=[
                    NonlinearConstraint(
                        con_fun,
                        np.zeros(n_cons),
                        np.full(n_cons, np.inf),
                        jac=con_jac,
                        hess=BFGS(),
                    )
                ],
                options={
                    "maxiter": 8,
                    "gtol": 1e-5,
                    "xtol": 1e-5,
                    "initial_tr_radius": 0.35 * min_spacing,
                    "barrier_tol": 1e-4,
                    "verbose": 0,
                },
            )
            rz = jnp.asarray(res.x)
            return consider(best_x, best_y, best_obj, rz)
        except Exception:
            return best_x, best_y, best_obj

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    base_angle = mean_wind + 0.5 * jnp.pi
    starts = [
        hexagonal_lattice(base_angle, 1.000, 0.24, 0.000),
        hexagonal_lattice(base_angle + 0.16, 1.006, 0.58, 0.015),
    ]

    best_x, best_y = starts[0]
    best_obj = float(objective_xy(best_x, best_y)) if feasible(best_x, best_y) else 1.0e30
    for init_x, init_y in starts:
        if feasible(init_x, init_y):
            init_obj = float(objective_xy(init_x, init_y))
            if init_obj < best_obj:
                best_x, best_y, best_obj = init_x, init_y, init_obj
        z1 = main_run(jnp.concatenate([init_x, init_y]), 1.0)
        best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z1)

    z2 = polish_run(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z2)
    best_x, best_y, best_obj = trust_polish(best_x, best_y, best_obj)
    return best_x, best_y
