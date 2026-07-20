"""Nesterov lookahead optimizer with inverse-LR constraint ramp.

HYPOTHESIS: A Nesterov momentum loop can keep the long exploratory moves that
helped the Adam-like runs, but the lookahead gradient should reduce late-stage
oscillation around wake-interaction ridges and spacing-active layouts.

AXIS: nesterov_momentum with wind-aware staggered initialization and a rising
penalty schedule; no TopFarm packaged solver and no SciPy constrained polish.

LESSON: Pending score.
"""
import jax
import jax.numpy as jnp
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

    def objective_xy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def objective_z(z):
        return objective_xy(z[:n_target], z[n_target:])

    def constraint_z(z, spacing_scale):
        x = z[:n_target]
        y = z[n_target:]
        return 100.0 * boundary_penalty(x, y, boundary) + 100.0 * spacing_penalty(
            x, y, min_spacing * spacing_scale
        )

    obj_vg = jax.jit(jax.value_and_grad(objective_z))
    con_vg = jax.jit(jax.value_and_grad(constraint_z))

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

    def wind_staggered_grid(angle, spacing_mult, phase):
        ca = jnp.cos(angle)
        sa = jnp.sin(angle)
        rot = jnp.array([[ca, -sa], [sa, ca]])
        inv_rot = jnp.array([[ca, sa], [-sa, ca]])
        rb = (rot @ (boundary - center).T).T
        rx_min, ry_min = jnp.min(rb, axis=0)
        rx_max, ry_max = jnp.max(rb, axis=0)
        spacing = min_spacing * spacing_mult
        row_step = spacing * 0.8660254037844386
        nx = max(4, int(jnp.ceil((rx_max - rx_min) / spacing)) + 3)
        ny = max(4, int(jnp.ceil((ry_max - ry_min) / row_step)) + 3)
        gx = rx_min - spacing + spacing * phase + spacing * jnp.arange(nx)
        gy = ry_min - row_step + row_step * (0.37 + phase) + row_step * jnp.arange(ny)
        xx, yy = jnp.meshgrid(gx, gy)
        xx = xx + (jnp.arange(ny) % 2)[:, None] * 0.5 * spacing
        pts_r = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        pts = (inv_rot @ pts_r.T).T + center
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        if len(px) >= n_target:
            rank = (
                (px - center[0]) * jnp.cos(angle + 0.19)
                + (py - center[1]) * jnp.sin(angle + 0.19)
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
            _, g_obj0 = obj_vg(z0)
            alpha0 = jnp.mean(jnp.abs(g_obj0)) / lr0
            v0 = jnp.zeros_like(z0)

            def body(carry, t):
                z, v, lr, alpha = carry
                look = z + momentum * v
                _, g_obj = obj_vg(look)
                _, g_con = con_vg(look, spacing_scale)
                g = g_obj + alpha * g_con
                mean_abs = jnp.mean(jnp.abs(g)) + 1e-12
                v = momentum * v - lr * g / mean_abs
                z = z + v
                z = jnp.concatenate(
                    [
                        jnp.clip(z[:n_target], x_min - 0.10 * span_x, x_max + 0.10 * span_x),
                        jnp.clip(z[n_target:], y_min - 0.10 * span_y, y_max + 0.10 * span_y),
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

    main_run = make_runner(
        lr0=42.0,
        gamma_min=0.020,
        const_steps=900,
        decay_steps=3200,
        momentum=0.72,
    )
    polish_run = make_runner(
        lr0=10.0,
        gamma_min=0.080,
        const_steps=120,
        decay_steps=520,
        momentum=0.45,
    )

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

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    base_angle = mean_wind + 0.5 * jnp.pi
    starts = [
        wind_staggered_grid(base_angle, 1.000, 0.25),
        wind_staggered_grid(base_angle - 0.20, 1.006, 0.52),
    ]

    best_x, best_y = starts[0]
    best_obj = float(objective_xy(best_x, best_y)) if feasible(best_x, best_y) else 1.0e30

    for init_x, init_y in starts:
        z0 = jnp.concatenate([init_x, init_y])
        if feasible(init_x, init_y):
            init_obj = float(objective_xy(init_x, init_y))
            if init_obj < best_obj:
                best_x, best_y, best_obj = init_x, init_y, init_obj
        z1 = main_run(z0, 1.0)
        best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z1)

    z_polish = polish_run(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z_polish)

    return best_x, best_y
