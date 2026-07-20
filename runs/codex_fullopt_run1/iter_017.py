"""Tiny particle swarm over hex-lattice parameters with incumbent retention.

HYPOTHESIS: pso can cheaply explore coupled orientation/phase/shear changes in
the low-dimensional hex-lattice parameter space, and retaining the two exact
best-known starts should recover the 5560.8 GWh basin if the swarm seed is not
better.

AXIS: particle_swarm search for one extra hex-lattice seed plus the two
incumbent wind-aware hex starts and nesterov_momentum local penalty polish.

LESSON: Pending score.
"""
import numpy as np
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

    objective_xy_jit = jax.jit(objective_xy)

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

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    base_angle = mean_wind + 0.5 * jnp.pi

    def feasible(x, y):
        bnd = boundary_penalty(x, y, boundary)
        spc = spacing_penalty(x, y, min_spacing * 0.99)
        return float(bnd) < 1e-3 and float(spc) < 1e-3

    def particle_value(params):
        x, y = hexagonal_lattice(
            base_angle + float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
        )
        if not feasible(x, y):
            return 1.0e8
        return float(objective_xy_jit(x, y))

    def particle_swarm_seed():
        lo = np.array([-0.42, 0.995, 0.00, -0.018], dtype=float)
        hi = np.array([0.42, 1.012, 1.00, 0.018], dtype=float)
        pos = np.array(
            [
                [0.16, 1.006, 0.58, 0.015],
                [0.00, 1.000, 0.24, 0.000],
                [-0.18, 1.002, 0.82, -0.010],
                [0.31, 0.997, 0.12, 0.009],
                [-0.34, 1.010, 0.46, 0.004],
            ],
            dtype=float,
        )
        vel = np.zeros_like(pos)
        pbest = pos.copy()
        pval = np.array([particle_value(p) for p in pbest], dtype=float)
        best_i = int(np.argmin(pval))
        gbest = pbest[best_i].copy()
        gval = float(pval[best_i])

        for step in range(3):
            grid = np.arange(pos.size, dtype=float).reshape(pos.shape)
            r1 = np.mod(0.37 * grid + 0.19 * (step + 1), 1.0)
            r2 = np.mod(0.53 * grid + 0.11 * (step + 2), 1.0)
            vel = 0.42 * vel + 0.85 * r1 * (pbest - pos) + 0.65 * r2 * (gbest - pos)
            pos = np.minimum(np.maximum(pos + vel, lo), hi)
            vals = np.array([particle_value(p) for p in pos], dtype=float)
            improved = vals < pval
            pbest[improved] = pos[improved]
            pval[improved] = vals[improved]
            best_i = int(np.argmin(pval))
            if float(pval[best_i]) < gval:
                gbest = pbest[best_i].copy()
                gval = float(pval[best_i])
        return gbest

    swarm_params = particle_swarm_seed()

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
                _, gc = con_vg(look, spacing_scale)
                g = go + alpha * gc
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
                body, (z0, v0, lr0, alpha0), jnp.arange(total_steps)
            )
            return z

        return run

    main_run = make_runner(95.0, 0.0035, 1400, 3600, 0.38)
    swarm_run = make_runner(88.0, 0.0042, 900, 2600, 0.35)
    polish_run = make_runner(18.0, 0.10, 80, 520, 0.18)

    def consider(best_x, best_y, best_obj, cand_z):
        x = cand_z[:n_target]
        y = cand_z[n_target:]
        if feasible(x, y):
            obj = float(objective_xy_jit(x, y))
            if obj < best_obj:
                return x, y, obj
        return best_x, best_y, best_obj

    start0 = hexagonal_lattice(base_angle, 1.000, 0.24, 0.000)
    start1 = hexagonal_lattice(base_angle + 0.16, 1.006, 0.58, 0.015)
    start2 = hexagonal_lattice(
        base_angle + float(swarm_params[0]),
        float(swarm_params[1]),
        float(swarm_params[2]),
        float(swarm_params[3]),
    )

    best_x, best_y = start0
    best_obj = float(objective_xy_jit(best_x, best_y)) if feasible(best_x, best_y) else 1.0e30
    for init_x, init_y in (start0, start1, start2):
        if feasible(init_x, init_y):
            init_obj = float(objective_xy_jit(init_x, init_y))
            if init_obj < best_obj:
                best_x, best_y, best_obj = init_x, init_y, init_obj

    for init_x, init_y in (start0, start1):
        z = main_run(jnp.concatenate([init_x, init_y]), 1.0)
        best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)

    z = swarm_run(jnp.concatenate([start2[0], start2[1]]), 1.0)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)

    z = polish_run(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)
    return best_x, best_y
