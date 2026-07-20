"""Differential evolution over wind-aware hex-lattice parameters.

HYPOTHESIS: A low-dimensional scipy_differential_evolution search over
hex-lattice orientation, phase, shear, and selection ranking can find a
different basin than hand-picked hex starts while staying inside the 60s
scoring budget when followed by one Nesterov polish.

AXIS: scipy_differential_evolution global search for initialization plus
wind-aware hexagonal lattice and short nesterov_momentum penalty polish.

LESSON: Pending score.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import differential_evolution
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

    def inside_mask(x, y, edge_margin):
        return jnp.min(edge_distances(x, y), axis=0) > edge_margin

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    base_angle = mean_wind + 0.5 * jnp.pi

    def hexagonal_lattice(params):
        angle = base_angle + float(params[0])
        spacing_mult = float(params[1])
        phase_x = float(params[2])
        phase_y = float(params[3])
        shear = float(params[4])
        rank_angle = base_angle + float(params[5])
        radial_bias = float(params[6])

        ca = jnp.cos(angle)
        sa = jnp.sin(angle)
        rot = jnp.array([[ca, -sa], [sa, ca]])
        inv_rot = jnp.array([[ca, sa], [-sa, ca]])
        rb = (rot @ (boundary - center).T).T
        rx_min, ry_min = jnp.min(rb, axis=0)
        rx_max, ry_max = jnp.max(rb, axis=0)

        spacing = min_spacing * spacing_mult
        row_step = spacing * jnp.sqrt(3.0) * 0.5
        nx = max(4, int(jnp.ceil((rx_max - rx_min) / spacing)) + 6)
        ny = max(4, int(jnp.ceil((ry_max - ry_min) / row_step)) + 6)
        gx = rx_min - 2.0 * spacing + phase_x * spacing + spacing * jnp.arange(nx)
        gy = ry_min - 2.0 * row_step + phase_y * row_step + row_step * jnp.arange(ny)
        xx, yy = jnp.meshgrid(gx, gy)
        row = jnp.arange(ny)[:, None]
        xx = xx + (row % 2) * 0.5 * spacing + shear * (yy - jnp.mean(rb[:, 1]))
        pts_r = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        pts = (inv_rot @ pts_r.T).T + center

        margin = max(0.0, min(0.05, 0.018 + 0.006 * abs(radial_bias))) * min_spacing
        mask = inside_mask(pts[:, 0], pts[:, 1], margin)
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        if len(px) < n_target:
            mask = inside_mask(pts[:, 0], pts[:, 1], 0.0)
            px = pts[:, 0][mask]
            py = pts[:, 1][mask]
        if len(px) < n_target:
            return None, None

        ux = jnp.cos(rank_angle)
        uy = jnp.sin(rank_angle)
        rx = (px - center[0]) / (span_x + 1e-9)
        ry = (py - center[1]) / (span_y + 1e-9)
        rank = (
            (px - center[0]) * ux
            + (py - center[1]) * uy
            + radial_bias * min_spacing * (rx * rx + ry * ry)
            + 0.055 * min_spacing * jnp.sin((px + 0.37 * py) / (2.6 * min_spacing))
        )
        order = jnp.argsort(rank)
        px = px[order]
        py = py[order]
        idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
        return px[idx], py[idx]

    def feasible(x, y):
        if x is None:
            return False
        bnd = boundary_penalty(x, y, boundary)
        spc = spacing_penalty(x, y, min_spacing * 0.99)
        return float(bnd) < 1e-3 and float(spc) < 1e-3

    def de_objective(params):
        x, y = hexagonal_lattice(params)
        if not feasible(x, y):
            return 1.0e8
        value = float(objective_xy_jit(x, y))
        return value

    bounds = [
        (-0.55, 0.55),   # angle offset
        (0.992, 1.018),  # spacing multiplier; benchmark allows 0.99 spacing
        (0.0, 1.0),      # x phase
        (0.0, 1.0),      # y phase
        (-0.018, 0.018), # shear
        (-1.0, 1.0),     # rank-angle offset
        (-0.55, 0.55),   # edge/interior selection bias
    ]

    init_pop = np.array(
        [
            [0.00, 1.000, 0.24, 0.65, 0.000, 0.31, 0.00],
            [0.16, 1.006, 0.58, 0.99, 0.015, 0.31, 0.05],
            [-0.11, 0.997, 0.76, 0.17, -0.012, 0.31, -0.10],
            [0.08, 0.995, 0.12, 0.42, 0.008, -0.24, 0.25],
            [-0.22, 1.003, 0.91, 0.20, 0.004, 0.72, -0.30],
            [0.31, 0.999, 0.37, 0.74, -0.006, -0.66, 0.15],
            [-0.38, 1.010, 0.69, 0.08, 0.012, 0.48, -0.42],
            [0.46, 0.994, 0.05, 0.88, -0.014, -0.08, 0.38],
            [-0.04, 1.014, 0.83, 0.33, 0.017, 0.91, -0.18],
            [0.24, 1.001, 0.48, 0.56, -0.002, -0.91, 0.48],
            [-0.50, 0.996, 0.31, 0.13, 0.010, 0.08, -0.52],
            [0.53, 1.016, 0.73, 0.47, -0.010, -0.42, 0.31],
        ],
        dtype=float,
    )

    try:
        de_res = differential_evolution(
            de_objective,
            bounds,
            strategy="best1bin",
            maxiter=2,
            popsize=3,
            init=init_pop,
            mutation=(0.35, 0.85),
            recombination=0.72,
            polish=False,
            seed=12012,
            tol=0.0,
            updating="immediate",
            workers=1,
        )
        de_params = np.asarray(de_res.x, dtype=float)
    except Exception:
        de_params = init_pop[0]

    de_x, de_y = hexagonal_lattice(de_params)
    if not feasible(de_x, de_y):
        de_x, de_y = hexagonal_lattice(init_pop[0])

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

    main_run = make_runner(90.0, 0.0040, 1100, 2800, 0.36)
    polish_run = make_runner(16.0, 0.11, 70, 420, 0.16)

    def consider(best_x, best_y, best_obj, cand_z):
        x = cand_z[:n_target]
        y = cand_z[n_target:]
        if feasible(x, y):
            obj = float(objective_xy_jit(x, y))
            if obj < best_obj:
                return x, y, obj
        return best_x, best_y, best_obj

    best_x, best_y = de_x, de_y
    best_obj = float(objective_xy_jit(best_x, best_y)) if feasible(best_x, best_y) else 1.0e30

    z1 = main_run(jnp.concatenate([de_x, de_y]), 1.0)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z1)

    z2 = polish_run(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z2)

    return best_x, best_y
