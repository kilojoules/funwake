"""BayesianOptimization staggered-seed search with incumbent recovery.

HYPOTHESIS: The 5560.8 GWh recovery path comes from the two staggered
incumbent starts; a small GP surrogate can replace the previous wind-sector
third start and test bayesian_optimization without disrupting that basin.

AXIS: BayesianOptimization-style GP/LCB search over staggered lattice
angle/spacing/phase/skew/ranking parameters, followed by established
nesterov_momentum incumbent recovery starts.

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

    def staggered_candidates(angle, spacing_mult, phase, skew):
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
        return pts[:, 0][mask], pts[:, 1][mask]

    def staggered_seed(angle, spacing_mult, phase, skew):
        px, py = staggered_candidates(angle, spacing_mult, phase, skew)
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

    param_bounds = np.array(
        [
            [-0.46, 0.46],
            [0.994, 1.012],
            [0.00, 1.00],
            [-0.018, 0.018],
            [0.05, 0.78],
            [0.00, 1.00],
        ],
        dtype=float,
    )
    param_span = param_bounds[:, 1] - param_bounds[:, 0]

    def clamp_params(params):
        params = np.asarray(params, dtype=float)
        return np.minimum(np.maximum(params, param_bounds[:, 0]), param_bounds[:, 1])

    def to_unit(params):
        return (clamp_params(params) - param_bounds[:, 0]) / param_span

    def from_unit(unit_params):
        unit_params = np.minimum(np.maximum(np.asarray(unit_params, dtype=float), 0.0), 1.0)
        return param_bounds[:, 0] + unit_params * param_span

    def ranked_staggered_seed(params):
        params = clamp_params(params)
        angle = base_angle + float(params[0])
        px, py = staggered_candidates(angle, float(params[1]), float(params[2]), float(params[3]))
        if len(px) < n_target:
            return staggered_seed(angle, float(params[1]), float(params[2]), float(params[3]))
        rank = (
            (px - center[0]) * jnp.cos(angle + float(params[4]))
            + (py - center[1]) * jnp.sin(angle + float(params[4]))
            + 0.08
            * min_spacing
            * jnp.sin(px / (2.7 * min_spacing) + 2.0 * jnp.pi * float(params[5]))
        )
        order = jnp.argsort(rank)
        px = px[order]
        py = py[order]
        idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
        return px[idx], py[idx]

    def seed_objective(params):
        x, y = ranked_staggered_seed(params)
        if not feasible(x, y):
            return 1.0e8
        return float(objective_xy_jit(x, y))

    def deterministic_pool(count, dim):
        multipliers = np.array([0.754877666, 0.569840291, 0.438543333,
                                0.315964877, 0.245122333, 0.184308110])
        offsets = np.array([0.11, 0.37, 0.61, 0.83, 0.29, 0.53])
        i = np.arange(1, count + 1, dtype=float)[:, None]
        return np.mod(i * multipliers[:dim] + offsets[:dim], 1.0)

    def bayesian_seed():
        seeds = [
            np.array([0.00, 1.000, 0.24, 0.000, 0.31, 0.00]),
            np.array([0.16, 1.006, 0.58, 0.015, 0.47, 0.20]),
            np.array([-0.18, 1.002, 0.74, -0.010, 0.22, 0.63]),
            np.array([0.31, 0.997, 0.09, 0.006, 0.68, 0.42]),
            np.array([-0.36, 1.010, 0.43, 0.012, 0.53, 0.87]),
            np.array([0.07, 0.995, 0.86, -0.015, 0.14, 0.31]),
            np.array([0.42, 1.011, 0.33, -0.003, 0.38, 0.72]),
        ]
        xs = []
        ys = []
        for params in seeds:
            unit = to_unit(params)
            xs.append(unit)
            ys.append(seed_objective(params))

        base_pool = deterministic_pool(120, 6)
        length_scale = np.array([0.22, 0.16, 0.25, 0.30, 0.24, 0.26])
        for step in range(6):
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            finite = np.isfinite(y_arr) & (y_arr < 1.0e7)
            y_mean = float(np.mean(y_arr[finite])) if np.any(finite) else 0.0
            y_std = float(np.std(y_arr[finite])) + 1.0e-9 if np.any(finite) else 1.0
            y_scaled = (y_arr - y_mean) / y_std
            diffs = (x_arr[:, None, :] - x_arr[None, :, :]) / length_scale
            k_mat = np.exp(-0.5 * np.sum(diffs * diffs, axis=2))
            k_mat += np.eye(len(x_arr)) * 2.0e-6
            try:
                chol = np.linalg.cholesky(k_mat)
                alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_scaled))
            except np.linalg.LinAlgError:
                break
            best_unit = x_arr[int(np.argmin(y_arr))]
            local = np.mod(best_unit[None, :] + (deterministic_pool(36, 6) - 0.5) * 0.26, 1.0)
            pool = np.vstack([base_pool, local])
            sep = np.min(np.sum((pool[:, None, :] - x_arr[None, :, :]) ** 2, axis=2), axis=1)
            pool = pool[sep > 0.016]
            if len(pool) == 0:
                break
            k_star = np.exp(
                -0.5
                * np.sum(((pool[:, None, :] - x_arr[None, :, :]) / length_scale) ** 2, axis=2)
            )
            mu = k_star @ alpha
            v = np.linalg.solve(chol, k_star.T)
            sigma = np.sqrt(np.maximum(1.0 - np.sum(v * v, axis=0), 1.0e-8))
            next_unit = pool[int(np.argmin(mu - (1.25 - 0.06 * step) * sigma))]
            xs.append(next_unit)
            ys.append(seed_objective(from_unit(next_unit)))

        best_i = int(np.argmin(np.asarray(ys, dtype=float)))
        return ranked_staggered_seed(from_unit(np.asarray(xs[best_i], dtype=float)))

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
    bayes_run = make_runner(86.0, 0.0045, 700, 2500, 0.32)
    polish_run = make_runner(18.0, 0.10, 80, 520, 0.18)

    def consider(best_x, best_y, best_obj, cand_z):
        x = cand_z[:n_target]
        y = cand_z[n_target:]
        if feasible(x, y):
            obj = float(objective_xy_jit(x, y))
            if obj < best_obj:
                return x, y, obj
        return best_x, best_y, best_obj

    start0 = staggered_seed(base_angle, 1.000, 0.24, 0.000)
    start1 = staggered_seed(base_angle + 0.16, 1.006, 0.58, 0.015)
    start2 = bayesian_seed()

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

    z = bayes_run(jnp.concatenate([start2[0], start2[1]]), 1.0)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)

    z = polish_run(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)
    return best_x, best_y
