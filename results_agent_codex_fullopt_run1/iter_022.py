"""BayesianOptimization-style surrogate search with incumbent basin recovery.

HYPOTHESIS: Attempt 21 lost score by drifting the incumbent lattice geometry;
restoring the known basin while letting a GP surrogate pick the second start
can keep the 5560.8 GWh recovery path and still test bayesian_optimization.

AXIS: bayesian_optimization with a hand-built GP/LCB acquisition over
hex-lattice angle/spacing/phase/shear/ranking parameters, followed by two
nesterov_momentum penalty polishes.

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

    def inside_mask(x, y, edge_margin):
        return jnp.min(edge_distances(x, y), axis=0) > edge_margin

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    base_angle = mean_wind + 0.5 * jnp.pi

    bounds = np.array(
        [
            [-0.52, 0.52],    # angle offset
            [0.993, 1.014],   # spacing multiplier
            [0.00, 1.00],     # x phase
            [0.00, 1.00],     # y phase
            [-0.017, 0.017],  # shear
            [-0.90, 0.90],    # ranking-angle offset
            [-0.45, 0.45],    # edge/interior ranking bias
        ],
        dtype=float,
    )
    span_params = bounds[:, 1] - bounds[:, 0]

    def clamp_params(params):
        params = np.asarray(params, dtype=float)
        return np.minimum(np.maximum(params, bounds[:, 0]), bounds[:, 1])

    def to_unit(params):
        return (clamp_params(params) - bounds[:, 0]) / span_params

    def from_unit(unit_params):
        unit_params = np.minimum(np.maximum(np.asarray(unit_params, dtype=float), 0.0), 1.0)
        return bounds[:, 0] + unit_params * span_params

    def hexagonal_lattice(params):
        params = clamp_params(params)
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

        margin = max(0.0, min(0.05, 0.012 + 0.008 * abs(radial_bias))) * min_spacing
        mask = inside_mask(pts[:, 0], pts[:, 1], margin)
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        if len(px) < n_target:
            mask = inside_mask(pts[:, 0], pts[:, 1], 0.0)
            px = pts[:, 0][mask]
            py = pts[:, 1][mask]
        if len(px) < n_target:
            gx, gy = jnp.meshgrid(
                jnp.linspace(x_min + 0.08 * span_x, x_max - 0.08 * span_x, n_target),
                jnp.linspace(y_min + 0.08 * span_y, y_max - 0.08 * span_y, n_target),
            )
            pts = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
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
            + 0.06 * min_spacing * jnp.sin((px + 0.41 * py) / (2.8 * min_spacing))
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

    def init_objective(params):
        x, y = hexagonal_lattice(params)
        if not feasible(x, y):
            return 1.0e8
        return float(objective_xy_jit(x, y))

    incumbent_params = np.array([0.00, 1.000, 0.24, 0.65, 0.000, 0.31, 0.00])
    alternate_params = np.array([0.16, 1.006, 0.58, 0.99, 0.015, 0.47, 0.05])

    def deterministic_pool(count, dim):
        multipliers = np.array([0.754877666, 0.569840291, 0.438543333, 0.315964877,
                                0.245122333, 0.184308110, 0.128982912])
        offsets = np.array([0.11, 0.37, 0.61, 0.83, 0.29, 0.53, 0.73])
        i = np.arange(1, count + 1, dtype=float)[:, None]
        return np.mod(i * multipliers[:dim] + offsets[:dim], 1.0)

    def gp_lcb_search():
        seeds = [
            incumbent_params,
            alternate_params,
            np.array([-0.22, 1.002, 0.11, 0.34, -0.006, -0.18, 0.30]),
            np.array([0.31, 0.998, 0.77, 0.18, 0.010, 0.72, -0.22]),
            np.array([-0.39, 1.011, 0.42, 0.86, 0.017, -0.63, 0.46]),
            np.array([0.08, 0.994, 0.91, 0.46, -0.014, -0.82, -0.38]),
            np.array([0.45, 1.015, 0.28, 0.73, -0.002, 0.05, 0.12]),
            np.array([-0.08, 1.008, 0.66, 0.08, 0.006, 0.89, -0.08]),
        ]

        xs = []
        ys = []
        for params in seeds:
            unit = to_unit(params)
            if xs and np.min(np.sum((np.asarray(xs) - unit) ** 2, axis=1)) < 1.0e-6:
                continue
            xs.append(unit)
            ys.append(init_objective(params))

        base_pool = deterministic_pool(144, 7)
        length_scale = np.array([0.20, 0.13, 0.24, 0.24, 0.28, 0.24, 0.30])
        jitter = 2.0e-6

        for step in range(7):
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            finite = np.isfinite(y_arr) & (y_arr < 1.0e7)
            if np.any(finite):
                y_mean = float(np.mean(y_arr[finite]))
                y_std = float(np.std(y_arr[finite])) + 1.0e-9
            else:
                y_mean = 0.0
                y_std = 1.0
            y_scaled = (y_arr - y_mean) / y_std

            diffs = (x_arr[:, None, :] - x_arr[None, :, :]) / length_scale
            k_mat = np.exp(-0.5 * np.sum(diffs * diffs, axis=2))
            k_mat += np.eye(len(x_arr)) * jitter
            try:
                chol = np.linalg.cholesky(k_mat)
                alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_scaled))
            except np.linalg.LinAlgError:
                best_i = int(np.argmin(y_arr))
                return clamp_params(from_unit(x_arr[best_i]))

            best_unit = x_arr[int(np.argmin(y_arr))]
            local = np.mod(
                best_unit[None, :]
                + (deterministic_pool(48, 7) - 0.5) * (0.32 - 0.025 * step),
                1.0,
            )
            pool = np.vstack([base_pool, local])
            sep = np.min(np.sum((pool[:, None, :] - x_arr[None, :, :]) ** 2, axis=2), axis=1)
            pool = pool[sep > 0.018]
            if len(pool) == 0:
                break

            k_star = np.exp(
                -0.5
                * np.sum(
                    ((pool[:, None, :] - x_arr[None, :, :]) / length_scale) ** 2,
                    axis=2,
                )
            )
            mu = k_star @ alpha
            v = np.linalg.solve(chol, k_star.T)
            var = np.maximum(1.0 - np.sum(v * v, axis=0), 1.0e-8)
            sigma = np.sqrt(var)
            explore = 1.35 - 0.07 * step
            acquisition = mu - explore * sigma
            next_unit = pool[int(np.argmin(acquisition))]
            xs.append(next_unit)
            ys.append(init_objective(from_unit(next_unit)))

        best_i = int(np.argmin(np.asarray(ys, dtype=float)))
        return clamp_params(from_unit(np.asarray(xs[best_i], dtype=float)))

    try:
        bo_params = gp_lcb_search()
    except Exception:
        bo_params = alternate_params

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
    bo_run = make_runner(88.0, 0.0045, 850, 2550, 0.34)
    polish_run = make_runner(18.0, 0.10, 80, 520, 0.18)

    def consider(best_x, best_y, best_obj, cand_z):
        x = cand_z[:n_target]
        y = cand_z[n_target:]
        if feasible(x, y):
            obj = float(objective_xy_jit(x, y))
            if obj < best_obj:
                return x, y, obj
        return best_x, best_y, best_obj

    inc_x, inc_y = hexagonal_lattice(incumbent_params)
    bo_x, bo_y = hexagonal_lattice(bo_params)
    if not feasible(bo_x, bo_y):
        bo_x, bo_y = hexagonal_lattice(alternate_params)

    best_x, best_y = inc_x, inc_y
    best_obj = float(objective_xy_jit(best_x, best_y)) if feasible(best_x, best_y) else 1.0e30
    if feasible(bo_x, bo_y):
        bo_obj = float(objective_xy_jit(bo_x, bo_y))
        if bo_obj < best_obj:
            best_x, best_y, best_obj = bo_x, bo_y, bo_obj

    z1 = main_run(jnp.concatenate([inc_x, inc_y]), 1.0)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z1)

    z2 = bo_run(jnp.concatenate([bo_x, bo_y]), 1.0)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z2)

    z3 = polish_run(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z3)
    return best_x, best_y
