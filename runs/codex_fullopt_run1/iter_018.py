"""Compact cmaes search over a richer wind-aware hex lattice seed.

HYPOTHESIS: cmaes can learn coupled angle/phase/shear/ranking moves in the
low-dimensional lattice space that PSO did not sample, while incumbent
retention preserves the known 5560.8 GWh basin if the covariance search is
only neutral.

AXIS: full-covariance cmaes over six hex-lattice seed parameters, followed by
the established nesterov_momentum penalty polish on the best seed and one
incumbent start.

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

    def hexagonal_lattice(angle, spacing_mult, phase_x, phase_y, skew, rank_shift):
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
        gx = rx_min - 1.5 * spacing + spacing * phase_x + spacing * jnp.arange(nx)
        gy = ry_min - 1.5 * row_step + row_step * phase_y + row_step * jnp.arange(ny)
        xx, yy = jnp.meshgrid(gx, gy)
        row = jnp.arange(ny)[:, None]
        xx = xx + (row % 2) * 0.5 * spacing + skew * (yy - center[1])
        pts_r = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        pts = (inv_rot @ pts_r.T).T + center
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        if len(px) >= n_target:
            rank_angle = angle + rank_shift
            rank = (
                (px - center[0]) * jnp.cos(rank_angle)
                + (py - center[1]) * jnp.sin(rank_angle)
                + 0.07 * min_spacing * jnp.sin(px / (2.9 * min_spacing) + phase_y)
                + 0.05 * min_spacing * jnp.cos(py / (3.3 * min_spacing) + phase_x)
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

    lo = np.array([-0.50, 0.994, 0.00, 0.00, -0.022, -0.10], dtype=float)
    hi = np.array([0.50, 1.016, 1.00, 1.00, 0.022, 0.85], dtype=float)

    def unpack(unit_params):
        u = np.minimum(np.maximum(np.asarray(unit_params, dtype=float), 0.0), 1.0)
        return lo + u * (hi - lo)

    def lattice_from_unit(unit_params):
        p = unpack(unit_params)
        return hexagonal_lattice(
            base_angle + float(p[0]),
            float(p[1]),
            float(p[2]),
            float(p[3]),
            float(p[4]),
            float(p[5]),
        )

    def seed_value(unit_params):
        x, y = lattice_from_unit(unit_params)
        if not feasible(x, y):
            return 1.0e8
        return float(objective_xy_jit(x, y))

    def to_unit(params):
        return np.minimum(np.maximum((np.asarray(params, dtype=float) - lo) / (hi - lo), 0.0), 1.0)

    def cmaes_seed():
        dim = 6
        known_a = to_unit([0.00, 1.000, 0.24, 0.65, 0.000, 0.31])
        known_b = to_unit([0.16, 1.006, 0.58, 0.99, 0.015, 0.31])
        mean = 0.55 * known_a + 0.45 * known_b
        sigma = 0.23
        cov = np.eye(dim, dtype=float)
        lam = 8
        mu = 4
        weights_mu = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        weights_mu = weights_mu / np.sum(weights_mu)
        best = known_a.copy()
        best_val = seed_value(best)
        for candidate in (known_b, mean):
            value = seed_value(candidate)
            if value < best_val:
                best = candidate.copy()
                best_val = value

        rng = np.random.default_rng(18)
        for gen in range(4):
            vals = []
            samples = []
            eigval, eigvec = np.linalg.eigh(cov)
            eigval = np.maximum(eigval, 1e-8)
            transform = eigvec @ np.diag(np.sqrt(eigval))
            for _ in range(lam):
                step = transform @ rng.standard_normal(dim)
                u = np.minimum(np.maximum(mean + sigma * step, 0.0), 1.0)
                vals.append(seed_value(u))
                samples.append(u)
            order = np.argsort(np.asarray(vals))
            elites = np.asarray([samples[int(i)] for i in order[:mu]])
            elite_vals = np.asarray([vals[int(i)] for i in order[:mu]])
            if float(elite_vals[0]) < best_val:
                best = elites[0].copy()
                best_val = float(elite_vals[0])

            old_mean = mean.copy()
            mean = np.sum(elites * weights_mu[:, None], axis=0)
            centered = elites - old_mean
            cov_update = np.zeros_like(cov)
            for w, c in zip(weights_mu, centered):
                cov_update += w * np.outer(c, c)
            cov = 0.72 * cov + 0.28 * cov_update / max(sigma * sigma, 1e-8)
            cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(dim)
            sigma *= 0.86 if gen >= 1 else 0.94
        return best

    cma_unit = cmaes_seed()

    def compute_mid(lr0, gamma_min, n_steps):
        lo_mid = 0.0
        hi_mid = 0.1
        for _ in range(80):
            mid = 0.5 * (lo_mid + hi_mid)
            lr = lr0
            for t in range(1, n_steps + 1):
                lr *= 1.0 / (1.0 + mid * t)
            if lr < gamma_min:
                hi_mid = mid
            else:
                lo_mid = mid
        return 0.5 * (lo_mid + hi_mid)

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
    cma_run = make_runner(90.0, 0.0040, 1000, 2800, 0.35)
    polish_run = make_runner(18.0, 0.10, 80, 520, 0.18)

    def consider(best_x, best_y, best_obj, cand_z):
        x = cand_z[:n_target]
        y = cand_z[n_target:]
        if feasible(x, y):
            obj = float(objective_xy_jit(x, y))
            if obj < best_obj:
                return x, y, obj
        return best_x, best_y, best_obj

    start0 = hexagonal_lattice(base_angle, 1.000, 0.24, 0.65, 0.000, 0.31)
    start1 = hexagonal_lattice(base_angle + 0.16, 1.006, 0.58, 0.99, 0.015, 0.31)
    start2 = lattice_from_unit(cma_unit)

    best_x, best_y = start0
    best_obj = float(objective_xy_jit(best_x, best_y)) if feasible(best_x, best_y) else 1.0e30
    for init_x, init_y in (start0, start1, start2):
        if feasible(init_x, init_y):
            init_obj = float(objective_xy_jit(init_x, init_y))
            if init_obj < best_obj:
                best_x, best_y, best_obj = init_x, init_y, init_obj

    for init_x, init_y, runner in ((start0[0], start0[1], main_run), (start2[0], start2[1], cma_run)):
        z = runner(jnp.concatenate([init_x, init_y]), 1.0)
        best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)

    z = polish_run(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)
    return best_x, best_y
