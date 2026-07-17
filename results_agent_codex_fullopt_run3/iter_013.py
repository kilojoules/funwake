"""SHGO lattice basin search with projected gradient wake-relief finish.

HYPOTHESIS: The BO lattice search has repeatedly returned the same polished
basin near 5584 GWh. A compact scipy_shgo pass over the normalized lattice
parameter box can sample the basin topology differently and may expose a
nearby high-clearance lattice that the surrogate expected-improvement loop
missed.

AXIS: scipy_shgo exact-size lattice parameter search plus staged AL-NAdam
halving, projection ensemble, strict projected Adam polish, single-turbine
sparse finish, row/band-level wind-frame moves, a nesterov_momentum projected
finish, and a very small scipy_basin_hopping affine polish.

LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.boundary import polygon_sdf
from scipy.optimize import basinhopping, shgo


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    @jax.jit
    def aep_obj(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    wd_rad = jnp.deg2rad(wd)
    vx = jnp.sum(jnp.cos(wd_rad) * weights)
    vy = jnp.sum(jnp.sin(wd_rad) * weights)
    dominant_wd = jnp.arctan2(vy, vx)

    @jax.jit
    def generate_grid(params):
        sx, sy, theta_off, ox, oy, shear, aspect = params
        theta = dominant_wd + theta_off
        sy_actual = sy * aspect
        n_side = int(np.sqrt(n_target)) + 15
        ii, jj = jnp.meshgrid(
            jnp.arange(n_side) - n_side // 2, jnp.arange(n_side) - n_side // 2
        )
        ix = ii.ravel()
        iy = jj.ravel()
        hx = ix * sx * min_spacing + (iy % 2) * sx * min_spacing * 0.5
        hy = iy * sy_actual * min_spacing * jnp.sqrt(3.0) * 0.5
        hx = hx + shear * hy
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        sdf = polygon_sdf(rx, ry, boundary)
        idx = jnp.argsort(sdf)[:n_target]
        return rx[idx], ry[idx]

    def grid_score(params):
        gx, gy = generate_grid(params)
        return -aep_obj(gx, gy)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    bounds_low = jnp.array([1.02, 1.02, -jnp.pi / 4, x_min, y_min, -0.6, 0.8])
    bounds_high = jnp.array([5.0, 5.0, jnp.pi / 4, x_max, y_max, 0.6, 1.3])

    def scale_params(p):
        return bounds_low + p * (bounds_high - bounds_low)

    def raw_lattice_objective(raw):
        raw = np.clip(np.asarray(raw, dtype=np.float64), 0.0, 1.0)
        return -float(grid_score(scale_params(jnp.asarray(raw))))

    seed_raw = np.array(
        [
            [0.02, 0.03, 0.50, 0.50, 0.50, 0.50, 0.20],
            [0.04, 0.06, 0.32, 0.43, 0.46, 0.43, 0.24],
            [0.05, 0.02, 0.69, 0.56, 0.52, 0.57, 0.23],
            [0.14, 0.10, 0.18, 0.45, 0.58, 0.50, 0.30],
            [0.10, 0.16, 0.84, 0.53, 0.42, 0.58, 0.36],
            [0.19, 0.05, 0.61, 0.48, 0.50, 0.63, 0.18],
            [0.06, 0.20, 0.39, 0.58, 0.48, 0.37, 0.35],
            [0.00, 0.00, 0.50, 0.47, 0.55, 0.50, 0.18],
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(1313)
    scored_raw = [(raw_lattice_objective(row), row.copy()) for row in seed_raw]
    for row in rng.random((8, 7)):
        scored_raw.append((raw_lattice_objective(row), row.copy()))

    shgo_n = 34 if n_target <= 55 else 24
    try:
        result = shgo(
            raw_lattice_objective,
            bounds=[(0.0, 1.0)] * 7,
            n=shgo_n,
            iters=1,
            sampling_method="simplicial",
            minimizer_kwargs={
                "method": "Nelder-Mead",
                "options": {
                    "maxiter": 14,
                    "xatol": 1e-3,
                    "fatol": 1e-2,
                    "disp": False,
                },
            },
            options={
                "maxfev": 125 if n_target <= 55 else 80,
                "maxev": 80 if n_target <= 55 else 55,
                "local_iter": 4,
                "minimize_every_iter": True,
            },
        )
        row = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
        scored_raw.append((raw_lattice_objective(row), row))
        try:
            local_rows = np.asarray(result.xl, dtype=np.float64)
        except Exception:
            local_rows = np.empty((0, 7), dtype=np.float64)
        for row in local_rows[:10]:
            row = np.clip(row, 0.0, 1.0)
            scored_raw.append((raw_lattice_objective(row), row))
    except Exception:
        pass

    top_rows = [row for _, row in sorted(scored_raw, key=lambda item: item[0])[:10]]
    top_params = jnp.asarray(top_rows)

    if n_target < 40:
        def quick_project(x, y):
            n_edges = boundary.shape[0]
            for _ in range(60):
                for e in range(n_edges):
                    x1, y1 = boundary[e]
                    x2, y2 = boundary[(e + 1) % n_edges]
                    ex = x2 - x1
                    ey = y2 - y1
                    el = jnp.sqrt(ex * ex + ey * ey) + 1e-12
                    nx = -ey / el
                    ny = ex / el
                    clearance = (x - x1) * nx + (y - y1) * ny
                    push = jnp.maximum(0.0, 0.12 - clearance)
                    x = x + push * nx
                    y = y + push * ny
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(n_target) * 1e18)
                force = jnp.maximum(0.0, min_spacing * 1.002 - dist)
                x = x + jnp.sum(force * dx / dist, axis=1) * 0.55
                y = y + jnp.sum(force * dy / dist, axis=1) * 0.55
            return x, y

        small_layouts = [quick_project(*generate_grid(scale_params(p))) for p in top_params[:4]]
        sx = jnp.stack([layout[0] for layout in small_layouts])
        sy = jnp.stack([layout[1] for layout in small_layouts])
        vals = jax.vmap(aep_obj)(sx, sy)

        def small_feasible(x, y):
            sdf = polygon_sdf(x, y, boundary)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(n_target) * 1e18)
            return (jnp.max(sdf) <= -1e-4) & (jnp.min(dist) >= min_spacing * 0.999)

        feas = jax.vmap(small_feasible)(sx, sy)
        idx = jnp.argmin(jnp.where(feas, vals, vals + 1e9))
        return sx[idx], sy[idx]

    @jax.jit
    def constraints_penalty(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        vb = jnp.maximum(0.0, sdf + 0.01)
        pen_b = jnp.sum(lam_b * vb + 0.5 * mu * vb**2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_target, n_target)), k=1)
        vs = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        pen_s = jnp.sum(mask * (lam_s * vs + 0.5 * mu * vs**2))
        return pen_b + pen_s

    @jax.jit
    def total_obj(x, y, lam_b, lam_s, mu):
        return aep_obj(x, y) + constraints_penalty(x, y, lam_b, lam_s, mu)

    grad_total = jax.jit(
        jax.vmap(jax.grad(total_obj, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None))
    )

    def run_al_nadam_batch(x, y, n_steps, lr, mu_init, mu_rate):
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx_acc = jnp.zeros_like(x)
        vy_acc = jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_target, n_target))
        mu = mu_init
        cx, cy = x, y
        n_inner = 5
        steps_per_inner = n_steps // n_inner
        lr_arr = jnp.asarray(lr)
        if lr_arr.ndim == 0:
            lr_arr = jnp.full((x.shape[0],), lr_arr)
        lr_arr = lr_arr[:, None]

        for i in range(n_inner):
            def step(carry, t):
                px, py, ax, ay, bx, by = carry
                gx, gy = grad_total(px, py, lam_b, lam_s, mu)
                b1, b2 = 0.9, 0.999
                ax = b1 * ax + (1.0 - b1) * gx
                ay = b1 * ay + (1.0 - b1) * gy
                bx = b2 * bx + (1.0 - b2) * gx**2
                by = b2 * by + (1.0 - b2) * gy**2
                ahx = (b1 * ax + (1.0 - b1) * gx) / (1.0 - b1 ** (t + 1))
                ahy = (b1 * ay + (1.0 - b1) * gy) / (1.0 - b1 ** (t + 1))
                vhx = bx / (1.0 - b2 ** (t + 1))
                vhy = by / (1.0 - b2 ** (t + 1))
                px = px - lr_arr * ahx / (jnp.sqrt(vhx) + 1e-8)
                py = py - lr_arr * ahy / (jnp.sqrt(vhy) + 1e-8)
                return (px, py, ax, ay, bx, by), None

            (cx, cy, mx, my, vx_acc, vy_acc), _ = jax.lax.scan(
                step,
                (cx, cy, mx, my, vx_acc, vy_acc),
                jnp.arange(i * steps_per_inner, (i + 1) * steps_per_inner),
            )
            sdf = jax.vmap(lambda px, py: polygon_sdf(px, py, boundary))(cx, cy)
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx = cx[:, :, None] - cx[:, None, :]
            dy = cy[:, :, None] - cy[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= mu_rate
        return cx, cy

    layouts = [generate_grid(scale_params(p)) for p in top_params]
    cur_x = jnp.stack([layout[0] for layout in layouts])
    cur_y = jnp.stack([layout[1] for layout in layouts])

    cur_x, cur_y = run_al_nadam_batch(cur_x, cur_y, 250, 15.0, 1e2, 2.5)
    idx = jnp.argsort(jax.vmap(aep_obj)(cur_x, cur_y))[:5]
    cur_x, cur_y = cur_x[idx], cur_y[idx]

    cur_x, cur_y = run_al_nadam_batch(cur_x, cur_y, 500, 8.0, 1e3, 3.0)
    idx = jnp.argsort(jax.vmap(aep_obj)(cur_x, cur_y))[:2]
    cur_x, cur_y = cur_x[idx], cur_y[idx]

    cur_x, cur_y = run_al_nadam_batch(cur_x, cur_y, 1000, 4.0, 1e4, 2.5)
    idx = jnp.argsort(jax.vmap(aep_obj)(cur_x, cur_y))[:2]
    base_x, base_y = cur_x[idx], cur_y[idx]
    cur_x = jnp.concatenate(
        [base_x[0:1], base_x[0:1], base_x[0:1], base_x[0:1], base_x[1:2], base_x[1:2]],
        axis=0,
    )
    cur_y = jnp.concatenate(
        [base_y[0:1], base_y[0:1], base_y[0:1], base_y[0:1], base_y[1:2], base_y[1:2]],
        axis=0,
    )
    lr_vec = jnp.array([1.4, 1.8, 2.0, 2.4, 1.6, 2.2])
    cur_x, cur_y = run_al_nadam_batch(cur_x, cur_y, 1200, lr_vec, 1e5, 2.0)
    best_idx = jnp.argmin(jax.vmap(aep_obj)(cur_x, cur_y))
    cur_x = cur_x[best_idx : best_idx + 1]
    cur_y = cur_y[best_idx : best_idx + 1]

    def project(x, y, n_steps, boundary_margin, spacing_margin, force_scale):
        for _ in range(n_steps):
            sdf = polygon_sdf(x, y, boundary)
            grad_b = jax.vmap(
                jax.grad(
                    lambda px, py: polygon_sdf(
                        jnp.array([px]), jnp.array([py]), boundary
                    )[0],
                    argnums=(0, 1),
                )
            )(x, y)
            x = x - jnp.maximum(0.0, sdf + boundary_margin) * grad_b[0]
            y = y - jnp.maximum(0.0, sdf + boundary_margin) * grad_b[1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            force = jnp.maximum(0.0, min_spacing * spacing_margin - dist)
            x = x + jnp.sum(force * (dx / dist), axis=1) * force_scale
            y = y + jnp.sum(force * (dy / dist), axis=1) * force_scale
        return x, y

    proj_specs = [
        (8, 0.004, 1.0004, 0.09),
        (12, 0.008, 1.0008, 0.11),
        (20, 0.020, 1.0020, 0.15),
        (30, 0.018, 1.0020, 0.10),
    ]
    def affine_transform(x, y, spec):
        tx, ty, rot, scale, shear = spec
        cx = jnp.mean(x)
        cy = jnp.mean(y)
        px = x - cx
        py = y - cy
        px = px * scale + shear * py
        py = py / scale
        rx = px * jnp.cos(rot) - py * jnp.sin(rot)
        ry = px * jnp.sin(rot) + py * jnp.cos(rot)
        return rx + cx + tx, ry + cy + ty

    affine_specs = [
        (0.0, 0.0, 0.0, 1.000, 0.000),
        (70.0, 0.0, 0.0, 1.000, 0.000),
        (-70.0, 0.0, 0.0, 1.000, 0.000),
        (0.0, 70.0, 0.0, 1.000, 0.000),
        (0.0, -70.0, 0.0, 1.000, 0.000),
        (0.0, 0.0, 0.012, 1.000, 0.000),
        (0.0, 0.0, -0.012, 1.000, 0.000),
        (0.0, 0.0, 0.0, 1.006, 0.000),
        (0.0, 0.0, 0.0, 0.994, 0.000),
        (0.0, 0.0, 0.0, 1.000, 0.010),
        (0.0, 0.0, 0.0, 1.000, -0.010),
    ]
    projected = []
    for affine_spec in affine_specs:
        ax, ay = affine_transform(cur_x[0], cur_y[0], affine_spec)
        for proj_spec in proj_specs:
            projected.append(project(ax, ay, *proj_spec))
    cand_x = jnp.stack([layout[0] for layout in projected])
    cand_y = jnp.stack([layout[1] for layout in projected])

    def feasible(x, y):
        sdf = polygon_sdf(x, y, boundary)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        dist = dist + jnp.eye(n_target) * 1e9
        return (jnp.max(sdf) <= 1e-4) & (jnp.min(dist) >= min_spacing * 0.9995)

    scores = jax.vmap(aep_obj)(cand_x, cand_y)
    feas = jax.vmap(feasible)(cand_x, cand_y)
    penalized = jnp.where(feas, scores, scores + 1e9)
    best_proj = jnp.argmin(penalized)
    best_x = cand_x[best_proj]
    best_y = cand_y[best_proj]

    def feasible_strict(x, y):
        sdf = polygon_sdf(x, y, boundary)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        dist = dist + jnp.eye(n_target) * 1e9
        return (jnp.max(sdf) <= -1e-5) & (jnp.min(dist) >= min_spacing * 1.00002)

    grad_aep_batch = jax.jit(jax.vmap(jax.grad(aep_obj, argnums=(0, 1))))

    def project_batch(x, y, n_steps, boundary_margin, spacing_margin, force_scale):
        for _ in range(n_steps):
            sdf = jax.vmap(lambda px, py: polygon_sdf(px, py, boundary))(x, y)
            grad_b = jax.vmap(
                jax.vmap(
                    jax.grad(
                        lambda px, py: polygon_sdf(
                            jnp.array([px]), jnp.array([py]), boundary
                        )[0],
                        argnums=(0, 1),
                    ),
                    in_axes=(0, 0),
                ),
                in_axes=(0, 0),
            )(x, y)
            x = x - jnp.maximum(0.0, sdf + boundary_margin) * grad_b[0]
            y = y - jnp.maximum(0.0, sdf + boundary_margin) * grad_b[1]
            dx = x[:, :, None] - x[:, None, :]
            dy = y[:, :, None] - y[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            force = jnp.maximum(0.0, min_spacing * spacing_margin - dist)
            x = x + jnp.sum(force * (dx / dist), axis=2) * force_scale
            y = y + jnp.sum(force * (dy / dist), axis=2) * force_scale
        return x, y

    def projected_adam_polish(seed_x, seed_y):
        starts = []
        for spec in [
            (0.0, 0.0, 0.0, 1.000, 0.000),
            (25.0, 0.0, 0.0, 1.000, 0.000),
            (-25.0, 0.0, 0.0, 1.000, 0.000),
            (0.0, 25.0, 0.0, 1.000, 0.000),
            (0.0, -25.0, 0.0, 1.000, 0.000),
            (0.0, 0.0, 0.006, 1.000, 0.000),
            (0.0, 0.0, -0.006, 1.000, 0.000),
            (0.0, 0.0, 0.0, 1.003, 0.000),
        ]:
            sx, sy = affine_transform(seed_x, seed_y, spec)
            starts.append(project(sx, sy, 14, 0.010, 1.0010, 0.11))

        x = jnp.stack([s[0] for s in starts])
        y = jnp.stack([s[1] for s in starts])
        lrs = jnp.array([0.8, 1.0, 1.2, 1.5, 1.8, 0.9, 1.1, 1.4])[:, None]
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx_acc = jnp.zeros_like(x)
        vy_acc = jnp.zeros_like(y)
        best_x_local = x
        best_y_local = y
        scores = jax.vmap(aep_obj)(x, y)
        feas = jax.vmap(feasible_strict)(x, y)
        best_scores = jnp.where(feas, scores, scores + 1e9)

        def step(carry, t):
            px, py, ax, ay, bx, by, bx_best, by_best, bs = carry
            gx, gy = grad_aep_batch(px, py)
            b1, b2 = 0.85, 0.995
            ax = b1 * ax + (1.0 - b1) * gx
            ay = b1 * ay + (1.0 - b1) * gy
            bx = b2 * bx + (1.0 - b2) * gx**2
            by = b2 * by + (1.0 - b2) * gy**2
            ax_hat = ax / (1.0 - b1 ** (t + 1))
            ay_hat = ay / (1.0 - b1 ** (t + 1))
            bx_hat = bx / (1.0 - b2 ** (t + 1))
            by_hat = by / (1.0 - b2 ** (t + 1))
            px = px - lrs * ax_hat / (jnp.sqrt(bx_hat) + 1e-8)
            py = py - lrs * ay_hat / (jnp.sqrt(by_hat) + 1e-8)
            px, py = project_batch(px, py, 3, 0.008, 1.0008, 0.10)
            cur_scores = jax.vmap(aep_obj)(px, py)
            cur_feas = jax.vmap(feasible_strict)(px, py)
            better = cur_feas & (cur_scores < bs)
            bx_best = jnp.where(better[:, None], px, bx_best)
            by_best = jnp.where(better[:, None], py, by_best)
            bs = jnp.where(better, cur_scores, bs)
            return (px, py, ax, ay, bx, by, bx_best, by_best, bs), None

        (x, y, mx, my, vx_acc, vy_acc, best_x_local, best_y_local, best_scores), _ = (
            jax.lax.scan(
                step,
                (x, y, mx, my, vx_acc, vy_acc, best_x_local, best_y_local, best_scores),
                jnp.arange(220),
            )
        )
        idx = jnp.argmin(best_scores)
        return best_x_local[idx], best_y_local[idx]

    def single_turbine_finish(seed_x, seed_y):
        ux = jnp.cos(dominant_wd)
        uy = jnp.sin(dominant_wd)
        vx = -uy
        vy = ux

        base_shifts = jnp.array(
            [
                [65.0, 0.0],
                [-65.0, 0.0],
                [0.0, 45.0],
                [0.0, -45.0],
            ]
        )
        idxs = jnp.repeat(jnp.arange(n_target), base_shifts.shape[0])
        shifts = jnp.tile(base_shifts, (n_target, 1))

        def transform_one(idx, shift):
            du_shift, dv_shift = shift
            mask = (jnp.arange(n_target) == idx).astype(seed_x.dtype)
            return (
                seed_x + mask * (du_shift * ux + dv_shift * vx),
                seed_y + mask * (du_shift * uy + dv_shift * vy),
            )

        raw_x, raw_y = jax.vmap(transform_one)(idxs, shifts)
        rep_x, rep_y = project_batch(raw_x, raw_y, 10, 0.008, 1.0008, 0.10)
        local_x = jnp.concatenate([seed_x[None, :], rep_x], axis=0)
        local_y = jnp.concatenate([seed_y[None, :], rep_y], axis=0)
        local_scores = jax.vmap(aep_obj)(local_x, local_y)
        local_feas = jax.vmap(feasible_strict)(local_x, local_y)
        local_idx = jnp.argmin(jnp.where(local_feas, local_scores, local_scores + 1e9))
        return local_x[local_idx], local_y[local_idx]

    def row_band_finish(seed_x, seed_y):
        ux = jnp.cos(dominant_wd)
        uy = jnp.sin(dominant_wd)
        vx = -uy
        vy = ux

        u = seed_x * ux + seed_y * uy
        v = seed_x * vx + seed_y * vy
        u_edges = jnp.quantile(u, jnp.array([0.0, 0.33, 0.66, 1.0]))
        v_edges = jnp.quantile(v, jnp.array([0.0, 0.33, 0.66, 1.0]))
        band_specs = jnp.array(
            [
                [0.0, u_edges[0] - 1.0, u_edges[1], 0.0],
                [0.0, u_edges[1], u_edges[2], 0.0],
                [0.0, u_edges[2], u_edges[3] + 1.0, 0.0],
                [1.0, v_edges[0] - 1.0, v_edges[1], 0.0],
                [1.0, v_edges[1], v_edges[2], 0.0],
                [1.0, v_edges[2], v_edges[3] + 1.0, 0.0],
            ]
        )
        move_specs = jnp.array(
            [
                [0.0, 38.0],
                [0.0, -38.0],
                [55.0, 0.0],
                [-55.0, 0.0],
                [45.0, 30.0],
                [45.0, -30.0],
                [-45.0, 30.0],
                [-45.0, -30.0],
                [80.0, 18.0],
                [-80.0, -18.0],
            ]
        )
        bands = jnp.repeat(band_specs, move_specs.shape[0], axis=0)
        moves = jnp.tile(move_specs, (band_specs.shape[0], 1))

        def transform_band(band, move):
            axis, lo, hi, _ = band
            du_shift, dv_shift = move
            coord = jnp.where(axis < 0.5, u, v)
            mask = ((coord >= lo) & (coord <= hi)).astype(seed_x.dtype)
            return (
                seed_x + mask * (du_shift * ux + dv_shift * vx),
                seed_y + mask * (du_shift * uy + dv_shift * vy),
            )

        raw_x, raw_y = jax.vmap(transform_band)(bands, moves)
        rep_x, rep_y = project_batch(raw_x, raw_y, 14, 0.008, 1.0008, 0.10)
        local_x = jnp.concatenate([seed_x[None, :], rep_x], axis=0)
        local_y = jnp.concatenate([seed_y[None, :], rep_y], axis=0)
        local_scores = jax.vmap(aep_obj)(local_x, local_y)
        local_feas = jax.vmap(feasible_strict)(local_x, local_y)
        local_idx = jnp.argmin(jnp.where(local_feas, local_scores, local_scores + 1e9))
        return local_x[local_idx], local_y[local_idx]

    def nesterov_finish(seed_x, seed_y):
        starts = []
        for spec in [
            (0.0, 0.0, 0.0, 1.000, 0.000),
            (18.0, 0.0, 0.0, 1.000, 0.000),
            (-18.0, 0.0, 0.0, 1.000, 0.000),
            (0.0, 18.0, 0.0, 1.000, 0.000),
            (0.0, -18.0, 0.0, 1.000, 0.000),
            (0.0, 0.0, 0.004, 1.000, 0.000),
        ]:
            sx, sy = affine_transform(seed_x, seed_y, spec)
            starts.append(project(sx, sy, 10, 0.008, 1.0008, 0.10))

        x = jnp.stack([s[0] for s in starts])
        y = jnp.stack([s[1] for s in starts])
        vx_m = jnp.zeros_like(x)
        vy_m = jnp.zeros_like(y)
        lrs = jnp.array([0.35, 0.45, 0.55, 0.70, 0.85, 1.05])[:, None]
        momentum = 0.88
        best_x_local = x
        best_y_local = y
        scores = jax.vmap(aep_obj)(x, y)
        feas = jax.vmap(feasible_strict)(x, y)
        best_scores = jnp.where(feas, scores, scores + 1e9)

        def step(carry, _):
            px, py, vx_cur, vy_cur, bx_best, by_best, bs = carry
            look_x = px + momentum * vx_cur
            look_y = py + momentum * vy_cur
            gx, gy = grad_aep_batch(look_x, look_y)
            vx_cur = momentum * vx_cur - lrs * gx
            vy_cur = momentum * vy_cur - lrs * gy
            px = px + vx_cur
            py = py + vy_cur
            px, py = project_batch(px, py, 2, 0.008, 1.0008, 0.09)
            cur_scores = jax.vmap(aep_obj)(px, py)
            cur_feas = jax.vmap(feasible_strict)(px, py)
            better = cur_feas & (cur_scores < bs)
            bx_best = jnp.where(better[:, None], px, bx_best)
            by_best = jnp.where(better[:, None], py, by_best)
            bs = jnp.where(better, cur_scores, bs)
            return (px, py, vx_cur, vy_cur, bx_best, by_best, bs), None

        (x, y, vx_m, vy_m, best_x_local, best_y_local, best_scores), _ = jax.lax.scan(
            step,
            (x, y, vx_m, vy_m, best_x_local, best_y_local, best_scores),
            jnp.arange(160),
        )
        idx = jnp.argmin(best_scores)
        return best_x_local[idx], best_y_local[idx]

    def basin_hopping_finish(seed_x, seed_y):
        def layout_from_z(z):
            z = np.clip(np.asarray(z, dtype=np.float64), -1.0, 1.0)
            spec = (
                34.0 * z[0],
                34.0 * z[1],
                0.0055 * z[2],
                1.0 + 0.0035 * z[3],
                0.0070 * z[4],
            )
            bx, by = affine_transform(seed_x, seed_y, spec)
            return project(bx, by, 10, 0.008, 1.0008, 0.09)

        def scipy_score(z):
            bx, by = layout_from_z(z)
            score = aep_obj(bx, by)
            ok = feasible_strict(bx, by)
            return float(jnp.where(ok, score, score + 1e6))

        result = basinhopping(
            scipy_score,
            np.zeros(5, dtype=np.float64),
            niter=1,
            stepsize=0.65,
            minimizer_kwargs={
                "method": "Nelder-Mead",
                "options": {
                    "maxiter": 4,
                    "xatol": 1e-3,
                    "fatol": 1e-3,
                    "disp": False,
                },
            },
            seed=1217,
            disp=False,
        )
        return layout_from_z(result.x)

    polish_x, polish_y = projected_adam_polish(best_x, best_y)
    single_x, single_y = single_turbine_finish(polish_x, polish_y)
    double_x, double_y = single_turbine_finish(single_x, single_y)
    row_x, row_y = row_band_finish(polish_x, polish_y)
    cascade_x, cascade_y = row_band_finish(double_x, double_y)
    nesterov_x, nesterov_y = nesterov_finish(cascade_x, cascade_y)
    basin_x, basin_y = basin_hopping_finish(nesterov_x, nesterov_y)
    cand_x = jnp.stack(
        [best_x, polish_x, single_x, double_x, row_x, cascade_x, nesterov_x, basin_x]
    )
    cand_y = jnp.stack(
        [best_y, polish_y, single_y, double_y, row_y, cascade_y, nesterov_y, basin_y]
    )
    scores = jax.vmap(aep_obj)(cand_x, cand_y)
    feas = jax.vmap(feasible_strict)(cand_x, cand_y)
    idx = jnp.argmin(jnp.where(feas, scores, scores + 1e9))
    return cand_x[idx], cand_y[idx]
