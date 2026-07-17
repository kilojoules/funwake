"""BO lattice pool with particle swarm basin expansion.

HYPOTHESIS: SHGO's topology sampling was feasible but too weak as a primary
lattice finder. A small pso / particle_swarm phase seeded from the strongest
BO lattice candidates can explore continuous spacing, shear, phase, and
rotation neighborhoods while preserving the known-good BO basin as fallback.

AXIS: BO-tuned exact-size grid plus pso lattice parameter expansion, staged
AL-NAdam halving, projection ensemble, strict projected Adam polish,
single-turbine sparse finish, row/band-level wind-frame moves, and a
nesterov_momentum projected finish.

LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.boundary import polygon_sdf


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

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    x_raw = jax.random.uniform(subkey, (12, 7))
    y_raw = jnp.array([grid_score(scale_params(p)) for p in x_raw])

    @jax.jit
    def gp_predict(x_test, x_train, y_train, length=0.3):
        def kernel(a, b):
            d = jnp.sqrt(jnp.sum((a - b) ** 2) + 1e-8)
            return (
                1.0
                + jnp.sqrt(5.0) * d / length
                + 5.0 * d**2 / (3.0 * length**2)
            ) * jnp.exp(-jnp.sqrt(5.0) * d / length)

        k = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(x_train))(x_train)
        k = k + jnp.eye(len(x_train)) * 1e-5
        chol = jnp.linalg.cholesky(k)
        ks = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(x_train))(x_test)
        alpha = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, y_train))
        mu = ks @ alpha
        v = jnp.linalg.solve(chol, ks.T)
        var = jax.vmap(lambda a: kernel(a, a))(x_test) - jnp.sum(v**2, axis=0)
        return mu, jnp.sqrt(jnp.maximum(1e-9, var))

    for _ in range(18):
        key, subkey = jax.random.split(key)
        cand = jax.random.uniform(subkey, (800, 7))
        mu, sig = gp_predict(cand, x_raw, y_raw)
        incumbent = jnp.max(y_raw)
        z = (mu - incumbent) / sig
        cdf = 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))
        pdf = jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)
        ei = (mu - incumbent) * cdf + sig * pdf
        nxt = cand[jnp.argmax(ei)]
        val = grid_score(scale_params(nxt))
        x_raw = jnp.vstack([x_raw, nxt])
        y_raw = jnp.append(y_raw, val)

    @jax.jit
    def score_params_batch(raw_params):
        return jax.vmap(lambda p: grid_score(scale_params(p)))(raw_params)

    n_particles = 14 if n_target <= 55 else 10
    n_pso_steps = 6 if n_target <= 55 else 4
    key, subkey = jax.random.split(key)
    pso_pos = jax.random.uniform(subkey, (n_particles, 7))
    seed_count = min(6, n_particles)
    seed_idx = jnp.argsort(y_raw)[-seed_count:][::-1]
    key, subkey = jax.random.split(key)
    seed_noise = 0.075 * jax.random.normal(subkey, (seed_count, 7))
    pso_pos = pso_pos.at[:seed_count].set(jnp.clip(x_raw[seed_idx] + seed_noise, 0.0, 1.0))
    pso_vel = jnp.zeros_like(pso_pos)
    pbest_pos = pso_pos
    pbest_val = score_params_batch(pso_pos)
    gbest = pbest_pos[jnp.argmax(pbest_val)]
    gbest_val = jnp.max(pbest_val)

    for i in range(n_pso_steps):
        key, sub1, sub2 = jax.random.split(key, 3)
        r1 = jax.random.uniform(sub1, pso_pos.shape)
        r2 = jax.random.uniform(sub2, pso_pos.shape)
        inertia = 0.68 - 0.035 * i
        pso_vel = (
            inertia * pso_vel
            + 1.35 * r1 * (pbest_pos - pso_pos)
            + 1.55 * r2 * (gbest[None, :] - pso_pos)
        )
        pso_vel = jnp.clip(pso_vel, -0.20, 0.20)
        pso_pos = jnp.clip(pso_pos + pso_vel, 0.0, 1.0)
        cur_val = score_params_batch(pso_pos)
        better = cur_val > pbest_val
        pbest_pos = jnp.where(better[:, None], pso_pos, pbest_pos)
        pbest_val = jnp.where(better, cur_val, pbest_val)
        local_best = jnp.argmax(pbest_val)
        gbest = pbest_pos[local_best]
        gbest_val = pbest_val[local_best]

    x_raw = jnp.vstack([x_raw, pbest_pos, gbest[None, :]])
    y_raw = jnp.concatenate([y_raw, pbest_val, gbest_val[None]])
    top_params = x_raw[jnp.argsort(y_raw)[-10:][::-1]]

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

    polish_x, polish_y = projected_adam_polish(best_x, best_y)
    single_x, single_y = single_turbine_finish(polish_x, polish_y)
    double_x, double_y = single_turbine_finish(single_x, single_y)
    row_x, row_y = row_band_finish(polish_x, polish_y)
    cascade_x, cascade_y = row_band_finish(double_x, double_y)
    nesterov_x, nesterov_y = nesterov_finish(cascade_x, cascade_y)
    cand_x = jnp.stack([best_x, polish_x, single_x, double_x, row_x, cascade_x, nesterov_x])
    cand_y = jnp.stack([best_y, polish_y, single_y, double_y, row_y, cascade_y, nesterov_y])
    scores = jax.vmap(aep_obj)(cand_x, cand_y)
    feas = jax.vmap(feasible_strict)(cand_x, cand_y)
    idx = jnp.argmin(jnp.where(feas, scores, scores + 1e9))
    return cand_x[idx], cand_y[idx]
