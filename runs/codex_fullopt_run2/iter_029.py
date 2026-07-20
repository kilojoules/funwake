"""CMA-trained skopt-style BO surplus-grid search with NAdam pruning.

HYPOTHESIS: The incumbent CMA basin discovery is strong, but its sampled
history is also useful training data for a skopt-like BayesianOptimization
expected-improvement pass that can probe the best basin without abandoning the
known-good pruning and feasibility-multiplier NAdam polish.
AXIS: cmaes history plus skopt BayesianOptimization-style GP refinement over
surplus-grid parameters before NAdam surplus pruning.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erf
from pixwake.optim.boundary import polygon_sdf


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    n_extra = 8
    n_total = n_target + n_extra

    @jax.jit
    def aep_obj_total(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()
        aep = jnp.sum(p * weights[:, None]) * 8760.0 / 1e6
        return -aep

    wd_rad = jnp.deg2rad(wd)
    v_x = jnp.sum(jnp.cos(wd_rad) * weights)
    v_y = jnp.sum(jnp.sin(wd_rad) * weights)
    dominant_wd = jnp.arctan2(v_y, v_x)

    @jax.jit
    def generate_grid_surplus(params):
        sx, sy, theta_off, ox, oy, shear, aspect = params
        theta = dominant_wd + theta_off
        sy_actual = sy * aspect
        n_side = int(np.sqrt(n_total)) + 15
        ii, jj = jnp.meshgrid(
            jnp.arange(n_side) - n_side // 2, jnp.arange(n_side) - n_side // 2
        )
        ix = ii.flatten()
        iy = jj.flatten()
        hx = ix * sx * min_spacing + (iy % 2) * sx * min_spacing / 2.0
        hy = iy * sy_actual * min_spacing * jnp.sqrt(3.0) / 2.0
        hx = hx + shear * hy
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        sdf_vals = polygon_sdf(rx, ry, boundary)
        idx = jnp.argsort(sdf_vals)[:n_total]
        return rx[idx], ry[idx]

    def get_aep_grid(params):
        gx, gy = generate_grid_surplus(params)
        return -aep_obj_total(gx, gy)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    bounds_low = jnp.array([1.02, 1.02, -jnp.pi / 4.0, x_min, y_min, -0.5, 0.85])
    bounds_high = jnp.array([4.5, 4.5, jnp.pi / 4.0, x_max, y_max, 0.5, 1.2])

    def scale_params(p):
        return bounds_low + p * (bounds_high - bounds_low)

    def reflect_unit(z):
        z = np.where(z < 0.0, -z, z)
        z = np.where(z > 1.0, 2.0 - z, z)
        return np.clip(z, 0.0, 1.0)

    rng = np.random.default_rng(20020)
    dim = 7
    pop_size = 14
    elite_size = 6
    recomb = np.log(elite_size + 0.5) - np.log(np.arange(1, elite_size + 1))
    recomb = recomb / np.sum(recomb)

    mean = np.array([0.08, 0.08, 0.50, 0.50, 0.50, 0.50, 0.45])
    mean = reflect_unit(mean + rng.normal(0.0, 0.025, size=dim))
    sigma = 0.20
    cov = np.diag(np.array([0.25, 0.25, 0.45, 0.70, 0.70, 0.45, 0.25]) ** 2)
    best_value = -np.inf
    history_pos = []
    history_val = []

    anchors = np.array(
        [
            [0.08, 0.08, 0.50, 0.50, 0.50, 0.50, 0.45],
            [0.06, 0.10, 0.46, 0.48, 0.52, 0.42, 0.36],
            [0.10, 0.06, 0.54, 0.52, 0.48, 0.58, 0.54],
            [0.12, 0.08, 0.50, 0.42, 0.58, 0.50, 0.45],
        ],
        dtype=float,
    )

    for generation in range(7):
        if generation == 0:
            samples = anchors.copy()
            needed = pop_size - len(samples)
            chol = np.linalg.cholesky(cov + np.eye(dim) * 1e-8)
            random_steps = rng.normal(size=(needed, dim)) @ chol.T
            samples = np.vstack([samples, reflect_unit(mean + sigma * random_steps)])
        else:
            cov = (cov + cov.T) * 0.5 + np.eye(dim) * 1e-9
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-5, 1.5)
            transform = eigvecs @ np.diag(np.sqrt(eigvals))
            steps = rng.normal(size=(pop_size, dim)) @ transform.T
            samples = reflect_unit(mean + sigma * steps)

        values = np.array(
            [float(get_aep_grid(scale_params(jnp.asarray(p)))) for p in samples]
        )
        history_pos.append(samples.copy())
        history_val.append(values.copy())

        order = np.argsort(values)[::-1]
        elites = samples[order[:elite_size]]
        old_mean = mean.copy()
        mean = np.sum(elites * recomb[:, None], axis=0)
        centered = (elites - old_mean) / max(sigma, 1e-6)
        cov_update = np.zeros((dim, dim), dtype=float)
        for w, step in zip(recomb, centered):
            cov_update += w * np.outer(step, step)
        cov = 0.72 * cov + 0.28 * cov_update + np.eye(dim) * 1e-5

        gen_best = values[order[0]]
        if gen_best > best_value + 1e-6:
            sigma = min(0.30, sigma * 1.10)
            best_value = gen_best
        else:
            sigma = max(0.045, sigma * 0.82)

    def gp_expected_improvement(candidates, train_x, train_y, length):
        y_mean = np.mean(train_y)
        y_std = np.std(train_y) + 1e-9
        y_norm = (train_y - y_mean) / y_std
        diff = train_x[:, None, :] - train_x[None, :, :]
        k_train = np.exp(-0.5 * np.sum(diff * diff, axis=2) / (length * length))
        k_train += np.eye(train_x.shape[0]) * 3e-5
        try:
            chol = np.linalg.cholesky(k_train)
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_norm))
            diff_s = candidates[:, None, :] - train_x[None, :, :]
            k_s = np.exp(-0.5 * np.sum(diff_s * diff_s, axis=2) / (length * length))
            mu = k_s @ alpha
            v = np.linalg.solve(chol, k_s.T)
            sigma = np.sqrt(np.maximum(1e-10, 1.0 - np.sum(v * v, axis=0)))
        except np.linalg.LinAlgError:
            mu = np.zeros(candidates.shape[0])
            sigma = np.ones(candidates.shape[0])
        f_best = np.max(y_norm)
        z = (mu - f_best - 0.003) / sigma
        cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        return (mu - f_best - 0.003) * cdf + sigma * pdf

    x_train_np = np.vstack(history_pos)
    y_train_np = np.concatenate(history_val)
    best_train = x_train_np[np.argmax(y_train_np)]
    for i in range(10):
        radius = max(0.045, 0.16 * (0.91**i))
        local = best_train + rng.normal(
            0.0,
            radius * np.array([0.70, 0.70, 1.10, 1.35, 1.35, 1.10, 0.70]),
            size=(520, dim),
        )
        global_samples = mean + rng.normal(
            0.0,
            np.array([0.18, 0.18, 0.32, 0.55, 0.55, 0.32, 0.18]),
            size=(160, dim),
        )
        candidates = reflect_unit(np.vstack([local, global_samples, anchors]))
        acquisition = gp_expected_improvement(candidates, x_train_np, y_train_np, 0.18)
        nearest = np.min(
            np.sum((candidates[:, None, :] - x_train_np[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        acquisition = acquisition + 0.004 * np.sqrt(nearest)
        for pick in np.argsort(acquisition)[::-1]:
            if nearest[pick] > 6e-5:
                x_next = candidates[pick]
                break
        else:
            x_next = reflect_unit(best_train + rng.normal(0.0, radius, size=dim))
        y_next = float(get_aep_grid(scale_params(jnp.asarray(x_next))))
        x_train_np = np.vstack([x_train_np, x_next])
        y_train_np = np.append(y_train_np, y_next)
        if y_next > np.max(y_train_np[:-1]):
            best_train = x_next

    x_raw_np = x_train_np
    y_val_np = y_train_np
    top_indices = np.argsort(y_val_np)[-8:][::-1]
    top_params = jnp.asarray(x_raw_np[top_indices])

    @jax.jit
    def constraints_penalty(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        v_b = jnp.maximum(0.0, sdf + 0.01)
        pen_b = jnp.sum(lam_b * v_b + 0.5 * mu * v_b**2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_total, n_total)), k=1)
        v_s = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        pen_s = jnp.sum(mask * (lam_s * v_s + 0.5 * mu * v_s**2))
        return pen_b + pen_s

    @jax.jit
    def total_obj(x, y, lam_b, lam_s, mu):
        return aep_obj_total(x, y) + constraints_penalty(x, y, lam_b, lam_s, mu)

    grad_total = jax.jit(
        jax.vmap(jax.grad(total_obj, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None))
    )

    def run_al_nadam_surplus(x, y, lr):
        m_x = jnp.zeros_like(x)
        m_y = jnp.zeros_like(y)
        v_x2 = jnp.zeros_like(x)
        v_y2 = jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_total, n_total))
        mu = 100.0
        curr_x = x
        curr_y = y

        for i in range(4):

            def step(carry, t):
                cx, cy, mx, my, vx2, vy2 = carry
                gx, gy = grad_total(cx, cy, lam_b, lam_s, mu)
                b1 = 0.9
                b2 = 0.999
                mx = b1 * mx + (1.0 - b1) * gx
                my = b1 * my + (1.0 - b1) * gy
                vx2 = b2 * vx2 + (1.0 - b2) * gx**2
                vy2 = b2 * vy2 + (1.0 - b2) * gy**2
                mx_h = (b1 * mx + (1.0 - b1) * gx) / (1.0 - b1 ** (t + 1))
                my_h = (b1 * my + (1.0 - b1) * gy) / (1.0 - b1 ** (t + 1))
                vx_h = vx2 / (1.0 - b2 ** (t + 1))
                vy_h = vy2 / (1.0 - b2 ** (t + 1))
                nx = cx - lr * mx_h / (jnp.sqrt(vx_h) + 1e-8)
                ny = cy - lr * my_h / (jnp.sqrt(vy_h) + 1e-8)
                return (nx, ny, mx, my, vx2, vy2), None

            init_carry = (curr_x, curr_y, m_x, m_y, v_x2, v_y2)
            (curr_x, curr_y, m_x, m_y, v_x2, v_y2), _ = jax.lax.scan(
                step, init_carry, jnp.arange(i * 200, (i + 1) * 200)
            )
            sdf = jax.vmap(lambda px, py: polygon_sdf(px, py, boundary))(curr_x, curr_y)
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx = curr_x[:, :, None] - curr_x[:, None, :]
            dy = curr_y[:, :, None] - curr_y[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= 2.0
        return curr_x, curr_y

    layouts = [generate_grid_surplus(scale_params(p)) for p in top_params]
    cur_x = jnp.stack([layout[0] for layout in layouts])
    cur_y = jnp.stack([layout[1] for layout in layouts])
    cur_x, cur_y = run_al_nadam_surplus(cur_x, cur_y, 12.0)

    def prune(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()
        indiv_aep = jnp.sum(p * weights[:, None], axis=0)
        idx = jnp.argsort(indiv_aep)[-n_target:]
        return x[idx], y[idx]

    pruned = [prune(cur_x[i], cur_y[i]) for i in range(len(cur_x))]
    px = jnp.stack([layout[0] for layout in pruned])
    py = jnp.stack([layout[1] for layout in pruned])

    @jax.jit
    def aep_obj_final(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        aep = jnp.sum(p * weights[:, None]) * 8760.0 / 1e6
        return -aep

    @jax.jit
    def total_obj_final(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        v_b = jnp.maximum(0.0, sdf + 0.01)
        pen_b = jnp.sum(lam_b * v_b + 0.5 * mu * v_b**2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_target, n_target)), k=1)
        v_s = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        pen_s = jnp.sum(mask * (lam_s * v_s + 0.5 * mu * v_s**2))
        return aep_obj_final(x, y) + pen_b + pen_s

    grad_final = jax.jit(
        jax.vmap(jax.grad(total_obj_final, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None))
    )

    def run_al_nadam_final(x, y, lr):
        m_x = jnp.zeros_like(x)
        m_y = jnp.zeros_like(y)
        v_x2 = jnp.zeros_like(x)
        v_y2 = jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_target, n_target))
        mu = 1000.0
        curr_x = x
        curr_y = y

        for i in range(3):

            def step(carry, t):
                cx, cy, mx, my, vx2, vy2 = carry
                gx, gy = grad_final(cx, cy, lam_b, lam_s, mu)
                b1 = 0.9
                b2 = 0.999
                mx = b1 * mx + (1.0 - b1) * gx
                my = b1 * my + (1.0 - b1) * gy
                vx2 = b2 * vx2 + (1.0 - b2) * gx**2
                vy2 = b2 * vy2 + (1.0 - b2) * gy**2
                mx_h = (b1 * mx + (1.0 - b1) * gx) / (1.0 - b1 ** (t + 1))
                my_h = (b1 * my + (1.0 - b1) * gy) / (1.0 - b1 ** (t + 1))
                vx_h = vx2 / (1.0 - b2 ** (t + 1))
                vy_h = vy2 / (1.0 - b2 ** (t + 1))
                nx = cx - lr * mx_h / (jnp.sqrt(vx_h) + 1e-8)
                ny = cy - lr * my_h / (jnp.sqrt(vy_h) + 1e-8)
                return (nx, ny, mx, my, vx2, vy2), None

            init_carry = (curr_x, curr_y, m_x, m_y, v_x2, v_y2)
            (curr_x, curr_y, m_x, m_y, v_x2, v_y2), _ = jax.lax.scan(
                step, init_carry, jnp.arange(i * 400, (i + 1) * 400)
            )
            sdf = jax.vmap(lambda pxx, pyy: polygon_sdf(pxx, pyy, boundary))(
                curr_x, curr_y
            )
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx = curr_x[:, :, None] - curr_x[:, None, :]
            dy = curr_y[:, :, None] - curr_y[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= 3.0
        return curr_x, curr_y

    e = jax.vmap(aep_obj_final)(px, py)
    idx = jnp.argsort(e)[:2]
    fx, fy = run_al_nadam_final(px[idx], py[idx], 4.0)
    best_idx = jnp.argmin(jax.vmap(aep_obj_final)(fx, fy))
    final_x = fx[best_idx]
    final_y = fy[best_idx]

    def project(x, y):
        for _ in range(25):
            sdf = polygon_sdf(x, y, boundary)
            grad_b = jax.vmap(
                jax.grad(
                    lambda px0, py0: polygon_sdf(
                        jnp.array([px0]), jnp.array([py0]), boundary
                    )[0],
                    argnums=(0, 1),
                )
            )(x, y)
            x = x - jnp.maximum(0.0, sdf + 0.02) * grad_b[0]
            y = y - jnp.maximum(0.0, sdf + 0.02) * grad_b[1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            force = jnp.maximum(0.0, min_spacing * 1.002 - dist)
            x = x + jnp.sum(force * (dx / dist), axis=1) * 0.15
            y = y + jnp.sum(force * (dy / dist), axis=1) * 0.15
        return x, y

    return project(final_x, final_y)
