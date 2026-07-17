"""K-means seeded genetic surplus-grid search with AL pruning.

HYPOTHESIS: Weighted kmeans clustering of wind sectors can add distinct
orientation anchors that the tight genetic_algorithm search missed, while
crossover and mutation retain the strong dense-lattice traits.
AXIS: init_kmeans wind-sector anchors plus genetic_algorithm crossover and
mutation population search with augmented-Lagrangian NAdam refinement.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
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

    def eval_unit(p):
        return float(get_aep_grid(scale_params(jnp.asarray(p))))

    rng = np.random.default_rng(20020)
    anchors = np.array(
        [
            [0.08, 0.08, 0.50, 0.50, 0.50, 0.50, 0.45],
            [0.06, 0.10, 0.46, 0.48, 0.52, 0.42, 0.36],
            [0.10, 0.06, 0.54, 0.52, 0.48, 0.58, 0.54],
            [0.12, 0.08, 0.50, 0.42, 0.58, 0.50, 0.45],
        ],
        dtype=float,
    )
    wd_np = np.asarray(wd, dtype=float)
    weights_np = np.asarray(weights, dtype=float)
    weights_np = weights_np / np.maximum(np.sum(weights_np), 1e-12)
    wind_pts = np.column_stack(
        [np.cos(np.deg2rad(wd_np)), np.sin(np.deg2rad(wd_np))]
    )
    seed_ids = np.argsort(weights_np)[-3:]
    kmeans_centers = wind_pts[seed_ids].copy()
    for _ in range(8):
        sims = wind_pts @ kmeans_centers.T
        labels = np.argmax(sims, axis=1)
        for k in range(3):
            mask = labels == k
            if np.any(mask):
                center = np.sum(wind_pts[mask] * weights_np[mask, None], axis=0)
                norm = np.linalg.norm(center)
                if norm > 1e-12:
                    kmeans_centers[k] = center / norm

    dom = float(dominant_wd)
    kmeans_params = []
    for center in kmeans_centers:
        angle = np.arctan2(center[1], center[0])
        offset = (angle - dom + np.pi) % (2.0 * np.pi) - np.pi
        offset = np.clip(offset, -np.pi / 4.0, np.pi / 4.0)
        theta_gene = 0.5 + offset / (np.pi / 2.0)
        kmeans_params.append([0.075, 0.085, theta_gene, 0.48, 0.52, 0.42, 0.40])
        kmeans_params.append([0.090, 0.065, theta_gene, 0.52, 0.48, 0.58, 0.52])
    anchors = np.vstack([anchors, np.asarray(kmeans_params, dtype=float)])
    mean = np.array([0.08, 0.08, 0.50, 0.50, 0.50, 0.50, 0.45])
    spread = np.array([0.050, 0.050, 0.090, 0.140, 0.140, 0.090, 0.050])
    history_pos = []
    history_val = []

    def eval_and_store(z):
        p = reflect_unit(np.asarray(z, dtype=float))
        value = eval_unit(p)
        history_pos.append(p.copy())
        history_val.append(value)
        return p, value

    population_size = 16
    elite_size = 6
    population = [anchor.copy() for anchor in anchors]
    wide_spread = np.array([0.25, 0.25, 0.45, 0.70, 0.70, 0.45, 0.25])
    cov = np.diag(wide_spread**2)
    chol = np.linalg.cholesky(cov + np.eye(7) * 1e-8)
    while len(population) < population_size:
        step = rng.normal(size=7) @ chol.T
        population.append(mean + 0.20 * step)
    population = np.array([reflect_unit(p) for p in population])

    values = np.full(population_size, -np.inf, dtype=float)
    for generation in range(8):
        for i in range(population_size):
            population[i], values[i] = eval_and_store(population[i])

        order = np.argsort(values)[::-1]
        elites = population[order[:elite_size]].copy()
        elite_values = values[order[:elite_size]].copy()

        next_population = [elite.copy() for elite in elites]
        mutation_scale = spread * (0.82 - 0.065 * generation)
        mutation_rate = np.array([0.62, 0.62, 0.54, 0.48, 0.48, 0.54, 0.42])

        while len(next_population) < population_size:
            if rng.random() < 0.12:
                parent = anchors[rng.integers(0, len(anchors))]
                child = parent + rng.normal(0.0, mutation_scale * 0.75, size=7)
            else:
                a, b = rng.choice(elite_size, size=2, replace=False)
                alpha = rng.uniform(-0.08, 1.08, size=7)
                child = alpha * elites[a] + (1.0 - alpha) * elites[b]
                mask = rng.random(7) < mutation_rate
                child += mask * rng.normal(0.0, mutation_scale, size=7)

            if rng.random() < 0.16:
                donor = anchors[rng.integers(0, len(anchors))]
                dims = rng.random(7) < 0.35
                child = np.where(dims, donor, child)
            next_population.append(reflect_unit(child))

        population = np.asarray(next_population)
        values = np.concatenate(
            [elite_values, np.full(population_size - elite_size, -np.inf)]
        )

    x_raw_np = np.vstack(history_pos)
    y_val_np = np.asarray(history_val)
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
