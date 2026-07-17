"""TopFarm SGD with feasible farthest-point starts.

HYPOTHESIS: Feasible, well-spread interior starts give the local TopFarm-style SGD
solver better basins than a sparse row-major grid while preserving ROWP feasibility.
AXIS: topfarm_sgd_solve plus deterministic farthest-point initialization.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    n_verts = boundary.shape[0]

    def edge_clearance(px, py):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (px - x1) * (-ey / el) + (py - y1) * (ex / el)

        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0)

    def candidate_cloud(step_factor, margin_factor):
        step = min_spacing * step_factor
        margin = min_spacing * margin_factor
        nx = int(jnp.maximum(3, jnp.ceil((x_max - x_min) / step)))
        ny = int(jnp.maximum(3, jnp.ceil((y_max - y_min) / step)))
        gx, gy = jnp.meshgrid(
            jnp.linspace(x_min + margin, x_max - margin, nx),
            jnp.linspace(y_min + margin, y_max - margin, ny),
        )
        cand_x = gx.flatten()
        cand_y = gy.flatten()
        inside = edge_clearance(cand_x, cand_y) > margin * 0.15
        return cand_x[inside], cand_y[inside]

    def farthest_init(cand_x, cand_y, seed_mode):
        n_cand = len(cand_x)
        if n_cand < n_target:
            key = jax.random.PRNGKey(seed_mode + 17)
            key, kx = jax.random.split(key)
            key, ky = jax.random.split(key)
            return (
                jax.random.uniform(kx, (n_target,), minval=float(x_min), maxval=float(x_max)),
                jax.random.uniform(ky, (n_target,), minval=float(y_min), maxval=float(y_max)),
            )

        cx = jnp.mean(boundary[:, 0])
        cy = jnp.mean(boundary[:, 1])

        # Use wind-energy weighted direction to vary which edge seeds the packing.
        w_energy = weights * ws**3
        dom = wd[jnp.argmax(w_energy)] * jnp.pi / 180.0
        wx = jnp.sin(dom)
        wy = jnp.cos(dom)
        proj = (cand_x - cx) * wx + (cand_y - cy) * wy
        radial = (cand_x - cx) ** 2 + (cand_y - cy) ** 2

        if seed_mode == 0:
            first_idx = jnp.argmax(radial)
        elif seed_mode == 1:
            first_idx = jnp.argmin(proj)
        else:
            first_idx = jnp.argmax(proj)

        init_x = jnp.zeros(n_target)
        init_y = jnp.zeros(n_target)
        init_x = init_x.at[0].set(cand_x[first_idx])
        init_y = init_y.at[0].set(cand_y[first_idx])
        best_dist2 = (cand_x - init_x[0]) ** 2 + (cand_y - init_y[0]) ** 2

        for i in range(1, n_target):
            valid = best_dist2 >= (min_spacing * 1.01) ** 2
            score = jnp.where(valid, best_dist2, -1.0)
            next_idx = jnp.argmax(score)
            # If the candidate grid is too sparse for strict spacing, keep spreading
            # rather than collapsing to a duplicate point.
            next_idx = jnp.where(jnp.max(score) > 0.0, next_idx, jnp.argmax(best_dist2))
            init_x = init_x.at[i].set(cand_x[next_idx])
            init_y = init_y.at[i].set(cand_y[next_idx])
            dist2_new = (cand_x - init_x[i]) ** 2 + (cand_y - init_y[i]) ** 2
            best_dist2 = jnp.minimum(best_dist2, dist2_new)

        return init_x, init_y

    def regular_grid_init(cand_x, cand_y):
        if len(cand_x) >= n_target:
            order = jnp.lexsort((cand_x, cand_y))
            sx = cand_x[order]
            sy = cand_y[order]
            idx = jnp.round(jnp.linspace(0, len(sx) - 1, n_target)).astype(int)
            return sx[idx], sy[idx]
        return farthest_init(cand_x, cand_y, 0)

    def min_distance(x, y):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(len(x)) * 1e12)
        return jnp.min(dist)

    def feasible(x, y):
        return (
            (boundary_penalty(x, y, boundary) < 1e-3)
            & (spacing_penalty(x, y, min_spacing) < 1e-3)
            & (min_distance(x, y) >= min_spacing * 0.99)
        )

    cand_x, cand_y = candidate_cloud(0.46, 0.10)
    init0_x, init0_y = regular_grid_init(cand_x, cand_y)
    init1_x, init1_y = farthest_init(cand_x, cand_y, 1)
    init2_x, init2_y = farthest_init(cand_x, cand_y, 2)

    settings = SGDSettings(
        learning_rate=55.0,
        max_iter=3600,
        additional_constant_lr_iterations=1200,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.01,
        ks_rho=100.0,
        spacing_weight=1.6,
        boundary_weight=1.6,
    )

    best_x = init0_x
    best_y = init0_y
    best_aep = jnp.where(feasible(best_x, best_y), -objective(best_x, best_y), -jnp.inf)

    for init_x, init_y in ((init0_x, init0_y), (init1_x, init1_y), (init2_x, init2_y)):
        opt_x, opt_y = topfarm_sgd_solve(
            objective, init_x, init_y, boundary, min_spacing, settings
        )

        cand_aep = -objective(opt_x, opt_y)
        cand_ok = feasible(opt_x, opt_y)
        if cand_ok & (cand_aep > best_aep):
            best_aep = cand_aep
            best_x = opt_x
            best_y = opt_y

        init_aep = -objective(init_x, init_y)
        if feasible(init_x, init_y) & (init_aep > best_aep):
            best_aep = init_aep
            best_x = init_x
            best_y = init_y

    return best_x, best_y
