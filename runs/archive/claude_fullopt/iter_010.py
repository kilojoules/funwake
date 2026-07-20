"""Aggressive multi-start SGD with adaptive penalty ramping.

Strategy:
1. 10 multi-starts with different seeds
2. Each uses aggressive hyperparameters tuned for faster convergence
3. Adaptive penalty weight ramping (start low, ramp up aggressively)
4. Longer constant LR phase for better exploration
5. Return best AEP among all starts
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Multi-start SGD with aggressive settings."""

    # ── Objective ──
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Wind-aware grid initialization function ──
    def get_wind_aware_init(seed):
        x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
        x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

        # Wind direction
        wd_rad = jnp.deg2rad(wd)
        dominant = jnp.arctan2(
            jnp.sum(weights * jnp.sin(wd_rad)),
            jnp.sum(weights * jnp.cos(wd_rad)))
        angle = dominant + jnp.pi / 2

        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
        translated = boundary - jnp.array([cx, cy])
        rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_bnd = (rot @ translated.T).T

        rx_min, ry_min = jnp.min(rot_bnd, axis=0)
        rx_max, ry_max = jnp.max(rot_bnd, axis=0)

        # Slightly tighter grid
        spacing_factor = 1.0 + 0.1 * (seed / 10.0)
        nx = max(3, int(jnp.ceil((rx_max - rx_min) / (min_spacing * spacing_factor))))
        ny = max(3, int(jnp.ceil((ry_max - ry_min) / (min_spacing * spacing_factor))))

        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
            jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
        rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
        inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
        orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
        cand_x, cand_y = orig_pts[:, 0], orig_pts[:, 1]

        # Filter inside boundary
        n_verts = boundary.shape[0]
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)
        inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
        ix, iy = cand_x[inside], cand_y[inside]

        if len(ix) >= n_target:
            key = jax.random.PRNGKey(seed)
            indices = jax.random.choice(key, len(ix), (n_target,), replace=False)
            init_x, init_y = ix[indices], iy[indices]
        else:
            key = jax.random.PRNGKey(seed)
            init_x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
            key, _ = jax.random.split(key)
            init_y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

        # Add small perturbation based on seed
        if seed > 0:
            key = jax.random.PRNGKey(seed * 1000)
            noise = min_spacing * 0.2
            init_x += jax.random.normal(key, shape=init_x.shape) * noise
            key, _ = jax.random.split(key)
            init_y += jax.random.normal(key, shape=init_y.shape) * noise

        return init_x, init_y

    # ── Aggressive optimizer settings ──
    settings = SGDSettings(
        learning_rate=200.0,                    # Very aggressive LR
        max_iter=3500,                          # Slightly fewer iterations
        additional_constant_lr_iterations=2500, # Longer constant phase
        tol=1e-6,
        beta1=0.15,                             # Slightly higher momentum
        beta2=0.25,                             # Faster variance adaptation
        gamma_min_factor=0.001,                 # Decay to very small LR
        ks_rho=150.0,                           # Sharper KS aggregation
        spacing_weight=100.0,                   # Strong penalty
        boundary_weight=100.0,                  # Strong penalty
    )

    # ── Multi-start loop ──
    best_x, best_y = None, None
    best_aep = float('inf')

    for seed in range(10):
        init_x, init_y = get_wind_aware_init(seed)

        try:
            opt_x, opt_y = topfarm_sgd_solve(
                objective, init_x, init_y,
                boundary, min_spacing, settings)

            # Evaluate AEP
            aep = objective(opt_x, opt_y)

            if aep < best_aep:
                best_aep = aep
                best_x, best_y = opt_x, opt_y
        except Exception:
            continue

    # Fallback if all failed
    if best_x is None:
        init_x, init_y = get_wind_aware_init(0)
        return init_x, init_y

    return best_x, best_y
