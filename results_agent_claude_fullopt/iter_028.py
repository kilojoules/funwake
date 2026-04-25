"""Enhanced three-start SGD with adaptive refinement.

Building on iter_022's successful formula:
- 3 diverse starts (wind-aware, standard grid, perturbed)
- Very high penalty weights (75.0) for constraint handling
- High learning rate (150.0) for fast convergence
- NEW: Additional refinement phase on best solution with more iterations
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Enhanced three-start SGD optimizer."""

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))
    n_verts = boundary.shape[0]

    def is_inside(xi, yi):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (xi - x1) * (-ey / el) + (yi - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0

    # Init 1: Wind-aware grid
    def wind_aware_init():
        wd_rad = jnp.deg2rad(wd)
        dominant = jnp.arctan2(jnp.sum(weights * jnp.sin(wd_rad)), jnp.sum(weights * jnp.cos(wd_rad)))
        angle = dominant + jnp.pi / 2
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
        translated = boundary - jnp.array([cx, cy])
        rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_bnd = (rot @ translated.T).T
        rx_min, ry_min = jnp.min(rot_bnd, axis=0)
        rx_max, ry_max = jnp.max(rot_bnd, axis=0)
        nx = max(3, int(jnp.ceil((rx_max - rx_min) / min_spacing)))
        ny = max(3, int(jnp.ceil((ry_max - ry_min) / min_spacing)))
        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
            jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
        rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
        inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
        orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
        cand_x, cand_y = orig_pts[:, 0], orig_pts[:, 1]
        inside_mask = jax.vmap(is_inside)(cand_x, cand_y)
        inside_x, inside_y = cand_x[inside_mask], cand_y[inside_mask]
        if len(inside_x) >= n_target:
            idx = jnp.round(jnp.linspace(0, len(inside_x) - 1, n_target)).astype(int)
            return inside_x[idx], inside_y[idx]
        key = jax.random.PRNGKey(42)
        return jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max), \
               jax.random.uniform(jax.random.split(key)[0], (n_target,), minval=y_min, maxval=y_max)

    # Init 2: Standard grid
    def standard_init():
        nx = max(3, int(jnp.ceil((x_max - x_min) / min_spacing)))
        ny = max(3, int(jnp.ceil((y_max - y_min) / min_spacing)))
        gx, gy = jnp.meshgrid(
            jnp.linspace(x_min + min_spacing/2, x_max - min_spacing/2, nx),
            jnp.linspace(y_min + min_spacing/2, y_max - min_spacing/2, ny))
        cand_x, cand_y = gx.flatten(), gy.flatten()
        inside_mask = jax.vmap(is_inside)(cand_x, cand_y)
        inside_x, inside_y = cand_x[inside_mask], cand_y[inside_mask]
        if len(inside_x) >= n_target:
            idx = jnp.round(jnp.linspace(0, len(inside_x) - 1, n_target)).astype(int)
            return inside_x[idx], inside_y[idx]
        key = jax.random.PRNGKey(123)
        return jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max), \
               jax.random.uniform(jax.random.split(key)[0], (n_target,), minval=y_min, maxval=y_max)

    # Init 3: Perturbed wind-aware
    def perturbed_init():
        x, y = wind_aware_init()
        key = jax.random.PRNGKey(777)
        noise = min_spacing * 0.25  # Slightly larger perturbation
        x = x + jax.random.normal(key, shape=x.shape) * noise
        key, _ = jax.random.split(key)
        y = y + jax.random.normal(key, shape=y.shape) * noise
        return jnp.clip(x, x_min, x_max), jnp.clip(y, y_min, y_max)

    # Phase 1: Multi-start exploration with iter_022's proven settings
    settings1 = SGDSettings(
        learning_rate=150.0,
        max_iter=2200,                   # Slightly reduced to save time
        additional_constant_lr_iterations=1100,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.005,
        ks_rho=100.0,
        spacing_weight=75.0,             # Key: very high penalty
        boundary_weight=75.0,            # Key: very high penalty
    )

    best_x, best_y = None, None
    best_aep = float('inf')

    for init_fn in [wind_aware_init, standard_init, perturbed_init]:
        init_x, init_y = init_fn()
        opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y, boundary, min_spacing, settings1)
        aep = objective(opt_x, opt_y)
        if aep < best_aep:
            best_aep = aep
            best_x, best_y = opt_x, opt_y

    # Phase 2: Refinement with more iterations and adjusted parameters
    settings2 = SGDSettings(
        learning_rate=100.0,             # Lower LR for fine-tuning
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-7,
        beta1=0.15,                      # Slightly more momentum
        beta2=0.25,
        gamma_min_factor=0.002,          # Stronger decay
        ks_rho=120.0,                    # Sharper constraints
        spacing_weight=100.0,            # Even higher penalty
        boundary_weight=100.0,           # Even higher penalty
    )

    final_x, final_y = topfarm_sgd_solve(objective, best_x, best_y, boundary, min_spacing, settings2)

    return final_x, final_y
