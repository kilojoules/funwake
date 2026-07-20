"""Two-start SGD optimizer for 180s budget.

Building on iter_020's success:
- 2 starts instead of 1 (should fit in ~100s total)
- Slightly fewer iterations per start (3000 vs 4000)
- Wind-aware + standard grid initializations
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Two-start SGD optimizer."""

    # Objective
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

    # Initialization 1: Wind-aware grid
    def wind_aware_init():
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
        else:
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
            key, _ = jax.random.split(key)
            y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)
            return x, y

    # Initialization 2: Standard rectangular grid
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
        else:
            key = jax.random.PRNGKey(123)
            x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
            key, _ = jax.random.split(key)
            y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)
            return x, y

    # SGD settings (slightly fewer iterations for 2 starts)
    settings = SGDSettings(
        learning_rate=150.0,
        max_iter=3000,
        additional_constant_lr_iterations=1500,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.005,
        ks_rho=100.0,
        spacing_weight=75.0,
        boundary_weight=75.0,
    )

    # Run both starts
    best_x, best_y = None, None
    best_aep = float('inf')

    for init_fn in [wind_aware_init, standard_init]:
        init_x, init_y = init_fn()
        opt_x, opt_y = topfarm_sgd_solve(
            objective, init_x, init_y,
            boundary, min_spacing, settings)

        aep = objective(opt_x, opt_y)
        if aep < best_aep:
            best_aep = aep
            best_x, best_y = opt_x, opt_y

    return best_x, best_y
