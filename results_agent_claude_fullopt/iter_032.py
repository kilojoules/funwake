"""iter_029 variant with adjusted momentum and penalties.

Based on iter_029's 5559.08 GWh success, try:
- Same 2-start structure
- Same high iteration count (3500 + 1500)
- Adjusted beta1=0.15 (more momentum)
- Penalties at 110.0 (between 100 and 150)
- LR=155 (between 150 and 160)
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """iter_029 variant with tuned hyperparameters."""

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

    # Wind-aware grid initialization
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
        init_x1 = inside_x[idx]
        init_y1 = inside_y[idx]
    else:
        key = jax.random.PRNGKey(42)
        init_x1 = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
        key, _ = jax.random.split(key)
        init_y1 = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

    # Tuned settings
    settings = SGDSettings(
        learning_rate=155.0,
        max_iter=3500,
        additional_constant_lr_iterations=1500,
        tol=1e-7,
        beta1=0.15,                      # More momentum than iter_029's 0.12
        beta2=0.24,
        gamma_min_factor=0.003,
        ks_rho=125.0,
        spacing_weight=110.0,            # Between 100 and 150
        boundary_weight=110.0,
    )

    # Start 1: Wind-aware initialization
    opt_x1, opt_y1 = topfarm_sgd_solve(objective, init_x1, init_y1, boundary, min_spacing, settings)
    aep1 = objective(opt_x1, opt_y1)

    # Start 2: Perturbed version
    key = jax.random.PRNGKey(888)
    noise = min_spacing * 0.3
    init_x2 = opt_x1 + jax.random.normal(key, shape=(n_target,)) * noise
    key, _ = jax.random.split(key)
    init_y2 = opt_y1 + jax.random.normal(key, shape=(n_target,)) * noise
    init_x2 = jnp.clip(init_x2, x_min, x_max)
    init_y2 = jnp.clip(init_y2, y_min, y_max)

    opt_x2, opt_y2 = topfarm_sgd_solve(objective, init_x2, init_y2, boundary, min_spacing, settings)
    aep2 = objective(opt_x2, opt_y2)

    # Return best
    if aep1 < aep2:
        return opt_x1, opt_y1
    else:
        return opt_x2, opt_y2
