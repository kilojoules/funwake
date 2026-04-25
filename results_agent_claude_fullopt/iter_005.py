"""Aggressive 20-start SGD with tuned hyperparameters.

Strategy: Based on best_optimizer.py's approach but with:
1. 20 diverse starts instead of 3
2. Single-stage optimization with carefully tuned parameters
3. Mix of grid and random initializations
4. Optimized SGD settings from empirical testing
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """20-start SGD with optimized settings."""

    # ── Objective ──
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Initialization helpers ──
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    n_verts = boundary.shape[0]
    def is_inside(x, y):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (x - x1) * (-ey / el) + (y - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0

    initial_layouts = []

    # ── Strategy 1-2: Wind-aligned grids with different rotations ──
    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(jnp.sum(weights * jnp.sin(wd_rad)),
                           jnp.sum(weights * jnp.cos(wd_rad)))

    for angle_offset in [0, jnp.pi/4]:  # Two different rotation angles
        angle = dominant + jnp.pi / 2 + angle_offset
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

        inside_mask = jax.vmap(lambda pt: is_inside(pt[0], pt[1]))(orig_pts)
        inside_pts = orig_pts[inside_mask]

        if len(inside_pts) >= n_target:
            key = jax.random.PRNGKey(int(angle_offset * 100))
            indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
            initial_layouts.append((inside_pts[indices, 0], inside_pts[indices, 1]))

    # ── Strategy 3: Regular grid ──
    nx2 = max(3, int(jnp.ceil((x_max - x_min) / min_spacing)))
    ny2 = max(3, int(jnp.ceil((y_max - y_min) / min_spacing)))
    gx2, gy2 = jnp.meshgrid(
        jnp.linspace(x_min + min_spacing/2, x_max - min_spacing/2, nx2),
        jnp.linspace(y_min + min_spacing/2, y_max - min_spacing/2, ny2))
    grid_pts = jnp.stack([gx2.flatten(), gy2.flatten()], axis=-1)
    inside_mask2 = jax.vmap(lambda pt: is_inside(pt[0], pt[1]))(grid_pts)
    inside_pts2 = grid_pts[inside_mask2]

    if len(inside_pts2) >= n_target:
        idx2 = jnp.round(jnp.linspace(0, len(inside_pts2) - 1, n_target)).astype(int)
        initial_layouts.append((inside_pts2[idx2, 0], inside_pts2[idx2, 1]))

    # ── Strategy 4-20: Random with different seeds ──
    for seed in range(1000, 1000 + 17 * 50, 50):
        key = jax.random.PRNGKey(seed)
        rx = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        ry = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))
        initial_layouts.append((rx, ry))

    # ── Optimized SGD settings ──
    # Tuned for balance between exploration and exploitation
    settings = SGDSettings(
        learning_rate=125.0,              # Moderately aggressive
        max_iter=4000,
        additional_constant_lr_iterations=2000,
        tol=1e-6,
        beta1=0.1,                        # Low momentum (more exploration)
        beta2=0.2,                        # Low second moment decay
        gamma_min_factor=0.005,           # Aggressive LR decay
        ks_rho=100.0,                     # Sharp but not extreme
        spacing_weight=80.0,              # Strong spacing enforcement
        boundary_weight=80.0,             # Strong boundary enforcement
    )

    # ── Multi-start optimization ──
    best_aep = -jnp.inf
    best_x, best_y = None, None

    for init_x, init_y in initial_layouts:
        opt_x, opt_y = topfarm_sgd_solve(
            objective, init_x, init_y,
            boundary, min_spacing, settings)

        aep = -objective(opt_x, opt_y)
        if aep > best_aep:
            best_aep = aep
            best_x, best_y = opt_x, opt_y

    return best_x, best_y
