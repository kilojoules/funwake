"""Enhanced SGD with aggressive multi-start and two-stage optimization.

Strategy: Use proven topfarm_sgd_solve but with:
1. More aggressive learning rate and iterations
2. Two-stage: feasibility-focused first, then AEP-focused
3. 10 diverse starting points
4. Adaptive hyperparameter selection per stage
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Two-stage multi-start SGD optimizer."""

    # ── Objective ──
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Initialization helpers ──
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    n_verts = boundary.shape[0]
    def edge_dist(x, y, i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (x - x1) * (-ey / el) + (y - y1) * (ex / el)

    def is_inside(pt):
        dists = jax.vmap(lambda i: edge_dist(pt[0], pt[1], i))(jnp.arange(n_verts))
        return jnp.min(dists) > 0

    # ── Generate 10 diverse initial layouts ──
    initial_layouts = []

    # 1. Wind-aligned grid
    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(jnp.sum(weights * jnp.sin(wd_rad)),
                           jnp.sum(weights * jnp.cos(wd_rad)))
    angle = dominant + jnp.pi / 2

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)
    nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
    ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, max(nx, 3)),
        jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, max(ny, 3)))
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])

    inside_mask = jax.vmap(is_inside)(orig_pts)
    inside_pts = orig_pts[inside_mask]

    if len(inside_pts) >= n_target:
        key = jax.random.PRNGKey(0)
        indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
        initial_layouts.append((inside_pts[indices, 0], inside_pts[indices, 1]))

    # 2. Regular grid
    nx2 = max(3, int(jnp.ceil((x_max - x_min) / min_spacing)))
    ny2 = max(3, int(jnp.ceil((y_max - y_min) / min_spacing)))
    gx2, gy2 = jnp.meshgrid(
        jnp.linspace(x_min + min_spacing/2, x_max - min_spacing/2, nx2),
        jnp.linspace(y_min + min_spacing/2, y_max - min_spacing/2, ny2))
    grid_pts = jnp.stack([gx2.flatten(), gy2.flatten()], axis=-1)
    inside_mask2 = jax.vmap(is_inside)(grid_pts)
    inside_pts2 = grid_pts[inside_mask2]

    if len(inside_pts2) >= n_target:
        idx2 = jnp.round(jnp.linspace(0, len(inside_pts2) - 1, n_target)).astype(int)
        initial_layouts.append((inside_pts2[idx2, 0], inside_pts2[idx2, 1]))

    # 3-10. Random with different seeds
    for seed in range(100, 900, 100):
        key = jax.random.PRNGKey(seed)
        rx = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        ry = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))
        initial_layouts.append((rx, ry))

    # ── Two-stage optimization settings ──

    # Stage 1: Feasibility-focused (high constraint penalties)
    settings_stage1 = SGDSettings(
        learning_rate=100.0,
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.01,
        ks_rho=150.0,            # Very sharp penalties
        spacing_weight=100.0,     # High weight on spacing
        boundary_weight=100.0,    # High weight on boundary
    )

    # Stage 2: AEP-focused (lower constraint penalties, assume near-feasible)
    settings_stage2 = SGDSettings(
        learning_rate=75.0,
        max_iter=3000,
        additional_constant_lr_iterations=1500,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.005,
        ks_rho=80.0,             # Moderate sharpness
        spacing_weight=50.0,      # Lower weight, focus on AEP
        boundary_weight=50.0,
    )

    # ── Multi-start with two-stage optimization ──
    best_aep = -jnp.inf
    best_x, best_y = None, None

    for init_x, init_y in initial_layouts:
        # Stage 1: Get feasible
        feas_x, feas_y = topfarm_sgd_solve(
            objective, init_x, init_y,
            boundary, min_spacing, settings_stage1)

        # Stage 2: Optimize AEP from feasible point
        opt_x, opt_y = topfarm_sgd_solve(
            objective, feas_x, feas_y,
            boundary, min_spacing, settings_stage2)

        # Evaluate
        aep = -objective(opt_x, opt_y)
        if aep > best_aep:
            best_aep = aep
            best_x, best_y = opt_x, opt_y

    return best_x, best_y
