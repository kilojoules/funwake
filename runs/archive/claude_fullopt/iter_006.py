"""Hybrid: SGD with CMA-ES-inspired multi-start diversity.

Strategy: Use topfarm_sgd_solve but with CMA-ES-inspired initialization:
1. Start with best grid initialization
2. Sample perturbations around it with increasing variance
3. Run SGD from each perturbed start
4. Use aggressive hyperparameters tuned for the DEI farm
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """CMA-ES-inspired multi-start SGD."""

    # ── Objective ──
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Generate base initialization (wind-aligned grid) ──
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

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
    nx = max(3, int(jnp.ceil((rx_max - rx_min) / min_spacing)))
    ny = max(3, int(jnp.ceil((ry_max - ry_min) / min_spacing)))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
        jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])

    # Filter inside boundary
    n_verts = boundary.shape[0]
    def is_inside(pt):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (pt[0] - x1) * (-ey / el) + (pt[1] - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0

    inside_mask = jax.vmap(is_inside)(orig_pts)
    inside_pts = orig_pts[inside_mask]

    if len(inside_pts) >= n_target:
        key = jax.random.PRNGKey(0)
        indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
        base_x, base_y = inside_pts[indices, 0], inside_pts[indices, 1]
    else:
        key = jax.random.PRNGKey(0)
        base_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        base_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # ── Generate perturbed starts around base ──
    initial_layouts = [(base_x, base_y)]

    # Perturbation scales: small to large
    sigma_scales = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    base_sigma = float(min_spacing) * 0.5

    for i, scale in enumerate(sigma_scales):
        sigma = base_sigma * scale
        key = jax.random.PRNGKey(1000 + i)
        dx = jax.random.normal(key, (n_target,)) * sigma
        key, _ = jax.random.split(key)
        dy = jax.random.normal(key, (n_target,)) * sigma
        initial_layouts.append((base_x + dx, base_y + dy))

    # Add a few pure random starts for diversity
    for seed in [5000, 6000, 7000]:
        key = jax.random.PRNGKey(seed)
        rx = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        ry = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))
        initial_layouts.append((rx, ry))

    # ── Aggressive SGD settings ──
    settings = SGDSettings(
        learning_rate=175.0,              # Very aggressive LR
        max_iter=3500,
        additional_constant_lr_iterations=1500,
        tol=1e-6,
        beta1=0.15,                       # Slightly higher momentum
        beta2=0.25,
        gamma_min_factor=0.003,           # Aggressive decay
        ks_rho=90.0,
        spacing_weight=85.0,
        boundary_weight=85.0,
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
