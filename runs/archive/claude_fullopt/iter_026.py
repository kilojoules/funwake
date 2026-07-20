"""Triple-start SGD with progressive refinement.

Strategy:
- Three sequential SGD runs (~50-60s each after JIT)
- Start 1: Wind-aware hexagonal grid (proven initialization)
- Start 2: Perturbed version with increased LR for escape
- Start 3: Refined run with lower LR and more iterations
- Aggressive tuning: higher LR, more momentum, stronger penalties
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Triple-start progressive SGD optimizer."""

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))
    n_verts = boundary.shape[0]

    def is_inside_boundary(xi, yi):
        """Check if point is inside polygon."""
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (xi - x1) * (-ey / el) + (yi - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0

    # ── Start 1: Wind-aware hexagonal grid ────────────────────────
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

    h_spacing = min_spacing * 1.1
    v_spacing = h_spacing * jnp.sqrt(3) / 2

    nx = max(3, int(jnp.ceil((rx_max - rx_min) / h_spacing)))
    ny = max(3, int(jnp.ceil((ry_max - ry_min) / v_spacing)))

    candidates = []
    for i in range(ny):
        for j in range(nx):
            offset = (h_spacing / 2) if i % 2 == 1 else 0
            rx = float(rx_min) + j * float(h_spacing) + offset
            ry = float(ry_min) + i * float(v_spacing)
            candidates.append((rx, ry))

    candidates = jnp.array(candidates)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ candidates.T).T + jnp.array([cx, cy])

    inside_mask = jnp.array([is_inside_boundary(pt[0], pt[1]) for pt in orig_pts])
    inside_pts = orig_pts[inside_mask]

    if len(inside_pts) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(inside_pts) - 1, n_target)).astype(int)
        init_x1 = inside_pts[idx, 0]
        init_y1 = inside_pts[idx, 1]
    else:
        key = jax.random.PRNGKey(42)
        init_x1 = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
        key, _ = jax.random.split(key)
        init_y1 = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

    # Phase 1: Fast aggressive optimization
    settings1 = SGDSettings(
        learning_rate=120.0,             # High LR for fast convergence
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-7,
        beta1=0.2,                       # More momentum
        beta2=0.3,
        gamma_min_factor=0.01,
        ks_rho=150.0,                    # Sharp constraint aggregation
        spacing_weight=1.5,
        boundary_weight=1.5,
    )

    opt_x1, opt_y1 = topfarm_sgd_solve(
        objective, init_x1, init_y1, boundary, min_spacing, settings1
    )

    # Phase 2: Perturbed restart with high LR for escape
    key = jax.random.PRNGKey(123)
    perturbation = min_spacing * 0.4

    init_x2 = opt_x1 + jax.random.uniform(key, (n_target,), minval=-perturbation, maxval=perturbation)
    key, _ = jax.random.split(key)
    init_y2 = opt_y1 + jax.random.uniform(key, (n_target,), minval=-perturbation, maxval=perturbation)

    settings2 = SGDSettings(
        learning_rate=100.0,
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-7,
        beta1=0.25,
        beta2=0.35,
        gamma_min_factor=0.005,          # Stronger decay
        ks_rho=150.0,
        spacing_weight=2.0,              # Stronger penalties
        boundary_weight=2.0,
    )

    opt_x2, opt_y2 = topfarm_sgd_solve(
        objective, init_x2, init_y2, boundary, min_spacing, settings2
    )

    # Phase 3: Final refinement with lower LR and more iterations
    settings3 = SGDSettings(
        learning_rate=60.0,              # Lower LR for fine-tuning
        max_iter=3000,
        additional_constant_lr_iterations=1500,
        tol=1e-8,
        beta1=0.3,                       # More momentum for stability
        beta2=0.4,
        gamma_min_factor=0.001,
        ks_rho=200.0,                    # Very sharp constraints
        spacing_weight=2.5,
        boundary_weight=2.5,
    )

    opt_x3, opt_y3 = topfarm_sgd_solve(
        objective, opt_x2, opt_y2, boundary, min_spacing, settings3
    )

    return opt_x3, opt_y3
