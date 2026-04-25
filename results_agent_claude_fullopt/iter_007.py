"""Three-stage progressive refinement with momentum annealing.

Strategy:
1. Exploration phase: High LR, low penalties, find good regions
2. Convergence phase: Moderate LR, balanced penalties
3. Refinement phase: Low LR, high penalties, ensure feasibility
Each phase uses different Adam beta parameters for momentum control.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Three-stage progressive refinement optimizer."""

    # ── Objective ──
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Initialization: best grid approach ──
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
        key = jax.random.PRNGKey(42)
        indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
        init_x, init_y = inside_pts[indices, 0], inside_pts[indices, 1]
    else:
        key = jax.random.PRNGKey(42)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # ── Phase 1: Exploration (find good regions) ──
    settings_explore = SGDSettings(
        learning_rate=200.0,              # Very high LR for exploration
        max_iter=1500,
        additional_constant_lr_iterations=500,
        tol=1e-6,
        beta1=0.05,                       # Very low momentum (more stochastic)
        beta2=0.1,
        gamma_min_factor=0.1,             # Don't decay too much
        ks_rho=50.0,                      # Soft penalties
        spacing_weight=30.0,              # Low penalty weights
        boundary_weight=30.0,
    )

    x1, y1 = topfarm_sgd_solve(objective, init_x, init_y,
                                boundary, min_spacing, settings_explore)

    # ── Phase 2: Convergence (balance AEP and feasibility) ──
    settings_converge = SGDSettings(
        learning_rate=150.0,
        max_iter=2500,
        additional_constant_lr_iterations=1500,
        tol=1e-6,
        beta1=0.2,                        # Moderate momentum
        beta2=0.3,
        gamma_min_factor=0.01,
        ks_rho=100.0,
        spacing_weight=70.0,
        boundary_weight=70.0,
    )

    x2, y2 = topfarm_sgd_solve(objective, x1, y1,
                                boundary, min_spacing, settings_converge)

    # ── Phase 3: Refinement (ensure feasibility) ──
    settings_refine = SGDSettings(
        learning_rate=75.0,
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-7,
        beta1=0.3,                        # Higher momentum for stability
        beta2=0.5,
        gamma_min_factor=0.001,           # Strong decay
        ks_rho=150.0,                     # Sharp penalties
        spacing_weight=120.0,             # High penalty weights
        boundary_weight=120.0,
    )

    x_final, y_final = topfarm_sgd_solve(objective, x2, y2,
                                          boundary, min_spacing, settings_refine)

    return x_final, y_final
