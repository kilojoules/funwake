"""Coarse-to-fine optimization with adaptive constraint handling.

Strategy:
1. Start with relaxed spacing constraint (0.8x) and coarse optimization
2. Gradually tighten constraints while refining
3. Use SGD with adaptive penalty weight schedule
4. Three refinement stages with increasing fidelity
5. Wind-aware hexagonal initialization
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Coarse-to-fine multi-stage optimizer."""

    # ── Objective ──
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Hexagonal initialization ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    # Hexagonal grid spacing
    dx = min_spacing * 1.1
    dy = min_spacing * 1.1 * jnp.sqrt(3) / 2

    nx = max(3, int(jnp.ceil((x_max - x_min) / dx)) + 1)
    ny = max(3, int(jnp.ceil((y_max - y_min) / dy)) + 1)

    pts = []
    for i in range(ny):
        for j in range(nx):
            offset = dx / 2 if i % 2 == 1 else 0
            xi = x_min + j * dx + offset
            yi = y_min + i * dy
            pts.append([xi, yi])

    pts = jnp.array(pts)

    # Filter inside boundary
    n_verts = boundary.shape[0]
    def is_inside(pt):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (pt[0] - x1) * (-ey / el) + (pt[1] - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > min_spacing * 0.4

    inside_mask = jax.vmap(is_inside)(pts)
    inside_pts = pts[inside_mask]

    if len(inside_pts) >= n_target:
        # Select evenly distributed points
        indices = jnp.round(jnp.linspace(0, len(inside_pts) - 1, n_target)).astype(int)
        init_x = inside_pts[indices, 0]
        init_y = inside_pts[indices, 1]
    else:
        # Random fallback
        key = jax.random.PRNGKey(42)
        init_x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

    # ── Stage 1: Coarse optimization with relaxed constraints ──
    # Use lower spacing requirement and moderate penalties
    settings1 = SGDSettings(
        learning_rate=120.0,
        max_iter=1500,
        additional_constant_lr_iterations=1000,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.01,
        ks_rho=80.0,
        spacing_weight=30.0,       # Lower penalty for relaxed phase
        boundary_weight=50.0,
    )

    relaxed_spacing = min_spacing * 0.85
    x1, y1 = topfarm_sgd_solve(objective, init_x, init_y,
                                boundary, relaxed_spacing, settings1)

    # ── Stage 2: Medium refinement ──
    settings2 = SGDSettings(
        learning_rate=100.0,
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.005,
        ks_rho=120.0,
        spacing_weight=75.0,       # Moderate penalty
        boundary_weight=75.0,
    )

    medium_spacing = min_spacing * 0.95
    x2, y2 = topfarm_sgd_solve(objective, x1, y1,
                                boundary, medium_spacing, settings2)

    # ── Stage 3: Fine refinement with full constraints ──
    settings3 = SGDSettings(
        learning_rate=80.0,
        max_iter=2500,
        additional_constant_lr_iterations=1500,
        tol=1e-7,
        beta1=0.12,
        beta2=0.22,
        gamma_min_factor=0.002,
        ks_rho=150.0,
        spacing_weight=120.0,      # Strong penalty for final phase
        boundary_weight=120.0,
    )

    opt_x, opt_y = topfarm_sgd_solve(objective, x2, y2,
                                      boundary, min_spacing, settings3)

    return opt_x, opt_y
