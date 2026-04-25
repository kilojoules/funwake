"""Dual-start SGD with complementary initializations.

Strategy:
- Two sequential SGD runs (each gets ~90s after JIT)
- Start 1: Wind-aware hexagonal grid (proven good initialization)
- Start 2: Use best of start 1 with small random perturbations
- Tuned SGD settings: higher LR, more iterations
- 3000 iterations per start should fit in ~60-75s each
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Dual-start SGD optimizer."""

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
    angle = dominant + jnp.pi / 2  # perpendicular to wind

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)

    # Hexagonal grid with proper spacing
    h_spacing = min_spacing * 1.05
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

    # Filter inside boundary
    inside_mask = jnp.array([is_inside_boundary(pt[0], pt[1]) for pt in orig_pts])
    inside_pts = orig_pts[inside_mask]

    if len(inside_pts) >= n_target:
        # Select uniformly distributed subset
        idx = jnp.round(jnp.linspace(0, len(inside_pts) - 1, n_target)).astype(int)
        init_x1 = inside_pts[idx, 0]
        init_y1 = inside_pts[idx, 1]
    else:
        # Fallback: random
        key = jax.random.PRNGKey(42)
        init_x1 = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
        key, _ = jax.random.split(key)
        init_y1 = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

    # SGD settings optimized for ~60-75s runtime
    settings = SGDSettings(
        learning_rate=75.0,              # Higher LR for faster convergence
        max_iter=3000,                   # Main iterations
        additional_constant_lr_iterations=1500,  # Constant LR phase
        tol=1e-7,
        beta1=0.15,                      # Slightly more momentum
        beta2=0.25,                      # Adjusted for stability
        gamma_min_factor=0.005,          # Stronger LR decay
        ks_rho=120.0,                    # Sharper constraint aggregation
        spacing_weight=1.2,              # Slightly stronger spacing penalty
        boundary_weight=1.2,             # Slightly stronger boundary penalty
    )

    # Run first optimization
    opt_x1, opt_y1 = topfarm_sgd_solve(
        objective, init_x1, init_y1, boundary, min_spacing, settings
    )

    # ── Start 2: Perturbed version of best solution ───────────────
    # Small random perturbations to escape local minima
    key = jax.random.PRNGKey(123)
    perturbation = min_spacing * 0.3  # 30% of min spacing

    init_x2 = opt_x1 + jax.random.uniform(key, (n_target,), minval=-perturbation, maxval=perturbation)
    key, _ = jax.random.split(key)
    init_y2 = opt_y1 + jax.random.uniform(key, (n_target,), minval=-perturbation, maxval=perturbation)

    # Second run with same settings
    opt_x2, opt_y2 = topfarm_sgd_solve(
        objective, init_x2, init_y2, boundary, min_spacing, settings
    )

    # Return the second optimization (should be better or similar)
    return opt_x2, opt_y2
