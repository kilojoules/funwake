"""Hybrid SLSQP + SGD warmstart optimizer.

Strategy:
- Phase 1: Quick SGD warmup (500 iters, ~15-20s) to find good basin
- Phase 2: SLSQP refinement with JAX gradients (~150s)
- SGD handles global exploration, SLSQP handles precise constraint satisfaction
- Wind-aware hexagonal initialization
- Adaptive constraint tolerance
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Hybrid SGD warmstart + SLSQP refinement."""

    # Objective function
    @jax.jit
    def aep_jax(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def objective_sgd(x, y):
        return -aep_jax(x, y)

    # Gradient for SLSQP
    grad_aep = jax.jit(jax.grad(aep_jax, argnums=(0, 1)))

    def objective_scipy(z):
        x, y = z[:n_target], z[n_target:]
        return -float(aep_jax(jnp.array(x), jnp.array(y)))

    def gradient_scipy(z):
        x, y = z[:n_target], z[n_target:]
        gx, gy = grad_aep(jnp.array(x), jnp.array(y))
        return -np.concatenate([np.array(gx), np.array(gy)])

    # ── Constraint functions ────────────────────────────────────────
    n_verts = boundary.shape[0]

    @jax.jit
    def spacing_violations(x, y):
        """Min spacing constraint: dist >= min_spacing."""
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        i, j = jnp.triu_indices(n_target, k=1)
        return dist_sq[i, j] - min_spacing**2

    @jax.jit
    def boundary_distances(x, y):
        """Boundary constraint: signed distance (positive = inside)."""
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (x - x1) * (-ey / el) + (y - y1) * (ex / el)
        dists = jax.vmap(edge_dist)(jnp.arange(n_verts))
        return jnp.min(dists, axis=0)

    def spacing_constraint_scipy(z):
        x, y = z[:n_target], z[n_target:]
        return np.array(spacing_violations(jnp.array(x), jnp.array(y)))

    def boundary_constraint_scipy(z):
        x, y = z[:n_target], z[n_target:]
        return np.array(boundary_distances(jnp.array(x), jnp.array(y)))

    # ── Initialization: wind-aware hexagonal grid ───────────────────
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
    angle = float(dominant + jnp.pi / 2)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = float(jnp.mean(boundary[:, 0])), float(jnp.mean(boundary[:, 1]))

    h_spacing = min_spacing * 1.08
    v_spacing = h_spacing * np.sqrt(3) / 2

    nx = int(np.ceil((x_max - x_min) / h_spacing)) + 2
    ny = int(np.ceil((y_max - y_min) / v_spacing)) + 2

    candidates = []
    for i in range(ny):
        for j in range(nx):
            offset = (h_spacing / 2) if i % 2 == 1 else 0
            rx = (x_min - min_spacing) + j * h_spacing + offset
            ry = (y_min - min_spacing) + i * v_spacing
            dx, dy = rx - cx, ry - cy
            x_orig = cx + cos_a * dx - sin_a * dy
            y_orig = cy + sin_a * dx + cos_a * dy
            candidates.append((x_orig, y_orig))

    candidates = np.array(candidates)

    def is_inside(xi, yi):
        for i in range(n_verts):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = np.sqrt(ex**2 + ey**2) + 1e-10
            d = (xi - x1) * (-ey / el) + (yi - y1) * (ex / el)
            if d <= min_spacing * 0.15:
                return False
        return True

    inside_candidates = np.array([c for c in candidates if is_inside(c[0], c[1])])

    if len(inside_candidates) >= n_target:
        indices = np.linspace(0, len(inside_candidates) - 1, n_target, dtype=int)
        init_x = jnp.array(inside_candidates[indices, 0])
        init_y = jnp.array(inside_candidates[indices, 1])
    else:
        key = jax.random.PRNGKey(42)
        init_x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

    # ── Phase 1: SGD warmstart (~15-20s) ────────────────────────────
    settings = SGDSettings(
        learning_rate=100.0,
        max_iter=300,
        additional_constant_lr_iterations=200,
        tol=1e-6,
        beta1=0.2,
        beta2=0.3,
        gamma_min_factor=0.01,
        ks_rho=100.0,
        spacing_weight=1.5,
        boundary_weight=1.5,
    )

    warm_x, warm_y = topfarm_sgd_solve(
        objective_sgd, init_x, init_y, boundary, min_spacing, settings
    )

    # ── Phase 2: SLSQP refinement (~150s) ───────────────────────────
    z0 = np.concatenate([np.array(warm_x), np.array(warm_y)])

    spacing_con = NonlinearConstraint(
        spacing_constraint_scipy,
        lb=0, ub=np.inf,
    )

    boundary_con = NonlinearConstraint(
        boundary_constraint_scipy,
        lb=0, ub=np.inf,
    )

    result = minimize(
        objective_scipy,
        z0,
        method='SLSQP',
        jac=gradient_scipy,
        constraints=[spacing_con, boundary_con],
        options={
            'maxiter': 2000,
            'ftol': 1e-10,
        }
    )

    opt_x = jnp.array(result.x[:n_target])
    opt_y = jnp.array(result.x[n_target:])

    return opt_x, opt_y
