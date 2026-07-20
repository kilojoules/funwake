"""SLSQP with JAX Jacobians - single high-quality start with efficient constraint handling.

Strategy:
- Use scipy.optimize.minimize with SLSQP method
- JAX-compiled gradients for objective and constraints
- Wind-aware hexagonal initialization
- SLSQP handles constraints natively (no penalties)
- Focus on one high-quality run within 180s budget
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from functools import partial


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """SLSQP optimizer with JAX gradients."""

    # Objective function (AEP to maximize -> negate for minimization)
    @jax.jit
    def aep_jax(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # Gradient of AEP w.r.t. positions
    grad_aep = jax.jit(jax.grad(aep_jax, argnums=(0, 1)))

    def objective_scipy(z):
        """Scipy-compatible objective (negated AEP)."""
        x, y = z[:n_target], z[n_target:]
        return -float(aep_jax(x, y))

    def gradient_scipy(z):
        """Scipy-compatible gradient."""
        x, y = z[:n_target], z[n_target:]
        gx, gy = grad_aep(x, y)
        return -np.concatenate([np.array(gx), np.array(gy)])

    # Spacing constraints: pairwise distances >= min_spacing
    @jax.jit
    def spacing_violations(x, y):
        """Returns array of squared distances minus min_spacing^2."""
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        # Extract upper triangular (avoid diagonal and duplicates)
        i, j = jnp.triu_indices(n_target, k=1)
        return dist_sq[i, j] - min_spacing**2

    grad_spacing = jax.jit(jax.grad(lambda x, y: jnp.sum(spacing_violations(x, y)), argnums=(0, 1)))

    def spacing_constraint_scipy(z):
        """Scipy-compatible spacing constraint."""
        x, y = z[:n_target], z[n_target:]
        return np.array(spacing_violations(x, y))

    def spacing_jacobian_scipy(z):
        """Scipy-compatible spacing Jacobian."""
        x, y = z[:n_target], z[n_target:]
        gx, gy = grad_spacing(x, y)
        # This is gradient of sum; we need Jacobian of each constraint
        # Recompute numerically for now (SLSQP will approximate if not provided)
        return np.zeros((len(spacing_constraint_scipy(z)), len(z)))

    # Boundary constraints: all turbines inside polygon
    @jax.jit
    def boundary_distances(x, y):
        """Returns signed distances to boundary (positive = inside)."""
        n_verts = boundary.shape[0]
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (x - x1) * (-ey / el) + (y - y1) * (ex / el)
        dists = jax.vmap(edge_dist)(jnp.arange(n_verts))
        return jnp.min(dists, axis=0)  # min distance for each turbine

    def boundary_constraint_scipy(z):
        """Scipy-compatible boundary constraint."""
        x, y = z[:n_target], z[n_target:]
        return np.array(boundary_distances(x, y))

    # Initialize with wind-aware hexagonal grid
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    # Dominant wind direction
    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
    angle = float(dominant + jnp.pi / 2)  # perpendicular to wind

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = float(jnp.mean(boundary[:, 0])), float(jnp.mean(boundary[:, 1]))

    # Create hexagonal grid in rotated space
    h_spacing = min_spacing * 1.1  # slightly larger to avoid violations
    v_spacing = h_spacing * np.sqrt(3) / 2

    nx = int(np.ceil((x_max - x_min) / h_spacing)) + 2
    ny = int(np.ceil((y_max - y_min) / v_spacing)) + 2

    candidates = []
    for i in range(ny):
        for j in range(nx):
            offset = (h_spacing / 2) if i % 2 == 1 else 0
            rx = (x_min - min_spacing) + j * h_spacing + offset
            ry = (y_min - min_spacing) + i * v_spacing
            # Rotate back to original coordinates
            dx, dy = rx - cx, ry - cy
            x_orig = cx + cos_a * dx - sin_a * dy
            y_orig = cy + sin_a * dx + cos_a * dy
            candidates.append((x_orig, y_orig))

    # Filter to keep only inside boundary
    candidates = np.array(candidates)
    n_verts = boundary.shape[0]

    def is_inside(xi, yi):
        for i in range(n_verts):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = np.sqrt(ex**2 + ey**2) + 1e-10
            d = (xi - x1) * (-ey / el) + (yi - y1) * (ex / el)
            if d <= min_spacing * 0.1:  # small margin
                return False
        return True

    inside_candidates = np.array([c for c in candidates if is_inside(c[0], c[1])])

    if len(inside_candidates) >= n_target:
        # Select n_target points uniformly
        indices = np.linspace(0, len(inside_candidates) - 1, n_target, dtype=int)
        init_x = inside_candidates[indices, 0]
        init_y = inside_candidates[indices, 1]
    else:
        # Fallback: random initialization
        np.random.seed(42)
        init_x = np.random.uniform(x_min + min_spacing, x_max - min_spacing, n_target)
        init_y = np.random.uniform(y_min + min_spacing, y_max - min_spacing, n_target)

    z0 = np.concatenate([init_x, init_y])

    # Define constraints for SLSQP
    spacing_con = NonlinearConstraint(
        spacing_constraint_scipy,
        lb=0, ub=np.inf,  # distances >= min_spacing^2
    )

    boundary_con = NonlinearConstraint(
        boundary_constraint_scipy,
        lb=0, ub=np.inf,  # inside boundary
    )

    # Run SLSQP optimization
    result = minimize(
        objective_scipy,
        z0,
        method='SLSQP',
        jac=gradient_scipy,
        constraints=[spacing_con, boundary_con],
        options={
            'maxiter': 1000,
            'ftol': 1e-9,
        }
    )

    opt_x = jnp.array(result.x[:n_target])
    opt_y = jnp.array(result.x[n_target:])

    return opt_x, opt_y
