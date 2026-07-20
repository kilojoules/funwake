"""Augmented Lagrangian method with adaptive multiplier updates.

Strategy:
1. Augmented Lagrangian: penalty + Lagrange multipliers for constraints
2. Inner loop: L-BFGS-B optimization with JAX gradients
3. Outer loop: update multipliers and penalty weight
4. Adaptive penalty ramping based on constraint satisfaction
5. Wind-aware hexagonal initialization
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Augmented Lagrangian optimizer."""

    # ── Objective ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Constraint functions ──
    # Boundary violations (negative = violating)
    def boundary_constraints(x, y):
        n_verts = boundary.shape[0]
        def min_edge_dist(xi, yi):
            def edge_dist(i):
                x1, y1 = boundary[i]
                x2, y2 = boundary[(i + 1) % n_verts]
                ex, ey = x2 - x1, y2 - y1
                el = jnp.sqrt(ex**2 + ey**2) + 1e-10
                return (xi - x1) * (-ey / el) + (yi - y1) * (ex / el)
            return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)))
        return jax.vmap(min_edge_dist)(x, y)

    # Spacing violations (negative = violating)
    def spacing_constraints(x, y):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = jnp.sqrt(dx**2 + dy**2 + 1e-10)
        i_idx, j_idx = jnp.triu_indices(n_target, k=1)
        pair_dists = dists[i_idx, j_idx]
        return pair_dists - min_spacing

    # Augmented Lagrangian
    def aug_lagrangian(x, y, lambda_b, lambda_s, rho):
        """L(x, lambda, rho) = f(x) + lambda^T c(x) + (rho/2) ||max(0, c(x))||^2"""
        aep = aep_objective(x, y)

        # Boundary constraints
        cb = boundary_constraints(x, y)
        vb = jnp.maximum(0, -cb)  # violation (positive when violated)
        aug_b = jnp.sum(lambda_b * vb) + (rho / 2) * jnp.sum(vb**2)

        # Spacing constraints
        cs = spacing_constraints(x, y)
        vs = jnp.maximum(0, -cs)
        aug_s = jnp.sum(lambda_s * vs) + (rho / 2) * jnp.sum(vs**2)

        return aep + aug_b + aug_s

    # For scipy
    def aug_lag_flat(xy_flat, lambda_b, lambda_s, rho):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        return float(aug_lagrangian(x, y, lambda_b, lambda_s, rho))

    # JAX gradient
    grad_aug = jax.grad(aug_lagrangian, argnums=(0, 1))

    def aug_lag_jac_flat(xy_flat, lambda_b, lambda_s, rho):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        gx, gy = grad_aug(x, y, lambda_b, lambda_s, rho)
        return np.concatenate([np.array(gx), np.array(gy)])

    # ── Hexagonal initialization ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    dx = min_spacing * 1.12
    dy = min_spacing * 1.12 * np.sqrt(3) / 2
    nx = max(3, int(np.ceil((x_max - x_min) / dx)) + 1)
    ny = max(3, int(np.ceil((y_max - y_min) / dy)) + 1)

    pts = []
    for i in range(ny):
        for j in range(nx):
            offset = dx / 2 if i % 2 == 1 else 0
            xi = x_min + j * dx + offset
            yi = y_min + i * dy
            pts.append([xi, yi])

    pts = jnp.array(pts)

    # Filter inside
    n_verts = boundary.shape[0]
    def is_inside(pt):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (pt[0] - x1) * (-ey / el) + (pt[1] - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > min_spacing * 0.35

    inside_mask = jax.vmap(is_inside)(pts)
    inside_pts = pts[inside_mask]

    if len(inside_pts) >= n_target:
        indices = jnp.round(jnp.linspace(0, len(inside_pts) - 1, n_target)).astype(int)
        x_init = inside_pts[indices, 0]
        y_init = inside_pts[indices, 1]
    else:
        key = jax.random.PRNGKey(42)
        x_init = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
        key, _ = jax.random.split(key)
        y_init = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

    init = np.concatenate([np.array(x_init), np.array(y_init)])

    # ── Bounds ──
    bounds = [(x_min, x_max)] * n_target + [(y_min, y_max)] * n_target

    # ── Augmented Lagrangian outer loop ──
    # Initialize multipliers
    n_boundary_constraints = n_target
    n_spacing_constraints = n_target * (n_target - 1) // 2
    lambda_b = np.zeros(n_boundary_constraints)
    lambda_s = np.zeros(n_spacing_constraints)
    rho = 10.0  # Initial penalty weight

    current = init
    max_outer_iter = 8
    rho_increase = 2.0

    for outer in range(max_outer_iter):
        # Inner optimization: minimize augmented Lagrangian
        try:
            result = minimize(
                lambda xy: aug_lag_flat(xy, jnp.array(lambda_b), jnp.array(lambda_s), rho),
                current,
                method='L-BFGS-B',
                jac=lambda xy: aug_lag_jac_flat(xy, jnp.array(lambda_b), jnp.array(lambda_s), rho),
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            current = result.x
        except Exception:
            break

        # Evaluate constraints
        x_curr = jnp.array(current[:n_target])
        y_curr = jnp.array(current[n_target:])

        cb = np.array(boundary_constraints(x_curr, y_curr))
        cs = np.array(spacing_constraints(x_curr, y_curr))

        vb = np.maximum(0, -cb)
        vs = np.maximum(0, -cs)

        # Check convergence
        max_violation = max(np.max(vb), np.max(vs))
        if max_violation < 0.1:  # Good enough feasibility
            break

        # Update multipliers
        lambda_b = np.maximum(0, lambda_b + rho * vb)
        lambda_s = np.maximum(0, lambda_s + rho * vs)

        # Increase penalty
        rho *= rho_increase

    opt_x = jnp.array(current[:n_target])
    opt_y = jnp.array(current[n_target:])

    return opt_x, opt_y
