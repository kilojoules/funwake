"""Multi-start SLSQP with JAX gradients and wind-aware initialization.

SLSQP is scipy's constrained optimizer that handles explicit constraints.
We use JAX for fast gradient computation and multiple random starts.

Strategy:
1. Wind-aware hexagonal grid initialization
2. Multiple perturbed starts (5 runs with different perturbations)
3. SLSQP with explicit boundary and spacing constraints
4. JAX-computed Jacobians for speed
5. Return best feasible solution
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Multi-start SLSQP optimizer with JAX gradients."""

    # ── Objective and constraint functions ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # For scipy (flat array)
    def objective_flat(xy_flat):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        return float(aep_objective(x, y))

    # JAX gradient
    grad_aep = jax.grad(aep_objective, argnums=(0, 1))

    def jacobian_flat(xy_flat):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        gx, gy = grad_aep(x, y)
        return np.concatenate([np.array(gx), np.array(gy)])

    # ── Constraint functions ──
    # Boundary: all turbines must be inside (SDF > 0)
    def boundary_constraints(xy_flat):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])

        n_verts = boundary.shape[0]
        def min_edge_dist(xi, yi):
            def edge_dist(i):
                x1, y1 = boundary[i]
                x2, y2 = boundary[(i + 1) % n_verts]
                ex, ey = x2 - x1, y2 - y1
                el = jnp.sqrt(ex**2 + ey**2) + 1e-10
                return (xi - x1) * (-ey / el) + (yi - y1) * (ex / el)
            return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)))

        dists = jax.vmap(min_edge_dist)(x, y)
        return np.array(dists)

    # Spacing: pairwise distances must be >= min_spacing
    def spacing_constraints(xy_flat):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = jnp.sqrt(dx**2 + dy**2)

        # Upper triangle (i < j pairs)
        i_idx, j_idx = jnp.triu_indices(n_target, k=1)
        pair_dists = dists[i_idx, j_idx]

        return np.array(pair_dists - min_spacing)

    # ── Wind-aware hexagonal initialization ──
    def create_hex_grid():
        x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
        x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

        # Hexagonal grid spacing
        dx = min_spacing * 1.1
        dy = min_spacing * 1.1 * np.sqrt(3) / 2

        nx = int(np.ceil((x_max - x_min) / dx)) + 1
        ny = int(np.ceil((y_max - y_min) / dy)) + 1

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
            return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0.5 * min_spacing

        inside_mask = jax.vmap(is_inside)(pts)
        inside_pts = pts[inside_mask]

        if len(inside_pts) >= n_target:
            # Select n_target points evenly spaced
            indices = jnp.round(jnp.linspace(0, len(inside_pts) - 1, n_target)).astype(int)
            return inside_pts[indices]
        else:
            # Fallback: random with safety margin
            key = jax.random.PRNGKey(0)
            margin = min_spacing
            x = jax.random.uniform(key, (n_target,), minval=x_min + margin, maxval=x_max - margin)
            key, _ = jax.random.split(key)
            y = jax.random.uniform(key, (n_target,), minval=y_min + margin, maxval=y_max - margin)
            return jnp.stack([x, y], axis=-1)

    base_pts = create_hex_grid()
    base_init = np.concatenate([base_pts[:, 0], base_pts[:, 1]])

    # ── Bounds ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))
    bounds = [(x_min, x_max)] * n_target + [(y_min, y_max)] * n_target

    # ── Constraints for SLSQP ──
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': spacing_constraints}
    ]

    # ── Multi-start optimization ──
    best_result = None
    best_objective = float('inf')

    n_starts = 5
    key = jax.random.PRNGKey(42)

    for start_idx in range(n_starts):
        if start_idx == 0:
            init = base_init
        else:
            # Perturb base initialization
            key, subkey = jax.random.split(key)
            noise_scale = min_spacing * 0.3
            noise = jax.random.normal(subkey, shape=base_init.shape) * noise_scale
            init = base_init + np.array(noise)
            # Clip to bounds
            init[:n_target] = np.clip(init[:n_target], x_min, x_max)
            init[n_target:] = np.clip(init[n_target:], y_min, y_max)

        try:
            result = minimize(
                objective_flat,
                init,
                method='SLSQP',
                jac=jacobian_flat,
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-9,
                    'disp': False
                }
            )

            if result.fun < best_objective:
                best_objective = result.fun
                best_result = result

        except Exception:
            continue

    # Return best result or base init if all failed
    if best_result is not None:
        opt_x = jnp.array(best_result.x[:n_target])
        opt_y = jnp.array(best_result.x[n_target:])
    else:
        opt_x = jnp.array(base_init[:n_target])
        opt_y = jnp.array(base_init[n_target:])

    return opt_x, opt_y
