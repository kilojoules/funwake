"""Two-stage L-BFGS-B: feasibility then AEP optimization.

Strategy:
1. Stage 1: Use L-BFGS-B to find feasible layout (minimize constraint violations)
2. Stage 2: From feasible start, use L-BFGS-B to maximize AEP with soft penalties
3. Multi-start with 3 different initializations
4. JAX gradients for fast computation
5. Bounds keep turbines roughly in farm area
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Two-stage L-BFGS-B optimizer."""

    # ── Objective functions ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def constraint_violation(x, y):
        return (boundary_penalty(x, y, boundary) +
                spacing_penalty(x, y, min_spacing))

    # For scipy (flat arrays)
    def aep_flat(xy_flat, alpha=0.0):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        aep = aep_objective(x, y)
        if alpha > 0:
            penalty = constraint_violation(x, y)
            return float(aep + alpha * penalty)
        return float(aep)

    def constraint_flat(xy_flat):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        return float(constraint_violation(x, y))

    # JAX gradients
    grad_aep = jax.grad(aep_objective, argnums=(0, 1))
    grad_con = jax.grad(constraint_violation, argnums=(0, 1))

    def aep_jac_flat(xy_flat, alpha=0.0):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        gx, gy = grad_aep(x, y)
        if alpha > 0:
            cx, cy = grad_con(x, y)
            gx = gx + alpha * cx
            gy = gy + alpha * cy
        return np.concatenate([np.array(gx), np.array(gy)])

    def constraint_jac_flat(xy_flat):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        cx, cy = grad_con(x, y)
        return np.concatenate([np.array(cx), np.array(cy)])

    # ── Initialization function ──
    def get_hex_init(seed):
        x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
        x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

        # Hexagonal grid
        dx = min_spacing * 1.15
        dy = min_spacing * 1.15 * np.sqrt(3) / 2

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

        # Filter inside
        n_verts = boundary.shape[0]
        def is_inside(pt):
            def edge_dist(i):
                x1, y1 = boundary[i]
                x2, y2 = boundary[(i + 1) % n_verts]
                ex, ey = x2 - x1, y2 - y1
                el = jnp.sqrt(ex**2 + ey**2) + 1e-10
                return (pt[0] - x1) * (-ey / el) + (pt[1] - y1) * (ex / el)
            return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > min_spacing * 0.3

        inside_mask = jax.vmap(is_inside)(pts)
        inside_pts = pts[inside_mask]

        if len(inside_pts) >= n_target:
            key = jax.random.PRNGKey(seed)
            indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
            selected = inside_pts[indices]
        else:
            # Fallback: random uniform
            key = jax.random.PRNGKey(seed)
            margin = min_spacing * 0.5
            x = jax.random.uniform(key, (n_target,), minval=x_min + margin, maxval=x_max - margin)
            key, _ = jax.random.split(key)
            y = jax.random.uniform(key, (n_target,), minval=y_min + margin, maxval=y_max - margin)
            selected = jnp.stack([x, y], axis=-1)

        return selected

    # ── Bounds ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))
    bounds = [(x_min, x_max)] * n_target + [(y_min, y_max)] * n_target

    # ── Multi-start optimization ──
    best_result = None
    best_objective = float('inf')

    for seed in [42, 123, 777]:
        pts = get_hex_init(seed)
        init = np.concatenate([pts[:, 0], pts[:, 1]])

        # Stage 1: Minimize constraint violation
        try:
            result1 = minimize(
                constraint_flat,
                init,
                method='L-BFGS-B',
                jac=constraint_jac_flat,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-7}
            )

            # Check if reasonably feasible
            if result1.fun < 1.0:
                # Stage 2: Optimize AEP with small penalty
                result2 = minimize(
                    lambda xy: aep_flat(xy, alpha=50.0),
                    result1.x,
                    method='L-BFGS-B',
                    jac=lambda xy: aep_jac_flat(xy, alpha=50.0),
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-7}
                )

                # Evaluate pure AEP
                pure_aep = aep_flat(result2.x, alpha=0.0)
                if pure_aep < best_objective:
                    best_objective = pure_aep
                    best_result = result2.x
            else:
                # If stage 1 fails, try direct optimization with heavy penalty
                result2 = minimize(
                    lambda xy: aep_flat(xy, alpha=200.0),
                    result1.x,
                    method='L-BFGS-B',
                    jac=lambda xy: aep_jac_flat(xy, alpha=200.0),
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-7}
                )

                pure_aep = aep_flat(result2.x, alpha=0.0)
                if pure_aep < best_objective:
                    best_objective = pure_aep
                    best_result = result2.x

        except Exception:
            continue

    # Return best or fallback
    if best_result is not None:
        opt_x = jnp.array(best_result[:n_target])
        opt_y = jnp.array(best_result[n_target:])
    else:
        pts = get_hex_init(42)
        opt_x, opt_y = pts[:, 0], pts[:, 1]

    return opt_x, opt_y
