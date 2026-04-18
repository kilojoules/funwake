"""Trust-region constrained optimization with explicit constraints.

Strategy:
1. Use scipy's trust-constr method (handles nonlinear constraints well)
2. Explicit boundary and spacing constraints (no penalty method)
3. JAX gradients and Hessians via finite differences
4. Multi-start with 3 different initializations
5. Wind-aware initialization
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Trust-region constrained optimizer."""

    # ── Objective and constraints ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # For scipy
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
    # Boundary constraints: min edge distance > 0
    def boundary_constraints_flat(xy_flat):
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

    # Spacing constraints: all pairwise distances >= min_spacing
    def spacing_constraints_flat(xy_flat):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = jnp.sqrt(dx**2 + dy**2)

        # Upper triangle pairs
        i_idx, j_idx = jnp.triu_indices(n_target, k=1)
        pair_dists = dists[i_idx, j_idx]

        return np.array(pair_dists)

    # ── Wind-aware initialization ──
    def get_wind_init(seed):
        x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
        x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

        # Dominant wind direction
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

        # Grid in rotated space
        nx = max(3, int(jnp.ceil((rx_max - rx_min) / min_spacing)))
        ny = max(3, int(jnp.ceil((ry_max - ry_min) / min_spacing)))

        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
            jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
        rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
        inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
        orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])

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

        inside_mask = jax.vmap(is_inside)(orig_pts)
        inside_pts = orig_pts[inside_mask]

        if len(inside_pts) >= n_target:
            key = jax.random.PRNGKey(seed)
            indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
            return inside_pts[indices]
        else:
            key = jax.random.PRNGKey(seed)
            margin = min_spacing * 0.5
            x = jax.random.uniform(key, (n_target,), minval=x_min + margin, maxval=x_max - margin)
            key, _ = jax.random.split(key)
            y = jax.random.uniform(key, (n_target,), minval=y_min + margin, maxval=y_max - margin)
            return jnp.stack([x, y], axis=-1)

    # ── Bounds ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))
    bounds = [(x_min, x_max)] * n_target + [(y_min, y_max)] * n_target

    # ── Constraints for trust-constr ──
    # Boundary: each turbine has positive distance to boundary
    boundary_nlc = NonlinearConstraint(
        boundary_constraints_flat,
        lb=0.0,
        ub=np.inf,
        keep_feasible=False
    )

    # Spacing: all pairs have distance >= min_spacing
    spacing_nlc = NonlinearConstraint(
        spacing_constraints_flat,
        lb=min_spacing,
        ub=np.inf,
        keep_feasible=False
    )

    # ── Multi-start optimization ──
    best_result = None
    best_objective = float('inf')

    for seed in [42, 123, 777]:
        pts = get_wind_init(seed)
        init = np.concatenate([pts[:, 0], pts[:, 1]])

        try:
            result = minimize(
                objective_flat,
                init,
                method='trust-constr',
                jac=jacobian_flat,
                bounds=bounds,
                constraints=[boundary_nlc, spacing_nlc],
                options={
                    'maxiter': 1000,
                    'verbose': 0,
                    'gtol': 1e-7,
                    'xtol': 1e-8,
                    'barrier_tol': 1e-8
                }
            )

            if result.fun < best_objective:
                best_objective = result.fun
                best_result = result

        except Exception:
            continue

    # Return best or fallback
    if best_result is not None:
        opt_x = jnp.array(best_result.x[:n_target])
        opt_y = jnp.array(best_result.x[n_target:])
    else:
        pts = get_wind_init(42)
        opt_x, opt_y = pts[:, 0], pts[:, 1]

    return opt_x, opt_y
