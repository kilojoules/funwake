"""Multi-start SLSQP with JAX Jacobians and constraint handling.

Strategy: SLSQP is the community standard for constrained optimization.
We use JAX to compute exact Jacobians for both objective and constraints,
and run multiple starts with diverse initialization strategies.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from pixwake.optim.boundary import polygon_sdf


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Multi-start SLSQP optimizer with JAX gradients."""

    # ── Objective: AEP (to maximize, so return negative for minimize) ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # Wrapper for scipy (takes flat array)
    def objective_flat(xy):
        x, y = xy[:n_target], xy[n_target:]
        return float(aep_objective(x, y))

    # JAX gradient for objective
    grad_aep = jax.grad(aep_objective, argnums=(0, 1))

    def jacobian_flat(xy):
        x, y = xy[:n_target], xy[n_target:]
        gx, gy = grad_aep(x, y)
        return np.concatenate([np.array(gx), np.array(gy)])

    # ── Constraints: boundary and spacing ──
    def boundary_constraint(xy):
        """All turbines must be inside boundary (>0 means feasible)."""
        x, y = xy[:n_target], xy[n_target:]
        coords = jnp.stack([x, y], axis=-1)
        dists = jax.vmap(lambda pt: polygon_sdf(pt, boundary))(coords)
        return np.array(dists)  # Each element > 0 means inside

    def spacing_constraint(xy):
        """All pairwise distances must exceed min_spacing (>0 means feasible)."""
        x, y = xy[:n_target], xy[n_target:]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = jnp.sqrt(dx**2 + dy**2)
        # Extract upper triangle (avoid diagonal and duplicates)
        i_upper, j_upper = jnp.triu_indices(n_target, k=1)
        pair_dists = dists[i_upper, j_upper]
        return np.array(pair_dists - min_spacing)

    # JAX Jacobians for constraints
    jac_boundary = jax.jacfwd(lambda xy: boundary_constraint(xy))
    jac_spacing = jax.jacfwd(lambda xy: spacing_constraint(xy))

    constraints = [
        {'type': 'ineq', 'fun': boundary_constraint, 'jac': jac_boundary},
        {'type': 'ineq', 'fun': spacing_constraint, 'jac': jac_spacing}
    ]

    # ── Initialization strategies ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    # Helper: check if point is inside boundary
    n_verts = boundary.shape[0]
    def is_inside(x, y):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (x - x1) * (-ey / el) + (y - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0

    initial_layouts = []

    # Strategy 1: Wind-aligned grid
    wd_rad = jnp.deg2rad(wd)
    dominant_angle = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
    angle = dominant_angle + jnp.pi / 2

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)
    nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
    ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
        jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])

    inside_mask = jax.vmap(lambda pt: is_inside(pt[0], pt[1]))(orig_pts)
    inside_pts = orig_pts[inside_mask]

    if len(inside_pts) >= n_target:
        key = jax.random.PRNGKey(0)
        indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
        init_1 = inside_pts[indices]
        initial_layouts.append(init_1)

    # Strategy 2: Regular grid
    nx2 = int(jnp.ceil((x_max - x_min) / min_spacing))
    ny2 = int(jnp.ceil((y_max - y_min) / min_spacing))
    gx2, gy2 = jnp.meshgrid(
        jnp.linspace(x_min + min_spacing/2, x_max - min_spacing/2, nx2),
        jnp.linspace(y_min + min_spacing/2, y_max - min_spacing/2, ny2))
    grid_pts = jnp.stack([gx2.flatten(), gy2.flatten()], axis=-1)
    inside_mask2 = jax.vmap(lambda pt: is_inside(pt[0], pt[1]))(grid_pts)
    inside_pts2 = grid_pts[inside_mask2]

    if len(inside_pts2) >= n_target:
        idx2 = jnp.round(jnp.linspace(0, len(inside_pts2) - 1, n_target)).astype(int)
        init_2 = inside_pts2[idx2]
        initial_layouts.append(init_2)

    # Strategy 3: Random with perturbation
    key3 = jax.random.PRNGKey(42)
    rand_pts = jax.random.uniform(key3, (n_target, 2))
    rand_pts = rand_pts * jnp.array([x_max - x_min, y_max - y_min]) + jnp.array([x_min, y_min])
    initial_layouts.append(rand_pts)

    # ── Multi-start optimization ──
    best_aep = -jnp.inf
    best_xy = None

    for i, init_pts in enumerate(initial_layouts):
        xy0 = np.concatenate([np.array(init_pts[:, 0]), np.array(init_pts[:, 1])])

        try:
            result = minimize(
                objective_flat,
                xy0,
                method='SLSQP',
                jac=jacobian_flat,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6}
            )

            if result.success or result.fun < -1000:  # Accept if decent AEP
                aep = -result.fun
                if aep > best_aep:
                    best_aep = aep
                    best_xy = result.x
        except Exception as e:
            # If optimization fails, continue to next start
            continue

    # If no successful optimization, return last attempt or fallback
    if best_xy is None:
        best_xy = xy0

    opt_x = jnp.array(best_xy[:n_target])
    opt_y = jnp.array(best_xy[n_target:])

    return opt_x, opt_y
