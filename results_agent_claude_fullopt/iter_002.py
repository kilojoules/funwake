"""L-BFGS-B with JAX gradients and adaptive constraint penalty.

Strategy: Use L-BFGS-B (quasi-Newton) with exact JAX gradients.
Instead of SLSQP's constraints, we use penalty methods with adaptive
weight ramping. Multiple starts with diverse initialization.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """L-BFGS-B optimizer with penalty method and multi-start."""

    # ── Objective: AEP ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Augmented objective with constraints ──
    def augmented_objective(xy, alpha):
        """Negative AEP + alpha * constraint_violation."""
        x, y = xy[:n_target], xy[n_target:]
        aep = aep_objective(x, y)
        con_penalty = boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)
        return aep + alpha * con_penalty

    # Wrapper for scipy
    def objective_flat(xy, alpha):
        return float(augmented_objective(xy, alpha))

    # JAX gradient
    def jacobian_flat(xy, alpha):
        grad_fn = jax.grad(augmented_objective)
        g = grad_fn(xy, alpha)
        return np.array(g)

    # ── Initialization strategies ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    # Helper: check if point is inside boundary
    n_verts = boundary.shape[0]
    def edge_dist(x, y, i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (x - x1) * (-ey / el) + (y - y1) * (ex / el)

    def is_inside(pt):
        dists = jax.vmap(lambda i: edge_dist(pt[0], pt[1], i))(jnp.arange(n_verts))
        return jnp.min(dists) > 0

    initial_layouts = []

    # Strategy 1: Wind-aligned grid
    wd_rad = jnp.deg2rad(wd)
    dominant_angle = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
    angle = dominant_angle + jnp.pi / 2

    cos_a, sin_a = float(jnp.cos(angle)), float(jnp.sin(angle))
    cx, cy = float(jnp.mean(boundary[:, 0])), float(jnp.mean(boundary[:, 1]))

    # Create rotated grid
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = float(jnp.min(rot_bnd[:, 0])), float(jnp.min(rot_bnd[:, 1]))
    rx_max, ry_max = float(jnp.max(rot_bnd[:, 0])), float(jnp.max(rot_bnd[:, 1]))

    spacing = float(min_spacing)
    nx = max(3, int(jnp.ceil((rx_max - rx_min) / spacing)))
    ny = max(3, int(jnp.ceil((ry_max - ry_min) / spacing)))

    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + spacing/2, rx_max - spacing/2, nx),
        jnp.linspace(ry_min + spacing/2, ry_max - spacing/2, ny))
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])

    inside_mask = jax.vmap(is_inside)(orig_pts)
    inside_pts = orig_pts[inside_mask]

    if len(inside_pts) >= n_target:
        key = jax.random.PRNGKey(0)
        indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
        init_1 = inside_pts[indices]
        xy1 = np.concatenate([np.array(init_1[:, 0]), np.array(init_1[:, 1])])
        initial_layouts.append(xy1)

    # Strategy 2: Regular grid
    nx2 = max(3, int(jnp.ceil((x_max - x_min) / spacing)))
    ny2 = max(3, int(jnp.ceil((y_max - y_min) / spacing)))
    gx2, gy2 = jnp.meshgrid(
        jnp.linspace(x_min + spacing/2, x_max - spacing/2, nx2),
        jnp.linspace(y_min + spacing/2, y_max - spacing/2, ny2))
    grid_pts = jnp.stack([gx2.flatten(), gy2.flatten()], axis=-1)
    inside_mask2 = jax.vmap(is_inside)(grid_pts)
    inside_pts2 = grid_pts[inside_mask2]

    if len(inside_pts2) >= n_target:
        idx2 = jnp.round(jnp.linspace(0, len(inside_pts2) - 1, n_target)).astype(int)
        init_2 = inside_pts2[idx2]
        xy2 = np.concatenate([np.array(init_2[:, 0]), np.array(init_2[:, 1])])
        initial_layouts.append(xy2)

    # Strategy 3-5: Random with different seeds
    for seed in [42, 123, 456]:
        key = jax.random.PRNGKey(seed)
        rand_pts = jax.random.uniform(key, (n_target, 2))
        rand_pts = rand_pts * jnp.array([x_max - x_min, y_max - y_min]) + jnp.array([x_min, y_min])
        xy_rand = np.concatenate([np.array(rand_pts[:, 0]), np.array(rand_pts[:, 1])])
        initial_layouts.append(xy_rand)

    # ── Adaptive penalty optimization ──
    best_aep = -jnp.inf
    best_xy = None

    # Try increasing penalty weights
    alphas = [10.0, 50.0, 100.0]

    for xy0 in initial_layouts:
        for alpha in alphas:
            try:
                result = minimize(
                    objective_flat,
                    xy0,
                    args=(alpha,),
                    method='L-BFGS-B',
                    jac=jacobian_flat,
                    options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7}
                )

                # Check feasibility
                x_res = result.x[:n_target]
                y_res = result.x[n_target:]
                bnd_pen = float(boundary_penalty(jnp.array(x_res), jnp.array(y_res), boundary))
                spc_pen = float(spacing_penalty(jnp.array(x_res), jnp.array(y_res), min_spacing))

                if bnd_pen < 1e-2 and spc_pen < 1e-2:  # Feasible enough
                    aep = -float(aep_objective(jnp.array(x_res), jnp.array(y_res)))
                    if aep > best_aep:
                        best_aep = aep
                        best_xy = result.x
            except Exception:
                continue

    # If no feasible solution, return best infeasible
    if best_xy is None and len(initial_layouts) > 0:
        best_xy = initial_layouts[0]

    opt_x = jnp.array(best_xy[:n_target])
    opt_y = jnp.array(best_xy[n_target:])

    return opt_x, opt_y
