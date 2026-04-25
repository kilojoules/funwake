"""Adaptive coordinate descent with line search.

Strategy:
1. Custom optimization loop: alternating coordinate descent
2. For each turbine, optimize (x, y) while keeping others fixed
3. Use line search along gradient direction for that turbine
4. Adaptive step size based on improvement
5. Multiple sweeps through all turbines
6. Wind-aware initialization
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Adaptive coordinate descent optimizer."""

    # ── Objective and constraints ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def penalized_objective(x, y, alpha=80.0):
        aep = aep_objective(x, y)
        penalty = boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)
        return aep + alpha * penalty

    # Gradient for single turbine
    grad_pen = jax.grad(penalized_objective, argnums=(0, 1))

    # ── Hexagonal initialization ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    dx = min_spacing * 1.12
    dy = min_spacing * 1.12 * jnp.sqrt(3) / 2
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
        x = inside_pts[indices, 0]
        y = inside_pts[indices, 1]
    else:
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
        key, _ = jax.random.split(key)
        y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

    # ── Coordinate descent with line search ──
    alpha = 80.0  # Penalty weight
    n_sweeps = 100  # Number of full sweeps through all turbines
    base_lr = 50.0

    for sweep in range(n_sweeps):
        # Decay learning rate
        lr = base_lr * (1.0 - sweep / n_sweeps) ** 0.5

        # Randomize order for this sweep
        key = jax.random.PRNGKey(sweep)
        order = jax.random.permutation(key, n_target)

        for idx in order:
            # Compute gradient for entire layout
            gx, gy = grad_pen(x, y, alpha)

            # Extract gradient for this turbine
            gx_i = gx[idx]
            gy_i = gy[idx]
            g_norm = jnp.sqrt(gx_i**2 + gy_i**2)

            if g_norm > 1e-8:
                # Normalize direction
                dx_dir = -gx_i / (g_norm + 1e-10)
                dy_dir = -gy_i / (g_norm + 1e-10)

                # Current objective
                obj_current = penalized_objective(x, y, alpha)

                # Try different step sizes (line search)
                best_step = 0.0
                best_obj = obj_current

                for scale in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
                    step = lr * scale
                    x_new = x.at[idx].set(x[idx] + step * dx_dir)
                    y_new = y.at[idx].set(y[idx] + step * dy_dir)

                    # Clip to bounds
                    x_new = x_new.at[idx].set(jnp.clip(x_new[idx], x_min, x_max))
                    y_new = y_new.at[idx].set(jnp.clip(y_new[idx], y_min, y_max))

                    obj_new = penalized_objective(x_new, y_new, alpha)

                    if obj_new < best_obj:
                        best_obj = obj_new
                        best_step = step

                # Apply best step
                if best_step > 0:
                    x = x.at[idx].set(jnp.clip(x[idx] + best_step * dx_dir, x_min, x_max))
                    y = y.at[idx].set(jnp.clip(y[idx] + best_step * dy_dir, y_min, y_max))

        # Early stopping if converged
        if sweep > 20 and sweep % 10 == 0:
            gx, gy = grad_pen(x, y, alpha)
            grad_norm = jnp.sqrt(jnp.mean(gx**2 + gy**2))
            if grad_norm < 1e-4:
                break

    return x, y
