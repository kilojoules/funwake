"""Simulated annealing with local SGD refinement.

Strategy:
1. Use scipy's basin-hopping (simulated annealing variant) for global search
2. Each local step uses L-BFGS-B with penalties
3. Acceptance criterion allows escaping local minima
4. Final refinement with SGD for fine-tuning
5. Wind-aware initialization
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import basinhopping, minimize
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Basin-hopping + SGD refinement optimizer."""

    # ── Objective and constraints ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def penalized_objective_flat(xy_flat, alpha=100.0):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        aep = aep_objective(x, y)
        penalty = boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)
        return float(aep + alpha * penalty)

    # Gradient for local minimizer
    grad_aep = jax.grad(aep_objective, argnums=(0, 1))
    grad_pen = jax.grad(lambda x, y: boundary_penalty(x, y, boundary) +
                        spacing_penalty(x, y, min_spacing), argnums=(0, 1))

    def penalized_jac_flat(xy_flat, alpha=100.0):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        gx, gy = grad_aep(x, y)
        px, py = grad_pen(x, y)
        return np.concatenate([np.array(gx + alpha * px), np.array(gy + alpha * py)])

    # ── Hexagonal initialization ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    dx = min_spacing * 1.1
    dy = min_spacing * 1.1 * np.sqrt(3) / 2
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
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > min_spacing * 0.3

    inside_mask = jax.vmap(is_inside)(pts)
    inside_pts = pts[inside_mask]

    if len(inside_pts) >= n_target:
        indices = jnp.round(jnp.linspace(0, len(inside_pts) - 1, n_target)).astype(int)
        init_pts = inside_pts[indices]
    else:
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
        key, _ = jax.random.split(key)
        y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)
        init_pts = jnp.stack([x, y], axis=-1)

    init = np.concatenate([init_pts[:, 0], init_pts[:, 1]])

    # ── Bounds ──
    bounds = [(x_min, x_max)] * n_target + [(y_min, y_max)] * n_target

    # ── Local minimizer for basin-hopping ──
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'jac': lambda xy: penalized_jac_flat(xy, alpha=100.0),
        'bounds': bounds,
        'options': {'maxiter': 200, 'ftol': 1e-8}
    }

    # ── Basin-hopping (limited iterations for speed) ──
    try:
        result = basinhopping(
            lambda xy: penalized_objective_flat(xy, alpha=100.0),
            init,
            minimizer_kwargs=minimizer_kwargs,
            niter=30,                    # 30 basin-hopping iterations
            T=1.0,                       # Temperature for acceptance
            stepsize=min_spacing * 0.5,  # Step size for random displacement
            seed=42
        )
        bh_x = jnp.array(result.x[:n_target])
        bh_y = jnp.array(result.x[n_target:])
    except Exception:
        # Fallback to init
        bh_x = jnp.array(init[:n_target])
        bh_y = jnp.array(init[n_target:])

    # ── Final refinement with SGD ──
    settings = SGDSettings(
        learning_rate=120.0,
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.005,
        ks_rho=120.0,
        spacing_weight=80.0,
        boundary_weight=80.0,
    )

    opt_x, opt_y = topfarm_sgd_solve(
        aep_objective, bh_x, bh_y,
        boundary, min_spacing, settings)

    return opt_x, opt_y
