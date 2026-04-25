"""Hybrid: Differential Evolution for initialization + SGD refinement.

Strategy:
1. Use scipy's differential_evolution for global search (population-based)
2. Run with penalty method for constraints
3. Take best solution and refine with SGD
This combines global exploration with local gradient-based refinement.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Differential Evolution + SGD hybrid optimizer."""

    # ── Objective functions ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # For DE: penalized objective (flat array input)
    def penalized_objective_flat(xy_flat, alpha=100.0):
        x = jnp.array(xy_flat[:n_target])
        y = jnp.array(xy_flat[n_target:])
        aep = aep_objective(x, y)
        penalty = boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)
        return float(aep + alpha * penalty)

    # ── Bounds for DE ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    bounds = [(x_min, x_max)] * n_target + [(y_min, y_max)] * n_target

    # ── Run Differential Evolution (limited iterations for speed) ──
    try:
        result = differential_evolution(
            penalized_objective_flat,
            bounds,
            strategy='best1bin',
            maxiter=100,          # Limited for speed
            popsize=10,           # Small population
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            workers=1,
            updating='deferred',
            polish=False          # We'll polish with SGD
        )

        de_x = jnp.array(result.x[:n_target])
        de_y = jnp.array(result.x[n_target:])
    except Exception:
        # Fallback to grid initialization if DE fails
        wd_rad = jnp.deg2rad(wd)
        dominant = jnp.arctan2(jnp.sum(weights * jnp.sin(wd_rad)),
                               jnp.sum(weights * jnp.cos(wd_rad)))
        angle = dominant + jnp.pi / 2

        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
        translated = boundary - jnp.array([cx, cy])
        rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_bnd = (rot @ translated.T).T

        rx_min, ry_min = jnp.min(rot_bnd, axis=0)
        rx_max, ry_max = jnp.max(rot_bnd, axis=0)
        nx = max(3, int(jnp.ceil((rx_max - rx_min) / min_spacing)))
        ny = max(3, int(jnp.ceil((ry_max - ry_min) / min_spacing)))
        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
            jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
        rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
        inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
        orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])

        n_verts = boundary.shape[0]
        def is_inside(pt):
            def edge_dist(i):
                x1, y1 = boundary[i]
                x2, y2 = boundary[(i + 1) % n_verts]
                ex, ey = x2 - x1, y2 - y1
                el = jnp.sqrt(ex**2 + ey**2) + 1e-10
                return (pt[0] - x1) * (-ey / el) + (pt[1] - y1) * (ex / el)
            return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0

        inside_mask = jax.vmap(is_inside)(orig_pts)
        inside_pts = orig_pts[inside_mask]

        if len(inside_pts) >= n_target:
            key = jax.random.PRNGKey(42)
            indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
            de_x, de_y = inside_pts[indices, 0], inside_pts[indices, 1]
        else:
            key = jax.random.PRNGKey(42)
            de_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
            key, _ = jax.random.split(key)
            de_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # ── Refine with SGD ──
    settings = SGDSettings(
        learning_rate=150.0,
        max_iter=4000,
        additional_constant_lr_iterations=2000,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.005,
        ks_rho=100.0,
        spacing_weight=75.0,
        boundary_weight=75.0,
    )

    opt_x, opt_y = topfarm_sgd_solve(
        aep_objective, de_x, de_y,
        boundary, min_spacing, settings)

    return opt_x, opt_y
