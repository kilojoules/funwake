"""TopFarm SGD baseline with benchmark-style feasible grid initialization.

HYPOTHESIS: Recreating the benchmark grid from only boundary/min_spacing and running one full TopFarm SGD pass should match the known strong baseline while staying within the runner timeout.
AXIS: topfarm_sgd_solve with conservative single-start settings and deterministic convex-polygon grid initialization.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, boundary_penalty, spacing_penalty
from pixwake.optim.sgd import topfarm_sgd_solve


def _inside_convex(cand_x, cand_y, boundary):
    n_verts = boundary.shape[0]

    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex * ex + ey * ey) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)

    return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 1e-6


def _grid_candidates(boundary, spacing, margin):
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    gx, gy = jnp.meshgrid(
        jnp.arange(x_min + margin, x_max - margin + 1e-6, spacing),
        jnp.arange(y_min + margin, y_max - margin + 1e-6, spacing),
    )
    cand_x = gx.ravel()
    cand_y = gy.ravel()
    inside = _inside_convex(cand_x, cand_y, boundary)
    return cand_x[inside], cand_y[inside]


def _initial_layout(n_target, boundary, min_spacing):
    inside_x, inside_y = _grid_candidates(
        boundary, min_spacing, 0.5 * min_spacing
    )
    if len(inside_x) < n_target:
        inside_x, inside_y = _grid_candidates(
            boundary, 0.7 * min_spacing, 0.25 * min_spacing
        )

    if len(inside_x) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(inside_x) - 1, n_target)).astype(int)
        return inside_x[idx], inside_y[idx]

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    idx = jnp.arange(n_target, dtype=jnp.float64)
    fx = (0.61803398875 * (idx + 0.5)) % 1.0
    fy = (0.75487766625 * (idx + 0.5)) % 1.0
    return x_min + fx * (x_max - x_min), y_min + fy * (y_max - y_min)


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    init_x, init_y = _initial_layout(n_target, boundary, min_spacing)

    settings = SGDSettings(
        learning_rate=50.0,
        max_iter=4000,
        additional_constant_lr_iterations=2000,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.01,
        spacing_weight=1.0,
        boundary_weight=1.0,
    )

    opt_x, opt_y = topfarm_sgd_solve(
        objective, init_x, init_y, boundary, min_spacing, settings
    )

    feasible = jnp.logical_and(
        boundary_penalty(opt_x, opt_y, boundary) < 1e-3,
        spacing_penalty(opt_x, opt_y, min_spacing) < 1e-3,
    )
    improved = objective(opt_x, opt_y) <= objective(init_x, init_y)
    use_opt = jnp.logical_and(feasible, improved)
    return jnp.where(use_opt, opt_x, init_x), jnp.where(use_opt, opt_y, init_y)
