"""
HYPOTHESIS: The BO-grid/custom-Adam basin has plateaued near 5584 GWh; a
stochastic TopFarm random-restart basin can land in a different feasible
layout family and may beat the incumbent despite using fewer moving parts.
AXIS: Three random bounding-box starts with high-learning-rate
topfarm_sgd_solve, then feasible-first AEP selection.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp

from pixwake.optim.sgd import (
    SGDSettings,
    boundary_penalty,
    spacing_penalty,
    topfarm_sgd_solve,
)


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    settings = SGDSettings(
        learning_rate=100.0,
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-6,
        beta1=0.9,
        beta2=0.999,
        gamma_min_factor=0.01,
        ks_rho=100.0,
        spacing_weight=1.0,
        boundary_weight=1.0,
    )

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    best_score = jnp.inf
    best_x = jnp.zeros((n_target,))
    best_y = jnp.zeros((n_target,))

    key = jax.random.PRNGKey(0)
    for _ in range(3):
        key, x_key, y_key = jax.random.split(key, 3)
        init_x = jax.random.uniform(
            x_key, (n_target,), minval=float(x_min), maxval=float(x_max)
        )
        init_y = jax.random.uniform(
            y_key, (n_target,), minval=float(y_min), maxval=float(y_max)
        )

        opt_x, opt_y = topfarm_sgd_solve(
            objective, init_x, init_y, boundary, min_spacing, settings
        )

        score = objective(opt_x, opt_y)
        penalty = boundary_penalty(opt_x, opt_y, boundary) + spacing_penalty(
            opt_x, opt_y, min_spacing
        )
        selection_score = score + jnp.where(penalty <= 1e-5, 0.0, 1e9 + penalty)

        better = selection_score < best_score
        best_score = jnp.where(better, selection_score, best_score)
        best_x = jnp.where(better, opt_x, best_x)
        best_y = jnp.where(better, opt_y, best_y)

    return best_x, best_y
