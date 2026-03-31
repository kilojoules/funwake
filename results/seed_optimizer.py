"""Baseline optimizer — single-start topfarm_sgd_solve."""

import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, init_x, init_y, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    settings = SGDSettings(learning_rate=50.0, max_iter=4000,
                           additional_constant_lr_iterations=2000, tol=1e-6)
    opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y,
                                      boundary, min_spacing, settings)
    return opt_x, opt_y
