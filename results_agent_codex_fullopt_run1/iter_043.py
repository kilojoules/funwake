"""Feasibility-filtered random topfarm multistart.

HYPOTHESIS: The incumbent affine basin has plateaued around 5564 GWh. A small
set of random infeasible bounding-box starts can let topfarm_sgd_solve settle
into a different basin that the deterministic staggered starts never enter.

AXIS: three random bounding-box initial layouts, topfarm_sgd_solve projection,
and explicit benchmark-feasibility filtering when selecting the returned layout.

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
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    objective_jit = jax.jit(objective)

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

    def feasible_benchmark(x, y):
        bnd = boundary_penalty(x, y, boundary)
        spc = spacing_penalty(x, y, min_spacing * 0.99)
        return float(bnd) < 1e-3 and float(spc) < 1e-3

    best_x = None
    best_y = None
    best_aep = -jnp.inf
    best_any_x = None
    best_any_y = None
    best_any_aep = -jnp.inf
    key = jax.random.PRNGKey(0)

    for _ in range(3):
        key, subkey_x, subkey_y = jax.random.split(key, 3)
        init_x = jax.random.uniform(
            subkey_x, (n_target,), minval=float(x_min), maxval=float(x_max)
        )
        init_y = jax.random.uniform(
            subkey_y, (n_target,), minval=float(y_min), maxval=float(y_max)
        )
        opt_x, opt_y = topfarm_sgd_solve(
            objective, init_x, init_y, boundary, min_spacing, settings
        )
        aep = -objective_jit(opt_x, opt_y)

        if aep > best_any_aep:
            best_any_aep = aep
            best_any_x = opt_x
            best_any_y = opt_y

        if feasible_benchmark(opt_x, opt_y) and aep > best_aep:
            best_aep = aep
            best_x = opt_x
            best_y = opt_y

    if best_x is None:
        return best_any_x, best_any_y
    return best_x, best_y
