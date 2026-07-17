"""Sequential-key random topfarm multistart.

HYPOTHESIS: The high-scoring random-start basin depends on the exact sequential
PRNG split pattern used by the earlier simple topfarm experiment. Reproducing
that start sequence may escape the current deterministic affine plateau.

AXIS: three sequential-key random bounding-box starts with topfarm_sgd_solve,
selecting the highest-AEP returned layout.

LESSON: Pending score.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
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

    best_aep = -jnp.inf
    opt_x_final = None
    opt_y_final = None
    key = jax.random.PRNGKey(0)

    for _ in range(3):
        key, subkey = jax.random.split(key)
        x_min, y_min = jnp.min(boundary, axis=0)
        x_max, y_max = jnp.max(boundary, axis=0)
        init_x = jax.random.uniform(
            subkey, (n_target,), minval=float(x_min), maxval=float(x_max)
        )
        key, subkey = jax.random.split(key)
        init_y = jax.random.uniform(
            subkey, (n_target,), minval=float(y_min), maxval=float(y_max)
        )
        opt_x, opt_y = topfarm_sgd_solve(
            objective, init_x, init_y, boundary, min_spacing, settings
        )
        aep = -objective(opt_x, opt_y)
        if aep > best_aep:
            best_aep = aep
            opt_x_final = opt_x
            opt_y_final = opt_y

    return opt_x_final, opt_y_final
