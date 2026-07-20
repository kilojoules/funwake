"""HYPOTHESIS: A long high-mobility plateau followed by smooth exponential cooling can use the successful strong inverse-LR penalty coupling without relying on the previous cosine/gaussian reheat path.
AXIS: exp_plateau_coupled_penalty_momentum_anneal
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    plateau_end = 0.36
    u = jnp.maximum(t - plateau_end, 0.0) / (1.0 - plateau_end)

    # Hold the wide-search step size early, then cool exponentially to a
    # small but nonzero polishing rate.
    lr_floor = lr_init / 18000.0
    exp_cool = jnp.exp(-9.4 * u ** 1.35)
    lr = lr_floor + (lr_init - lr_floor) * exp_cool

    alpha_base = 6.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.52, 0.0) / 0.48
    final = 1.0 / (1.0 + jnp.exp(-32.0 * (t - 0.78)))
    alpha = alpha_base + 4.5 * alpha0 * late ** 2 + 8.0 * alpha0 * final

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
