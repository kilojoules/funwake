"""HYPOTHESIS: A rational LR decay with a mid-late logistic shelf can mimic TopFarm's stable penalty coupling but retain more controllable mobility than the failed exponential plateau.
AXIS: rational_shelf_coupled_penalty
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 12000.0
    base = lr_init / (1.0 + 120.0 * t ** 3.2)
    base = jnp.maximum(base, lr_min)

    rise = 1.0 / (1.0 + jnp.exp(-34.0 * (t - 0.58)))
    fall = 1.0 / (1.0 + jnp.exp(34.0 * (t - 0.75)))
    lr = base * (1.0 + 0.38 * rise * fall)

    alpha_base = 5.4 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    final = 1.0 / (1.0 + jnp.exp(-30.0 * (t - 0.78)))
    alpha = alpha_base + 4.0 * alpha0 * late ** 2 + 4.0 * alpha0 * final

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
