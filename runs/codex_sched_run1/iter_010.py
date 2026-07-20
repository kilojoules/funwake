"""HYPOTHESIS: A delayed cosine cool-down with no discrete reheat can spend more steps in the productive high-mobility regime, while the same inverse-LR penalty handles feasibility.
AXIS: delayed_cosine_coupled_penalty
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 3.9 * lr0
    lr_min = lr_init / 10000.0
    hold = 0.12
    q = jnp.maximum(t - hold, 0.0) / (1.0 - hold)
    q = jnp.minimum(q, 1.0)

    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * q))
    lr = jnp.where(t < hold, lr_init, cosine_lr)

    alpha_base = 5.45 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.48, 0.0) / 0.52
    final = 1.0 / (1.0 + jnp.exp(-30.0 * (t - 0.78)))
    alpha = alpha_base + 3.8 * alpha0 * late ** 2 + 3.0 * alpha0 * final

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
