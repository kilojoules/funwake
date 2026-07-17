"""HYPOTHESIS: A slightly stronger alpha coupling and later bump center can preserve the high-scoring path while reducing small late constraint corrections that may cost objective value.
AXIS: cosine_bump_late_alpha_tune
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.32 * lr_init * jnp.exp(-0.5 * ((t - 0.715) / 0.055) ** 2)
    lr = lr_base + bump

    alpha_base = 5.25 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.50, 0.0) / 0.50
    alpha = alpha_base + 3.4 * alpha0 * late ** 2

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
