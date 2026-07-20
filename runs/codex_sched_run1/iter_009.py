"""HYPOTHESIS: Replacing the single Gaussian reheat with a broad logistic LR shoulder can preserve mid-late mobility for longer while avoiding the infeasible penalty dips seen earlier.
AXIS: cosine_logistic_shoulder_coupled_penalty
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    rise = 1.0 / (1.0 + jnp.exp(-36.0 * (t - 0.60)))
    fall = 1.0 / (1.0 + jnp.exp(36.0 * (t - 0.77)))
    shoulder = rise * fall
    lr = lr_base * (1.0 + 0.23 * shoulder)

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.2 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
