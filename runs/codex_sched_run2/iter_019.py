"""Softer-tail exponential cooling.

HYPOTHESIS: The first exp_decay schedule improved train AEP and generalized;
its final repair may be stronger than necessary, so a slightly slower tail
could retain more wake-objective progress while preserving feasibility.
AXIS: lr_exponential_decay tail-softening with the same inverse penalty
coupling.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00032 * lr0
    curved_t = t * t
    exp_decay = jnp.exp(-8.70 * curved_t)
    lr = jnp.maximum(lr_start * exp_decay, lr_floor)

    lr_safe = jnp.maximum(lr, 1e-10)
    alpha = 20.0 * alpha0 * lr0 / lr_safe

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.0 * late + 52.0 * late * late)

    beta1 = 1.0 / 10.0
    beta2 = 1.0 / 5.0

    return lr, alpha, beta1, beta2
