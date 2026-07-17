"""Monotone exponential cooling with late constraint repair.

HYPOTHESIS: A high-start exp_decay schedule can keep the useful large early
layout movement from the best runs while avoiding discrete escape features;
the late penalty ramp should recover feasibility as the exponential tail
shrinks to very small refinement steps.
AXIS: lr_exponential_decay with inverse penalty coupling and smooth late alpha
repair.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00032 * lr0
    curved_t = t * t
    exp_decay = jnp.exp(-9.43 * curved_t)
    lr = jnp.maximum(lr_start * exp_decay, lr_floor)

    lr_safe = jnp.maximum(lr, 1e-10)
    alpha = 20.0 * alpha0 * lr0 / lr_safe

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.0 * late + 52.0 * late * late)

    beta1 = 1.0 / 10.0
    beta2 = 1.0 / 5.0

    return lr, alpha, beta1, beta2
