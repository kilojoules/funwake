"""High-beta2 Adam on softened exponential cooling.

HYPOTHESIS: A long-horizon second moment may smooth coordinate-wise Adam
scaling enough to preserve the strong exponential schedule's late refinement
while reducing sensitivity to local wake-gradient spikes.
AXIS: adam_high_beta2 on the best feasible exponential cooling backbone.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00032 * lr0
    exp_decay = jnp.exp(-8.00 * t * t)
    lr = jnp.maximum(lr_start * exp_decay, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr, 1e-10)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.0 * late + 52.0 * late * late)

    beta1 = 0.1
    beta2 = 0.995

    return lr, alpha, beta1, beta2
