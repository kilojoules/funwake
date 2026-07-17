"""Exponential cooling with fading stochastic LR noise.

HYPOTHESIS: Small multiplicative LR noise on the strong exponential backbone
can shake early wake interactions without disrupting the final constraint
repair phase.
AXIS: lr_noise_injection on high-start exponential cooling with inverse
penalty coupling.
LESSON: pending
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00032 * lr0
    curved_t = t * t
    exp_decay = jnp.exp(-8.00 * curved_t)
    lr = jnp.maximum(lr_start * exp_decay, lr_floor)

    key = jax.random.fold_in(jax.random.PRNGKey(21021), step.astype(jnp.uint32))
    lr_noise = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)  # lr noise
    fade = jnp.maximum(0.0, 1.0 - t / 0.72)
    fade = fade * fade * (3.0 - 2.0 * fade)
    lr += lr * (0.16 * fade) * lr_noise
    lr = jnp.maximum(lr, lr_floor)

    lr_safe = jnp.maximum(lr, 1e-10)
    alpha = 20.0 * alpha0 * lr0 / lr_safe

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.0 * late + 52.0 * late * late)

    beta1 = 1.0 / 10.0
    beta2 = 1.0 / 5.0

    return lr, alpha, beta1, beta2
