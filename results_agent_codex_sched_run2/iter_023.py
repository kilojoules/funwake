"""Feasibility-biased mid-course LR noise.

HYPOTHESIS: Attempt 22 reached a better AEP basin but under-repaired
constraints; a smaller earlier-ending LR noise window plus a stronger late
penalty ramp may retain the basin while recovering feasibility.
AXIS: lr_noise_injection with feasibility-biased late repair.
LESSON: pending
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00032 * lr0
    exp_decay = jnp.exp(-8.00 * t * t)
    lr_base = jnp.maximum(lr_start * exp_decay, lr_floor)

    key = jax.random.fold_in(jax.random.PRNGKey(23023), step.astype(jnp.uint32))
    lr_noise = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)  # lr noise
    rise = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.12) / 0.18))
    fall = jnp.minimum(1.0, jnp.maximum(0.0, (0.64 - t) / 0.20))
    noise_gate = rise * rise * (3.0 - 2.0 * rise)
    noise_gate = noise_gate * fall * fall * (3.0 - 2.0 * fall)

    lr = lr_base
    lr += lr_base * (0.055 * noise_gate) * lr_noise
    lr = jnp.maximum(lr, lr_floor)

    alpha = 21.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.66) / 0.34))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 8.0 * late + 70.0 * late * late)

    beta1 = 1.0 / 10.0
    beta2 = 1.0 / 5.0

    return lr, alpha, beta1, beta2
