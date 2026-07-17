"""Momentum-annealed noisy exponential schedule.

HYPOTHESIS: The best backbone is close to a sign-gradient method; briefly
raising beta2 during the noisy exploration window may smooth wake-basin
selection, then dropping below the TopFarm pair late may sharpen feasibility
repair without the Gaussian-bump strategy used in the previous attempts.
AXIS: adam_high_beta2 annealed into adam_topfarm_low on a noisy exponential
backbone with quadratic alpha lift.
LESSON: pending
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00032 * lr0
    lr_base = jnp.maximum(lr_start * jnp.exp(-8.0 * t * t), lr_floor)

    key = jax.random.fold_in(jax.random.PRNGKey(26064), step.astype(jnp.uint32))
    lr_noise = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
    rise = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.10) / 0.18))
    fall = jnp.minimum(1.0, jnp.maximum(0.0, (0.74 - t) / 0.24))
    noise_gate = rise * rise * (3.0 - 2.0 * rise)
    noise_gate = noise_gate * fall * fall * (3.0 - 2.0 * fall)

    lr = lr_base * (1.0 + 0.065 * noise_gate * lr_noise)
    lr = jnp.maximum(lr, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.28 * t + 0.16 * t * t)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.0 * late + 58.0 * late * late)

    beta1 = 0.08 + 0.05 * noise_gate
    beta2 = 0.16 + 0.18 * noise_gate - 0.06 * late
    beta2 = jnp.maximum(beta2, 0.08)

    return lr, alpha, beta1, beta2
