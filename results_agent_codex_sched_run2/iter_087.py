"""Noisy exponential search with short feasibility settle windows.

HYPOTHESIS: The incumbent noisy exponential envelope repeatedly finds the best
train basin, but mid-run drift may leave some constraint repair until the very
late high-alpha phase. Two brief low-LR/high-alpha settle windows can clean the
layout while preserving most of the noisy exploration path.
AXIS: periodic settle windows layered onto proven noisy exponential envelope.
LESSON: pending
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00038 * lr0
    lr_base = jnp.maximum(lr_start * jnp.exp(-8.0 * t * t), lr_floor)

    key = jax.random.fold_in(jax.random.PRNGKey(22022), step.astype(jnp.uint32))
    lr_noise = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
    rise = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.12) / 0.18))
    fall = jnp.minimum(1.0, jnp.maximum(0.0, (0.72 - t) / 0.22))
    noise_gate = rise * rise * (3.0 - 2.0 * rise)
    noise_gate = noise_gate * fall * fall * (3.0 - 2.0 * fall)

    settle1 = jnp.exp(-0.5 * ((t - 0.38) / 0.045) ** 2)
    settle2 = jnp.exp(-0.5 * ((t - 0.58) / 0.050) ** 2)
    settle = jnp.minimum(1.0, 0.72 * settle1 + 0.58 * settle2)

    lr = lr_base + lr_base * (0.08 * noise_gate) * lr_noise
    lr = lr * (1.0 - 0.105 * noise_gate * settle)
    lr = jnp.maximum(lr, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.35 * t)
    alpha = alpha * (1.0 + 0.155 * noise_gate * settle)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.0 * late + 52.0 * late * late)

    beta1 = 0.1 - 0.018 * noise_gate * settle
    beta2 = 0.2 - 0.035 * noise_gate * settle
    return lr, alpha, beta1, beta2
