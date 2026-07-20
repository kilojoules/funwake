"""Desynchronized micro-jitter on penalty pressure.

HYPOTHESIS: The best basin comes from the proven noisy LR path, but the
constraint path may be too deterministic after the same displacement sequence.
A tiny independently seeded alpha jitter during the active search window can
break LR/penalty synchronization without changing the validated late repair.
AXIS: incumbent LR-noise backbone plus very small alpha jitter.
LESSON: pending
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00038 * lr0
    lr_base = jnp.maximum(lr_start * jnp.exp(-8.0 * t * t), lr_floor)

    lr_key = jax.random.fold_in(jax.random.PRNGKey(22022), step.astype(jnp.uint32))
    lr_noise = jax.random.uniform(lr_key, (), minval=-1.0, maxval=1.0)
    rise = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.12) / 0.18))
    fall = jnp.minimum(1.0, jnp.maximum(0.0, (0.72 - t) / 0.22))
    noise_gate = rise * rise * (3.0 - 2.0 * rise)
    noise_gate = noise_gate * fall * fall * (3.0 - 2.0 * fall)

    lr = lr_base + lr_base * (0.08 * noise_gate) * lr_noise
    lr = jnp.maximum(lr, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.35 * t)

    alpha_key = jax.random.fold_in(jax.random.PRNGKey(78077), step.astype(jnp.uint32))
    alpha_noise = jax.random.uniform(alpha_key, (), minval=-1.0, maxval=1.0)
    alpha = alpha * (1.0 + 0.012 * noise_gate * alpha_noise)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.0 * late + 52.0 * late * late)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
