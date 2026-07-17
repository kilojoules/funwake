"""Two-phase noisy search then steadier polish.

HYPOTHESIS: The current best may get most of its benefit from early/mid LR
noise; switching to a quieter second phase may retain basin selection while
reducing late objective/constraint jitter.
AXIS: schedule_two_phase on the best noisy exponential backbone.
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

    key = jax.random.fold_in(jax.random.PRNGKey(22022), step.astype(jnp.uint32))
    lr_noise = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
    rise = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.12) / 0.18))
    fall = jnp.minimum(1.0, jnp.maximum(0.0, (0.72 - t) / 0.22))
    noise_gate = rise * rise * (3.0 - 2.0 * rise)
    noise_gate = noise_gate * fall * fall * (3.0 - 2.0 * fall)

    early_noise = 0.090 * noise_gate
    polish_noise = 0.030 * noise_gate
    noise_amp = jnp.where(t < 0.42, early_noise, polish_noise)
    lr = lr_base + lr_base * noise_amp * lr_noise
    lr = jnp.maximum(lr, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.34 * t)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.2 * late + 54.0 * late * late)

    beta1 = 1.0 / 10.0
    beta2 = 1.0 / 5.0

    return lr, alpha, beta1, beta2
