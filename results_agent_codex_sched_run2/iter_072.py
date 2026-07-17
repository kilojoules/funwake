"""Delayed penalty lift on the high-AEP noisy basin entry.

HYPOTHESIS: The incumbent's LR perturbation finds the best basin, but its
linear alpha lift may still spend too much constraint pressure early. Delaying
that extra lift until the layout has moved should preserve wake gains while a
slightly steeper final lift keeps feasibility.
AXIS: delayed alpha ramp layered onto the proven noisy LR backbone.
LESSON: pending
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00032 * lr0
    lr_base = jnp.maximum(lr_start * jnp.exp(-8.0 * t * t), lr_floor)

    key = jax.random.fold_in(jax.random.PRNGKey(22022), step.astype(jnp.uint32))
    lr_noise = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
    rise = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.12) / 0.18))
    fall = jnp.minimum(1.0, jnp.maximum(0.0, (0.72 - t) / 0.22))
    noise_gate = rise * rise * (3.0 - 2.0 * rise)
    noise_gate = noise_gate * fall * fall * (3.0 - 2.0 * fall)

    lr = lr_base + lr_base * (0.08 * noise_gate) * lr_noise
    lr = jnp.maximum(lr, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    delay = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.24) / 0.52))
    delay = delay * delay * (3.0 - 2.0 * delay)
    alpha = alpha * (1.0 + 0.42 * delay)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.69) / 0.31))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.4 * late + 58.0 * late * late)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
