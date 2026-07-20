import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00034 * lr0
    lr_base = jnp.maximum(lr_start * jnp.exp(-8.0 * t * t), lr_floor)

    phase = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.50) / 0.18))
    phase = phase * phase * (3.0 - 2.0 * phase)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)

    alpha_search = alpha0 * (15.0 + 8.0 * t)
    alpha_repair = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha_repair = alpha_repair * (1.0 + 0.32 * t)
    alpha_repair = alpha_repair * (1.0 + 7.0 * late + 52.0 * late * late)
    alpha = alpha_search * (1.0 - phase) + alpha_repair * phase

    beta1 = 0.1
    beta2 = 0.2

    return lr_base, alpha, beta1, beta2
