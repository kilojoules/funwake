import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00034 * lr0
    lr = jnp.maximum(lr_start * jnp.exp(-8.0 * t * t), lr_floor)

    phase = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.49) / 0.19))
    phase = phase * phase * (3.0 - 2.0 * phase)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.67) / 0.33))
    late = late * late * (3.0 - 2.0 * late)

    alpha_search = alpha0 * (16.5 + 9.0 * t)
    alpha_repair = 20.0 * alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    alpha_repair = alpha_repair * (1.0 + 0.36 * t)
    alpha_repair = alpha_repair * (1.0 + 9.0 * late + 76.0 * late * late)
    alpha = alpha_search * (1.0 - phase) + alpha_repair * phase

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
