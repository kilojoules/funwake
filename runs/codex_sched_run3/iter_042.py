import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr = 0.220 * lr0

    early = jnp.clip(t / 0.14, 0.0, 1.0)
    mid = jnp.clip((t - 0.20) / 0.30, 0.0, 1.0)
    late = jnp.clip((t - 0.56) / 0.34, 0.0, 1.0)
    tail = jnp.clip((t - 0.84) / 0.16, 0.0, 1.0)

    early = early * early * (3.0 - 2.0 * early)
    mid = mid * mid * (3.0 - 2.0 * mid)
    late = late * late * (3.0 - 2.0 * late)
    tail = tail * tail * (3.0 - 2.0 * tail)

    alpha = alpha0 * (1.4 + 7.0 * early + 24.0 * mid + 120.0 * late + 430.0 * tail)

    beta1 = 0.04 + 0.04 * late
    beta2 = 0.12 + 0.08 * late

    return lr, alpha, beta1, beta2
