import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    s = jnp.asarray(step, dtype=jnp.float32)
    T = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    c = 200.0 / 240.0
    lr_peak = 1.07 * c * D

    t = jnp.clip(s / jnp.maximum(T - 1.0, 1.0), 0.0, 1.0)

    warm = 0.06
    hold = 0.42
    final = 0.08

    warm_u = jnp.clip(t / warm, 0.0, 1.0)
    warm_shape = 0.82 + 0.18 * (warm_u * warm_u * (3.0 - 2.0 * warm_u))

    cool_u = jnp.clip((t - hold) / jnp.maximum(1.0 - hold - final, 1e-6), 0.0, 1.0)
    cool_shape = (1.0 - cool_u) ** 1.35
    lr_floor = jnp.maximum(gamma_min, 1e-9)
    lr_main = lr_floor + (lr_peak - lr_floor) * cool_shape

    term_u = jnp.clip((t - (1.0 - final)) / final, 0.0, 1.0)
    term_shape = term_u * term_u * (3.0 - 2.0 * term_u)
    lr_terminal = lr_floor * (1.0 + 0.22 * (1.0 - term_shape))

    lr = jnp.where(t < warm, lr_peak * warm_shape, lr_main)
    lr = jnp.where(t > 1.0 - final, jnp.minimum(lr, lr_terminal), lr)
    lr = jnp.maximum(lr, lr_floor)

    base_alpha = alpha0 * D / jnp.maximum(lr_peak, 1e-30)

    ramp_u = jnp.clip((t - 0.34) / 0.34, 0.0, 1.0)
    ramp_shape = ramp_u * ramp_u * (3.0 - 2.0 * ramp_u)
    alpha_plateau = base_alpha * (1.0 + 2.4 * ramp_shape)

    native_alpha = alpha0 * D / jnp.maximum(lr, 1e-30)
    alpha_blend = 0.72 * alpha_plateau + 0.28 * native_alpha

    spike = 1.0 + 4.5 * term_shape * term_shape
    alpha = alpha_blend * spike

    beta1 = 0.24 - 0.16 * jnp.clip((t - 0.45) / 0.40, 0.0, 1.0)
    beta1 = beta1 - 0.035 * term_shape
    beta2 = 0.14 + 0.16 * (t * t * (3.0 - 2.0 * t)) + 0.05 * term_shape
    beta1 = jnp.clip(beta1, 0.05, 0.26)
    beta2 = jnp.clip(beta2, 0.14, 0.36)

    return lr, alpha, beta1, beta2