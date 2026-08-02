import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = (205.0 / 240.0) * D
    lr_peak = 1.145 * lr_scale

    warm = jnp.clip(t / 0.085, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    cool = jnp.clip((t - 0.435) / 0.49, 0.0, 1.0)
    cool_s = cool * cool * (3.0 - 2.0 * cool)

    lr_body = lr_peak * (0.715 + 0.285 * warm_s) * (1.0 - 0.952 * cool_s)
    lr_body = jnp.maximum(lr_body, 2.75 * gamma_min)

    terminal = jnp.clip((t - 0.92) / 0.08, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    delay = jnp.clip((t - 0.425) / 0.305, 0.0, 1.0)
    delay_s = delay * delay * (3.0 - 2.0 * delay)

    alpha_plateau = alpha0 * D / jnp.maximum(0.405 * lr_scale, 1e-30)
    alpha_late = alpha0 * D / jnp.maximum(2.18 * gamma_min, 1e-30)

    alpha_base = alpha0 * (1.055 + 3.05 * delay_s)
    alpha_main = jnp.minimum(alpha_base, alpha_plateau)

    alpha_spike = alpha_late * (1.0 + 1.48 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.205 - 0.115 * delay_s - 0.048 * terminal_s
    beta2 = 0.145 + 0.205 * delay_s + 0.115 * terminal_s

    return lr, alpha, beta1, beta2