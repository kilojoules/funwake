import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = (212.0 / 240.0) * D
    lr_peak = 1.205 * lr_scale

    warm = jnp.clip(t / 0.090, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    hold = 1.0 + 0.045 * jnp.exp(-0.5 * ((t - 0.300) / 0.170) ** 2)
    early_lift = 1.0 + 0.018 * jnp.exp(-0.5 * ((t - 0.155) / 0.090) ** 2)

    cool = jnp.clip((t - 0.505) / 0.410, 0.0, 1.0)
    cool_s = cool * cool * (3.0 - 2.0 * cool)

    lr_body = lr_peak * (0.655 + 0.345 * warm_s) * hold * early_lift * (1.0 - 0.958 * cool_s)
    lr_body = jnp.maximum(lr_body, 3.12 * gamma_min)

    terminal = jnp.clip((t - 0.915) / 0.085, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    delay = jnp.clip((t - 0.465) / 0.270, 0.0, 1.0)
    delay_s = delay * delay * (3.0 - 2.0 * delay)

    alpha_plateau = alpha0 * D / jnp.maximum(0.365 * lr_scale, 1e-30)
    alpha_late = alpha0 * D / jnp.maximum(2.06 * gamma_min, 1e-30)

    alpha_base = alpha0 * (1.035 + 3.30 * delay_s)
    alpha_main = jnp.minimum(alpha_base, alpha_plateau)

    alpha_spike = alpha_late * (1.0 + 1.80 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.225 - 0.132 * delay_s - 0.058 * terminal_s
    beta2 = 0.132 + 0.230 * delay_s + 0.128 * terminal_s

    return lr, alpha, beta1, beta2