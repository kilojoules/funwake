import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr0 = (200.0 / 240.0) * D
    lr_peak = 1.08 * lr0

    warm = jnp.clip(t / 0.10, 0.0, 1.0)
    hold = 1.0 - jnp.clip((t - 0.36) / 0.64, 0.0, 1.0)
    smooth_hold = hold * hold * (3.0 - 2.0 * hold)

    lr_body = lr_peak * (0.72 + 0.28 * warm) * (0.04 + 0.96 * smooth_hold)
    lr_body = jnp.maximum(lr_body, 2.5 * gamma_min)

    terminal = jnp.clip((t - 0.92) / 0.08, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    delay = jnp.clip((t - 0.38) / 0.34, 0.0, 1.0)
    delay_s = delay * delay * (3.0 - 2.0 * delay)

    alpha_plateau = alpha0 * D / jnp.maximum(0.42 * lr0, 1e-30)
    alpha_late = alpha0 * D / jnp.maximum(2.5 * gamma_min, 1e-30)
    alpha_base = alpha0 * (1.15 + 2.85 * delay_s)
    alpha_main = jnp.minimum(alpha_base, alpha_plateau)

    alpha_spike = alpha_late * (1.0 + 1.25 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.18 - 0.10 * delay_s - 0.04 * terminal_s
    beta2 = 0.16 + 0.18 * delay_s + 0.10 * terminal_s

    return lr, alpha, beta1, beta2