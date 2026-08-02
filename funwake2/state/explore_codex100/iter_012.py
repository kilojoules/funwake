import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    c = 0.875
    lr_peak = c * D
    warm = 0.055
    hold = 0.43
    spike_start = 0.925

    warm_s = jnp.clip(t / warm, 0.0, 1.0)
    warm_s = warm_s * warm_s * (3.0 - 2.0 * warm_s)
    lr_warm = lr_peak * (0.72 + 0.28 * warm_s)

    decay_u = jnp.clip((t - hold) / jnp.maximum(spike_start - hold, 1e-6), 0.0, 1.0)
    decay_s = decay_u * decay_u * (3.0 - 2.0 * decay_u)
    lr_main = gamma_min + (lr_peak - gamma_min) * (1.0 - decay_s) ** 1.35

    lr_pre = jnp.where(t < hold, lr_warm, lr_main)

    end_u = jnp.clip((t - spike_start) / jnp.maximum(1.0 - spike_start, 1e-6), 0.0, 1.0)
    end_s = end_u * end_u * (3.0 - 2.0 * end_u)
    lr_terminal = gamma_min + (lr_pre - gamma_min) * (1.0 - end_s) ** 2.25
    lr = jnp.maximum(lr_terminal, gamma_min)

    mid_u = jnp.clip((t - 0.48) / 0.30, 0.0, 1.0)
    mid_s = mid_u * mid_u * (3.0 - 2.0 * mid_u)
    terminal_s = end_s * end_s

    native_alpha = alpha0 * D / jnp.maximum(lr, 1e-30)
    plateau_alpha = alpha0 * (1.15 + 3.65 * mid_s)
    spike_alpha = alpha0 * (5.0 + 10.0 * terminal_s)
    alpha = jnp.maximum(native_alpha, jnp.maximum(plateau_alpha, spike_alpha))

    beta1 = 0.16 - 0.09 * jnp.clip((t - 0.52) / 0.40, 0.0, 1.0) - 0.035 * end_s
    beta2 = 0.18 + 0.10 * mid_s + 0.08 * end_s

    return lr, alpha, beta1, beta2