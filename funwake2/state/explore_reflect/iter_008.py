import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = (212.0 / 240.0) * D
    lr_floor = 2.65 * gamma_min
    lr_peak = 1.20 * lr_scale

    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    drift = jnp.clip((t - 0.075) / 0.775, 0.0, 1.0)
    drift_s = drift * drift * (3.0 - 2.0 * drift)
    envelope = lr_peak * (1.0 - 0.875 * drift_s)

    c1 = jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.075) / 0.285, 0.0, 1.0))
    c2 = jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.360) / 0.250, 0.0, 1.0))
    c3 = jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.610) / 0.240, 0.0, 1.0))
    burst = 1.0 + 0.105 * (0.5 + 0.5 * c1) * (1.0 - drift_s)
    burst = burst + 0.080 * (0.5 + 0.5 * c2) * (1.0 - 0.65 * drift_s)
    burst = burst + 0.045 * (0.5 + 0.5 * c3) * (1.0 - 0.25 * drift_s)

    lr_body = envelope * burst
    lr_body = lr_body * (0.61 + 0.39 * warm_s)
    lr_body = jnp.maximum(lr_body, lr_floor)

    terminal = jnp.clip((t - 0.915) / 0.085, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp = 1.0 / (1.0 + jnp.exp(-17.5 * (t - 0.515)))
    ramp0 = 1.0 / (1.0 + jnp.exp(17.5 * 0.515))
    ramp1 = 1.0 / (1.0 + jnp.exp(-17.5 * (1.0 - 0.515)))
    ramp_s = jnp.clip((ramp - ramp0) / jnp.maximum(ramp1 - ramp0, 1e-6), 0.0, 1.0)

    burst_a1 = jnp.exp(-0.5 * ((t - 0.475) / 0.055) * ((t - 0.475) / 0.055))
    burst_a2 = jnp.exp(-0.5 * ((t - 0.685) / 0.060) * ((t - 0.685) / 0.060))

    alpha_plateau = alpha0 * 4.05
    alpha_main = alpha0 * (0.96 + 2.62 * ramp_s + 0.34 * burst_a1 + 0.46 * burst_a2)
    alpha_main = jnp.minimum(alpha_main, alpha_plateau)

    alpha_restore = alpha0 * D / jnp.maximum(2.02 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.72 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.238 - 0.100 * ramp_s - 0.070 * terminal_s
    beta2 = 0.118 + 0.250 * ramp_s + 0.135 * terminal_s

    return lr, alpha, beta1, beta2