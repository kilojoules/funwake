import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = D * (218.0 / 240.0)
    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    macro = jnp.clip((t - 0.12) / 0.74, 0.0, 1.0)
    macro_s = macro * macro * (3.0 - 2.0 * macro)

    cyc_phase = jnp.mod(jnp.maximum(t - 0.08, 0.0) / 0.255, 1.0)
    cyc = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * cyc_phase)
    cyc_amp = (1.0 - macro_s) * (0.18 + 0.10 * (1.0 - t))

    lr_high = lr_scale * (0.72 + 0.34 * warm_s)
    lr_floor_body = jnp.maximum(3.35 * gamma_min, 0.040 * lr_scale)
    lr_trend = (1.0 - macro_s) * lr_high + macro_s * lr_floor_body
    lr_body = lr_trend * (1.0 + cyc_amp * cyc)

    terminal = jnp.clip((t - 0.905) / 0.095, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp = jnp.clip((t - 0.36) / 0.31, 0.0, 1.0)
    ramp_s = ramp * ramp * (3.0 - 2.0 * ramp)

    burst1 = jnp.exp(-0.5 * ((t - 0.55) / 0.055) ** 2)
    burst2 = jnp.exp(-0.5 * ((t - 0.735) / 0.045) ** 2)

    alpha_plateau = 3.35 * alpha0
    alpha_body = alpha0 * (0.92 + 2.43 * ramp_s)
    alpha_body = jnp.minimum(alpha_body, alpha_plateau)
    alpha_body = alpha_body * (1.0 + 0.24 * burst1 + 0.32 * burst2)

    alpha_late = alpha0 * D / jnp.maximum(1.82 * gamma_min, 1e-30)
    alpha_spike = alpha_late * (1.0 + 1.35 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_body + terminal_s * alpha_spike

    beta1 = 0.245 - 0.080 * warm_s - 0.082 * ramp_s - 0.045 * terminal_s
    beta2 = 0.125 + 0.095 * warm_s + 0.185 * ramp_s + 0.135 * terminal_s

    return lr, alpha, beta1, beta2