import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = D * (224.0 / 240.0)
    lr_floor = jnp.maximum(gamma_min, 1e-30)

    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    cosine1 = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.075) / 0.365, 0.0, 1.0))
    cosine2 = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.440) / 0.295, 0.0, 1.0))
    cosine3 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.735) / 0.185, 0.0, 1.0))

    gate1 = 1.0 - jnp.clip((t - 0.440) / 0.020, 0.0, 1.0)
    gate2 = jnp.clip((t - 0.420) / 0.020, 0.0, 1.0) * (1.0 - jnp.clip((t - 0.735) / 0.020, 0.0, 1.0))
    gate3 = jnp.clip((t - 0.715) / 0.020, 0.0, 1.0)

    lr_cycle1 = lr_scale * (0.78 + 0.33 * warm_s) * (0.54 + 0.46 * cosine1)
    lr_cycle2 = lr_scale * (0.47 + 0.40 * cosine2)
    lr_cycle3 = lr_scale * (0.18 + 0.32 * cosine3)

    lr_body = gate1 * lr_cycle1 + gate2 * lr_cycle2 + gate3 * lr_cycle3
    lr_body = jnp.maximum(lr_body, 3.15 * lr_floor)

    terminal = jnp.clip((t - 0.915) / 0.085, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * lr_floor
    lr = jnp.maximum(lr, lr_floor)

    ramp = jnp.clip((t - 0.345) / 0.245, 0.0, 1.0)
    ramp_s = ramp * ramp * (3.0 - 2.0 * ramp)

    burst1 = jnp.exp(-0.5 * ((t - 0.445) / 0.045) ** 2)
    burst2 = jnp.exp(-0.5 * ((t - 0.715) / 0.038) ** 2)

    alpha_plateau = alpha0 * 3.85
    alpha_main = alpha0 * (0.92 + 2.93 * ramp_s + 0.46 * burst1 + 0.36 * burst2)
    alpha_main = jnp.minimum(alpha_main, alpha_plateau)

    alpha_restore = alpha0 * D / jnp.maximum(1.82 * lr_floor, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.85 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.235 - 0.060 * warm_s - 0.070 * ramp_s - 0.065 * terminal_s
    beta2 = 0.125 + 0.135 * warm_s + 0.165 * ramp_s + 0.150 * terminal_s

    return lr, alpha, beta1, beta2