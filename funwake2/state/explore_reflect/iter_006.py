import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = (214.0 / 240.0) * D
    lr_floor = 2.65 * gamma_min

    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    sgdr1 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip(t / 0.34, 0.0, 1.0))
    sgdr2_u = jnp.clip((t - 0.34) / 0.315, 0.0, 1.0)
    sgdr2 = 0.5 + 0.5 * jnp.cos(jnp.pi * sgdr2_u)
    sgdr3_u = jnp.clip((t - 0.655) / 0.255, 0.0, 1.0)
    sgdr3 = 0.5 + 0.5 * jnp.cos(jnp.pi * sgdr3_u)

    cycle1 = 0.72 + 0.34 * sgdr1
    cycle2 = 0.61 + 0.30 * sgdr2
    cycle3 = 0.37 + 0.22 * sgdr3

    p2 = jnp.clip((t - 0.30) / 0.08, 0.0, 1.0)
    p2 = p2 * p2 * (3.0 - 2.0 * p2)
    p3 = jnp.clip((t - 0.615) / 0.09, 0.0, 1.0)
    p3 = p3 * p3 * (3.0 - 2.0 * p3)

    lr_cyclic = lr_scale * ((1.0 - p2) * cycle1 + p2 * ((1.0 - p3) * cycle2 + p3 * cycle3))
    lr_body = lr_cyclic * (0.64 + 0.36 * warm_s)

    settle = jnp.clip((t - 0.77) / 0.145, 0.0, 1.0)
    settle_s = settle * settle * (3.0 - 2.0 * settle)
    lr_body = (1.0 - settle_s) * lr_body + settle_s * lr_floor
    lr_body = jnp.maximum(lr_body, lr_floor)

    terminal = jnp.clip((t - 0.912) / 0.088, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp = jnp.clip((t - 0.39) / 0.30, 0.0, 1.0)
    ramp_s = ramp * ramp * (3.0 - 2.0 * ramp)

    burst1_u = (t - 0.515) / 0.070
    burst2_u = (t - 0.735) / 0.060
    burst1 = jnp.exp(-0.5 * burst1_u * burst1_u)
    burst2 = jnp.exp(-0.5 * burst2_u * burst2_u)

    alpha_plateau = 3.38 * alpha0
    alpha_base = alpha0 * (0.92 + 2.46 * ramp_s)
    alpha_bursts = alpha0 * (0.42 * burst1 + 0.66 * burst2)
    alpha_main = jnp.minimum(alpha_base + alpha_bursts, alpha_plateau)

    alpha_restore = alpha0 * D / jnp.maximum(2.05 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.68 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.245 - 0.100 * ramp_s - 0.075 * terminal_s + 0.030 * (p2 - p3)
    beta1 = jnp.clip(beta1, 0.045, 0.285)

    beta2 = 0.120 + 0.250 * ramp_s + 0.130 * terminal_s
    beta2 = jnp.clip(beta2, 0.110, 0.535)

    return lr, alpha, beta1, beta2