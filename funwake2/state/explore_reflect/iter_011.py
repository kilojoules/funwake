import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_anchor = (226.0 / 240.0) * D
    lr_floor = 2.70 * gamma_min

    warm = jnp.clip(t / 0.085, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    envelope = 1.08 - 0.20 * jnp.clip(t / 0.50, 0.0, 1.0)
    decay = jnp.clip((t - 0.50) / 0.405, 0.0, 1.0)
    decay_s = decay * decay * (3.0 - 2.0 * decay)
    envelope = envelope * (1.0 - 0.87 * decay_s)

    c1_u = jnp.clip(t / 0.255, 0.0, 1.0)
    c2_u = jnp.clip((t - 0.255) / 0.235, 0.0, 1.0)
    c3_u = jnp.clip((t - 0.490) / 0.220, 0.0, 1.0)
    c4_u = jnp.clip((t - 0.710) / 0.195, 0.0, 1.0)

    c1 = 0.5 + 0.5 * jnp.cos(jnp.pi * c1_u)
    c2 = 0.5 + 0.5 * jnp.cos(jnp.pi * c2_u)
    c3 = 0.5 + 0.5 * jnp.cos(jnp.pi * c3_u)
    c4 = 0.5 + 0.5 * jnp.cos(jnp.pi * c4_u)

    p2 = jnp.clip((t - 0.230) / 0.060, 0.0, 1.0)
    p2 = p2 * p2 * (3.0 - 2.0 * p2)
    p3 = jnp.clip((t - 0.455) / 0.070, 0.0, 1.0)
    p3 = p3 * p3 * (3.0 - 2.0 * p3)
    p4 = jnp.clip((t - 0.680) / 0.070, 0.0, 1.0)
    p4 = p4 * p4 * (3.0 - 2.0 * p4)

    cycle12 = (1.0 - p2) * (0.78 + 0.38 * c1) + p2 * (0.72 + 0.34 * c2)
    cycle34 = (1.0 - p4) * (0.62 + 0.29 * c3) + p4 * (0.44 + 0.22 * c4)
    cycle = (1.0 - p3) * cycle12 + p3 * cycle34

    lr_body = lr_anchor * (0.66 + 0.34 * warm_s) * envelope * cycle
    lr_body = jnp.maximum(lr_body, lr_floor)

    terminal = jnp.clip((t - 0.910) / 0.090, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp = jnp.clip((t - 0.360) / 0.285, 0.0, 1.0)
    ramp_s = ramp * ramp * (3.0 - 2.0 * ramp)

    hold = jnp.clip((t - 0.705) / 0.120, 0.0, 1.0)
    hold_s = hold * hold * (3.0 - 2.0 * hold)

    b1_u = (t - 0.455) / 0.052
    b2_u = (t - 0.640) / 0.050
    b3_u = (t - 0.805) / 0.044
    burst1 = jnp.exp(-0.5 * b1_u * b1_u)
    burst2 = jnp.exp(-0.5 * b2_u * b2_u)
    burst3 = jnp.exp(-0.5 * b3_u * b3_u)

    alpha_plateau = 3.85 * alpha0
    alpha_base = alpha0 * (0.86 + 3.05 * ramp_s + 0.36 * hold_s)
    alpha_cyclic = alpha0 * (0.30 * burst1 + 0.48 * burst2 + 0.70 * burst3)
    alpha_main = jnp.minimum(alpha_base + alpha_cyclic, alpha_plateau)

    alpha_restore = alpha0 * D / jnp.maximum(2.05 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.72 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.255 - 0.095 * ramp_s - 0.065 * hold_s - 0.070 * terminal_s
    beta1 = beta1 + 0.026 * (c2 - c4) * (1.0 - terminal_s)
    beta1 = jnp.clip(beta1, 0.050, 0.285)

    beta2 = 0.118 + 0.205 * ramp_s + 0.082 * hold_s + 0.132 * terminal_s
    beta2 = beta2 + 0.030 * (burst2 + burst3)
    beta2 = jnp.clip(beta2, 0.110, 0.545)

    return lr, alpha, beta1, beta2