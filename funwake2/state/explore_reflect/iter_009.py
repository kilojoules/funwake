import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.maximum(jnp.asarray(gamma_min, dtype=jnp.float32), 1e-30)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_unit = (212.0 / 240.0) * D

    warm = jnp.clip(t / 0.070, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    c1 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.070) / 0.175, 0.0, 1.0))
    c2 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.245) / 0.215, 0.0, 1.0))
    c3 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.460) / 0.235, 0.0, 1.0))
    c4 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.695) / 0.175, 0.0, 1.0))

    s12 = jnp.clip((t - 0.245) / 0.004, 0.0, 1.0)
    s23 = jnp.clip((t - 0.460) / 0.004, 0.0, 1.0)
    s34 = jnp.clip((t - 0.695) / 0.004, 0.0, 1.0)

    w1 = 1.0 - s12
    w2 = s12 * (1.0 - s23)
    w3 = s23 * (1.0 - s34)
    w4 = s34

    peak = lr_unit * (1.20 * w1 + 1.02 * w2 + 0.76 * w3 + 0.42 * w4)
    trough = lr_unit * (0.54 * w1 + 0.43 * w2 + 0.30 * w3 + 0.155 * w4)
    cyc = c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4

    lr_body = (trough + (peak - trough) * cyc) * (0.58 + 0.42 * warm_s)

    late_cool = jnp.clip((t - 0.870) / 0.060, 0.0, 1.0)
    late_cool_s = late_cool * late_cool * (3.0 - 2.0 * late_cool)
    lr_body = (1.0 - late_cool_s) * lr_body + late_cool_s * (3.40 * gamma_min)

    terminal = jnp.clip((t - 0.930) / 0.070, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * jnp.maximum(lr_body, 3.15 * gamma_min) + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    gate = 1.0 / (1.0 + jnp.exp(-18.0 * (t - 0.520)))
    plateau = alpha0 * 4.45
    alpha_main = alpha0 * (0.92 + (plateau / jnp.maximum(alpha0, 1e-30) - 0.92) * gate)

    restore1_x = (t - 0.360) / 0.040
    restore2_x = (t - 0.650) / 0.050
    restore3_x = (t - 0.805) / 0.035
    restore = (
        0.42 * jnp.exp(-0.5 * restore1_x * restore1_x)
        + 0.34 * jnp.exp(-0.5 * restore2_x * restore2_x)
        + 0.48 * jnp.exp(-0.5 * restore3_x * restore3_x)
    )
    alpha_main = alpha_main * (1.0 + restore)

    alpha_terminal = alpha0 * D / jnp.maximum(1.74 * gamma_min, 1e-30)
    alpha_spike = alpha_terminal * (1.0 + 1.62 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.255 - 0.060 * warm_s - 0.085 * gate - 0.060 * terminal_s
    beta2 = 0.120 + 0.090 * warm_s + 0.155 * gate + 0.160 * terminal_s

    return lr, alpha, beta1, beta2