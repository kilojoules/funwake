import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.maximum(jnp.asarray(gamma_min, dtype=jnp.float32), 1e-30)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_unit = (218.0 / 240.0) * D

    warm = jnp.clip(t / 0.060, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    macro = jnp.clip((t - 0.090) / 0.815, 0.0, 1.0)
    macro_s = macro * macro * (3.0 - 2.0 * macro)
    envelope = 1.22 - 1.06 * macro_s

    p1 = jnp.clip((t - 0.060) / 0.165, 0.0, 1.0)
    p2 = jnp.clip((t - 0.225) / 0.205, 0.0, 1.0)
    p3 = jnp.clip((t - 0.430) / 0.235, 0.0, 1.0)
    p4 = jnp.clip((t - 0.665) / 0.210, 0.0, 1.0)

    c1 = 0.5 + 0.5 * jnp.cos(jnp.pi * p1)
    c2 = 0.5 + 0.5 * jnp.cos(jnp.pi * p2)
    c3 = 0.5 + 0.5 * jnp.cos(jnp.pi * p3)
    c4 = 0.5 + 0.5 * jnp.cos(jnp.pi * p4)

    s12 = jnp.clip((t - 0.225) / 0.018, 0.0, 1.0)
    s23 = jnp.clip((t - 0.430) / 0.020, 0.0, 1.0)
    s34 = jnp.clip((t - 0.665) / 0.022, 0.0, 1.0)

    w1 = 1.0 - s12
    w2 = s12 * (1.0 - s23)
    w3 = s23 * (1.0 - s34)
    w4 = s34

    cyc = c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4
    floor_mult = 0.52 * w1 + 0.47 * w2 + 0.34 * w3 + 0.205 * w4
    amp_mult = 0.58 * w1 + 0.48 * w2 + 0.34 * w3 + 0.175 * w4

    lr_body = lr_unit * (floor_mult + amp_mult * cyc) * (0.64 + 0.36 * warm_s) * envelope
    pre_terminal = jnp.clip((t - 0.875) / 0.050, 0.0, 1.0)
    pre_terminal_s = pre_terminal * pre_terminal * (3.0 - 2.0 * pre_terminal)
    lr_body = (1.0 - pre_terminal_s) * lr_body + pre_terminal_s * (3.15 * gamma_min)

    terminal = jnp.clip((t - 0.925) / 0.075, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * jnp.maximum(lr_body, 2.80 * gamma_min) + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    alpha_gate = 1.0 / (1.0 + jnp.exp(-20.0 * (t - 0.505)))
    alpha_plateau = alpha0 * 4.85
    alpha_main = alpha0 * (0.88 + (alpha_plateau / jnp.maximum(alpha0, 1e-30) - 0.88) * alpha_gate)

    b1x = (t - 0.335) / 0.034
    b2x = (t - 0.585) / 0.043
    b3x = (t - 0.790) / 0.035
    bursts = (
        0.34 * jnp.exp(-0.5 * b1x * b1x)
        + 0.46 * jnp.exp(-0.5 * b2x * b2x)
        + 0.58 * jnp.exp(-0.5 * b3x * b3x)
    )
    alpha_main = alpha_main * (1.0 + bursts)

    alpha_terminal = alpha0 * D / jnp.maximum(1.62 * gamma_min, 1e-30)
    alpha_spike = alpha_terminal * (1.0 + 1.82 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.275 - 0.050 * warm_s - 0.100 * alpha_gate - 0.070 * terminal_s
    beta2 = 0.115 + 0.075 * warm_s + 0.175 * alpha_gate + 0.175 * terminal_s

    return lr, alpha, beta1, beta2