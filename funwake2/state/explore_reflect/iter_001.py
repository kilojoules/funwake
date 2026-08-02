import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)
    pi = jnp.asarray(jnp.pi, dtype=jnp.float32)

    lr_scale = D * (min_spacing / jnp.maximum(5.0 * D, 1e-30))
    lr_scale = jnp.clip(lr_scale, 0.72 * D, 1.10 * D)

    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    long_decay = jnp.clip((t - 0.34) / 0.565, 0.0, 1.0)
    long_cos = 0.5 + 0.5 * jnp.cos(pi * long_decay)

    cyc1 = 0.5 + 0.5 * jnp.cos(2.0 * pi * jnp.clip((t - 0.12) / 0.28, 0.0, 1.0))
    cyc2 = 0.5 + 0.5 * jnp.cos(2.0 * pi * jnp.clip((t - 0.40) / 0.24, 0.0, 1.0))
    cyc3 = 0.5 + 0.5 * jnp.cos(2.0 * pi * jnp.clip((t - 0.62) / 0.18, 0.0, 1.0))

    gate1 = jnp.clip((t - 0.12) / 0.28, 0.0, 1.0)
    gate2 = jnp.clip((t - 0.40) / 0.24, 0.0, 1.0)
    gate3 = jnp.clip((t - 0.62) / 0.18, 0.0, 1.0)
    pulse = (
        0.105 * cyc1 * gate1 * (1.0 - gate1)
        + 0.075 * cyc2 * gate2 * (1.0 - gate2)
        + 0.045 * cyc3 * gate3 * (1.0 - gate3)
    )

    lr_high = lr_scale * (0.73 + 0.37 * warm_s + pulse)
    lr_floor = jnp.maximum(3.15 * gamma_min, 0.030 * lr_scale)
    lr_body = lr_floor + (lr_high - lr_floor) * long_cos
    lr_body = jnp.maximum(lr_body, 2.75 * gamma_min)

    terminal = jnp.clip((t - 0.905) / 0.095, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp = 1.0 / (1.0 + jnp.exp(-18.0 * (t - 0.515)))
    alpha_plateau = alpha0 * 4.65
    alpha_main = alpha0 * (0.82 + 3.83 * ramp)

    burst1 = jnp.exp(-0.5 * ((t - 0.58) / 0.055) ** 2)
    burst2 = jnp.exp(-0.5 * ((t - 0.755) / 0.045) ** 2)
    alpha_burst = alpha0 * (1.10 * burst1 + 1.55 * burst2)

    alpha_late = alpha0 * D / jnp.maximum(2.05 * gamma_min, 1e-30)
    alpha_spike = alpha_late * (1.0 + 1.85 * terminal_s)
    alpha = (1.0 - terminal_s) * (alpha_main + alpha_burst) + terminal_s * alpha_spike
    alpha = jnp.minimum(alpha, (1.0 - terminal_s) * (alpha_plateau + alpha_burst) + terminal_s * alpha_spike)

    phase = jnp.clip((t - 0.46) / 0.36, 0.0, 1.0)
    phase_s = phase * phase * (3.0 - 2.0 * phase)

    beta1 = 0.245 - 0.120 * warm_s - 0.070 * phase_s - 0.040 * terminal_s
    beta2 = 0.125 + 0.165 * phase_s + 0.165 * terminal_s

    return lr, alpha, beta1, beta2