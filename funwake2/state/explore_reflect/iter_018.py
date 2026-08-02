import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = (214.0 / 240.0) * D
    floor_lr = 2.55 * gamma_min

    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    long_cool = jnp.clip((t - 0.52) / 0.365, 0.0, 1.0)
    long_cool_s = long_cool * long_cool * (3.0 - 2.0 * long_cool)

    cyc1 = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.10) / 0.245, 0.0, 1.0))
    cyc2 = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.345) / 0.235, 0.0, 1.0))
    cyc3 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.58) / 0.305, 0.0, 1.0))

    amp1 = 0.145 * jnp.clip((0.36 - t) / 0.26, 0.0, 1.0)
    amp2 = 0.105 * jnp.clip((0.62 - t) / 0.275, 0.0, 1.0)
    amp3 = 0.055 * jnp.clip((0.89 - t) / 0.31, 0.0, 1.0)

    lr_body = lr_scale * (0.72 + 0.38 * warm_s)
    lr_body = lr_body * (1.0 + amp1 * cyc1 + amp2 * cyc2 + amp3 * cyc3)
    lr_body = lr_body * (1.0 - 0.915 * long_cool_s)
    lr_body = jnp.maximum(lr_body, floor_lr)

    terminal = jnp.clip((t - 0.91) / 0.09, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp = jnp.clip((t - 0.37) / 0.245, 0.0, 1.0)
    ramp_s = ramp * ramp * (3.0 - 2.0 * ramp)

    burst1_x = jnp.clip(1.0 - jnp.abs(t - 0.58) / 0.075, 0.0, 1.0)
    burst2_x = jnp.clip(1.0 - jnp.abs(t - 0.755) / 0.065, 0.0, 1.0)
    burst1 = burst1_x * burst1_x * (3.0 - 2.0 * burst1_x)
    burst2 = burst2_x * burst2_x * (3.0 - 2.0 * burst2_x)

    alpha_plateau = alpha0 * 3.55
    alpha_main = alpha0 * (1.02 + 2.35 * ramp_s)
    alpha_main = jnp.minimum(alpha_main, alpha_plateau)
    alpha_main = alpha_main * (1.0 + 0.34 * burst1 + 0.48 * burst2)

    alpha_late = alpha0 * D / jnp.maximum(2.05 * gamma_min, 1e-30)
    alpha_spike = alpha_late * (1.0 + 1.85 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.235 - 0.105 * ramp_s - 0.070 * terminal_s
    beta2 = 0.125 + 0.215 * ramp_s + 0.145 * terminal_s

    return lr, alpha, beta1, beta2