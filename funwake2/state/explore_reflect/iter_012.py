import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)
    pi = jnp.asarray(jnp.pi, dtype=jnp.float32)

    lr_base = (218.0 / 240.0) * D
    floor = jnp.maximum(gamma_min, 1e-30)

    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    global_cool = jnp.clip((t - 0.18) / 0.74, 0.0, 1.0)
    global_s = global_cool * global_cool * (3.0 - 2.0 * global_cool)
    envelope = 1.10 - 1.055 * global_s

    cycle1 = 0.5 + 0.5 * jnp.cos(pi * jnp.clip((t - 0.10) / 0.24, 0.0, 1.0))
    cycle2 = 0.5 + 0.5 * jnp.cos(pi * jnp.clip((t - 0.38) / 0.20, 0.0, 1.0))
    cycle3 = 0.5 + 0.5 * jnp.cos(pi * jnp.clip((t - 0.62) / 0.17, 0.0, 1.0))
    restart = (
        1.0
        + 0.115 * cycle1 * (1.0 - cycle1)
        + 0.085 * cycle2 * (1.0 - cycle2)
        + 0.055 * cycle3 * (1.0 - cycle3)
    )

    lr_body = lr_base * (0.58 + 0.42 * warm_s) * envelope * restart
    lr_body = jnp.maximum(lr_body, 3.15 * floor)

    terminal = jnp.clip((t - 0.905) / 0.095, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * floor
    lr = jnp.maximum(lr, floor)

    ramp = jnp.clip((t - 0.36) / 0.34, 0.0, 1.0)
    ramp_s = ramp * ramp * (3.0 - 2.0 * ramp)

    burst1_x = (t - 0.50) / 0.075
    burst2_x = (t - 0.735) / 0.060
    burst1 = jnp.exp(-0.5 * burst1_x * burst1_x)
    burst2 = jnp.exp(-0.5 * burst2_x * burst2_x)

    alpha_plateau = alpha0 * (1.18 + 3.85 * ramp_s)
    alpha_bursts = alpha0 * (0.72 * burst1 + 1.05 * burst2)
    alpha_main = alpha_plateau + alpha_bursts

    alpha_cap = alpha0 * D / jnp.maximum(0.245 * lr_base, 1e-30)
    alpha_main = jnp.minimum(alpha_main, alpha_cap)

    alpha_restore = alpha0 * D / jnp.maximum(1.72 * floor, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.85 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.235 - 0.090 * ramp_s - 0.085 * terminal_s
    beta2 = 0.115 + 0.185 * ramp_s + 0.175 * terminal_s

    return lr, alpha, beta1, beta2