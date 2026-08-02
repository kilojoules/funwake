import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = (212.0 / 240.0) * D
    lr_floor = jnp.maximum(gamma_min, 1e-30)

    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    cycle1 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.075) / 0.245, 0.0, 1.0))
    cycle2 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.320) / 0.255, 0.0, 1.0))
    cycle3 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.575) / 0.265, 0.0, 1.0))

    w1 = 1.0 - jnp.clip((t - 0.320) / 0.001, 0.0, 1.0)
    w2 = jnp.clip((t - 0.320) / 0.001, 0.0, 1.0) * (1.0 - jnp.clip((t - 0.575) / 0.001, 0.0, 1.0))
    w3 = jnp.clip((t - 0.575) / 0.001, 0.0, 1.0)

    amp = lr_scale * (0.74 * w1 + 0.48 * w2 + 0.27 * w3)
    base = lr_scale * (0.36 * w1 + 0.27 * w2 + 0.135 * w3)
    cyc = cycle1 * w1 + cycle2 * w2 + cycle3 * w3

    lr_body = (base + amp * cyc) * (0.50 + 0.50 * warm_s)

    final_cool = jnp.clip((t - 0.840) / 0.095, 0.0, 1.0)
    final_cool_s = final_cool * final_cool * (3.0 - 2.0 * final_cool)
    lr_body = (1.0 - final_cool_s) * lr_body + final_cool_s * (3.15 * lr_floor)

    terminal = jnp.clip((t - 0.925) / 0.075, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * jnp.maximum(lr_body, 3.0 * lr_floor) + terminal_s * lr_floor
    lr = jnp.maximum(lr, lr_floor)

    ramp = jnp.clip((t - 0.365) / 0.245, 0.0, 1.0)
    ramp_s = ramp * ramp * (3.0 - 2.0 * ramp)

    burst1_x = (t - 0.515) / 0.055
    burst2_x = (t - 0.740) / 0.045
    burst1 = jnp.exp(-0.5 * burst1_x * burst1_x)
    burst2 = jnp.exp(-0.5 * burst2_x * burst2_x)

    alpha_plateau = 3.35 * alpha0
    alpha_low = 0.88 * alpha0
    alpha_main = alpha_low + (alpha_plateau - alpha_low) * ramp_s
    alpha_main = alpha_main * (1.0 + 0.24 * burst1 + 0.34 * burst2)

    alpha_terminal = alpha0 * D / jnp.maximum(1.72 * lr_floor, 1e-30)
    alpha_spike = alpha_terminal * (1.0 + 1.35 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.245 - 0.050 * warm_s - 0.070 * ramp_s - 0.060 * terminal_s
    beta2 = 0.125 + 0.090 * warm_s + 0.165 * ramp_s + 0.155 * terminal_s

    return lr, alpha, beta1, beta2