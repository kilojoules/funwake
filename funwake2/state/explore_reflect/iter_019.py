import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    min_spacing = jnp.asarray(min_spacing, dtype=jnp.float32)
    n_turbines = jnp.asarray(n_turbines, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)
    pi = jnp.asarray(jnp.pi, dtype=jnp.float32)

    lr_scale = D * (0.78 + 0.10 * jnp.clip(min_spacing / jnp.maximum(5.0 * D, 1e-30), 0.75, 1.25))
    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    cycle_pos = jnp.mod(jnp.maximum(t - 0.075, 0.0) / 0.255, 1.0)
    cycle_cos = 0.5 + 0.5 * jnp.cos(pi * cycle_pos)
    cycle_decay = jnp.exp(-1.35 * jnp.maximum(t - 0.075, 0.0))
    restart_floor = 0.58 - 0.34 * t

    lr_explore = lr_scale * (
        0.58
        + 0.54 * warm_s
        + 0.26 * cycle_decay * cycle_cos * (1.0 - 0.45 * warm_s)
    )
    lr_explore = lr_explore * (1.0 - 0.18 * jnp.clip(t / 0.56, 0.0, 1.0))
    lr_explore = jnp.maximum(lr_explore, restart_floor * lr_scale)

    cool = jnp.clip((t - 0.56) / 0.345, 0.0, 1.0)
    cool_s = cool * cool * (3.0 - 2.0 * cool)
    lr_floor = 3.15 * gamma_min
    lr_body = (1.0 - cool_s) * lr_explore + cool_s * lr_floor
    lr_body = jnp.maximum(lr_body, lr_floor)

    terminal = jnp.clip((t - 0.915) / 0.085, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp_mid = 1.0 / (1.0 + jnp.exp(-18.0 * (t - 0.50)))
    ramp_late = 1.0 / (1.0 + jnp.exp(-28.0 * (t - 0.72)))
    alpha_plateau = alpha0 * (1.02 + 2.95 * ramp_mid + 0.72 * ramp_late)

    burst1_x = jnp.clip(1.0 - jnp.abs(t - 0.48) / 0.075, 0.0, 1.0)
    burst2_x = jnp.clip(1.0 - jnp.abs(t - 0.70) / 0.070, 0.0, 1.0)
    burst1 = burst1_x * burst1_x * (3.0 - 2.0 * burst1_x)
    burst2 = burst2_x * burst2_x * (3.0 - 2.0 * burst2_x)
    size_soften = 1.0 / jnp.sqrt(jnp.maximum(n_turbines / 50.0, 0.25))
    alpha_main = alpha_plateau * (1.0 + size_soften * (0.18 * burst1 + 0.26 * burst2))

    alpha_restore = alpha0 * D / jnp.maximum(2.05 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.72 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    phase = jnp.clip((t - 0.44) / 0.36, 0.0, 1.0)
    phase_s = phase * phase * (3.0 - 2.0 * phase)
    beta1 = 0.225 - 0.095 * phase_s - 0.070 * terminal_s + 0.020 * cycle_decay * cycle_cos
    beta2 = 0.135 + 0.215 * phase_s + 0.125 * terminal_s

    return lr, alpha, beta1, beta2