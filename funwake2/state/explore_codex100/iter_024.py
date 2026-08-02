import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    t = jnp.asarray(step, dtype=jnp.float32)
    T = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    Dv = jnp.asarray(D, dtype=jnp.float32)
    gm = jnp.asarray(gamma_min, dtype=jnp.float32)
    a0 = jnp.asarray(alpha0, dtype=jnp.float32)

    progress = jnp.clip(t / T, 0.0, 1.0)

    base_lr = (200.0 / 240.0) * Dv
    peak_lr = 1.10 * base_lr

    warm = jnp.clip(progress / 0.08, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    hold = 1.0 - jnp.clip((progress - 0.46) / 0.46, 0.0, 1.0)
    hold_s = hold * hold * (3.0 - 2.0 * hold)

    explore_lr = base_lr + (peak_lr - base_lr) * warm_s
    decay_lr = gm + (peak_lr - gm) * hold_s
    lr_pre_terminal = jnp.minimum(explore_lr, decay_lr)

    terminal = jnp.clip((progress - 0.92) / 0.08, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = gm + (lr_pre_terminal - gm) * (1.0 - terminal_s) ** 1.6

    ramp = jnp.clip((progress - 0.38) / 0.42, 0.0, 1.0)
    ramp_s = ramp * ramp * (3.0 - 2.0 * ramp)

    alpha_floor = a0 * Dv / jnp.maximum(peak_lr, 1e-30)
    alpha_plateau = 3.2 * a0
    alpha_base = alpha_floor + (alpha_plateau - alpha_floor) * ramp_s

    spike = 1.0 + 3.5 * terminal_s
    native_guard = a0 * Dv / jnp.maximum(lr, 1e-30)
    alpha = jnp.maximum(alpha_base * spike, 0.62 * native_guard * terminal_s)

    beta1 = 0.18 - 0.10 * terminal_s + 0.05 * (1.0 - warm_s)
    beta2 = 0.16 + 0.22 * ramp_s + 0.10 * terminal_s

    return lr, alpha, beta1, beta2