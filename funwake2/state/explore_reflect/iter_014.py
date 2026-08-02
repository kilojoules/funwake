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

    warm = jnp.clip(t / 0.085, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    early = jnp.clip(t / 0.52, 0.0, 1.0)
    cycle_phase = jnp.mod(3.0 * early, 1.0)
    cycle_cos = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * cycle_phase)
    cycle_decay = 1.0 - 0.30 * early
    cyclic_lr = lr_scale * (0.76 + 0.43 * cycle_decay * cycle_cos)

    cruise = lr_scale * (1.08 - 0.18 * jnp.clip((t - 0.10) / 0.44, 0.0, 1.0))
    lr_explore = (0.62 + 0.38 * warm_s) * jnp.maximum(cyclic_lr, cruise)

    cool = jnp.clip((t - 0.54) / 0.38, 0.0, 1.0)
    cool_s = cool * cool * (3.0 - 2.0 * cool)
    lr_mid = (1.0 - cool_s) * lr_explore + cool_s * jnp.maximum(3.35 * gamma_min, 0.070 * lr_scale)

    terminal = jnp.clip((t - 0.915) / 0.085, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_mid + terminal_s * lr_floor
    lr = jnp.maximum(lr, lr_floor)

    ramp = 1.0 / (1.0 + jnp.exp(-18.0 * (t - 0.50)))
    alpha_plateau = alpha0 * 3.85
    alpha_main = alpha0 * (0.96 + 0.92 * ramp + 1.97 * ramp * ramp)
    alpha_main = jnp.minimum(alpha_main, alpha_plateau)

    burst1 = jnp.exp(-0.5 * ((t - 0.58) / 0.045) ** 2)
    burst2 = jnp.exp(-0.5 * ((t - 0.76) / 0.040) ** 2)
    alpha_burst = alpha_main * (1.0 + 0.34 * burst1 + 0.48 * burst2)

    alpha_late = alpha0 * D / jnp.maximum(2.08 * gamma_min, 1e-30)
    alpha_spike = alpha_late * (1.0 + 1.72 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_burst + terminal_s * alpha_spike

    beta1 = 0.245 - 0.105 * ramp - 0.075 * terminal_s
    beta2 = 0.135 + 0.185 * ramp + 0.145 * terminal_s

    return lr, alpha, beta1, beta2