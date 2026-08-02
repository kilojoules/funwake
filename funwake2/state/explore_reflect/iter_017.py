import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = (214.0 / 240.0) * D
    lr_floor = 2.65 * gamma_min

    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    decay = jnp.clip((t - 0.18) / 0.72, 0.0, 1.0)
    cosine_decay = 0.5 + 0.5 * jnp.cos(jnp.pi * decay)

    cycle_phase = jnp.mod(jnp.maximum(t - 0.105, 0.0) / 0.215, 1.0)
    cycle = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * cycle_phase)
    cycle_gate = jnp.clip((0.74 - t) / 0.20, 0.0, 1.0)
    cycle_gate = cycle_gate * cycle_gate * (3.0 - 2.0 * cycle_gate)

    base_lr = lr_scale * (0.58 + 0.60 * warm_s) * (0.11 + 0.89 * cosine_decay)
    lr_body = base_lr * (1.0 + 0.105 * cycle * cycle_gate)
    lr_body = jnp.maximum(lr_body, lr_floor)

    terminal = jnp.clip((t - 0.905) / 0.095, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp = jnp.clip((t - 0.36) / 0.26, 0.0, 1.0)
    ramp_s = ramp * ramp * (3.0 - 2.0 * ramp)

    settle = jnp.clip((t - 0.66) / 0.20, 0.0, 1.0)
    settle_s = settle * settle * (3.0 - 2.0 * settle)

    burst1 = jnp.exp(-0.5 * jnp.square((t - 0.57) / 0.045))
    burst2 = jnp.exp(-0.5 * jnp.square((t - 0.765) / 0.038))

    alpha_plateau = alpha0 * (1.02 + 2.35 * ramp_s + 0.78 * settle_s)
    alpha_bursts = alpha0 * (0.54 * burst1 + 0.72 * burst2)
    alpha_main = alpha_plateau + alpha_bursts

    alpha_restore = alpha0 * D / jnp.maximum(2.05 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.85 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta1 = 0.245 - 0.075 * warm_s - 0.085 * ramp_s - 0.060 * terminal_s
    beta2 = 0.128 + 0.100 * ramp_s + 0.120 * settle_s + 0.125 * terminal_s

    return lr, alpha, beta1, beta2