import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = D
    warm = jnp.clip(t / 0.075, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    prog = jnp.clip(t / 0.88, 0.0, 1.0)
    base_decay = 0.24 + 0.76 * 0.5 * (1.0 + jnp.cos(jnp.pi * prog))

    cycle_phase = jnp.mod(jnp.maximum(t - 0.07, 0.0) / 0.235, 1.0)
    cycle = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_phase))
    cycle_gate = jnp.clip((0.86 - t) / 0.18, 0.0, 1.0)
    cycle_amp = 0.185 * cycle_gate * cycle_gate * (3.0 - 2.0 * cycle_gate)

    mid = jnp.clip((t - 0.32) / 0.36, 0.0, 1.0)
    mid_s = mid * mid * (3.0 - 2.0 * mid)
    lr_body = lr_scale * (0.72 + 0.43 * warm_s) * base_decay * (1.0 + cycle_amp * cycle)
    lr_body = lr_body * (1.0 - 0.16 * mid_s)
    lr_body = jnp.maximum(lr_body, 3.35 * gamma_min)

    terminal = jnp.clip((t - 0.91) / 0.09, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp = 1.0 / (1.0 + jnp.exp(-18.0 * (t - 0.47)))
    alpha_plateau = alpha0 * (0.86 + 2.85 * ramp)

    burst1 = jnp.exp(-0.5 * ((t - 0.58) / 0.055) ** 2)
    burst2 = jnp.exp(-0.5 * ((t - 0.73) / 0.045) ** 2)
    alpha_bursts = alpha0 * (0.82 * burst1 + 0.68 * burst2)

    alpha_main = alpha_plateau + alpha_bursts
    alpha_cap = alpha0 * 5.45
    alpha_main = jnp.minimum(alpha_main, alpha_cap)

    alpha_restore = alpha0 * D / jnp.maximum(1.92 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.85 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    beta_phase = 1.0 / (1.0 + jnp.exp(-16.0 * (t - 0.50)))
    burst_mom = jnp.maximum(burst1, burst2)
    beta1 = 0.235 - 0.105 * beta_phase - 0.038 * burst_mom - 0.060 * terminal_s
    beta2 = 0.118 + 0.235 * beta_phase + 0.045 * burst_mom + 0.130 * terminal_s

    return lr, alpha, beta1, beta2