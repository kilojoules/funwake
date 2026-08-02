import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total - 1.0, 1.0), 0.0, 1.0)

    lr_scale = D * (226.0 / 240.0)

    warm = jnp.clip(t / 0.085, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    sgdr_pos = jnp.mod(jnp.maximum(t - 0.06, 0.0) / 0.205, 1.0)
    sgdr_cos = 0.5 + 0.5 * jnp.cos(jnp.pi * sgdr_pos)

    envelope = jnp.clip((t - 0.18) / 0.66, 0.0, 1.0)
    envelope_s = envelope * envelope * (3.0 - 2.0 * envelope)

    lr_peak = lr_scale * (0.76 + 0.36 * warm_s)
    lr_floor = jnp.maximum(3.10 * gamma_min, 0.034 * lr_scale)
    lr_cycle = lr_peak * (0.82 + (0.30 * (1.0 - envelope_s)) * sgdr_cos)
    lr_body = (1.0 - envelope_s) * lr_cycle + envelope_s * lr_floor

    restore = jnp.clip((t - 0.905) / 0.095, 0.0, 1.0)
    restore_s = restore * restore * (3.0 - 2.0 * restore)
    lr = (1.0 - restore_s) * lr_body + restore_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    gate = 1.0 / (1.0 + jnp.exp(-24.0 * (t - 0.52)))
    alpha_plateau = 3.55 * alpha0
    alpha_base = alpha0 * (0.88 + 2.67 * gate)

    a_phase = jnp.mod(jnp.maximum(t - 0.34, 0.0) / 0.185, 1.0)
    a_cycle = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * a_phase)
    a_amp = (1.0 - restore_s) * (0.10 + 0.14 * gate) * (1.0 - 0.55 * envelope_s)

    burst_a = jnp.exp(-0.5 * ((t - 0.61) / 0.040) ** 2)
    burst_b = jnp.exp(-0.5 * ((t - 0.78) / 0.032) ** 2)

    alpha_main = jnp.minimum(alpha_base, alpha_plateau)
    alpha_main = alpha_main * (1.0 + a_amp * a_cycle + 0.26 * burst_a + 0.34 * burst_b)

    alpha_late = alpha0 * D / jnp.maximum(2.05 * gamma_min, 1e-30)
    alpha_spike = alpha_late * (1.0 + 1.70 * restore_s)
    alpha = (1.0 - restore_s) * alpha_main + restore_s * alpha_spike

    beta1 = 0.265 - 0.050 * warm_s - 0.112 * gate - 0.058 * restore_s
    beta2 = 0.118 + 0.070 * warm_s + 0.205 * gate + 0.150 * restore_s

    return lr, alpha, beta1, beta2