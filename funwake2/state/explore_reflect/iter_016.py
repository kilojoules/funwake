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

    def smooth01(x):
        x = jnp.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    spacing_ratio = jnp.clip(min_spacing / jnp.maximum(D, 1e-30), 2.0, 8.0)
    farm_scale = jnp.sqrt(jnp.maximum(n_turbines, 1.0) / 50.0)

    lr_anchor = D * (0.86 + 0.020 * (spacing_ratio - 4.0)) / jnp.maximum(0.88, farm_scale)
    warm_s = smooth01(t / 0.085)

    cycle1 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip(t / 0.32, 0.0, 1.0))
    cycle2 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.32) / 0.29, 0.0, 1.0))
    cycle3 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.61) / 0.24, 0.0, 1.0))

    gate1 = 1.0 - smooth01((t - 0.32) / 0.035)
    gate2 = smooth01((t - 0.32) / 0.035) * (1.0 - smooth01((t - 0.61) / 0.04))
    gate3 = smooth01((t - 0.61) / 0.04)

    lr_cycles = lr_anchor * (
        gate1 * (0.78 + 0.28 * cycle1)
        + gate2 * (0.50 + 0.31 * cycle2)
        + gate3 * (0.18 + 0.28 * cycle3)
    )
    lr_body = lr_cycles * (0.66 + 0.34 * warm_s)

    late_cool = smooth01((t - 0.74) / 0.17)
    lr_body = lr_body * (1.0 - 0.64 * late_cool)
    lr_body = jnp.maximum(lr_body, 3.05 * gamma_min)

    terminal_s = smooth01((t - 0.915) / 0.085)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    alpha_ramp = smooth01((t - 0.38) / 0.24)
    alpha_plateau = alpha0 * (1.08 + 3.65 * alpha_ramp)

    alpha_cycle = (
        0.18 * jnp.exp(-0.5 * jnp.square((t - 0.36) / 0.040))
        + 0.32 * jnp.exp(-0.5 * jnp.square((t - 0.61) / 0.048))
        + 0.42 * jnp.exp(-0.5 * jnp.square((t - 0.78) / 0.050))
    )
    alpha_main = alpha_plateau * (1.0 + alpha_cycle)

    alpha_cap = alpha0 * (5.05 + 0.13 * spacing_ratio)
    alpha_main = jnp.minimum(alpha_main, alpha_cap)

    alpha_restore = alpha0 * D / jnp.maximum(2.04 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.85 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    phase = smooth01((t - 0.36) / 0.34)
    burst_phase = smooth01((t - 0.70) / 0.15)
    beta1 = 0.235 - 0.120 * phase - 0.030 * burst_phase - 0.055 * terminal_s
    beta2 = 0.130 + 0.225 * phase + 0.055 * burst_phase + 0.125 * terminal_s

    return lr, alpha, beta1, beta2