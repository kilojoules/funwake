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

    lr_scale = D * (0.72 + 0.036 * spacing_ratio) / jnp.maximum(farm_scale, 0.82)
    warm_s = smooth01(t / 0.075)

    c1 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip(t / 0.31, 0.0, 1.0))
    c2 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.31) / 0.27, 0.0, 1.0))
    c3 = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip((t - 0.58) / 0.29, 0.0, 1.0))

    gate1 = 1.0 - smooth01((t - 0.31) / 0.035)
    gate2 = smooth01((t - 0.31) / 0.035) * (1.0 - smooth01((t - 0.58) / 0.04))
    gate3 = smooth01((t - 0.58) / 0.04)

    lr_cyclic = lr_scale * (
        gate1 * (0.78 + 0.30 * c1)
        + gate2 * (0.54 + 0.34 * c2)
        + gate3 * (0.20 + 0.34 * c3)
    )
    lr_body = lr_cyclic * (0.62 + 0.38 * warm_s)

    global_cool = smooth01((t - 0.72) / 0.20)
    lr_body = lr_body * (1.0 - 0.72 * global_cool)
    lr_body = jnp.maximum(lr_body, 3.15 * gamma_min)

    terminal_s = smooth01((t - 0.925) / 0.075)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    ramp_s = smooth01((t - 0.34) / 0.28)
    alpha_plateau = alpha0 * (1.02 + 3.75 * ramp_s)

    burst1 = jnp.exp(-0.5 * jnp.square((t - 0.49) / 0.045))
    burst2 = jnp.exp(-0.5 * jnp.square((t - 0.70) / 0.052))
    alpha_mid = alpha_plateau * (1.0 + 0.34 * burst1 + 0.46 * burst2)

    alpha_cap = alpha0 * (4.95 + 0.18 * spacing_ratio)
    alpha_main = jnp.minimum(alpha_mid, alpha_cap)

    alpha_restore = alpha0 * D / jnp.maximum(1.95 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.85 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    moment_phase = smooth01((t - 0.38) / 0.36)
    beta1 = 0.24 - 0.13 * moment_phase - 0.065 * terminal_s
    beta2 = 0.13 + 0.25 * moment_phase + 0.13 * terminal_s

    return lr, alpha, beta1, beta2