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

    base_lr = D * (0.78 + 0.025 * spacing_ratio) / jnp.maximum(0.92, 0.82 + 0.18 * farm_scale)

    warm_s = smooth01(t / 0.085)
    long_hold = 1.0 - 0.36 * smooth01((t - 0.30) / 0.34)
    late_cool = 1.0 - 0.74 * smooth01((t - 0.66) / 0.25)

    cyc1 = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.10) / 0.34, 0.0, 1.0))
    cyc2 = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.44) / 0.30, 0.0, 1.0))
    cyc3 = 0.5 + 0.5 * jnp.cos(2.0 * jnp.pi * jnp.clip((t - 0.70) / 0.18, 0.0, 1.0))

    gate1 = (1.0 - smooth01((t - 0.42) / 0.045))
    gate2 = smooth01((t - 0.38) / 0.050) * (1.0 - smooth01((t - 0.68) / 0.050))
    gate3 = smooth01((t - 0.65) / 0.045) * (1.0 - smooth01((t - 0.88) / 0.035))

    restart_gain = gate1 * (0.93 + 0.17 * cyc1) + gate2 * (0.72 + 0.24 * cyc2) + gate3 * (0.34 + 0.26 * cyc3)
    lr_body = base_lr * (0.58 + 0.42 * warm_s) * long_hold * late_cool * restart_gain
    lr_body = jnp.maximum(lr_body, 3.25 * gamma_min)

    terminal_s = smooth01((t - 0.925) / 0.075)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    alpha_ramp = 1.0 / (1.0 + jnp.exp(-18.0 * (t - 0.56)))
    alpha_plateau = alpha0 * (1.08 + 3.85 * alpha_ramp)

    burst_a = jnp.exp(-0.5 * jnp.square((t - 0.52) / 0.040))
    burst_b = jnp.exp(-0.5 * jnp.square((t - 0.735) / 0.050))
    alpha_bursts = alpha0 * (0.58 * burst_a + 0.82 * burst_b)

    alpha_cap = alpha0 * (5.35 + 0.12 * spacing_ratio)
    alpha_main = jnp.minimum(alpha_plateau + alpha_bursts, alpha_cap)

    alpha_restore = alpha0 * D / jnp.maximum(2.00 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.95 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    feasibility_phase = smooth01((t - 0.50) / 0.34)
    beta1 = 0.255 - 0.110 * warm_s - 0.075 * feasibility_phase - 0.050 * terminal_s
    beta2 = 0.120 + 0.235 * feasibility_phase + 0.145 * terminal_s

    return lr, alpha, beta1, beta2