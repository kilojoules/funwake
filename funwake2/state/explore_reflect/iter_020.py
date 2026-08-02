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

    spacing_ratio = jnp.clip(min_spacing / jnp.maximum(D, 1e-30), 1.5, 6.0)
    farm_scale = jnp.clip(jnp.sqrt(jnp.maximum(n_turbines, 1.0) / 50.0), 0.82, 1.22)

    lr_anchor = D * (0.82 + 0.035 * spacing_ratio) / farm_scale
    lr_floor = 2.70 * gamma_min

    warm = jnp.clip(t / 0.085, 0.0, 1.0)
    warm_s = warm * warm * (3.0 - 2.0 * warm)

    global_cool = jnp.clip((t - 0.32) / 0.58, 0.0, 1.0)
    global_cool_s = global_cool * global_cool * (3.0 - 2.0 * global_cool)

    phase1 = jnp.clip(t / 0.30, 0.0, 1.0)
    phase2 = jnp.clip((t - 0.30) / 0.25, 0.0, 1.0)
    phase3 = jnp.clip((t - 0.55) / 0.22, 0.0, 1.0)
    phase4 = jnp.clip((t - 0.77) / 0.13, 0.0, 1.0)

    c1 = 0.5 + 0.5 * jnp.cos(jnp.pi * phase1)
    c2 = 0.5 + 0.5 * jnp.cos(jnp.pi * phase2)
    c3 = 0.5 + 0.5 * jnp.cos(jnp.pi * phase3)
    c4 = 0.5 + 0.5 * jnp.cos(jnp.pi * phase4)

    g1 = 1.0 - jnp.clip((t - 0.30) / 0.001, 0.0, 1.0)
    g2 = jnp.clip((t - 0.30) / 0.001, 0.0, 1.0) * (1.0 - jnp.clip((t - 0.55) / 0.001, 0.0, 1.0))
    g3 = jnp.clip((t - 0.55) / 0.001, 0.0, 1.0) * (1.0 - jnp.clip((t - 0.77) / 0.001, 0.0, 1.0))
    g4 = jnp.clip((t - 0.77) / 0.001, 0.0, 1.0)

    restart_wave = g1 * c1 + g2 * c2 + g3 * c3 + g4 * c4
    restart_amp = 0.34 * g1 + 0.25 * g2 + 0.17 * g3 + 0.08 * g4

    lr_body = lr_anchor * (0.70 + 0.34 * warm_s) * (1.0 - 0.78 * global_cool_s)
    lr_body = lr_body * (1.0 + restart_amp * restart_wave)
    lr_body = jnp.maximum(lr_body, lr_floor)

    terminal = jnp.clip((t - 0.910) / 0.090, 0.0, 1.0)
    terminal_s = terminal * terminal * (3.0 - 2.0 * terminal)
    lr = (1.0 - terminal_s) * lr_body + terminal_s * gamma_min
    lr = jnp.maximum(lr, gamma_min)

    logistic = 1.0 / (1.0 + jnp.exp(-18.0 * (t - 0.545)))
    late_logistic = 1.0 / (1.0 + jnp.exp(-24.0 * (t - 0.710)))

    alpha_plateau = alpha0 * (1.10 + 2.55 * logistic + 0.95 * late_logistic)

    burst_a = jnp.exp(-0.5 * jnp.square((t - 0.405) / 0.040))
    burst_b = jnp.exp(-0.5 * jnp.square((t - 0.635) / 0.045))
    burst_c = jnp.exp(-0.5 * jnp.square((t - 0.805) / 0.035))

    alpha_cycle = alpha0 * (0.30 * burst_a + 0.46 * burst_b + 0.54 * burst_c)
    alpha_main = alpha_plateau + alpha_cycle

    alpha_cap = alpha0 * (5.25 + 0.40 * jnp.clip(spacing_ratio - 2.0, 0.0, 3.0))
    alpha_main = jnp.minimum(alpha_main, alpha_cap)

    alpha_restore = alpha0 * D / jnp.maximum(2.08 * gamma_min, 1e-30)
    alpha_spike = alpha_restore * (1.0 + 1.92 * terminal_s)
    alpha = (1.0 - terminal_s) * alpha_main + terminal_s * alpha_spike

    alpha_phase = jnp.clip((t - 0.48) / 0.34, 0.0, 1.0)
    alpha_phase_s = alpha_phase * alpha_phase * (3.0 - 2.0 * alpha_phase)

    beta1 = 0.285 - 0.055 * warm_s - 0.125 * alpha_phase_s - 0.055 * terminal_s
    beta2 = 0.118 + 0.185 * alpha_phase_s + 0.145 * terminal_s

    return lr, alpha, beta1, beta2