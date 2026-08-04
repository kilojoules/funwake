import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    step = jnp.asarray(step, dtype=jnp.float32)
    total_steps = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    D = jnp.asarray(D, dtype=jnp.float32)
    min_spacing = jnp.asarray(min_spacing, dtype=jnp.float32)
    n_turbines = jnp.asarray(n_turbines, dtype=jnp.float32)
    gamma_min = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0 = jnp.asarray(alpha0, dtype=jnp.float32)

    t = jnp.clip(step / jnp.maximum(total_steps - 1.0, 1.0), 0.0, 1.0)
    lr_floor = jnp.maximum(gamma_min, 1.0e-6 * D)

    def smooth01(x):
        x = jnp.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def sigmoid(x):
        return 1.0 / (1.0 + jnp.exp(-x))

    spacing_scale = jnp.clip(min_spacing / jnp.maximum(2.0 * D, 1.0e-6), 0.85, 2.35)
    farm_scale = jnp.clip(jnp.sqrt(jnp.maximum(n_turbines, 1.0)) / 7.0, 0.75, 1.40)

    warm = smooth01(t / 0.055)
    global_cool = smooth01((t - 0.60) / 0.30)
    terminal = smooth01((t - 0.915) / 0.085)

    cycle_pos = jnp.mod(t, 0.255) / 0.255
    cycle_id = jnp.floor(t / 0.255)
    cycle_decay = jnp.power(0.78, cycle_id)
    cosine_cycle = 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.clip(cycle_pos, 0.0, 1.0))

    envelope_hi = (1.48 * D) * cycle_decay
    envelope_lo = (0.54 * D) * cycle_decay + 0.16 * D
    sgdr_lr = envelope_lo + (envelope_hi - envelope_lo) * cosine_cycle

    launch_lr = (0.30 * D) * (1.0 - warm) + (1.50 * D) * warm
    lr = launch_lr * (1.0 - smooth01((t - 0.09) / 0.075)) + sgdr_lr * smooth01((t - 0.09) / 0.075)

    linear_tail = (0.62 * D) * (1.0 - global_cool) + jnp.maximum(0.038 * D, lr_floor) * global_cool
    lr = lr * (1.0 - global_cool) + linear_tail * global_cool

    restore_1 = jnp.exp(-jnp.square((t - 0.575) / 0.038))
    restore_2 = jnp.exp(-jnp.square((t - 0.735) / 0.034))
    restore_3 = jnp.exp(-jnp.square((t - 0.850) / 0.030))
    restore_gate = smooth01((t - 0.50) / 0.18)
    lr = lr * (1.0 - restore_gate * (0.34 * restore_1 + 0.45 * restore_2 + 0.56 * restore_3))

    lr = lr * (1.0 - terminal) + lr_floor * terminal
    lr = jnp.maximum(lr, lr_floor)

    alpha_ramp = sigmoid(17.0 * (t - 0.46))
    alpha_plateau = 1.00 + (3.55 + 0.55 * spacing_scale + 0.22 * farm_scale) * alpha_ramp

    alpha_cycle_pos = jnp.mod(t + 0.045, 0.255) / 0.255
    alpha_cycle = jnp.square(0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * alpha_cycle_pos))
    alpha_cycle_gate = smooth01((t - 0.28) / 0.20) * (1.0 - terminal)

    alpha_burst = (
        1.00 * jnp.exp(-jnp.square((t - 0.585) / 0.045))
        + 1.45 * jnp.exp(-jnp.square((t - 0.745) / 0.038))
        + 1.85 * jnp.exp(-jnp.square((t - 0.855) / 0.032))
    )

    alpha = alpha0 * (
        alpha_plateau
        + (0.62 + 0.16 * spacing_scale) * alpha_cycle * alpha_cycle_gate
        + alpha_burst
    )

    late_push = sigmoid(26.0 * (t - 0.865))
    alpha = alpha * (1.0 + 0.62 * late_push)

    tol_ratio = jnp.maximum(D / jnp.maximum(lr_floor, 1.0e-30), 1.0)
    alpha = alpha + alpha0 * terminal * (28.0 + 6.2 * tol_ratio + 2.0 * spacing_scale)

    beta1_phase = sigmoid(13.0 * (t - 0.43))
    beta1 = 0.32 * (1.0 - beta1_phase) + 0.070 * beta1_phase
    beta1 = beta1 + 0.050 * (1.0 - global_cool) * cosine_cycle * (1.0 - alpha_ramp)
    beta1 = beta1 - 0.030 * restore_gate * (restore_1 + restore_2 + restore_3)
    beta1 = beta1 * (1.0 - terminal) + 0.014 * terminal
    beta1 = jnp.clip(beta1, 0.014, 0.36)

    beta2_phase = sigmoid(11.0 * (t - 0.36))
    beta2 = 0.085 * (1.0 - beta2_phase) + 0.42 * beta2_phase
    beta2 = beta2 + 0.11 * global_cool + 0.060 * restore_gate * (restore_1 + restore_2 + restore_3)
    beta2 = beta2 * (1.0 - terminal) + 0.72 * terminal
    beta2 = jnp.clip(beta2, 0.08, 0.76)

    return lr, alpha, beta1, beta2