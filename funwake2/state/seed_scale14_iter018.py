import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    dtype = jnp.result_type(D, min_spacing, n_turbines, gamma_min, alpha0, 1.0)

    step = jnp.asarray(step, dtype=dtype)
    total_steps = jnp.asarray(total_steps, dtype=dtype)
    D = jnp.asarray(D, dtype=dtype)
    min_spacing = jnp.asarray(min_spacing, dtype=dtype)
    n_turbines = jnp.asarray(n_turbines, dtype=dtype)
    gamma_min = jnp.asarray(gamma_min, dtype=dtype)
    alpha0 = jnp.asarray(alpha0, dtype=dtype)

    t = jnp.clip(step / jnp.maximum(total_steps - 1.0, 1.0), 0.0, 1.0)
    gamma = jnp.maximum(gamma_min, 0.0)
    gamma_safe = jnp.maximum(gamma, 1e-9 * jnp.maximum(D, 1.0))

    s_ratio = min_spacing / jnp.maximum(D, 1e-9)
    n_dense = jnp.clip((n_turbines - 45.0) / 35.0, 0.0, 1.0)
    n_small = jnp.clip((22.0 - n_turbines) / 18.0, 0.0, 1.0)
    tight = jnp.clip((2.35 - s_ratio) / 0.75, 0.0, 1.0)
    restore = jnp.clip(0.65 * n_dense + 0.35 * tight, 0.0, 1.0)

    warm_x = jnp.clip(t / (0.050 + 0.012 * n_small), 0.0, 1.0)
    warm = warm_x * warm_x * (3.0 - 2.0 * warm_x)

    lr_start = (0.54 + 0.06 * n_small - 0.03 * restore) * D
    lr_peak = (0.84 + 0.05 * n_small - 0.04 * restore) * D
    lr_warm = lr_start + (lr_peak - lr_start) * warm

    cool_start = 0.34 - 0.035 * restore + 0.035 * n_small
    cool_width = jnp.maximum(1.0 - cool_start, 1e-6)
    cool_x = jnp.clip((t - cool_start) / cool_width, 0.0, 1.0)
    cool = cool_x * cool_x * (3.0 - 2.0 * cool_x)

    lr_floor_mid = gamma + (0.060 + 0.050 * restore + 0.025 * n_small) * D
    floor_x = jnp.clip((t - 0.86) / 0.14, 0.0, 1.0)
    floor_phase = floor_x * floor_x * (3.0 - 2.0 * floor_x)
    lr_floor = (1.0 - floor_phase) * lr_floor_mid + floor_phase * gamma

    lr_decay = lr_floor + (lr_peak - lr_floor) * jnp.power(
        jnp.maximum(1.0 - cool, 0.0), 1.04 + 0.10 * n_small
    )
    lr = jnp.minimum(lr_warm, lr_decay)

    penalty_start = 0.38 - 0.075 * restore + 0.030 * n_small
    penalty_width = 0.34 + 0.04 * n_small
    penalty_x = jnp.clip((t - penalty_start) / penalty_width, 0.0, 1.0)
    penalty_phase = penalty_x * penalty_x * (3.0 - 2.0 * penalty_x)

    alpha_base = 0.72 + 0.12 * restore - 0.04 * n_small
    alpha_gain = 5.05 + 1.35 * restore - 0.55 * n_small
    alpha_plateau = alpha0 * (alpha_base + alpha_gain * penalty_phase)

    terminal_start = 0.765 - 0.075 * restore - 0.010 * n_small
    terminal_width = jnp.maximum(1.0 - terminal_start, 1e-6)
    terminal_x = jnp.clip((t - terminal_start) / terminal_width, 0.0, 1.0)
    terminal_phase = terminal_x * terminal_x * (3.0 - 2.0 * terminal_x)

    terminal_scale = 5.8 + 1.7 * restore - 0.55 * n_small
    tolerance_scale = (0.135 + 0.055 * restore) * D / gamma_safe
    terminal_target = alpha0 * (terminal_scale + tolerance_scale)
    alpha = (1.0 - terminal_phase) * alpha_plateau + terminal_phase * jnp.maximum(
        alpha_plateau, terminal_target
    )

    moment_start = 0.50 - 0.055 * restore
    moment_width = jnp.maximum(0.90 - moment_start, 1e-6)
    moment_x = jnp.clip((t - moment_start) / moment_width, 0.0, 1.0)
    moment_phase = moment_x * moment_x * (3.0 - 2.0 * moment_x)

    beta1 = 0.165 + 0.020 * n_small - (0.095 + 0.030 * restore) * moment_phase
    beta2 = 0.175 + (0.380 + 0.070 * restore) * moment_phase

    beta1 = jnp.clip(beta1, 0.035, 0.20)
    beta2 = jnp.clip(beta2, 0.16, 0.64)

    return lr, alpha, beta1, beta2