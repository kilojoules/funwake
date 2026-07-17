"""HYPOTHESIS: Deterministic per-step LR noise can shake the strong gated
warmup/cosine repair schedule out of repeated basins, while a decaying noise
amplitude and inverse-LR alpha coupling preserve the terminal feasibility path.
AXIS: lr_noise_injection on alpha0-gated warmup/cosine repair.
LESSON: Pending score.
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    key = jax.random.fold_in(jax.random.PRNGKey(20260426), step)
    random = jax.random.normal(key, ())
    random = jnp.clip(random, -1.75, 1.75) / 1.75

    real_init = 4.327099 * lr0
    real_min = real_init / (10.0 ** 3.420468)
    warmup_end = 0.061433
    warmup_lr = real_init * t / jnp.maximum(warmup_end, 1e-6)
    cosine_t = (t - warmup_end) / jnp.maximum(1.0 - warmup_end, 1e-6)
    cosine_lr = real_min + (real_init - real_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    real_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
    real_mid = jnp.exp(-0.5 * ((t - 0.546162) / 0.120347) ** 2)
    real_late = jnp.exp(-0.5 * ((t - 0.923544) / 0.112811) ** 2)
    lr_real = real_base + 0.458205 * real_init * real_mid + 0.165784 * real_init * real_late

    repair_init = 4.24 * lr0
    repair_min = repair_init / 11000.0
    repair_warm = 0.055
    repair_shelf = 0.175
    repair_warm_lr = repair_init * t / jnp.maximum(repair_warm, 1e-6)
    shelf_u = jnp.clip((t - repair_warm) / jnp.maximum(repair_shelf - repair_warm, 1e-6), 0.0, 1.0)
    shelf_lr = repair_init * (1.0 - 0.08 * shelf_u)
    decay_u = jnp.clip((t - repair_shelf) / jnp.maximum(1.0 - repair_shelf, 1e-6), 0.0, 1.0)
    decay_lr = repair_min + (0.92 * repair_init - repair_min) * (1.0 - decay_u) ** 1.65
    repair_base = jnp.where(t < repair_warm, repair_warm_lr,
                            jnp.where(t < repair_shelf, shelf_lr, decay_lr))
    mid_on = jnp.clip((t - 0.385) / 0.070, 0.0, 1.0)
    mid_off = jnp.clip((0.655 - t) / 0.080, 0.0, 1.0)
    mid_on = mid_on * mid_on * (3.0 - 2.0 * mid_on)
    mid_off = mid_off * mid_off * (3.0 - 2.0 * mid_off)
    repair_mid = mid_on * mid_off
    repair_late = jnp.exp(-0.5 * ((t - 0.895) / 0.065) ** 2)
    lr_repair = repair_base + 0.22 * repair_init * repair_mid + 0.13 * repair_init * repair_late
    tail = jnp.clip((t - 0.855) / 0.145, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    lr_repair = jnp.maximum(lr_repair * (1.0 - 0.82 * tail), 1e-10)

    low_scale = alpha0 < 1.0e-4
    noise_window = jnp.clip((t - 0.018) / 0.050, 0.0, 1.0)
    noise_window = noise_window * noise_window * (3.0 - 2.0 * noise_window)
    noise_fade = jnp.clip((0.82 - t) / 0.82, 0.0, 1.0)
    noise_amp = 0.22 * noise_window * noise_fade * noise_fade
    noise_amp = jnp.where(low_scale, 0.0, noise_amp)
    noise_mult = jnp.clip(1.0 + noise_amp * random, 0.82, 1.24)

    lr_base = jnp.where(low_scale, lr_repair, lr_real)
    lr = jnp.maximum(lr_base * noise_mult, 1e-10)

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_real = 2.879478 * alpha0 * real_init / lr
    alpha_real = alpha_real + 16.850946 * alpha0 * late ** 2

    repair_late_ramp = jnp.maximum(t - 0.50, 0.0) / 0.50
    alpha_repair = 3.55 * alpha0 * repair_init / lr
    alpha_repair = alpha_repair + 24.00 * alpha0 * repair_late_ramp ** 2
    alpha_repair = alpha_repair + 1.10 * alpha0 * repair_mid
    alpha_repair = alpha_repair + 36.00 * alpha0 * tail ** 2

    alpha = jnp.where(low_scale, alpha_repair, alpha_real)
    beta1 = jnp.where(low_scale, 0.245, 0.239994)
    beta2_repair = 0.60 - 0.05 * repair_mid + 0.03 * tail
    beta2 = jnp.where(low_scale, beta2_repair, 0.635963)

    return lr, alpha, beta1, beta2
