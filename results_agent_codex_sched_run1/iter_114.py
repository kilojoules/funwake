"""HYPOTHESIS: Hard alpha0 regime switches are brittle near farm-scale
boundaries. Keep the proven train-scale dual-window path, but replace the last
attempt's discrete low/mid/high selection with smooth alpha0 gates so held-out
farms near a threshold blend repair behavior instead of jumping abruptly.
AXIS: smooth_alpha0_gated_dual_window
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Train-scale path: validated warmup + dual mobility windows.
    real_init = 4.327099 * lr0
    real_min = real_init / (10.0 ** 3.420468)
    warmup_end = 0.061433
    warmup_lr = real_init * t / jnp.maximum(warmup_end, 1e-6)
    cosine_t = (t - warmup_end) / jnp.maximum(1.0 - warmup_end, 1e-6)
    cosine_lr = real_min + (real_init - real_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    real_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
    real_mid = jnp.exp(-0.5 * ((t - 0.546162) / 0.120347) ** 2)
    real_late = jnp.exp(-0.5 * ((t - 0.923544) / 0.112811) ** 2)
    lr_real = jnp.maximum(real_base + 0.458205 * real_init * real_mid
                          + 0.165784 * real_init * real_late, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_real = 2.879478 * alpha0 * real_init / lr_real
    alpha_real = alpha_real + 16.850946 * alpha0 * late**2

    # Low-gradient-scale path: strong terminal repair for tight polygons.
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
    repair_late_ramp = jnp.maximum(t - 0.50, 0.0) / 0.50
    alpha_repair = 3.55 * alpha0 * repair_init / lr_repair
    alpha_repair = alpha_repair + 24.00 * alpha0 * repair_late_ramp**2
    alpha_repair = alpha_repair + 1.10 * alpha0 * repair_mid
    alpha_repair = alpha_repair + 36.00 * alpha0 * tail**2
    beta2_repair = 0.60 - 0.05 * repair_mid + 0.03 * tail

    # High-alpha path: ROWP-robust single late bump with local counter-ramp.
    high_init = 4.0 * lr0
    high_min = high_init / 10000.0
    high_base = high_min + (high_init - high_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    high_bump = jnp.exp(-0.5 * ((t - 0.692) / 0.050) ** 2)
    lr_high = jnp.maximum(high_base + 0.30 * high_init * high_bump, 1e-10)
    alpha_high = 5.0 * alpha0 * high_init / lr_high
    alpha_high = alpha_high + 3.0 * alpha0 * late**2
    alpha_high = alpha_high + 1.18 * alpha0 * high_bump

    low_u = jnp.clip((1.30e-4 - alpha0) / 0.60e-4, 0.0, 1.0)
    high_u = jnp.clip((alpha0 - 4.00e-4) / 2.00e-4, 0.0, 1.0)
    low_w = low_u * low_u * (3.0 - 2.0 * low_u)
    high_w = high_u * high_u * (3.0 - 2.0 * high_u)
    real_w = jnp.maximum(1.0 - low_w - high_w, 0.0)
    norm = jnp.maximum(low_w + real_w + high_w, 1e-9)
    low_w = low_w / norm
    real_w = real_w / norm
    high_w = high_w / norm

    lr = low_w * lr_repair + real_w * lr_real + high_w * lr_high
    alpha = low_w * alpha_repair + real_w * alpha_real + high_w * alpha_high
    beta1 = low_w * 0.245 + real_w * 0.239994 + high_w * 0.3
    beta2 = low_w * beta2_repair + real_w * 0.635963 + high_w * 0.5

    return lr, alpha, beta1, beta2
