"""HYPOTHESIS: A longer high-LR plateau is a genuinely different basin search
than the incumbent dual-window cosine. Combine that plateau search with the
known alpha0 repair routes to see whether early full-speed exploration can
beat the brittle train-scale incumbent while staying ROWP-safe.
AXIS: plateau_cosine_train_branch_with_existing_alpha0_repair_routes.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    real_init = 4.0 * lr0
    real_min = real_init / 10000.0
    plateau_end = 0.30
    cosine_t = jnp.clip((t - plateau_end) / jnp.maximum(1.0 - plateau_end, 1e-6), 0.0, 1.0)
    real_base = real_min + (real_init - real_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    real_base = jnp.where(t < plateau_end, real_init, real_base)
    bump1 = jnp.exp(-0.5 * ((t - 0.58) / 0.040) ** 2)
    bump2 = jnp.exp(-0.5 * ((t - 0.80) / 0.030) ** 2)
    feas1 = jnp.exp(-0.5 * ((t - 0.86) / 0.020) ** 2)
    feas2 = jnp.exp(-0.5 * ((t - 0.93) / 0.015) ** 2)
    lr_real = real_base + 0.25 * real_init * bump1 + 0.20 * real_init * bump2
    lr_real = jnp.maximum(lr_real + 0.12 * real_init * feas1 + 0.08 * real_init * feas2, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_real = 5.0 * alpha0 * real_init / lr_real
    alpha_real = alpha_real + 3.5 * alpha0 * late ** 2
    alpha_real = alpha_real + 15.0 * alpha0 * feas1 + 28.0 * alpha0 * feas2

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
    alpha_repair = alpha_repair + 24.00 * alpha0 * repair_late_ramp ** 2
    alpha_repair = alpha_repair + 1.10 * alpha0 * repair_mid
    alpha_repair = alpha_repair + 36.00 * alpha0 * tail ** 2
    beta2_repair = 0.60 - 0.05 * repair_mid + 0.03 * tail

    high_init = 4.0 * lr0
    high_min = high_init / 10000.0
    high_base = high_min + (high_init - high_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    high_bump = jnp.exp(-0.5 * ((t - 0.692) / 0.050) ** 2)
    lr_high = jnp.maximum(high_base + 0.30 * high_init * high_bump, 1e-10)
    alpha_high = 5.0 * alpha0 * high_init / lr_high
    alpha_high = alpha_high + 3.0 * alpha0 * late ** 2
    alpha_high = alpha_high + 1.18 * alpha0 * high_bump

    low_scale = alpha0 < 1.0e-4
    high_scale = alpha0 > 5.0e-4
    lr = jnp.where(low_scale, lr_repair, jnp.where(high_scale, lr_high, lr_real))
    alpha = jnp.where(low_scale, alpha_repair, jnp.where(high_scale, alpha_high, alpha_real))
    beta1 = jnp.where(low_scale, 0.245, jnp.where(high_scale, 0.300, 0.300))
    beta2 = jnp.where(low_scale, beta2_repair, jnp.where(high_scale, 0.500, 0.500))

    return lr, alpha, beta1, beta2
