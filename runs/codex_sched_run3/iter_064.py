"""HYPOTHESIS: The Farm 1 basin is brittle, so preserve the exact validated
train-scale trajectory and spend the next experiment on high-alpha
generalization. For ROWP-sized alpha0 values, replace the last attempt's
single cosine bump with a plateau, two mobility windows, and a late constraint
lock so irregular farms can separate first and harden feasibility at the end.
AXIS: train_band_preserved_with_high_scale_plateau_reheat_constraint_lock.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

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
    alpha_real = alpha_real + 16.850946 * alpha0 * late ** 2

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

    high_init = 4.08 * lr0
    high_min = high_init / 12500.0
    high_warm = 0.050
    high_shelf = 0.205
    high_warm_lr = high_init * t / jnp.maximum(high_warm, 1e-6)
    high_shelf_u = jnp.clip((t - high_warm) / jnp.maximum(high_shelf - high_warm, 1e-6), 0.0, 1.0)
    high_shelf_lr = high_init * (1.0 - 0.055 * high_shelf_u)
    high_decay_u = jnp.clip((t - high_shelf) / jnp.maximum(1.0 - high_shelf, 1e-6), 0.0, 1.0)
    high_decay_lr = high_min + (0.945 * high_init - high_min) * (1.0 - high_decay_u) ** 1.72
    high_base = jnp.where(t < high_warm, high_warm_lr,
                          jnp.where(t < high_shelf, high_shelf_lr, high_decay_lr))
    high_mid = jnp.exp(-0.5 * ((t - 0.575) / 0.085) ** 2)
    high_polish = jnp.exp(-0.5 * ((t - 0.865) / 0.055) ** 2)
    high_tail = jnp.clip((t - 0.880) / 0.120, 0.0, 1.0)
    high_tail = high_tail * high_tail * (3.0 - 2.0 * high_tail)
    lr_high = high_base + 0.18 * high_init * high_mid + 0.105 * high_init * high_polish
    lr_high = jnp.maximum(lr_high * (1.0 - 0.70 * high_tail), 1e-10)
    alpha_high = 4.55 * alpha0 * high_init / jnp.maximum(high_base, 1e-10)
    alpha_high = alpha_high + 4.00 * alpha0 * late ** 2
    alpha_high = alpha_high + 1.35 * alpha0 * high_mid
    alpha_high = alpha_high + 1.65 * alpha0 * high_polish
    alpha_high = alpha_high + 30.00 * alpha0 * high_tail ** 2
    beta1_high = 0.285 - 0.060 * high_tail
    beta2_high = 0.535 + 0.085 * high_tail - 0.030 * high_mid

    low_scale = alpha0 < 1.0e-4
    high_scale = alpha0 > 5.0e-4
    lr = jnp.where(low_scale, lr_repair, jnp.where(high_scale, lr_high, lr_real))
    alpha = jnp.where(low_scale, alpha_repair, jnp.where(high_scale, alpha_high, alpha_real))
    beta1 = jnp.where(low_scale, 0.245, jnp.where(high_scale, beta1_high, 0.239994))
    beta2 = jnp.where(low_scale, beta2_repair, jnp.where(high_scale, beta2_high, 0.635963))

    return lr, alpha, beta1, beta2
