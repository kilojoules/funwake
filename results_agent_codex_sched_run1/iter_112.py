"""HYPOTHESIS: The plateau family is a distinct alternative to the last gated
dual-window schedule. Use a short high-LR plateau and two narrower mobility
kicks, but add the codex low/high-alpha robustness gates so it does not inherit
the older schedule's held-out infeasibility.
AXIS: plateau_double_kick_with_alpha0_robust_gates
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Train-scale path: aggressive plateau with two compact mobility kicks.
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0
    plateau_end = 0.22
    cosine_t = jnp.clip((t - plateau_end) / jnp.maximum(1.0 - plateau_end, 1e-6), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_plateau = jnp.where(t < plateau_end, lr_init, cosine_lr)
    kick1 = jnp.exp(-0.5 * ((t - 0.52) / 0.045) ** 2)
    kick2 = jnp.exp(-0.5 * ((t - 0.77) / 0.035) ** 2)
    lr_plateau = jnp.maximum(lr_plateau + 0.30 * lr_init * kick1 + 0.18 * lr_init * kick2, 1e-10)
    late = jnp.maximum(t - 0.50, 0.0) / 0.50
    alpha_plateau = 5.35 * alpha0 * lr_init / lr_plateau
    alpha_plateau = alpha_plateau + 4.0 * alpha0 * late**2
    alpha_plateau = alpha_plateau + 0.75 * alpha0 * (kick1 + kick2)

    # Low-gradient-scale path: terminal-repair shelf from the robust prior.
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

    # High-alpha held-out path from the robust prior.
    high_init = 4.0 * lr0
    high_min = high_init / 10000.0
    high_base = high_min + (high_init - high_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    high_bump = jnp.exp(-0.5 * ((t - 0.692) / 0.050) ** 2)
    lr_high = jnp.maximum(high_base + 0.30 * high_init * high_bump, 1e-10)
    alpha_high = 5.0 * alpha0 * high_init / lr_high
    alpha_high = alpha_high + 3.0 * alpha0 * late**2
    alpha_high = alpha_high + 1.18 * alpha0 * high_bump

    low_scale = alpha0 < 1.0e-4
    high_scale = alpha0 > 5.0e-4

    lr = jnp.where(low_scale, lr_repair, jnp.where(high_scale, lr_high, lr_plateau))
    alpha = jnp.where(low_scale, alpha_repair, jnp.where(high_scale, alpha_high, alpha_plateau))
    beta1 = jnp.where(low_scale, 0.245, jnp.where(high_scale, 0.3, 0.3))
    beta2 = jnp.where(low_scale, 0.60 - 0.05 * repair_mid + 0.03 * tail,
                      jnp.where(high_scale, 0.5, 0.5))

    return lr, alpha, beta1, beta2
