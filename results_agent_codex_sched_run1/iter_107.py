"""HYPOTHESIS: The last scored single late-bump attempt underperformed the
dual-bump incumbent, but repeated Gaussian tweaks have saturated. Try a
different trust-region shape: warmup into a short high-LR shelf, a broad smooth
rectangular mid-run mobility window, then a restrained late reheat and stronger
terminal penalty closure for feasibility.
AXIS: smooth_rectangular_shelf_reheat_trust_region
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.24 * lr0
    lr_min = lr_init / 11000.0

    warmup_end = 0.055
    shelf_end = 0.175
    warmup_lr = lr_init * t / jnp.maximum(warmup_end, 1e-6)

    shelf_u = jnp.clip((t - warmup_end) / jnp.maximum(shelf_end - warmup_end, 1e-6), 0.0, 1.0)
    shelf_lr = lr_init * (1.0 - 0.08 * shelf_u)

    decay_u = jnp.clip((t - shelf_end) / jnp.maximum(1.0 - shelf_end, 1e-6), 0.0, 1.0)
    decay_lr = lr_min + (0.92 * lr_init - lr_min) * (1.0 - decay_u) ** 1.65
    base_lr = jnp.where(t < warmup_end, warmup_lr, jnp.where(t < shelf_end, shelf_lr, decay_lr))

    mid_on = jnp.clip((t - 0.385) / 0.070, 0.0, 1.0)
    mid_off = jnp.clip((0.655 - t) / 0.080, 0.0, 1.0)
    mid_on = mid_on * mid_on * (3.0 - 2.0 * mid_on)
    mid_off = mid_off * mid_off * (3.0 - 2.0 * mid_off)
    mid_window = mid_on * mid_off

    late_shape = jnp.exp(-0.5 * ((t - 0.895) / 0.065) ** 2)
    lr = base_lr + 0.22 * lr_init * mid_window + 0.13 * lr_init * late_shape

    tail = jnp.clip((t - 0.855) / 0.145, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    lr = jnp.maximum(lr * (1.0 - 0.82 * tail), 1e-10)

    late = jnp.maximum(t - 0.50, 0.0) / 0.50
    alpha = 3.55 * alpha0 * lr_init / lr
    alpha = alpha + 24.00 * alpha0 * late**2
    alpha = alpha + 1.10 * alpha0 * mid_window
    alpha = alpha + 36.00 * alpha0 * tail**2

    beta1 = 0.245
    beta2 = 0.60 - 0.05 * mid_window + 0.03 * tail

    return lr, alpha, beta1, beta2
