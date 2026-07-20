"""HYPOTHESIS: The dual-Gaussian incumbent has plateaued, so test a more
piecewise trust-region path: warmup into a short high-LR shelf, one broad
smooth rectangular mobility window for layout relocation, then a small late
reheat for wake polishing while inverse-LR and late penalties preserve
feasibility.
AXIS: smooth_shelf_rectangular_reheat
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.18 * lr0
    lr_min = lr_init / 9000.0

    warmup_end = 0.050
    shelf_end = 0.185
    warmup_lr = lr_init * t / jnp.maximum(warmup_end, 1e-6)

    shelf_u = (t - warmup_end) / jnp.maximum(shelf_end - warmup_end, 1e-6)
    shelf_lr = lr_init * (1.0 - 0.10 * jnp.clip(shelf_u, 0.0, 1.0))

    decay_u = jnp.clip((t - shelf_end) / jnp.maximum(1.0 - shelf_end, 1e-6), 0.0, 1.0)
    decay_lr = lr_min + (0.90 * lr_init - lr_min) * (1.0 - decay_u) ** 1.75

    base_lr = jnp.where(t < warmup_end, warmup_lr, jnp.where(t < shelf_end, shelf_lr, decay_lr))

    mid_on = jnp.clip((t - 0.365) / 0.055, 0.0, 1.0)
    mid_off = jnp.clip((0.650 - t) / 0.070, 0.0, 1.0)
    mid_on = mid_on * mid_on * (3.0 - 2.0 * mid_on)
    mid_off = mid_off * mid_off * (3.0 - 2.0 * mid_off)
    mid_window = mid_on * mid_off

    late_shape = jnp.exp(-0.5 * ((t - 0.865) / 0.050) ** 2)
    lr = base_lr + 0.20 * lr_init * mid_window + 0.11 * lr_init * late_shape
    tail = jnp.clip((t - 0.90) / 0.10, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    lr = lr * (1.0 - 0.45 * tail)
    lr = jnp.maximum(lr, 1e-10)

    late = jnp.maximum(t - 0.48, 0.0) / 0.52
    alpha = 4.20 * alpha0 * lr_init / lr
    alpha = alpha + 18.0 * alpha0 * late**2
    alpha = alpha + 1.0 * alpha0 * jnp.maximum(t - 0.72, 0.0) / 0.28
    alpha = alpha + 12.0 * alpha0 * tail**2

    beta1 = 0.24
    beta2 = 0.58 - 0.06 * mid_window

    return lr, alpha, beta1, beta2
