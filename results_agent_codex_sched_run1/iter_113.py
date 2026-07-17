"""HYPOTHESIS: The robust dual-window basin may be over-constrained during the
late wake-polish bump. Use a small alpha relief localized to that late LR
window, then over-recover with terminal quadratic pressure. This differs from
the previous regime gate by changing the train-scale alpha phase directly.
AXIS: dual_window_late_alpha_relief_recovery
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.327099 * lr0
    lr_min = lr_init / (10.0 ** 3.420468)

    warmup_end = 0.061433
    warmup_lr = lr_init * t / jnp.maximum(warmup_end, 1e-6)
    cosine_t = (t - warmup_end) / jnp.maximum(1.0 - warmup_end, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    mid_shape = jnp.exp(-0.5 * ((t - 0.546162) / 0.120347) ** 2)
    late_shape = jnp.exp(-0.5 * ((t - 0.923544) / 0.112811) ** 2)
    lr = lr_base + 0.458205 * lr_init * mid_shape + 0.165784 * lr_init * late_shape
    lr = jnp.maximum(lr, 1e-10)

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    relief = 1.0 - 0.10 * late_shape
    terminal = jnp.clip((t - 0.955) / 0.045, 0.0, 1.0)
    terminal = terminal * terminal * (3.0 - 2.0 * terminal)
    alpha = 2.879478 * alpha0 * lr_init / lr
    alpha = alpha * relief
    alpha = alpha + 16.850946 * alpha0 * late**2
    alpha = alpha + 8.0 * alpha0 * terminal**2

    return lr, alpha, 0.239994, 0.635963
