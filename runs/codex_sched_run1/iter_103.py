"""HYPOTHESIS: The single-bump counter-ramp has plateaued, while the rounded
dual-bump path is the best robust basin. A small smooth time warp should
change when the two mobility windows are encountered without changing their
validated relative shape, potentially avoiding terminal over-correction.
AXIS: time_warped_rounded_dual_bump
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Smoothly advance the middle of the run and give the tail a little more
    # polishing time. Endpoints remain fixed at 0 and 1.
    warp = 0.018 * jnp.sin(2.0 * jnp.pi * t) * (1.0 - 0.35 * t)
    tw = jnp.clip(t + warp, 0.0, 1.0)

    lr_init = 4.327099 * lr0
    lr_min = lr_init / (10.0 ** 3.420468)

    warmup_end = 0.061433
    warmup_lr = lr_init * tw / jnp.maximum(warmup_end, 1e-6)
    cosine_t = (tw - warmup_end) / jnp.maximum(1.0 - warmup_end, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(tw < warmup_end, warmup_lr, cosine_lr)

    late_shape = jnp.exp(-0.5 * ((tw - 0.923544) / 0.112811) ** 2)
    mid_shape = jnp.exp(-0.5 * ((tw - 0.546162) / 0.120347) ** 2)
    lr = lr_base + 0.165784 * lr_init * late_shape + 0.458205 * lr_init * mid_shape
    lr = jnp.maximum(lr, 1e-10)

    late = jnp.maximum(tw - 0.5, 0.0) / 0.5
    alpha = 2.879478 * alpha0 * lr_init / lr
    alpha = alpha + 16.850946 * alpha0 * late**2

    return lr, alpha, 0.239994, 0.635963
