"""HYPOTHESIS: A broad two-bump schedule with weaker inverse-LR coupling but stronger terminal alpha ramp may outperform the narrow late-bump basin while keeping feasibility.
AXIS: broad_dual_bump_terminal_alpha
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.327099 * lr0
    lr_min = lr_init / (10.0 ** 3.420468)

    warmup_end = 0.061433
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    mid_bump = 0.458205 * lr_init * jnp.exp(-0.5 * ((t - 0.546162) / 0.120347) ** 2)
    late_bump = 0.165784 * lr_init * jnp.exp(-0.5 * ((t - 0.923544) / 0.112811) ** 2)
    lr = jnp.maximum(lr_base + mid_bump + late_bump, 1e-10)

    alpha = 2.879478 * alpha0 * lr_init / lr
    alpha = alpha + 16.850946 * alpha0 * (jnp.maximum(t - 0.5, 0.0) / 0.5) ** 2

    return lr, alpha, 0.239994, 0.635963
