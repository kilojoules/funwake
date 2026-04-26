"""HYPOTHESIS: A very short warmup can reduce the first-step shock from the
strong bump schedule while preserving the early basin-changing LR peak.
AXIS: lr_gaussian_bumps with warmup and established late repair.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    base = 0.00125 + 0.95 * (1.0 - t) * (1.0 - t)
    gaussian_bump_1 = 3.00 * jnp.exp(-0.5 * ((t - 0.10) / 0.080) ** 2)
    gaussian_bump_2 = 2.20 * jnp.exp(-0.5 * ((t - 0.32) / 0.085) ** 2)
    gaussian_bump_3 = 1.15 * jnp.exp(-0.5 * ((t - 0.55) / 0.080) ** 2)
    gaussian_bump_4 = 0.46 * jnp.exp(-0.5 * ((t - 0.735) / 0.060) ** 2)
    bumps = gaussian_bump_1 + gaussian_bump_2 + gaussian_bump_3 + gaussian_bump_4

    warmup_frac = 0.012
    warmup = jnp.clip(t / warmup_frac, 0.0, 1.0)
    warmup = warmup * warmup * (3.0 - 2.0 * warmup)
    lr = lr0 * (base + bumps) * warmup

    late = jnp.clip((t - 0.62) / 0.38, 0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    tail = jnp.clip((t - 0.87) / 0.13, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    squeeze = jnp.clip((t - 0.978) / 0.022, 0.0, 1.0)
    squeeze = squeeze * squeeze * (3.0 - 2.0 * squeeze)

    lr_safe = jnp.maximum(lr + 0.038 * lr0, 1e-10)
    alpha = alpha0 * 3.28 * lr0 / lr_safe
    bump_norm = bumps / (1.0 + bumps)
    alpha = alpha * (1.0 - 0.62 * (1.0 - 0.48 * t) * bump_norm)
    alpha = alpha * (1.0 + 2.9 * t + 23.0 * late * late + 170.0 * tail * tail)
    alpha = alpha * (1.0 + 70.0 * squeeze * squeeze)

    lr = jnp.where(squeeze > 0.0, lr0 * (0.00107 - 0.00064 * squeeze), lr)

    beta1 = 0.09 + 0.105 * t
    beta2 = 0.30 + 0.25 * t

    return lr, alpha, beta1, beta2
