"""Iter 232: Scheduled betas + exponential LR + sigmoidal alpha + late ramp.

NEW STRATEGY: Every prior attempt uses constant beta1=0.3, beta2=0.5.
This schedules betas through phases:
- Early: High momentum (beta1~0.65) for fast exploration, beta2~0.85 for stable Adam
- Late: Low momentum (beta1~0.15, beta2~0.3) for precise refinement

Uses exponential LR decay, sigmoidal alpha + cubic late ramp for feasibility.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup
    warmup_end = 0.05
    warmup_lr = lr_init * (t / warmup_end)

    # Exponential decay after warmup
    decay_rate = jnp.log(lr_init / lr_min)
    decay_t = jnp.clip((t - warmup_end) / (1.0 - warmup_end), 0.0, 1.0)
    exp_lr = lr_init * jnp.exp(-decay_rate * decay_t)
    lr_base = jnp.where(t < warmup_end, warmup_lr, exp_lr)

    # Single bump at 0.65 for escape
    bump = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.65) / 0.05) ** 2)
    lr = lr_base + bump

    # Hybrid alpha: sigmoidal base + cubic late ramp for feasibility
    sigmoid = 1.0 / (1.0 + jnp.exp(-12.0 * (t - 0.4)))
    alpha_base = alpha0 * (1.0 + 49.0 * sigmoid)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 5.0 * alpha0 * lr_init / jnp.maximum(lr_min, 1e-10) * late ** 3
    alpha = alpha_base + alpha_extra

    # Scheduled betas: smooth transition from high to low momentum
    beta1 = 0.15 + 0.50 * (1.0 - t) ** 1.5
    beta2 = 0.30 + 0.55 * (1.0 - t) ** 1.5

    return lr, alpha, beta1, beta2
