"""Iter 233: Proven cosine LR + coupled alpha, BUT with scheduled betas.

Keep the winning formula (4x LR, cosine, 5x coupled alpha, bump at 0.7)
but explore the one untouched dimension: vary beta1/beta2 over time.
- Early: beta1=0.6, beta2=0.8 (high momentum for fast exploration)
- Late: beta1=0.15, beta2=0.3 (low momentum for precise convergence)
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Proven warmup + cosine
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    # Proven bump at 0.7
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Proven 5x coupled alpha + late quadratic
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    # NEW: Scheduled betas (smooth cosine interpolation)
    beta1 = 0.15 + 0.45 * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    beta2 = 0.30 + 0.50 * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    return lr, alpha, beta1, beta2
