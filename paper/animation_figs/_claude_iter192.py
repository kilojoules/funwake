"""Iter 192: Best-of-all combo: warmup + cosine + dual bumps + alpha dip.

Combines proven elements from top 5 performers:
- Warmup 5% (from iter_179)
- Cosine base (all top performers)
- Bump at t=0.5 AND t=0.75 (exploratory phases)
- Alpha dip at t=0.6 between bumps (from iter_183)
- 5x coupling + quadratic ramp (standard)
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup + cosine
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    # Two bumps
    bump1 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
    bump2 = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
    lr = lr_base + bump1 + bump2

    # Alpha with dip between bumps
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2

    dip = 0.5 * jnp.exp(-0.5 * ((t - 0.6) / 0.04) ** 2)
    alpha = (alpha_base + alpha_extra) * (1.0 - dip)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
