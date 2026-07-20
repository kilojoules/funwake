"""Iter 294: Cosine with 5x LR + stronger bump + heavy late alpha.

Variation of best pattern but with:
- 5x initial LR (instead of 4-4.5x)
- Larger bump at t=0.65
- Much stronger alpha enforcement in final 20%
Goal: higher AEP from more exploration while still landing feasible.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 5.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup
    warmup_end = 0.03
    warmup_lr = lr_init * t / warmup_end
    cosine_t = jnp.clip((t - warmup_end) / (1.0 - warmup_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    # Larger bump at 0.65
    bump = 0.4 * lr_init * jnp.exp(-0.5 * ((t - 0.65) / 0.04) ** 2)
    lr = lr_base + bump

    # Base inverse coupling
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    # Heavy enforcement in final 20%
    late = jnp.maximum(t - 0.8, 0.0) / 0.2
    alpha_extra = 10.0 * alpha0 * late ** 3
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
