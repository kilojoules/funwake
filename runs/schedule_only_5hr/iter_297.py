"""Iter 297: Standard Adam betas (0.9/0.999) early → TopFarm betas late.

Hypothesis: standard Adam converges better in early exploration phase.
Combined with best cosine schedule. Phase transition at t=0.4.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = jnp.clip((t - warmup_end) / (1.0 - warmup_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    # Smooth transition from standard Adam to TopFarm betas
    transition = jnp.clip((t - 0.3) / 0.2, 0.0, 1.0)  # sigmoid-like at 0.3-0.5
    beta1 = (1.0 - transition) * 0.9 + transition * 0.1
    beta2 = (1.0 - transition) * 0.999 + transition * 0.2

    return lr, alpha, beta1, beta2
