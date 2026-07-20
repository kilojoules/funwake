"""Iter 184: Sqrt-warped cosine + double bump + exponential alpha ramp + beta anneal.

Different from previous best (iter_179):
- Sqrt-warped cosine spends more time at high LR (slower initial decay)
- Two exploration bumps (t=0.55 and t=0.8) instead of one
- Exponential alpha ramp in last 30% (faster growth than quadratic)
- Beta1 anneals from 0.4 to 0.2, beta2 from 0.6 to 0.4
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.5 * lr0
    lr_min = lr_init / 10000.0

    # Warmup (first 5%)
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end

    # Sqrt-warped cosine: slower decay early, faster late
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    warped_t = jnp.sqrt(jnp.maximum(cosine_t, 0.0))
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * warped_t))

    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    # Double bump for re-exploration
    bump1 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.55) / 0.04) ** 2)
    bump2 = 0.15 * lr_init * jnp.exp(-0.5 * ((t - 0.82) / 0.03) ** 2)
    lr = lr_base + bump1 + bump2

    # Alpha: stronger coupling + exponential ramp in last 30%
    alpha_base = 6.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.7, 0.0) / 0.3
    alpha_exp = 8.0 * alpha0 * (jnp.exp(3.0 * late) - 1.0) / (jnp.exp(3.0) - 1.0)
    alpha = alpha_base + alpha_exp

    # Beta annealing: high momentum early, low late
    beta1 = 0.4 - 0.2 * t
    beta2 = 0.6 - 0.2 * t

    return lr, alpha, beta1, beta2
