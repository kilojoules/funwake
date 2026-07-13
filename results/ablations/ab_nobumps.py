"""Ablation: iter_192 with the two Gaussian LR bumps removed."""
import jax.numpy as jnp
def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0; lr_min = lr_init / 10000.0
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr = jnp.where(t < warmup_end, warmup_lr, cosine_lr)          # NO bumps
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    dip = 0.5 * jnp.exp(-0.5 * ((t - 0.6) / 0.04) ** 2)
    alpha = (alpha_base + alpha_extra) * (1.0 - dip)
    return lr, alpha, 0.3, 0.5
