"""Ablation: iter_192 with standard Adam betas (0.9, 0.999) vs Claude's (0.3,0.5)."""
import jax.numpy as jnp
def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0; lr_min = lr_init / 10000.0
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
    bump1 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
    bump2 = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
    lr = lr_base + bump1 + bump2
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = (alpha_base + 3.0 * alpha0 * late ** 2) * (1.0 - 0.5 * jnp.exp(-0.5 * ((t - 0.6) / 0.04) ** 2))
    return lr, alpha, 0.9, 0.999                                    # standard betas
