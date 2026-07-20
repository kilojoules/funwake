"""Iter 210: Compressed cosine (80% period) + long refinement tail.

Cosine decay compressed into first 80%, then flat minimum for last 20%.
The long tail at low LR with high alpha allows better convergence to
a feasible optimum without the noise of cosine's final oscillation.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end

    # Cosine compressed into 5%-80% range
    cos_end = 0.80
    cos_t = (t - warmup_end) / (cos_end - warmup_end)
    cos_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cos_t))

    lr_base = jnp.where(t < warmup_end, warmup_lr,
              jnp.where(t < cos_end, cos_lr, lr_min))

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
