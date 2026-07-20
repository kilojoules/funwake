"""Iter 120: Cosine LR, 4x, lr/10000, (0.3,0.5), with LR bump at 60% (micro restart)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Standard cosine
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Small Gaussian bump centered at t=0.6 with width 0.05
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.6) / 0.05) ** 2)
    lr = lr_base + bump

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
