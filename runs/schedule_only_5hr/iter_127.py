"""Iter 127: Cosine LR, alpha warm-up then 1/lr. 4x, lr/10000, (0.3,0.5).

Alpha starts at 0.1*alpha0 and linearly ramps to alpha0 over first 20%,
then follows standard 1/lr coupling.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Alpha warmup: linearly ramp from 0.1*alpha0 to alpha0 over first 20%
    warmup_frac = 0.20
    alpha_base = alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    warmup_scale = jnp.minimum(t / warmup_frac, 1.0) * 0.9 + 0.1
    alpha = alpha_base * warmup_scale

    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
