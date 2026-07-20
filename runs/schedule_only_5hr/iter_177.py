"""Iter 177: Low alpha early (2x) + sigmoid ramp to very high alpha late.

Explore aggressively in first half with minimal constraints,
then sharp sigmoid transition to strong constraint enforcement.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Cosine LR decay with bump at 0.7
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Base coupling (always present but scaled)
    coupling = lr_init / jnp.maximum(lr, 1e-10)

    # Sigmoid transition: alpha multiplier goes from 2x to 12x around t=0.6
    sigmoid = 1.0 / (1.0 + jnp.exp(-20.0 * (t - 0.6)))
    alpha_mult = 2.0 + 10.0 * sigmoid
    alpha = alpha_mult * alpha0 * coupling

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
