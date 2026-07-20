"""Iter 005: Constant LR for 40% + steep cosine decay + stronger alpha coupling.

Longer exploration phase (40% vs seed's 33%), then steeper decay.
Alpha ramps more aggressively in final 20%.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Constant phase: 40% of steps
    const_frac = 0.4
    in_const = t < const_frac

    # Decay phase: cosine from lr0 to 0.005*lr0
    decay_t = (t - const_frac) / (1.0 - const_frac)
    lr_min = 0.005 * lr0
    lr_decay = lr_min + 0.5 * (lr0 - lr_min) * (1 + jnp.cos(jnp.pi * decay_t))

    lr = jnp.where(in_const, lr0, lr_decay)

    # Alpha: constant during exploration, then ramp inversely with LR
    alpha = jnp.where(in_const, alpha0,
                      alpha0 * lr0 / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
