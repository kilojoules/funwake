"""Iter 049: iter_039 + oscillating LR (small cosine perturbation on constant phase).

Add LR oscillation during constant phase to explore more of landscape.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    const_frac = 0.35
    in_const = t < const_frac
    lr_init = 3.0 * lr0

    # During constant phase: oscillate LR between 2.5x and 3.5x lr0
    explore_t = t / jnp.maximum(const_frac, 1e-6)
    lr_osc = lr_init + 0.5 * lr0 * jnp.sin(2 * jnp.pi * 4 * explore_t)

    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_decay = lr_init / (1 + 29999.0 * decay_t)
    lr = jnp.where(in_const, lr_osc, lr_decay)

    alpha = jnp.where(in_const, 3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
