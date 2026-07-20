"""Iter 070: Polynomial (quadratic) LR decay + 5x initial LR.

Quadratic decay concentrates more iterations at higher LR,
decaying slowly at first and fast at end.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 5.0 * lr0
    lr_min = lr_init / 30000.0
    const_frac = 0.25

    in_const = t < const_frac
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)

    # Quadratic decay: stays high longer, drops fast at end
    lr_poly = lr_init * (1.0 - decay_t) ** 2 + lr_min
    lr = jnp.where(in_const, lr_init, lr_poly)

    # Alpha coupled to 1/lr
    alpha = jnp.where(in_const, 3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
