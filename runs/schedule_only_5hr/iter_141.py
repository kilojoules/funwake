"""Iter 141: Cosine + bump at t=0.7, with beta1 drop during bump.

During the bump (mini restart), temporarily drop beta1 to 0.1 (less momentum)
so the optimizer reacts quickly to the fresh gradient landscape.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump_weight = jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    bump = 0.3 * lr_init * bump_weight
    lr = lr_base + bump

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Drop beta1 during bump for faster adaptation
    beta1 = 0.3 - 0.2 * bump_weight
    beta2 = 0.5

    return lr, alpha, beta1, beta2
