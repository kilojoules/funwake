"""Iter 145: Constant LR for 35% then cosine decay + bump at t=0.7.

Extended constant plateau gives more exploration, then cosine decay.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0
    plateau_end = 0.35

    in_plateau = t < plateau_end
    decay_t = (t - plateau_end) / (1.0 - plateau_end)
    lr_decay = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * decay_t))

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)

    lr = jnp.where(in_plateau, lr_init, lr_decay) + bump

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
