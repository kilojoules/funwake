"""Iter 140: Exponential decay with constant plateau first 20%.

Plateau at full LR for 20%, then exponential decay for remaining 80%.
This gives more time for initial exploration than cosine.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    plateau_end = 0.20
    in_plateau = t < plateau_end

    # Exponential decay: lr_init * exp(-k * t_decay) = lr_min at t=1
    # k = -ln(lr_min/lr_init) / 0.8
    k = -jnp.log(lr_min / lr_init) / (1.0 - plateau_end)
    t_decay = jnp.maximum(t - plateau_end, 0.0)
    lr_decay = lr_init * jnp.exp(-k * t_decay)

    lr = jnp.where(in_plateau, lr_init, lr_decay)

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
