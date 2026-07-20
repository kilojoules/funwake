"""Iter 091: Exponential decay, 4x LR, lr/10000 final."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Exponential: lr = lr_init * exp(-k*t) where k = ln(lr_init/lr_min)
    k = jnp.log(lr_init / lr_min)
    lr = lr_init * jnp.exp(-k * t)
    lr = jnp.maximum(lr, lr_min)

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
