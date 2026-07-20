"""Iter 004: Cosine LR + aggressive alpha ramp + low betas.

Keep TopFarm-style low betas (0.1, 0.2) since standard Adam caused infeasibility.
Use cosine LR decay (smoother than multiplicative) but much stronger alpha ramp.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Cosine annealing LR
    lr_min = 0.005 * lr0
    lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + jnp.cos(jnp.pi * t))

    # Aggressive alpha: coupled to 1/lr but with floor
    alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
