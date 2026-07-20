"""Iter 161: Plateau + aggressive cosine tail.

50% at near-full LR (slow linear decay), then aggressive cosine for last 50%.
This maximizes exploration time before convergence.
5x alpha, bump at 0.75 (during transition).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Phase 1 (0-50%): slow linear decay from lr_init to 0.7*lr_init
    # Phase 2 (50-100%): cosine from 0.7*lr_init to lr_min
    switch = 0.50
    lr_mid = 0.7 * lr_init

    t1 = t / switch  # 0 to 1 in phase 1
    t2 = (t - switch) / (1.0 - switch)  # 0 to 1 in phase 2

    lr_phase1 = lr_init - (lr_init - lr_mid) * t1
    lr_phase2 = lr_min + (lr_mid - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t2))

    lr_base = jnp.where(t < switch, lr_phase1, lr_phase2)

    # Bump at t=0.75
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
    lr = lr_base + bump

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
