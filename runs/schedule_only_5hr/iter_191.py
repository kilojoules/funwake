"""Iter 191: Alpha crash at t=0.5 — near-zero constraints briefly.

Radical constraint relaxation: at t=0.5, alpha drops to ~5% of its
coupled value for a brief window, allowing turbines to cross boundaries
and re-explore. Followed by very aggressive alpha recovery.
LR gets a simultaneous bump to exploit the unconstrained window.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup + cosine (iter_179 base)
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    # LR bump at alpha crash point
    bump = 0.4 * lr_init * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
    lr = lr_base + bump

    # Alpha: base coupling + late ramp
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 5.0 * alpha0 * late ** 2

    # Alpha crash at t=0.5: drop to 5% for brief window
    crash = 0.95 * jnp.exp(-0.5 * ((t - 0.5) / 0.02) ** 2)
    alpha = (alpha_base + alpha_extra) * (1.0 - crash)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
