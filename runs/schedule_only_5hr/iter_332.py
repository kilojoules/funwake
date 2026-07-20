"""Iter 332: Adapt iter_262 (best AEP=5599.97) with much stronger feasibility.

262 used asymmetric cosine t^0.7 + bump at 0.7 but was infeasible.
This adds: aggressive alpha ramp from t=0.3, strong late enforcement,
and a final polishing phase.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    warmup_lr = lr_init * t / jnp.maximum(warmup_end, 1e-10)
    cosine_t = jnp.clip((t - warmup_end) / (1.0 - warmup_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t ** 0.7))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    bump = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    feas1 = 0.1 * lr_init * jnp.exp(-0.5 * ((t - 0.85) / 0.02) ** 2)
    feas2 = 0.06 * lr_init * jnp.exp(-0.5 * ((t - 0.93) / 0.015) ** 2)
    lr = lr_base + bump + feas1 + feas2

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.3, 0.0) / 0.7
    alpha_extra = 4.0 * alpha0 * late ** 2.5
    fb1 = 20.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.85) / 0.02) ** 2)
    fb2 = 35.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.93) / 0.015) ** 2)
    polish = 50.0 * alpha0 * jnp.maximum(t - 0.97, 0.0) / 0.03
    alpha = alpha_base + alpha_extra + fb1 + fb2 + polish

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
