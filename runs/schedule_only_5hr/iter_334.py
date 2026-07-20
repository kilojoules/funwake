"""Iter 334: Lighter feasibility version of 262's asymmetric cosine.

262 got 5599.97 (infeasible). This keeps the high-AEP schedule
but adds just enough alpha enforcement to stay feasible, without
the heavy-handed approach of 332.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Asymmetric cosine from 262 (no warmup, just clip)
    cosine_t = jnp.clip(t, 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t ** 0.7))
    lr_base = cosine_lr

    # Bump at 0.7 (from 262)
    bump = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    # Small feas bump
    feas = 0.1 * lr_init * jnp.exp(-0.5 * ((t - 0.88) / 0.02) ** 2)
    lr = lr_base + bump + feas

    # Alpha: same base as 262 but with moderate extra enforcement
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    # One feasibility burst at 0.88
    fb = 18.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.88) / 0.02) ** 2)
    # Gentle ramp in final 5%
    polish = 15.0 * alpha0 * jnp.maximum(t - 0.95, 0.0) / 0.05
    alpha = alpha_base + alpha_extra + fb + polish

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
