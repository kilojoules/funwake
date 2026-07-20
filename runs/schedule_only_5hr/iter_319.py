"""Iter 319: Plateau + bumps + triple feas bursts at 0.83, 0.90, 0.95.

Three small feasibility corrections spread across the final 17%.
Each progressively stronger alpha.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    plateau_end = 0.25
    cosine_t = jnp.clip((t - plateau_end) / (1.0 - plateau_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < plateau_end, lr_init, cosine_lr)

    bump1 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.55) / 0.04) ** 2)
    bump2 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.78) / 0.03) ** 2)
    fb1_lr = 0.10 * lr_init * jnp.exp(-0.5 * ((t - 0.83) / 0.02) ** 2)
    fb2_lr = 0.08 * lr_init * jnp.exp(-0.5 * ((t - 0.90) / 0.015) ** 2)
    fb3_lr = 0.05 * lr_init * jnp.exp(-0.5 * ((t - 0.95) / 0.01) ** 2)
    lr = lr_base + bump1 + bump2 + fb1_lr + fb2_lr + fb3_lr

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    fb1_a = 10.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.83) / 0.02) ** 2)
    fb2_a = 20.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.90) / 0.015) ** 2)
    fb3_a = 30.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.95) / 0.01) ** 2)
    alpha = alpha_base + alpha_extra + fb1_a + fb2_a + fb3_a

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
