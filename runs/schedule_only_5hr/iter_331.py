"""Iter 331: Cosine warm restarts (SGDR) — 3 cycles with decaying amplitude.

Fundamentally different from single-cosine approaches. Each restart
lets the optimizer escape local minima. Final cycle has aggressive
alpha for feasibility.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # 3 cycles: [0, 0.4), [0.4, 0.7), [0.7, 1.0)
    c1_end = 0.4
    c2_end = 0.7

    in_c1 = t < c1_end
    in_c2 = (t >= c1_end) & (t < c2_end)

    t_local_c1 = t / c1_end
    t_local_c2 = (t - c1_end) / (c2_end - c1_end)
    t_local_c3 = (t - c2_end) / (1.0 - c2_end)

    lr_max_c1 = lr_init
    lr_max_c2 = 2.5 * lr0
    lr_max_c3 = 1.5 * lr0

    lr_c1 = lr_min + (lr_max_c1 - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_local_c1))
    lr_c2 = lr_min + (lr_max_c2 - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_local_c2))
    lr_c3 = lr_min + (lr_max_c3 - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_local_c3))

    lr = jnp.where(in_c1, lr_c1, jnp.where(in_c2, lr_c2, lr_c3))

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.6, 0.0) / 0.4
    alpha_extra = 5.0 * alpha0 * late ** 2
    pulse1 = 10.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.4) / 0.02) ** 2)
    pulse2 = 20.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.7) / 0.02) ** 2)
    fb = 30.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.93) / 0.02) ** 2)
    alpha = alpha_base + alpha_extra + pulse1 + pulse2 + fb

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
