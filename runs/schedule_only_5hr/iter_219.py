"""Iter 219: SGDR warm restarts - stronger alpha enforcement.

Fix feasibility from iter_218: higher base alpha, stronger coupling,
bigger late penalty. Keep the warm restart structure.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # 3 cycles: [0, 0.25), [0.25, 0.55), [0.55, 1.0)
    lr_min = lr0 / 200.0

    in_c0 = t < 0.25
    in_c1 = (t >= 0.25) & (t < 0.55)

    t_local = jnp.where(in_c0, t / 0.25,
              jnp.where(in_c1, (t - 0.25) / 0.3,
                        (t - 0.55) / 0.45))

    max_lr = jnp.where(in_c0, 5.0 * lr0,
             jnp.where(in_c1, 2.5 * lr0,
                       1.0 * lr0))

    lr = lr_min + (max_lr - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_local))

    # Stronger alpha: 6x base coupling, cap at 60x
    alpha_base = 6.0 * alpha0 * (5.0 * lr0) / jnp.maximum(lr, 1e-10)
    alpha = jnp.minimum(alpha_base, 60.0 * alpha0)
    # Extra penalty ramps from t=0.4
    late = jnp.maximum(t - 0.4, 0.0) / 0.6
    alpha = alpha + 8.0 * alpha0 * late ** 2

    beta1 = jnp.where(in_c0, 0.7,
            jnp.where(in_c1, 0.4, 0.2))
    beta2 = jnp.where(in_c0, 0.9,
            jnp.where(in_c1, 0.6, 0.4))

    return lr, alpha, beta1, beta2
