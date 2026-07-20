"""Iter 272: Cosine with flat plateau until 30% then rapid drop.

Piecewise: LR = lr_init for t<0.3, then cosine from lr_init to lr_min
for t in [0.3, 1.0]. Pure exploration for 30%, then converge.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    plateau_end = 0.3
    in_plateau = t < plateau_end
    decay_t = (t - plateau_end) / (1.0 - plateau_end)
    decay_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * decay_t))

    lr_base = jnp.where(in_plateau, lr_init, decay_lr)

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
