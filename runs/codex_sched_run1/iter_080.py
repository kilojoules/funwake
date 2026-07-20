"""HYPOTHESIS: The transferred warmup dual-bump schedule may gain a small
amount from restoring the full-precision optimized parameters rather than the
rounded source file constants.
AXIS: transferred_dual_bump_full_precision
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    K = 4.327099217214548
    logM = 3.4204682031708615
    W = 0.06143342573031323
    A1 = 0.1657844986712761
    A2 = 0.4582054854798933
    c1 = 0.9235435629977167
    c2 = 0.5461616321072262
    w1 = 0.11281119470891057
    w2 = 0.1203474900183133
    C = 2.87947841945131
    D = 16.85094598409327
    B1 = 0.23999415122738466
    B2 = 0.6359632116018108

    lr_init = K * lr0
    lr_min = lr_init / (10.0 ** logM)

    warmup_lr = lr_init * t / jnp.maximum(W, 1e-6)
    cosine_t = (t - W) / jnp.maximum(1.0 - W, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < W, warmup_lr, cosine_lr)

    bump1 = A1 * lr_init * jnp.exp(-0.5 * ((t - c1) / w1) ** 2)
    bump2 = A2 * lr_init * jnp.exp(-0.5 * ((t - c2) / w2) ** 2)
    lr = jnp.maximum(lr_base + bump1 + bump2, 1e-10)

    alpha_base = C * alpha0 * lr_init / lr
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha_base + D * alpha0 * late**2

    return lr, alpha, B1, B2
