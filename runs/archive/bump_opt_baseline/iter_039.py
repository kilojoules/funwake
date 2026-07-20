"""Bump-family schedule, optimized parameters."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    K = 3.226223
    logM = 3.881309
    W = 0.099647
    A1 = 0.163055
    A2 = 0.250396
    c1 = 0.241927
    c2 = 0.389291
    w1 = 0.029092
    w2 = 0.012036
    C = 5.156895
    D = 17.440662
    B1 = 0.475993
    B2 = 0.534675

    lr_init = K * lr0
    lr_min = lr_init / (10.0 ** logM)

    warmup_lr = lr_init * t / jnp.maximum(W, 1e-6)
    cosine_t = (t - W) / jnp.maximum(1.0 - W, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < W, warmup_lr, cosine_lr)

    bump1 = A1 * lr_init * jnp.exp(-0.5 * ((t - c1) / w1) ** 2)
    bump2 = A2 * lr_init * jnp.exp(-0.5 * ((t - c2) / w2) ** 2)
    lr = lr_base + bump1 + bump2
    lr = jnp.maximum(lr, 1e-10)

    alpha_base = C * alpha0 * lr_init / lr
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = D * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    return lr, alpha, B1, B2
