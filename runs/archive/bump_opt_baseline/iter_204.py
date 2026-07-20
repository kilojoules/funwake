"""Bump-family schedule, optimized parameters."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    K = 4.031629
    logM = 3.437812
    W = 0.049168
    A1 = 0.188589
    A2 = 0.147874
    c1 = 0.731921
    c2 = 0.650678
    w1 = 0.080346
    w2 = 0.144931
    C = 2.181959
    D = 13.431195
    B1 = 0.368955
    B2 = 0.607222

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
