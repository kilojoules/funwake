"""Iter 073: Cosine decay with 4x LR + short 10% warmup."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 30000.0
    warmup = 0.10

    # Linear warmup then cosine decay
    in_warmup = t < warmup
    warmup_lr = lr_init * (t / warmup)
    decay_t = (t - warmup) / (1.0 - warmup)
    cos_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * decay_t))

    lr = jnp.where(in_warmup, warmup_lr, cos_lr)
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
