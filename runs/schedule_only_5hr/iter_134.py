"""Iter 134: One-cycle policy (super-convergence style).

Linear warmup 0-15%, cosine decay 15-100%. Higher initial LR (5x).
Super-convergence uses aggressive warmup then strong decay.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 5.0 * lr0
    lr_min = lr_init / 10000.0
    warmup_end = 0.15

    in_warmup = t < warmup_end

    # Linear warmup from lr_min to lr_init
    lr_warmup = lr_min + (lr_init - lr_min) * (t / warmup_end)

    # Cosine decay from lr_init to lr_min
    decay_t = (t - warmup_end) / (1.0 - warmup_end)
    lr_decay = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * decay_t))

    lr = jnp.where(in_warmup, lr_warmup, lr_decay)

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
