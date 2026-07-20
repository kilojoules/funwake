"""Iter 045: iter_039 + LR warmup (5% linear ramp before constant)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    warmup_frac = 0.05
    const_frac = 0.35

    in_warmup = t < warmup_frac
    in_const = (t >= warmup_frac) & (t < const_frac)

    lr_init = 3.0 * lr0
    warmup_lr = lr_init * t / warmup_frac

    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_decay = lr_init / (1 + 29999.0 * decay_t)

    lr = jnp.where(in_warmup, warmup_lr,
         jnp.where(in_const, lr_init, lr_decay))

    alpha = jnp.where(t < const_frac, 3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
