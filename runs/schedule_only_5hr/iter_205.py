"""Iter 205: Cosine with warm restart at t=0.65 (single restart).

One restart at 65% — enough time in first cycle for good convergence,
then a fresh start for the final 35% to escape any local minimum.
Restart LR is 50% of initial (not full restart).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup (0-5%)
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end

    # Cycle 1: 5%-65% cosine decay
    restart_t = 0.65
    c1_t = (t - warmup_end) / (restart_t - warmup_end)
    c1_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * c1_t))

    # Cycle 2: 65%-100% cosine decay from 50% peak
    c2_peak = 0.5 * lr_init
    c2_t = (t - restart_t) / (1.0 - restart_t)
    c2_lr = lr_min + (c2_peak - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * c2_t))

    lr = jnp.where(t < warmup_end, warmup_lr,
         jnp.where(t < restart_t, c1_lr, c2_lr))

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
