"""Iter 011: Cyclic alpha with warm restarts.

Key insight: periodically RELAX constraints to allow turbines to jump
to better positions, then re-tighten. This mimics multi-start exploration
within a single run.

3 cycles with decreasing amplitude of alpha relaxation.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # LR: linear warmup 5%, then cosine to 0.001*lr0
    warmup_frac = 0.05
    in_warmup = t < warmup_frac

    lr_cos_t = (t - warmup_frac) / (1.0 - warmup_frac)
    lr_min = 0.001 * lr0
    lr_cos = lr_min + 0.5 * (lr0 - lr_min) * (1 + jnp.cos(jnp.pi * lr_cos_t))
    lr_warmup = lr0 * t / warmup_frac
    lr = jnp.where(in_warmup, jnp.maximum(lr_warmup, lr_min), lr_cos)

    # Cyclic alpha: 3 cycles of sinusoidal oscillation on top of ramp
    # Base ramp: alpha0 -> 200*alpha0
    base = alpha0 * (1.0 + 199.0 * t)
    # Oscillation amplitude decreases: 0.8 at start, 0.1 at end
    amplitude = 0.8 * (1.0 - t) + 0.1 * t
    cycle = jnp.sin(2 * jnp.pi * 3 * t)  # 3 full cycles
    alpha = base * (1.0 + amplitude * cycle)
    alpha = jnp.maximum(alpha, 0.5 * alpha0)  # floor

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
