"""Cosine annealing with warm restarts (SGDR-style).

Warm restarts temporarily boost LR, helping escape local minima from
a single starting point. Three cycles in the optimization phase,
then aggressive constraint enforcement.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps, 1.0)

    # 75% optimization with 3 warm restart cycles, 25% enforcement
    opt_end = 0.75
    n_cycles = 3.0

    # === Learning rate ===
    in_opt = t < opt_end

    # Cosine annealing with warm restarts during optimization
    opt_progress = t / opt_end  # 0 to 1 within opt phase
    cycle_progress = jnp.mod(opt_progress * n_cycles, 1.0)
    lr_opt = lr0 * (0.05 + 0.95 * 0.5 * (1 + jnp.cos(jnp.pi * cycle_progress)))

    # Enforcement: exponential decay
    enforce_progress = (t - opt_end) / (1.0 - opt_end)
    lr_enforce = lr0 * 0.05 * jnp.exp(-4.0 * enforce_progress)

    lr = jnp.where(in_opt, lr_opt, lr_enforce)

    # === Alpha ===
    base_alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    boost = jnp.where(t >= opt_end, 50.0, 1.0)
    alpha = base_alpha * boost

    # === Momentum: TopFarm-style ===
    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
