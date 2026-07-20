import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 157: 3-cycle Multi-Start-in-Time.
    - 3 cycles of 2666 steps.
    - Each cycle: 30% constant, 70% multiplicative decay (mid=49).
    - Global Alpha: Exponentially increasing to ensure final feasibility.
    - Beta: (0.1, 0.9).
    """
    
    t_global = step / (total_steps - 1.0)
    
    # ── Cycle Definitions (3 cycles) ───────────────────────────
    cycle_len = 2666
    t_c = (step % cycle_len) / (cycle_len - 1.0)
    is_const = (t_c < 0.30)
    t_decay = (t_c - 0.30) / 0.70
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 3.0 * (1.0 - 0.5 * t_global) # Peak decays globally
    lr_mult = lr_peak / (1.0 + 49.0 * t_decay)
    
    lr = jnp.where(is_const, lr_peak, lr_mult)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global scale ramp (Exponential)
    alpha_global_scale = jnp.exp(6.0 * t_global)
    alpha_global = alpha0 * alpha_global_scale
    
    # Couple to 1/LR
    alpha = alpha_global * (lr0 * 3.0 / jnp.maximum(lr, 1e-10))
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t_global > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
