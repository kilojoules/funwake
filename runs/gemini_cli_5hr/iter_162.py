import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 162: 2-cycle 154-Hybrid.
    - 2 cycles of 4000 steps.
    - Each cycle: 30% constant, 70% multiplicative decay (mid=49).
    - Global Alpha Ramp (Exponential).
    - Beta: (0.1, 0.9).
    """
    
    t_global = step / (total_steps - 1.0)
    
    # ── Cycle Definitions (2 cycles of 4000) ───────────────────
    cycle_len = 4000
    t_c = (step % cycle_len) / (cycle_len - 1.0)
    is_const = (t_c < 0.30)
    t_decay = (t_c - 0.30) / 0.70
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays globally slightly
    lr_peak = lr0 * 3.0 * (1.0 - 0.2 * t_global)
    lr_mult = lr_peak / (1.0 + 49.0 * t_decay)
    
    lr = jnp.where(is_const, lr_peak, lr_mult)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_base = 75.0 * alpha0
    alpha_global_scale = jnp.exp(5.0 * t_global)
    
    # Couple to LR
    alpha = alpha_base * alpha_global_scale * (lr0 * 3.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at start of second cycle
    is_c2 = (t_global >= 0.5)
    dip = 0.90 * jnp.exp(-0.5 * (t_c / 0.1)**2)
    alpha = jnp.where(is_c2, alpha * (1.0 - dip), alpha)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t_global > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
