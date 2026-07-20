import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 153: Multiplicative Decay Phase.
    - 30% constant LR (3.0 * lr0).
    - 70% multiplicative decay (standard TopFarm style).
    - Alpha: Coupled to 1/LR, starting at 75 * alpha0.
    - Beta: (0.2, 0.9).
    - Final Squeeze.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 3.0
    
    is_const = (t < 0.30)
    t_decay = (t - 0.30) / 0.70
    
    # Multiplicative decay: 0.01 = 1 / (1 + 99)
    # mid = 99
    lr_mult = lr_peak / (1.0 + 99.0 * t_decay)
    
    lr = jnp.where(is_const, lr_peak, lr_mult)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_base = 75.0 * alpha0
    
    # Couple to LR and increase globally
    alpha = alpha_base * (lr_peak / jnp.maximum(lr, 1e-10)) * (1.0 + 12.0 * t**2)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
