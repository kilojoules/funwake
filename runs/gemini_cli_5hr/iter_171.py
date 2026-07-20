import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 171: Zero-Momentum Discovery.
    - 30% constant LR (3.0 * lr0).
    - 70% multiplicative decay (mid=49).
    - Beta1: 0.0 during constant phase, 0.1 during decay.
    - Beta2: 0.9 throughout.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 3.0
    
    is_const = (t < 0.30)
    t_decay = (t - 0.30) / 0.70
    
    # Multiplicative decay
    lr_mult = lr_peak / (1.0 + 49.0 * t_decay)
    
    lr = jnp.where(is_const, lr_peak, lr_mult)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_base = 75.0 * alpha0
    
    # Couple to LR and increase globally
    alpha = alpha_base * (lr_peak / jnp.maximum(lr, 1e-10)) * (1.0 + 15.0 * t**2.5)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = jnp.where(is_const, 0.0, 0.1)
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
