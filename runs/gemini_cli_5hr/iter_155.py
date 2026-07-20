import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 155: 40% Constant Phase + High Peak.
    - 40% constant LR (4.0 * lr0).
    - 60% multiplicative decay (mid=49).
    - Alpha: Coupled to 1/LR, starting at 80 * alpha0.
    - Beta: (0.1, 0.9).
    - Final Squeeze.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 4.0
    
    is_const = (t < 0.40)
    t_decay = (t - 0.40) / 0.60
    
    # Multiplicative decay
    lr_mult = lr_peak / (1.0 + 49.0 * t_decay)
    
    lr = jnp.where(is_const, lr_peak, lr_mult)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_base = 80.0 * alpha0
    
    # Couple to LR and increase globally
    alpha = alpha_base * (lr_peak / jnp.maximum(lr, 1e-10)) * (1.0 + 10.0 * t**2)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
