import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 170: Tweak of 154.
    - 32% constant LR (3.2 * lr0).
    - 68% multiplicative decay (mid=49).
    - Alpha: Coupled to 1/LR, starting at 75 * alpha0.
    - Beta: (0.1, 0.9).
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 3.2
    
    is_const = (t < 0.32)
    t_decay = (t - 0.32) / 0.68
    
    # Multiplicative decay
    lr_mult = lr_peak / (1.0 + 49.0 * t_decay)
    
    lr = jnp.where(is_const, lr_peak, lr_mult)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_base = 75.0 * alpha0
    
    # Couple to LR and increase globally
    alpha = alpha_base * (lr_peak / jnp.maximum(lr, 1e-10)) * (1.0 + 15.0 * t**2.5)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
