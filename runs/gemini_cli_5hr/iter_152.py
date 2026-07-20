import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 152: 50% Constant Phase Strategy.
    - 50% constant LR (5.0 * lr0).
    - 50% linear decay to 0.001 * lr0.
    - Alpha: Coupled to 1/LR, starting at 100 * alpha0.
    - Beta: (0.2, 0.9).
    - Final Squeeze.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 5.0
    lr_min = lr0 * 0.001
    
    is_const = (t < 0.50)
    t_decay = (t - 0.50) / 0.50
    
    lr = jnp.where(is_const, lr_peak, lr_peak - (lr_peak - lr_min) * t_decay)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # weight=100, starting at alpha_base = 100 * alpha0
    alpha_base = 100.0 * alpha0
    
    # Couple to LR and increase globally
    alpha = alpha_base * (lr_peak / jnp.maximum(lr, 1e-10)) * (1.0 + 8.0 * t**2)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
