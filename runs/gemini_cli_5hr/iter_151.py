import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 151: Long Constant Phase (mimicking best_optimizer).
    - 30% constant LR (3.0 * lr0).
    - 70% linear decay to 0.01 * lr0.
    - Alpha: Coupled to 1/LR, starting at 75 * alpha0.
    - Beta: (0.2, 0.9) for more stable adaptive scaling.
    - Squeeze at end.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 3.0
    lr_min = lr0 * 0.01
    
    is_const = (t < 0.30)
    t_decay = (t - 0.30) / 0.70
    
    lr = jnp.where(is_const, lr_peak, lr_peak - (lr_peak - lr_min) * t_decay)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # We want a strong penalty similar to best_optimizer
    # Effective penalty = lr * alpha
    # If best_optimizer used weight 75, we target alpha_eff = 75 * lr_peak
    alpha_base = 75.0 * alpha0
    
    # Couple to LR and increase globally
    alpha = alpha_base * (lr_peak / jnp.maximum(lr, 1e-10)) * (1.0 + 10.0 * t**2)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e9, alpha)
    
    return lr, alpha, beta1, beta2
