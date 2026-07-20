import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 161: 154 + Late Shaking.
    - 30% constant LR (3.0 * lr0).
    - 70% multiplicative decay (mid=49).
    - Small LR bump at t=0.9 to shake out of local minima.
    - Alpha: Coupled to 1/LR, starting at 75 * alpha0.
    - Beta: (0.1, 0.9).
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 3.0
    
    is_const = (t < 0.30)
    t_decay = (t - 0.30) / 0.70
    
    # Base multiplicative decay
    lr_base = lr_peak / (1.0 + 49.0 * t_decay)
    
    # Late shake: Gaussian bump at t=0.9
    shake = 0.5 * lr_peak * jnp.exp(-0.5 * ((t - 0.90) / 0.03)**2)
    
    lr = jnp.where(is_const, lr_peak, lr_base + shake)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_base = 75.0 * alpha0
    
    # Couple to LR and increase globally
    # Dip alpha during shake
    dip = 0.8 * jnp.exp(-0.5 * ((t - 0.90) / 0.04)**2)
    alpha = alpha_base * (lr_peak / jnp.maximum(lr, 1e-10)) * (1.0 + 15.0 * t**2.5) * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
