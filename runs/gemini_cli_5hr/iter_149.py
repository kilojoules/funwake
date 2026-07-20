import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 149: Slow Cool Strategy.
    - 2000 steps: Constant high LR (3.0*lr0).
    - 4000 steps: Linear decay to 0.1*lr0.
    - 2000 steps: Constant low LR (0.01*lr0).
    - Alpha: Coupled to 1/LR, with global quadratic ramp.
    - Beta: Fixed at (0.2, 0.4).
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_high = lr0 * 3.0
    lr_mid = lr0 * 0.1
    lr_low = lr0 * 0.01
    
    lr = jnp.where(t < 0.25, lr_high,
                   jnp.where(t < 0.75, 
                             lr_high - (lr_high - lr_mid) * (t - 0.25) / 0.5,
                             lr_mid - (lr_mid - lr_low) * (t - 0.75) / 0.25))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global scale ramp
    alpha_global_scale = 1.0 + 150.0 * (t ** 2.5)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR (reference 3.0)
    alpha = alpha_global * (lr0 * 3.0 / jnp.maximum(lr, 1e-10))
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.4
    
    # ── Final Squeeze (last 1%) ─────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
