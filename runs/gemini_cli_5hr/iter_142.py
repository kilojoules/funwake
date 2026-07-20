import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 142: Standard Adam Hybrid.
    - Uses standard Adam parameters (beta1=0.9, beta2=0.999) for better momentum.
    - Long Cosine annealing for LR.
    - Global cubic Alpha ramp for late enforcement.
    - Coupled Alpha-LR relationship.
    - Final Squeeze.
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak LR is moderate since we have momentum now
    lr_peak = lr0 * 5.0
    lr_min = lr0 * 0.001
    
    # Simple Cosine annealing for the whole run
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_global))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Alpha increases globally
    alpha_global_scale = 1.0 + 300.0 * (t_global ** 3.0)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR
    alpha = alpha_global * (lr0 * 5.0 / jnp.maximum(lr, 1e-10))
    
    # ── Beta (Standard Adam) ────────────────────────────────────
    beta1 = 0.9
    beta2 = 0.999
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_squeeze = (t_global > 0.99)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e9, alpha)
    
    return lr, alpha, beta1, beta2
