import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 106: Single long cycle with standard Adam (0.9, 0.999).
    - No restarts.
    - High initial LR (20x).
    - Coupled alpha.
    """
    t = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 20.0
    lr_min = lr0 * 0.001
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global = alpha0 * (1.0 + 100.0 * (t ** 3))
    alpha = alpha_global * (lr_peak / jnp.maximum(lr, 1e-10))
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta (Standard Adam) ────────────────────────────────────
    beta1 = 0.9
    beta2 = 0.999
    
    return lr, alpha, beta1, beta2
