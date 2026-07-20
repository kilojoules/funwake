import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 148: Refined 118-structure.
    - 7 cycles of 700 steps, then 1 of 3100.
    - Higher peak LR (8.0*lr0) decaying globally.
    - Stronger global Alpha ramp (150x, t^3.2).
    - Deeper Alpha dips (0.96) for better reorganization.
    - Cyclic Beta (0.1->0.4, 0.2->0.9).
    - Final Squeeze.
    """
    
    t_global = step / (total_steps - 1.0)
    
    # ── Cycle Definitions (from 118) ───────────────────────────
    is_long_final = (step >= 4900)
    c_start = jnp.where(is_long_final, 4900.0, jnp.floor(step / 700.0) * 700.0)
    c_len = jnp.where(is_long_final, 3100.0, 700.0)
    t_cycle = (step - c_start) / (c_len - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays globally
    lr_peak = lr0 * (8.0 * (1.0 - 0.6 * t_global) + 3.2 * 0.6 * t_global)
    lr_min = lr0 * 0.002
    
    # Cosine annealing within cycle
    lr_cyclic = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global Alpha Ramp
    alpha_global_scale = 1.0 + 150.0 * (t_global ** 3.2)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR
    alpha_coupled = alpha_global * (lr0 * 8.0 / jnp.maximum(lr_cyclic, 1e-10))
    
    # Dip at start of cycle
    dip_mag = 0.96 * (1.0 - 0.4 * t_global)
    dip_width = 0.14
    dip = dip_mag * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    
    alpha_dip = alpha_coupled * (1.0 - dip)
    
    # ── Beta (Cyclic) ───────────────────────────────────────────
    beta1 = 0.1 + 0.3 * t_cycle
    beta2 = 0.2 + 0.7 * t_cycle
    
    # ── Final Selection ─────────────────────────────────────────
    # Final Squeeze (last 1%)
    is_sq = (t_global > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr_cyclic)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha_dip)
    
    return lr, alpha, beta1, beta2
