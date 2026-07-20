import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 195: High Beta Ramping + 4+1 Structure.
    - 4 cycles of 1200 steps + 1 final cycle of 3200 steps.
    - Beta ramps up to standard Adam values (0.9, 0.99).
    - Alpha surge starts earlier (0.65) to allow feasibility refinement.
    - LR peak 15.0 * lr0.
    """
    # ── Cycle Definitions ──────────────────────────────────────
    is_long_final = (step >= 4800)
    c_start = jnp.where(is_long_final, 4800.0, jnp.floor(step / 1200.0) * 1200.0)
    c_len = jnp.where(is_long_final, 3200.0, 1200.0)
    
    t_cycle = (step - c_start) / (c_len - 1.0)
    t_global = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 15.0 * (1.0 - 0.8 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Beta ────────────────────────────────────────────────────
    # Ramp to standard Adam values
    beta1 = 0.1 + 0.8 * t_global
    beta2 = 0.2 + 0.79 * t_global
    
    # ── Alpha (Penalty Weight) ──────────────────────────────────
    # Surge starts at 0.65
    alpha_global_scale = 1.0 + 1000.0 * (jax.nn.sigmoid(20.0 * (t_global - 0.65)))
    alpha_global = alpha0 * alpha_global_scale
    
    # Local dip in alpha at the start of each cycle
    dip_magnitude = 0.98 * (1.0 - 0.6 * t_global)
    dip_width = 0.15
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    
    # Coupled LR-penalty
    alpha_local = alpha_global * (lr0 * 12.0 / jnp.maximum(lr, 1e-10))
    alpha = alpha_local * (1.0 - dip)
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_squeeze = (t_global > 0.985)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e12, alpha)
    beta1 = jnp.where(is_squeeze, 0.9, beta1)
    beta2 = jnp.where(is_squeeze, 0.999, beta2)
    
    return lr, alpha, beta1, beta2
