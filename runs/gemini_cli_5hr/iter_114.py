import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 114: Refined 6-cycle approach with dynamic Beta and Alpha ramp.
    - 5 cycles of 1000 steps, then 1 cycle of 3000 steps.
    - Beta ramps up globally from (0.1, 0.2) to (0.4, 0.8).
    - Alpha global scale uses power 3.5 to delay the strictest constraints.
    - LR peak decay slightly more aggressive than 105.
    - Wider Alpha dip at cycle start to encourage repositioning.
    """
    is_long_final = (step >= 5000)
    c_start = jnp.where(is_long_final, 5000, (step // 1000) * 1000)
    c_len = jnp.where(is_long_final, 3000, 1000)
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays 6*lr0 to 2.4*lr0
    lr_peak = lr0 * 6.0 * (1.0 - 0.6 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Higher scale but later ramp
    alpha_global_scale = 1.0 + 120.0 * (t_global ** 3.5)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupled locally to 6*lr0 / lr
    alpha_coupled = alpha_global * (lr0 * 6.0 / jnp.maximum(lr, 1e-10))
    
    # Wider dip (0.15 vs 0.10)
    dip_magnitude = 0.95 * (1.0 - 0.5 * t_global)
    dip_width = 0.15
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 5000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1 + 0.3 * t_global
    beta2 = 0.2 + 0.6 * t_global
    
    return lr, alpha, beta1, beta2
