import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 115: 6-cycle approach with moderate peak LR and cubic Alpha.
    - 5 cycles of 1000 steps, then 1 cycle of 3000 steps.
    - Beta ramps globally: B1(0.15->0.4), B2(0.25->0.8).
    - Alpha global scale: 1 + 85 * t_global^3.
    - LR peak decay: 5.5*lr0 -> 2.75*lr0.
    - Alpha dip magnitude decreases globally.
    """
    is_long_final = (step >= 5000)
    c_start = jnp.where(is_long_final, 5000, (step // 1000) * 1000)
    c_len = jnp.where(is_long_final, 3000, 1000)
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 5.5 * (1.0 - 0.5 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global_scale = 1.0 + 85.0 * (t_global ** 3)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupled locally to peak LR of this global step
    alpha_coupled = alpha_global * (lr0 * 5.5 / jnp.maximum(lr, 1e-10))
    
    # Dip at cycle start
    dip_magnitude = 0.93 * (1.0 - 0.4 * t_global)
    dip_width = 0.13
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 5000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.15 + 0.25 * t_global
    beta2 = 0.25 + 0.55 * t_global
    
    return lr, alpha, beta1, beta2
