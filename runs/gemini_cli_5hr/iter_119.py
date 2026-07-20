import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 119: Refining 116 with linear dip decay and stricter Alpha.
    - 6 cycles (5x1000, 1x3000).
    - LR peak decay (6.5*lr0 to 2.6*lr0).
    - Alpha global: 1 + 105 * t^3.3.
    - Dip magnitude decays linearly to 0 at t_global = 1.0.
    - Global Beta ramp up to (0.45, 0.90).
    """
    is_long_final = (step >= 5000)
    c_start = jnp.where(is_long_final, 5000, (step // 1000) * 1000)
    c_len = jnp.where(is_long_final, 3000, 1000)
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 6.5 * (1.0 - 0.6 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global_scale = 1.0 + 105.0 * (t_global ** 3.3)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling
    alpha_coupled = alpha_global * (lr0 * 6.5 / jnp.maximum(lr, 1e-10))
    
    # Dip magnitude decays to 0 at the end
    dip_magnitude = 0.95 * (1.0 - t_global)
    dip_width = 0.15
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.15 + 0.30 * t_global
    beta2 = 0.25 + 0.65 * t_global
    
    return lr, alpha, beta1, beta2
