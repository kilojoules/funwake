import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 113: 6-cycle (5x1000, 1x3000) with moderate Beta and cubic Alpha.
    - Constant Beta (0.2, 0.4) for stability.
    - LR peak decay: 10*lr0 -> 4*lr0.
    - Alpha global: 1 + 200 * t_global^3.
    - Alpha coupled to 6*lr0 / lr.
    - Longer squeeze (4%) with moderate LR.
    """
    is_long_final = (step >= 5000)
    c_start = jnp.where(is_long_final, 5000, (step // 1000) * 1000)
    c_len = jnp.where(is_long_final, 3000, 1000)
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * (10.0 - 6.0 * t_global)
    lr_min = lr0 * 0.002
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global_scale = 1.0 + 200.0 * (t_global ** 3)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling
    alpha_local = alpha_global * (lr0 * 6.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at start of each cycle
    dip_magnitude = 0.92 * (1.0 - 0.5 * t_global)
    dip_width = 0.12
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    # 4% squeeze (320 steps)
    is_squeeze = (t_global > 0.96)
    lr = jnp.where(is_squeeze, lr0 * 0.0005, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 50000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.4
    
    return lr, alpha, beta1, beta2
