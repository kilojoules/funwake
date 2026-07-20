import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 118: 8-cycle approach with cyclic Beta.
    - 7 cycles of 700 steps, then 1 cycle of 3100 steps.
    - Cyclic Beta: resets at each cycle to (0.1, 0.2) and ramps up.
    - LR peak decay similar to 116.
    - Alpha global ramp: 1 + 100 * t^3.
    """
    is_long_final = (step >= 4900)
    c_start = jnp.where(is_long_final, 4900, (step // 700) * 700)
    c_len = jnp.where(is_long_final, 3100, 700)
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 6.5 * (1.0 - 0.5 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global_scale = 1.0 + 100.0 * (t_global ** 3)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling
    alpha_coupled = alpha_global * (lr0 * 6.5 / jnp.maximum(lr, 1e-10))
    
    # Dip at cycle start
    dip_magnitude = 0.94 * (1.0 - 0.4 * t_global)
    dip_width = 0.12
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 5000000.0, alpha)

    # ── Beta (Cyclic) ───────────────────────────────────────────
    beta1 = 0.1 + 0.3 * t_cycle
    beta2 = 0.2 + 0.7 * t_cycle
    
    return lr, alpha, beta1, beta2
