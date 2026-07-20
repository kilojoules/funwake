import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 129: Refined Iteration 118 with constant low Beta.
    - 7 cycles of 700 steps, then 3100 step final cycle.
    - Constant low Beta (0.1, 0.1).
    - LR peak decay (7.0*lr0 down to 3.0*lr0).
    - Alpha global ramp (t^3.5).
    - Coupling to LR.
    - Dip at cycle start (0.95 magnitude).
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions ──────────────────────────────────────
    is_long_final = (step >= 4900)
    c_start = jnp.where(is_long_final, 4900, (step // 700) * 700)
    c_len = jnp.where(is_long_final, 3100, 700)
    
    t_cycle = (step - c_start) / (c_len - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays from 7.0 to 3.0
    lr_peak = lr0 * (7.0 * (1.0 - 0.5 * t_global) + 3.0 * 0.5 * t_global)
    lr_min = lr0 * 0.005
    
    # Cyclic LR: starts high at cycle start
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global ramp increases aggressively
    alpha_global_scale = 1.0 + 150.0 * (t_global ** 3.5)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to LR
    alpha_coupled = alpha_global * (lr0 * 7.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at cycle start
    dip_magnitude = 0.95 * (1.0 - 0.4 * t_global)
    dip_width = 0.14
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.1
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_squeeze = (t_global > 0.99)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e9, alpha)
    
    return lr, alpha, beta1, beta2
