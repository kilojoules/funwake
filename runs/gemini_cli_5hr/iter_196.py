import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 196: Massive Jump Explorations.
    - 4 exploratory cycles (1200 steps) + 1 convergence cycle (3200 steps).
    - Cycles 1 and 3 (index 1 and 2) start with a huge LR jump (30.0 * lr0).
    - Other cycles start at 15.0 * lr0.
    - Global beta ramping (0.1, 0.2) to (0.5, 0.9).
    - Alpha sigmoid surge at 75%.
    """
    # ── Cycle Definitions ──────────────────────────────────────
    is_long_final = (step >= 4800)
    c_idx = jnp.floor(step / 1200.0)
    c_start = jnp.where(is_long_final, 4800.0, c_idx * 1200.0)
    c_len = jnp.where(is_long_final, 3200.0, 1200.0)
    
    t_cycle = (step - c_start) / (c_len - 1.0)
    t_global = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Massive jumps for even-numbered cycles (1 and 3)
    is_jump_cycle = (c_idx == 1) | (c_idx == 3)
    lr_peak_base = jnp.where(is_jump_cycle & (~is_long_final), 30.0, 15.0)
    lr_peak = lr0 * lr_peak_base * (1.0 - 0.8 * t_global)
    
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1 + 0.4 * t_global
    beta2 = 0.2 + 0.7 * t_global
    
    # ── Alpha (Penalty Weight) ──────────────────────────────────
    alpha_global_scale = 1.0 + 1000.0 * (jax.nn.sigmoid(20.0 * (t_global - 0.75)))
    alpha_global = alpha0 * alpha_global_scale
    
    # Local dip in alpha at the start of each cycle
    # Deeper dip during jump cycles
    dip_mag_base = jnp.where(is_jump_cycle, 0.995, 0.98)
    dip_magnitude = dip_mag_base * (1.0 - 0.6 * t_global)
    dip_width = 0.18
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
