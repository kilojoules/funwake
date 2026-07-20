import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 111: Modified 6-cycle approach with long final phase and delayed Alpha ramp.
    - 5 cycles of 1000 steps, then 1 cycle of 3000 steps.
    - Constant Beta (0.2, 0.4) as in best previous attempts.
    - Alpha global ramp delayed (t_global^4) to allow more early exploration.
    - Learning rate peak decay similar to iter_105 but with higher start.
    """
    is_long_final = (step >= 5000)
    c_start = jnp.where(is_long_final, 5000, (step // 1000) * 1000)
    c_len = jnp.where(is_long_final, 3000, 1000)
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak LR decays from 8*lr0 to 4*lr0
    lr_peak = lr0 * (8.0 - 4.0 * t_global)
    lr_min = lr0 * 0.002
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global alpha scale stays lower for longer (power 4)
    alpha_global_scale = 1.0 + 150.0 * (t_global ** 4)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to LR
    alpha_local = alpha_global * (lr0 * 5.0 / jnp.maximum(lr, 1e-10))
    
    # Dip alpha at the start of each cycle
    dip_magnitude = 0.95 * (1.0 - 0.5 * t_global)
    dip_width = 0.12
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 50000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.4
    
    return lr, alpha, beta1, beta2
