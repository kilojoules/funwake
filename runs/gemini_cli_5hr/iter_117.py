import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 117: Higher exploration with 5x800 + 4000 cycle structure.
    - 5 cycles of 800 steps, then 1 cycle of 4000 steps.
    - Initial peak LR 8.0*lr0 decaying to 3.2*lr0.
    - Alpha global ramp delayed further (t^4) with higher scale (150).
    - Beta ramp: B1(0.1->0.5), B2(0.2->0.95).
    - Wider Alpha dip (0.16).
    """
    is_long_final = (step >= 4000)
    c_start = jnp.where(is_long_final, 4000, (step // 800) * 800)
    c_len = jnp.where(is_long_final, 4000, 800)
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays 8.0*lr0 to 3.2*lr0
    lr_peak = lr0 * 8.0 * (1.0 - 0.6 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global_scale = 1.0 + 150.0 * (t_global ** 4)
    alpha_global = alpha0 * alpha_global_scale
    
    # Normalized local coupling to 8.0*lr0
    alpha_coupled = alpha_global * (lr0 * 8.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at cycle start
    dip_magnitude = 0.95 * (1.0 - 0.4 * t_global)
    dip_width = 0.16
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1 + 0.4 * t_global
    beta2 = 0.2 + 0.75 * t_global
    
    return lr, alpha, beta1, beta2
