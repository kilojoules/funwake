import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 192: 5 small cycles + 1 long final cycle.
    - 5 cycles of 800 steps (exploratory).
    - 1 final cycle of 4000 steps (convergence).
    - Global beta ramping (0.1, 0.2) to (0.5, 0.9).
    - Alpha sigmoid surge at 75% (middle of final cycle).
    - Even higher initial LR peak for broader exploration.
    """
    # ── Cycle Definitions ──────────────────────────────────────
    is_long_final = (step >= 4000)
    c_start = jnp.where(is_long_final, 4000.0, jnp.floor(step / 800.0) * 800.0)
    c_len = jnp.where(is_long_final, 4000.0, 800.0)
    
    t_cycle = (step - c_start) / (c_len - 1.0)
    t_global = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays from 18.0*lr0 to 3.0*lr0
    lr_peak = lr0 * 18.0 * (1.0 - 0.85 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1 + 0.4 * t_global
    beta2 = 0.2 + 0.7 * t_global
    
    # ── Alpha (Penalty Weight) ──────────────────────────────────
    # Sigmoid surge after 75% of global run
    alpha_global_scale = 1.0 + 1200.0 * (jax.nn.sigmoid(20.0 * (t_global - 0.75)))
    alpha_global = alpha0 * alpha_global_scale
    
    # Local dip in alpha at the start of each cycle (LR peaks)
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
