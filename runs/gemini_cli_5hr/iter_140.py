import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 140: 6-cycle Hybrid Convergent.
    - 5 early discovery cycles (800 steps each).
    - 1 long refinement cycle (4000 steps).
    - Initial peak LR multiplier: 12.0 (aggressive shaking).
    - Alpha global ramp: t^3.2 (aggressive enforcement).
    - Beta: Ramps within each cycle for stability.
    - Squeeze: last 1% for final feasibility.
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions (5 cycles of 800, then 1 of 4000) ──────
    is_long_final = (step >= 4000)
    c_start = jnp.where(is_long_final, 4000.0, jnp.floor(step / 800.0) * 800.0)
    c_len = jnp.where(is_long_final, 4000.0, 800.0)
    
    t_cycle = (step - c_start) / (c_len - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Global Peak LR (decays globally)
    lr_peak_global = lr0 * (12.0 * (1.0 - 0.7 * t_global) + 3.0 * 0.7 * t_global)
    lr_min = lr0 * 0.001
    
    # LR within cycle: starts at peak, decays to min
    lr_cyclic = lr_min + 0.5 * (lr_peak_global - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global Alpha Ramp (polynomial)
    alpha_global_scale = 1.0 + 180.0 * (t_global ** 3.2)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR
    # We use 12.0 as the reference peak
    alpha_coupled = alpha_global * (lr0 * 12.0 / jnp.maximum(lr_cyclic, 1e-10))
    
    # Dip at start of cycle
    # Shorter dip for the long final cycle
    dip_width = jnp.where(is_long_final, 0.04, 0.18)
    dip_mag = 0.97 * (1.0 - 0.4 * t_global)
    dip = dip_mag * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    
    alpha_dip = alpha_coupled * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1 + 0.4 * t_cycle
    beta2 = 0.2 + 0.79 * t_cycle
    
    # ── Final Selection ─────────────────────────────────────────
    # Final Squeeze (last 1%)
    is_squeeze = (t_global > 0.99)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr_cyclic)
    alpha = jnp.where(is_squeeze, alpha0 * 1e9, alpha_dip)
    
    return lr, alpha, beta1, beta2
