import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 138: Hierarchical Cyclic + Exponential Hardening.
    - 4 Large Cycles (2000 steps each).
    - Mini-Shakes every 500 steps.
    - Exponential Global Alpha Ramp: jnp.exp(6.0 * t_global) for stronger late pressure.
    - Final Enforcement: last 3% (240 steps) with balanced LR/Alpha.
    - Beta: Slightly higher base momentum (0.2, 0.4) ramping to (0.5, 0.9).
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions ──────────────────────────────────────
    large_cycle_len = 2000
    t_large = (step % large_cycle_len) / (large_cycle_len - 1)
    
    mini_cycle_len = 500
    t_mini = (step % mini_cycle_len) / (mini_cycle_len - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Global Peak LR
    lr_peak_global = lr0 * (10.0 * (1.0 - 0.7 * t_global) + 3.0 * 0.7 * t_global)
    
    # Base LR: decays within each large cycle
    lr_base = lr_peak_global * (1.0 - 0.9 * t_large)
    
    # Mini-shakes: LR spikes
    spike_mag = 0.5 * lr_peak_global * (1.0 - 0.4 * t_global)
    spike = spike_mag * jnp.exp(- (t_mini**2) / (2 * 0.12**2))
    
    lr = jnp.maximum(lr_base + spike, 1e-4 * lr0)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Exponential global scale
    alpha_global_scale = jnp.exp(6.0 * t_global)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR
    alpha_coupled = alpha_global * (lr0 * 10.0 / jnp.maximum(lr, 1e-10))
    
    # Dips during spikes
    dip_mag = 0.98 * (1.0 - 0.5 * t_global)
    dip = dip_mag * jnp.exp(- (t_mini**2) / (2 * 0.18**2))
    
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2 + 0.3 * t_global
    beta2 = 0.4 + 0.5 * t_global
    
    # ── Final Enforcement (last 3%) ─────────────────────────────
    # Smoothly transition to squeeze in the last 240 steps
    is_enforce = (t_global > 0.97)
    enforce_t = (t_global - 0.97) / 0.03
    
    lr_squeeze = lr0 * 0.0001
    alpha_squeeze = alpha0 * 5000000.0
    
    lr = jnp.where(is_enforce, 
                   lr * (1.0 - enforce_t) + lr_squeeze * enforce_t, 
                   lr)
    alpha = jnp.where(is_enforce, 
                      alpha * (1.0 - enforce_t) + alpha_squeeze * enforce_t, 
                      alpha)
    
    return lr, alpha, beta1, beta2
