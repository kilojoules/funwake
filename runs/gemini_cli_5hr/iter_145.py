import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 145: 8 Long Cycles (1000 steps).
    - Melt phase: 100 steps (High LR, low Alpha).
    - Freeze phase: 900 steps (Decaying LR, increasing Alpha).
    - Exponential global alpha ramp.
    - Beta ramping up globally.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Cycle Definitions (8 cycles of 1000 steps) ─────────────
    cycle_len = 1000
    t_c = (step % cycle_len) / (cycle_len - 1.0)
    is_melt = (t_c < 0.10)
    
    # ── Global Parameters ───────────────────────────────────────
    # Global peak LR decays
    lr_peak_global = lr0 * (15.0 * (1.0 - t) + 4.0 * t)
    # Exponential Alpha Ramp
    alpha_global_scale = jnp.exp(6.0 * t)
    alpha_global = alpha0 * alpha_global_scale
    
    # ── Melt Phase ─────────────────────────────────────────────
    lr_melt = lr_peak_global
    alpha_melt = alpha0 * 0.05 * (1.0 + 10.0 * t)
    
    # ── Freeze Phase ───────────────────────────────────────────
    t_f = (t_c - 0.10) / 0.90
    lr_freeze = lr_peak_global * 0.1 * (1.0 - t_f) + lr0 * 0.001
    
    # Couple Alpha to 1/LR
    alpha_freeze_coupled = alpha_global * (lr0 * 15.0 / jnp.maximum(lr_freeze, 1e-10))
    # Final pressure in each cycle
    alpha_freeze = alpha_freeze_coupled * (1.0 + 3.0 * t_f**2)
    
    # ── Selection ──────────────────────────────────────────────
    lr = jnp.where(is_melt, lr_melt, lr_freeze)
    alpha = jnp.where(is_melt, alpha_melt, alpha_freeze)
    
    beta1 = 0.1 + 0.4 * t
    beta2 = 0.2 + 0.75 * t
    
    # ── Final Squeeze ──────────────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
