import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 150: Super-Melt and Freeze.
    - 10 Cycles (800 steps each).
    - Super-Melt (first 10%): Extremely high LR (20*lr0), no momentum (0.05, 0.1).
    - Freeze (remaining 90%): Decaying LR, increasing Alpha.
    - Global Alpha Ramp (polynomial t^3.5).
    - Coupling to 20.0*lr0.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Cycle Definitions (10 cycles of 800) ────────────────────
    cycle_len = 800
    t_c = (step % cycle_len) / (cycle_len - 1.0)
    is_melt = (t_c < 0.10)
    
    # ── Global Parameters ───────────────────────────────────────
    # Global peak LR decays
    lr_peak_global = lr0 * (20.0 * (1.0 - t) + 5.0 * t)
    alpha_global_scale = 1.0 + 200.0 * (t ** 3.5)
    alpha_global = alpha0 * alpha_global_scale
    
    # ── Melt Phase (Super-Exploration) ──────────────────────────
    lr_melt = lr_peak_global
    alpha_melt = alpha0 * 0.01 * (1.0 + 20.0 * t)
    b1_melt = 0.05
    b2_melt = 0.1
    
    # ── Freeze Phase (Refinement) ───────────────────────────────
    t_f = (t_c - 0.10) / 0.90
    lr_freeze = lr_peak_global * 0.1 * (1.0 - t_f) + lr0 * 0.001
    
    # Couple Alpha to 1/LR
    alpha_freeze_coupled = alpha_global * (lr0 * 20.0 / jnp.maximum(lr_freeze, 1e-10))
    alpha_freeze = alpha_freeze_coupled * (1.0 + 5.0 * t_f**2)
    
    b1_freeze = 0.1 + 0.4 * t_f
    b2_freeze = 0.2 + 0.79 * t_f
    
    # ── Selection ──────────────────────────────────────────────
    lr = jnp.where(is_melt, lr_melt, lr_freeze)
    alpha = jnp.where(is_melt, alpha_melt, alpha_freeze)
    beta1 = jnp.where(is_melt, b1_melt, b1_freeze)
    beta2 = jnp.where(is_melt, b2_melt, b2_freeze)
    
    # ── Final Squeeze (last 1%) ─────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
