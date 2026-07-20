import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 144: Melt and Freeze Strategy.
    - 10 Cycles (800 steps each).
    - Melt phase (first 20%): High LR, very low Alpha.
    - Freeze phase (remaining 80%): Decaying LR, increasing Alpha.
    - Coupling to maintain gradient magnitude during freeze.
    - Squeeze at the very end.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Cycle Definitions ──────────────────────────────────────
    cycle_len = 800
    t_c = (step % cycle_len) / (cycle_len - 1.0)
    is_melt = (t_c < 0.20)
    
    # ── Global Parameters ───────────────────────────────────────
    lr_peak_global = lr0 * (12.0 * (1.0 - t) + 4.0 * t)
    alpha_global_scale = 1.0 + 200.0 * (t ** 3.0)
    alpha_global = alpha0 * alpha_global_scale
    
    # ── Melt Phase ─────────────────────────────────────────────
    lr_melt = lr_peak_global
    # Very low alpha during melt to allow reorganization
    alpha_melt = alpha0 * 0.05 * (1.0 + 10.0 * t**2)
    
    # ── Freeze Phase ───────────────────────────────────────────
    t_f = (t_c - 0.20) / 0.80
    lr_freeze = lr_peak_global * 0.1 * (1.0 - t_f) + lr0 * 0.001
    
    # Couple alpha to 1/LR during freeze
    alpha_freeze_coupled = alpha_global * (lr0 * 12.0 / jnp.maximum(lr_freeze, 1e-10))
    # Extra pressure towards end of cycle
    alpha_freeze = alpha_freeze_coupled * (1.0 + 5.0 * t_f**2)
    
    # ── Selection ──────────────────────────────────────────────
    lr = jnp.where(is_melt, lr_melt, lr_freeze)
    alpha = jnp.where(is_melt, alpha_melt, alpha_freeze)
    
    beta1 = 0.1 + 0.3 * t
    beta2 = 0.2 + 0.7 * t
    
    # ── Final Squeeze (last 1%) ─────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
