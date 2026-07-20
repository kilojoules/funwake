import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 122: 3-Phase Exploration Strategy.
    1. Feasibility Start (0-500): High alpha to quickly find valid space.
    2. Exploration (500-6500): 4 cycles of LR/Alpha.
    3. Final Convergence (6500-8000): Decay LR, ramp Alpha, high Beta2.
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Phase Definitions ──────────────────────────────────────
    is_start = (step < 500)
    is_mid = (step >= 500) & (step < 6500)
    is_end = (step >= 6500)
    
    # ── Phase 1: Start ─────────────────────────────────────────
    lr_start = lr0 * 2.0
    alpha_start = alpha0 * 1000.0
    beta1_start = 0.1
    beta2_start = 0.99
    
    # ── Phase 2: Exploration ───────────────────────────────────
    n_cycles = 4
    mid_step = jnp.maximum(step - 500.0, 0.0)
    mid_total = 6000.0
    t_mid = mid_step / (mid_total - 1)
    
    # Each cycle 1500 steps
    cycle_idx = mid_step // 1500
    t_cycle = (mid_step % 1500) / 1499.0
    
    lr_peak = lr0 * 5.0 * (1.0 - 0.5 * t_mid)
    lr_min = lr0 * 0.1
    lr_mid = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # Alpha is low at lr peak, high at lr troughs
    alpha_base = alpha0 * 100.0 * (1.0 + 200.0 * t_mid)
    alpha_mid = alpha_base * (lr0 / jnp.maximum(lr_mid, 1e-10))
    
    # ── Phase 3: End ───────────────────────────────────────────
    end_step = jnp.maximum(step - 6500.0, 0.0)
    end_total = 1500.0
    t_end = end_step / (end_total - 1)
    
    lr_end = lr_min * (1.0 - t_end) + lr0 * 0.001 * t_end
    alpha_end = alpha_base * 10.0 * (1.0 + 1000.0 * (t_end**2))
    
    # ── Selection ──────────────────────────────────────────────
    lr = jnp.where(is_start, lr_start, jnp.where(is_mid, lr_mid, lr_end))
    alpha = jnp.where(is_start, alpha_start, jnp.where(is_mid, alpha_mid, alpha_end))
    
    # Beta schedules
    beta1 = jnp.where(is_start, 0.1, jnp.where(is_mid, 0.1 + 0.4 * t_cycle, 0.9))
    beta2 = jnp.where(is_start, 0.99, jnp.where(is_mid, 0.99, 0.999))
    
    # ── SQUEEZE (Last 100 steps) ───────────────────────────────
    is_squeeze = (step > total_steps - 100)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e8, alpha)
    
    return lr, alpha, beta1, beta2
