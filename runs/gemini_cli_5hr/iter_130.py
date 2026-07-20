import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 130: Three-Phase Shake-Explore-Settle.
    1. Phase 1 (0-2000): Shake. High LR, low Beta, moderate Alpha.
    2. Phase 2 (2000-6000): Explore. Cyclic LR and Alpha to find local minima.
    3. Phase 3 (6000-8000): Settle. Decay LR, high Alpha, high Beta (Adam).
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Phase 1: Shake ──────────────────────────────────────────
    lr_phase1 = lr0 * 6.0
    alpha_phase1 = alpha0 * 50.0
    beta1_phase1 = 0.1
    beta2_phase1 = 0.2
    
    # ── Phase 2: Explore ────────────────────────────────────────
    # 3 cycles of 1333 steps
    t_phase2 = (step - 2000.0) / 4000.0
    t_phase2 = jnp.clip(t_phase2, 0.0, 1.0)
    cycle_idx = (step - 2000.0) // 1333
    t_cycle = ((step - 2000.0) % 1333) / 1332.0
    
    lr_peak2 = lr0 * 4.0 * (1.0 - 0.5 * t_phase2)
    lr_phase2 = lr0 * 0.1 + 0.5 * (lr_peak2 - lr0 * 0.1) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    alpha_phase2 = alpha0 * 500.0 * (1.0 + 10.0 * t_phase2) * (lr_peak2 / jnp.maximum(lr_phase2, 1e-10))
    
    beta1_phase2 = 0.1 + 0.4 * t_cycle
    beta2_phase2 = 0.2 + 0.75 * t_cycle
    
    # ── Phase 3: Settle ─────────────────────────────────────────
    t_phase3 = (step - 6000.0) / 1999.0
    t_phase3 = jnp.clip(t_phase3, 0.0, 1.0)
    
    lr_phase3 = lr0 * 0.1 * (0.01 ** t_phase3)
    alpha_phase3 = alpha0 * 50000.0 * (100.0 ** t_phase3)
    beta1_phase3 = 0.9
    beta2_phase3 = 0.999
    
    # ── Selection ──────────────────────────────────────────────
    is_p1 = (step < 2000)
    is_p2 = (step >= 2000) & (step < 6000)
    
    lr = jnp.where(is_p1, lr_phase1, jnp.where(is_p2, lr_phase2, lr_phase3))
    alpha = jnp.where(is_p1, alpha_phase1, jnp.where(is_p2, alpha_phase2, alpha_phase3))
    beta1 = jnp.where(is_p1, beta1_phase1, jnp.where(is_p2, beta1_phase2, beta1_phase3))
    beta2 = jnp.where(is_p1, beta2_phase1, jnp.where(is_p2, beta2_phase2, beta2_phase3))
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (step > total_steps - 100)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
