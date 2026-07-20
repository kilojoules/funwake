import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 123: Two-Phase 'Shake and Settle' Strategy.
    1. Phase 1 (0-4000): Shaking. High LR, low Beta1, low Alpha.
       Allows turbines to move past each other and explore the boundary.
    2. Phase 2 (4000-8000): Settling. Exponential LR decay, increasing Alpha, 
       increasing Beta1 (Adam-like).
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Phase 1: Shaking ────────────────────────────────────────
    # High constant LR, low alpha to allow movement
    lr_phase1 = lr0 * 8.0
    alpha_phase1 = alpha0 * 10.0
    beta1_phase1 = 0.05
    beta2_phase1 = 0.5
    
    # ── Phase 2: Settling ───────────────────────────────────────
    t_phase2 = (step - 4000.0) / 3999.0
    t_phase2 = jnp.maximum(t_phase2, 0.0)
    
    # Exponential decay from 8.0*lr0 down to 0.01*lr0
    lr_phase2 = lr0 * 8.0 * (0.001 ** t_phase2)
    
    # Alpha increases from 10.0*alpha0 to 100000.0*alpha0
    alpha_phase2 = alpha0 * 10.0 * (10000.0 ** t_phase2)
    
    # Beta increases towards standard Adam
    beta1_phase2 = 0.05 + 0.85 * t_phase2
    beta2_phase2 = 0.5 + 0.499 * t_phase2
    
    # ── Selection ──────────────────────────────────────────────
    is_phase1 = (step < 4000)
    
    lr = jnp.where(is_phase1, lr_phase1, lr_phase2)
    alpha = jnp.where(is_phase1, alpha_phase1, alpha_phase2)
    beta1 = jnp.where(is_phase1, beta1_phase1, beta1_phase2)
    beta2 = jnp.where(is_phase1, beta2_phase1, beta2_phase2)
    
    # ── Final Squeeze (Last 200 steps) ─────────────────────────
    is_squeeze = (step > total_steps - 200)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e8, alpha)
    
    return lr, alpha, beta1, beta2
