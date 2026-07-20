import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 092: 3-cycle Cosine Annealing with increasing cycle lengths.
    - Cycles: 1000, 2000, 5000 steps.
    - Focus on deep convergence in the final long cycle.
    - Adaptive alpha coupling and global ramp.
    """
    # ── Cycle Definitions ──────────────────────────────────────
    # Boundaries: 1000, 3000, 8000
    is_c0 = step < 1000
    is_c1 = (step >= 1000) & (step < 3000)
    is_c2 = step >= 3000
    
    c_idx = jnp.where(is_c0, 0, jnp.where(is_c1, 1, 2))
    c_start = jnp.where(is_c0, 0, jnp.where(is_c1, 1000, 3000))
    c_len = jnp.where(is_c0, 1000, jnp.where(is_c1, 2000, 5000))
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Decay peak LR per cycle
    lr_peak = lr0 * 6.0 * (0.7 ** c_idx)
    lr_min = lr0 * 0.002
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Strong global ramp, especially in the last cycle
    alpha_global = alpha0 * (1.0 + 100.0 * (t_global ** 3))
    
    # Local coupling
    alpha_local = alpha_global * (lr0 * 6.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at start of each cycle to escape local optima
    dip_mag = 0.9 * (1.0 - 0.5 * t_global)
    dip = dip_mag * jnp.exp(-(t_cycle**2) / (2 * 0.1**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    # Very long squeeze at the end of the last cycle
    is_squeeze = (t_global > 0.96)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.4
    
    return lr, alpha, beta1, beta2
