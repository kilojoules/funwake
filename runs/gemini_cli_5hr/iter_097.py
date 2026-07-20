import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 097: 3-cycle Cosine Annealing with higher beta2 for stability.
    - Cycles of 2000, 2000, 4000 steps.
    - Higher beta2 (0.9) for more stable adaptive scaling.
    - iter_087-like LR and Alpha coupling.
    """
    is_c0 = step < 2000
    is_c1 = (step >= 2000) & (step < 4000)
    is_c2 = step >= 4000
    
    c_start = jnp.where(is_c0, 0, jnp.where(is_c1, 2000, 4000))
    c_len = jnp.where(is_c0, 2000, jnp.where(is_c1, 2000, 4000))
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 5.0 * (1.0 - 0.5 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global = alpha0 * (1.0 + 80.0 * (t_global ** 3))
    alpha_local = alpha_global * (lr0 * 5.0 / jnp.maximum(lr, 1e-10))
    
    dip = 0.9 * (1.0 - 0.5 * t_global) * jnp.exp(-(t_cycle**2) / (2 * 0.1**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 5000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.9 # More stable than 0.4
    
    return lr, alpha, beta1, beta2
