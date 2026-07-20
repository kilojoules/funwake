import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 100: Refined iter_087 with non-uniform cycles.
    - 3 cycles of 1500 steps, then 1 final cycle of 3500 steps.
    - Adam beta1=0.1, beta2=0.2 (TopFarm style).
    - Refined alpha dip and global ramp.
    """
    is_c0 = step < 1500
    is_c1 = (step >= 1500) & (step < 3000)
    is_c2 = (step >= 3000) & (step < 4500)
    is_c3 = step >= 4500
    
    c_start = jnp.where(is_c0, 0, jnp.where(is_c1, 1500, jnp.where(is_c2, 3000, 4500)))
    c_len = jnp.where(is_c3, 3500, 1500)
    
    t_cycle = (step - c_start) / (c_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 5.0 * (1.0 - 0.4 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global = alpha0 * (1.0 + 100.0 * (t_global ** 3))
    alpha_local = alpha_global * (lr0 * 5.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at start of cycle
    dip = 0.95 * (1.0 - 0.6 * t_global) * jnp.exp(-(t_cycle**2) / (2 * 0.1**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.2
    
    return lr, alpha, beta1, beta2
