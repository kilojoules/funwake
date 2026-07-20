import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 101: 4-cycle constant peak LR.
    - Peaks: 5x, 5x, 5x, 2.5x.
    - Power-4 alpha ramp.
    - Beta1=0.15, Beta2=0.35.
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Keep high exploration longer
    lr_peak = jnp.where(cycle_idx < 3, lr0 * 5.0, lr0 * 2.5)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Steeper ramp: stay lower longer, then end higher.
    alpha_global = alpha0 * (1.0 + 120.0 * (t_global ** 4))
    
    # Coupling
    alpha_local = alpha_global * (lr0 * 5.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at start of cycle
    dip = 0.95 * (1.0 - 0.5 * t_global) * jnp.exp(-(t_cycle**2) / (2 * 0.1**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 5000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.15
    beta2 = 0.35
    
    return lr, alpha, beta1, beta2
