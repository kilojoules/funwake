import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 102: 6 cycles with iter_087 LR/Alpha.
    - 6 equal cycles of cosine decay.
    - iter_087 LR and Alpha coupling.
    - Beta1/Beta2 ramp from 0.1/0.2 to 0.5/0.8 within each cycle.
    """
    n_cycles = 6
    cycle_len = total_steps // n_cycles
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 5.0 * (1.0 - 0.6 * t_global)
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

    # ── Beta (Cyclic) ───────────────────────────────────────────
    beta1 = 0.1 + 0.4 * t_cycle
    beta2 = 0.2 + 0.6 * t_cycle
    
    return lr, alpha, beta1, beta2
