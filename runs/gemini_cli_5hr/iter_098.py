import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 098: Refined iter_087.
    - 4 cycles of cosine decay.
    - Peak LR 6x decaying to 3x.
    - Global alpha ramp 75x.
    - Beta1=0.15, Beta2=0.3.
    - Wider alpha dip (0.15) for more exploration.
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 6.0 * (1.0 - 0.5 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global = alpha0 * (1.0 + 75.0 * (t_global ** 3))
    alpha_local = alpha_global * (lr0 * 6.0 / jnp.maximum(lr, 1e-10))
    
    # Wider dip
    dip_mag = 0.95 * (1.0 - 0.5 * t_global)
    dip_width = 0.15
    dip = dip_mag * jnp.exp(-(t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.15
    beta2 = 0.3
    
    return lr, alpha, beta1, beta2
