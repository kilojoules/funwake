import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 094: Adam Restarts with iter_087-like schedule.
    - 4 cycles of cosine decay.
    - Adam beta1/beta2 set to 0.0 at the first step of each cycle to 'restart' moments.
    - Otherwise use beta1=0.2, beta2=0.4.
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 5.0 * (1.0 - 0.5 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global = alpha0 * (1.0 + 80.0 * (t_global ** 3))
    alpha_local = alpha_global * (lr0 * 5.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at start of cycle
    dip = 0.9 * (1.0 - 0.5 * t_global) * jnp.exp(-(t_cycle**2) / (2 * 0.1**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 5000000.0, alpha)

    # ── Beta (Adam Restarts) ────────────────────────────────────
    # Set to zero at exactly the start of a cycle
    is_cycle_start = (step % cycle_len == 0)
    
    # Use very low values for one step to restart
    beta1 = jnp.where(is_cycle_start, 0.0, 0.2)
    beta2 = jnp.where(is_cycle_start, 0.0, 0.4)
    
    return lr, alpha, beta1, beta2
