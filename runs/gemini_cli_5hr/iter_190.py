import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 190: 3-cycle Cosine Annealing with No-Dip Final Cycle.
    - 3 cycles of LR cosine annealing.
    - Alpha ramp: Sigmoid surge after 75% of run.
    - Alpha Dip: Active only in first two cycles to allow early exploration.
    - Beta2: Constant 0.2.
    """
    n_cycles = 3
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1.0)
    t_global = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 10.0 * (1.0 - 0.7 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.2
    
    # ── Alpha (Penalty Weight) ──────────────────────────────────
    # Sigmoid surge after 75% of global run
    alpha_global_scale = 1.0 + 800.0 * (jax.nn.sigmoid(15.0 * (t_global - 0.75)))
    alpha_global = alpha0 * alpha_global_scale
    
    # Local dip in alpha at the start of cycles 0 and 1.
    # No dip in cycle 2 (last cycle).
    is_last_cycle = (cycle_idx == n_cycles - 1)
    
    dip_magnitude = 0.95 * (1.0 - 0.5 * t_global)
    dip_width = 0.15
    dip_base = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    
    # Disable dip in last cycle
    dip = jnp.where(is_last_cycle, 0.0, dip_base)
    
    # Coupled LR-penalty
    alpha_local = alpha_global * (lr0 * 10.0 / jnp.maximum(lr, 1e-10))
    alpha = alpha_local * (1.0 - dip)
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e11, alpha)
    
    return lr, alpha, beta1, beta2
