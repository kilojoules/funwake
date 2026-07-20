import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 188: 3-cycle Cosine Annealing with Sigmoid Alpha.
    - 3 cycles of LR cosine annealing.
    - LR peak starts high for exploration.
    - Alpha ramp: Sigmoid-based (stays low for 75%, then surges).
    - Beta2: Constant 0.2 (TopFarm style).
    """
    n_cycles = 3
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1.0)
    t_global = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Higher initial peaks for early exploration
    lr_peak = lr0 * 12.0 * (1.0 - 0.75 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.2
    
    # ── Alpha (Penalty Weight) ──────────────────────────────────
    # Global alpha scale is lower in cycles 0 & 1, then surges in 2.
    # We use a sigmoid-like ramp for global alpha.
    alpha_global_scale = 1.0 + 500.0 * (jax.nn.sigmoid(15.0 * (t_global - 0.75)))
    alpha_global = alpha0 * alpha_global_scale
    
    # Local dip in alpha at the start of each cycle
    dip_magnitude = 0.95 * (1.0 - 0.6 * t_global)
    dip_width = 0.12
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    
    # Coupled LR-penalty
    alpha_local = alpha_global * (lr0 * 8.0 / jnp.maximum(lr, 1e-10))
    alpha = alpha_local * (1.0 - dip)
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
