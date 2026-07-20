import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 124: Cyclic Annealing with Adaptive Beta2.
    - 12 cycles of 600 steps, plus 800 final squeeze.
    - Peak LR decays over cycles (10.0 -> 2.0).
    - Beta2 increases over cycles (0.1 -> 0.99).
    - Alpha coupled to LR with global ramp (t^3).
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions ──────────────────────────────────────
    n_cycles = 12
    cycle_len = 600
    is_squeeze = (step >= n_cycles * cycle_len)
    
    c_step = jnp.where(is_squeeze, step - (n_cycles * cycle_len), step % cycle_len)
    c_total = jnp.where(is_squeeze, total_steps - (n_cycles * cycle_len), cycle_len)
    t_cycle = c_step / (c_total - 1)
    
    cycle_idx = jnp.floor(step / cycle_len)
    t_meta = cycle_idx / (n_cycles - 1)
    t_meta = jnp.minimum(t_meta, 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays from 10.0 to 2.0
    lr_peak = lr0 * (10.0 * (1.0 - t_meta) + 2.0 * t_meta)
    lr_min = lr0 * 0.001
    
    # Cyclic LR: start low, peak mid, end low
    lr_cyclic = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 - jnp.cos(2.0 * jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global scale increases
    alpha_global_scale = 1.0 + 200.0 * (t_global ** 3)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to LR
    alpha_coupled = alpha_global * (lr0 * 10.0 / jnp.maximum(lr_cyclic, 1e-10))
    
    # Dip at cycle peaks (when LR is high)
    # Wait, usually we want alpha high when LR is low.
    # If LR is high (t_cycle=0.5), cos(2pi*0.5) = cos(pi) = -1. So (1 - cos) = 2.
    # So alpha_coupled is lowest at t_cycle=0.5. This is good.
    alpha = alpha_coupled
    
    # ── Beta ────────────────────────────────────────────────────
    # Beta1 starts low, increases
    beta1 = 0.1 + 0.4 * t_meta
    
    # Beta2 starts VERY low (fast adaptation), increases to Adam-standard
    beta2 = 0.1 + 0.89 * t_meta
    
    # ── Selection ──────────────────────────────────────────────
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr_cyclic)
    alpha = jnp.where(is_squeeze, alpha0 * 1e8, alpha)
    beta1 = jnp.where(is_squeeze, 0.9, beta1)
    beta2 = jnp.where(is_squeeze, 0.999, beta2)
    
    return lr, alpha, beta1, beta2
