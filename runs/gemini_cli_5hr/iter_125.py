import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 125: Refined Cyclic Squeeze.
    - 10 cycles of 750 steps, then 500 step final squeeze.
    - LR peaks at start of cycle (8.0*lr0 down to 2.5*lr0 peak).
    - Alpha coupled to LR with global ramp (t^3.5).
    - Alpha dip at cycle start (0.95 dip magnitude).
    - Beta parameters ramp up across cycles.
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions ──────────────────────────────────────
    n_cycles = 10
    cycle_len = 750
    is_squeeze = (step >= n_cycles * cycle_len)
    
    c_step = jnp.where(is_squeeze, step - (n_cycles * cycle_len), step % cycle_len)
    c_total = jnp.where(is_squeeze, total_steps - (n_cycles * cycle_len), cycle_len)
    t_cycle = c_step / (c_total - 1)
    
    cycle_idx = jnp.floor(step / cycle_len)
    t_meta = cycle_idx / (n_cycles - 1)
    t_meta = jnp.minimum(t_meta, 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays from 8.0 to 2.5
    lr_peak_global = lr0 * (8.0 * (1.0 - 0.7 * t_global) + 2.5 * 0.7 * t_global)
    lr_min = lr0 * 0.005
    
    # LR starts high at cycle start, decays to low at cycle end
    lr_cyclic = lr_min + 0.5 * (lr_peak_global - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global scale increases aggressively
    alpha_global_scale = 1.0 + 150.0 * (t_global ** 3.5)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to LR (targeting the peak LR of 8.0*lr0)
    alpha_coupled = alpha_global * (lr0 * 8.0 / jnp.maximum(lr_cyclic, 1e-10))
    
    # Dip at cycle start (when LR is high)
    dip_magnitude = 0.95 * (1.0 - 0.4 * t_global)
    dip_width = 0.15
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    # Beta1 starts low, increases
    beta1 = 0.1 + 0.4 * t_global
    
    # Beta2 starts low, increases to help stabilization
    beta2 = 0.2 + 0.75 * t_global
    
    # ── Selection ──────────────────────────────────────────────
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr_cyclic)
    alpha = jnp.where(is_squeeze, alpha0 * 1e8, alpha)
    beta1 = jnp.where(is_squeeze, 0.9, beta1)
    beta2 = jnp.where(is_squeeze, 0.999, beta2)
    
    return lr, alpha, beta1, beta2
