import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 131: Meta-Cyclic Strategy.
    - 10 cycles of 700 steps, then 1000 step final settle.
    - First 5 cycles: Deep Alpha dips for global movement.
    - Next 5 cycles: Shallow Alpha dips for local refinement.
    - Cyclic Beta (0.1->0.5, 0.2->0.9) in every cycle.
    - Global LR peak decay.
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions ──────────────────────────────────────
    cycle_len = 700
    is_settle = (step >= 7000)
    
    c_step = jnp.where(is_settle, step - 7000.0, step % cycle_len)
    c_total = jnp.where(is_settle, total_steps - 7000.0, cycle_len)
    t_cycle = c_step / (c_total - 1)
    
    cycle_idx = step // cycle_len
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak LR decays from 8.0 to 2.0
    lr_peak = lr0 * (8.0 * (1.0 - 0.7 * t_global) + 2.0 * 0.7 * t_global)
    lr_min = lr0 * 0.002
    
    # LR starts high at cycle start
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global_scale = 1.0 + 100.0 * (t_global ** 3)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling
    alpha_coupled = alpha_global * (lr0 * 8.0 / jnp.maximum(lr, 1e-10))
    
    # Deep dip in first 5 cycles, shallow in last 5
    is_early = (cycle_idx < 5)
    dip_mag = jnp.where(is_early, 0.98, 0.6)
    dip_width = jnp.where(is_early, 0.2, 0.1)
    
    dip = dip_mag * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1 + 0.4 * t_cycle
    beta2 = 0.2 + 0.7 * t_cycle
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_squeeze = (step > total_steps - 100)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
