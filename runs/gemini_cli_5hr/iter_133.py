import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 133: Final Candidate - Meta-Cyclic Hybrid.
    - 7 cycles of 700 steps, then 3100 step final cycle (Total 8000).
    - Meta-Alpha Dips:
        - Cycles 1-4: Deep (0.98) for global exploration.
        - Cycles 5-7: Medium (0.75) for regional refinement.
        - Final Cycle: Shallow (0.4) for local convergence.
    - Cyclic Beta (0.1->0.4, 0.2->0.9).
    - LR peak decay (7.0*lr0 -> 3.0*lr0).
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions ──────────────────────────────────────
    is_long_final = (step >= 4900)
    c_start = jnp.where(is_long_final, 4900, (step // 700) * 700)
    c_len = jnp.where(is_long_final, 3100, 700)
    
    t_cycle = (step - c_start) / (c_len - 1)
    cycle_idx = jnp.where(is_long_final, 7, step // 700)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays from 7.0 to 3.0
    lr_peak = lr0 * (7.0 * (1.0 - 0.6 * t_global) + 3.0 * 0.6 * t_global)
    lr_min = lr0 * 0.005
    
    # LR starts high at cycle start
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_global_scale = 1.0 + 110.0 * (t_global ** 3.2)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling
    alpha_coupled = alpha_global * (lr0 * 7.0 / jnp.maximum(lr, 1e-10))
    
    # Meta-Dip Strategy
    # idx 0,1,2,3 -> mag 0.98, width 0.18
    # idx 4,5,6   -> mag 0.75, width 0.14
    # idx 7       -> mag 0.40, width 0.10
    dip_mag = jnp.where(cycle_idx < 4, 0.98, jnp.where(cycle_idx < 7, 0.75, 0.40))
    dip_width = jnp.where(cycle_idx < 4, 0.18, jnp.where(cycle_idx < 7, 0.14, 0.10))
    
    dip = dip_mag * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1 + 0.3 * t_cycle
    beta2 = 0.2 + 0.7 * t_cycle
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_squeeze = (t_global > 0.99)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
