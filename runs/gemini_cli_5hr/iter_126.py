import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 126: Hybrid Cycle Strategy.
    - 7 cycles of 700 steps, then 3100 step final cycle.
    - LR peaks at start of each cycle (9.0*lr0 -> 3.5*lr0).
    - Alpha coupled to LR with global ramp (t^3.2).
    - Alpha dip at cycle start (0.95 magnitude, 0.14 width).
    - Beta parameters ramp up WITHIN each cycle (t_cycle).
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions ──────────────────────────────────────
    is_long_final = (step >= 4900)
    c_start = jnp.where(is_long_final, 4900, (step // 700) * 700)
    c_len = jnp.where(is_long_final, 3100, 700)
    
    t_cycle = (step - c_start) / (c_len - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays from 9.0 to 3.5
    lr_peak = lr0 * (9.0 * (1.0 - 0.6 * t_global) + 3.5 * 0.6 * t_global)
    lr_min = lr0 * 0.005
    
    # LR starts high at cycle start, decays to low at cycle end
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global scale increases
    alpha_global_scale = 1.0 + 120.0 * (t_global ** 3.2)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to LR (targeting the initial peak LR)
    alpha_coupled = alpha_global * (lr0 * 9.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at cycle start (when LR is high)
    dip_magnitude = 0.95 * (1.0 - 0.4 * t_global)
    dip_width = 0.14
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Beta (Cyclic) ───────────────────────────────────────────
    # Beta1/Beta2 ramp up within each cycle to allow settlement
    beta1 = 0.1 + 0.4 * t_cycle
    beta2 = 0.2 + 0.75 * t_cycle
    
    # ── Final Squeeze (Last 150 steps) ─────────────────────────
    is_squeeze = (t_global > 0.985)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e9, alpha)
    beta1 = jnp.where(is_squeeze, 0.9, beta1)
    beta2 = jnp.where(is_squeeze, 0.999, beta2)
    
    return lr, alpha, beta1, beta2
