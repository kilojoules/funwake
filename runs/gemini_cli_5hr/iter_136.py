import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 136: 12-cycle "Shaking" strategy.
    - More frequent cycles (12 instead of 8) to escape local minima.
    - High initial peak LR (10.0 * lr0) decaying to 2.0 * lr0.
    - Global Alpha ramp: quadratic (t^2) for more pressure in mid-stages.
    - Sharp Alpha Dips (mag 0.98) to allow reorganization.
    - Beta: (0.1, 0.3) at cycle start, ramping to (0.4, 0.9) at cycle end.
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions (12 cycles) ──────────────────────────
    n_cycles = 12
    steps_per_cycle = total_steps // n_cycles
    
    # Handle the fact that total_steps might not be divisible by n_cycles
    # Use a long final cycle if necessary
    is_last_cycle = (step >= (n_cycles - 1) * steps_per_cycle)
    c_start = jnp.where(is_last_cycle, (n_cycles - 1) * steps_per_cycle, (step // steps_per_cycle) * steps_per_cycle)
    c_len = jnp.where(is_last_cycle, total_steps - (n_cycles - 1) * steps_per_cycle, steps_per_cycle)
    
    t_cycle = (step - c_start) / (c_len - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak decays from 10.0 to 2.0
    lr_peak = lr0 * (10.0 * (1.0 - t_global) + 2.0 * t_global)
    lr_min = lr0 * 0.001
    
    # Cosine annealing in each cycle
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Quadratic global scale
    alpha_global_scale = 1.0 + 150.0 * (t_global ** 2.0)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR
    # We use the initial peak LR (10.0) as the coupling constant
    alpha_coupled = alpha_global * (lr0 * 10.0 / jnp.maximum(lr, 1e-10))
    
    # Sharp Dip at start of cycle
    dip_mag = 0.98 * (1.0 - 0.4 * t_global)
    dip_width = 0.15
    dip = dip_mag * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    # Start each cycle with low momentum, end with higher stability
    beta1 = 0.1 + 0.3 * t_cycle
    beta2 = 0.3 + 0.6 * t_cycle
    
    # ── Final Squeeze (last 1%) ─────────────────────────────────
    is_squeeze = (t_global > 0.99)
    lr = jnp.where(is_squeeze, lr0 * 0.000001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e9, alpha)
    
    return lr, alpha, beta1, beta2
