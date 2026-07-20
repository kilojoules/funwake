import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 078: 8-cycle restart with power-3 alpha.
    Frequent restarts allow the layout to repeatedly shake and settle.
    Peak heights decay exponentially across cycles.
    """
    t = step / (total_steps - 1)
    
    # 8 Cycles
    num_cycles = 8
    cycle_idx = jnp.minimum((t * num_cycles).astype(int), num_cycles - 1)
    t_cycle = (t * num_cycles) % 1.0
    
    # Exponentially decaying peaks: starts at 1.6x, ends much lower
    lr_peak = 1.6 * (0.75)**cycle_idx * lr0
    
    # LR within each cycle (cosine decay)
    lr_min = 0.002 * lr0
    lr_in_cycle = lr_min + (lr_peak - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * t_cycle))
    
    # Cubic alpha ramp for late-stage constraint enforcement
    alpha_raw = alpha0 * (1.0 + 19999.0 * t**3)
    
    # Responsive momentum
    beta1 = 0.0
    beta2 = 0.992
    
    # Final surgical squeeze (99.5-100%)
    is_squeeze = (t > 0.995)
    lr = jnp.where(is_squeeze, 0.00005 * lr0, lr_in_cycle)
    alpha = jnp.where(is_squeeze, 2000000.0 * alpha0, alpha_raw)
    
    return lr, alpha, beta1, beta2
