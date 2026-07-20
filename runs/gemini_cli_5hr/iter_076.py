import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 076: 4-cycle restart with aggressive initial peak.
    Focuses on large-scale movement in the first cycle (1.8x lr0) 
    and progressive settling in subsequent cycles.
    """
    t = step / (total_steps - 1)
    
    # 4 Cycles
    num_cycles = 4
    cycle_idx = jnp.minimum((t * num_cycles).astype(int), num_cycles - 1)
    t_cycle = (t * num_cycles) % 1.0
    
    # Progressively smaller peaks for refinement
    peaks = jnp.array([1.8, 1.0, 0.5, 0.2])
    lr_peak = peaks[cycle_idx]
    
    # LR within each cycle
    lr_min = 0.002 * lr0
    lr = lr_min + (lr_peak * lr0 - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * t_cycle))
    
    # Steady alpha ramp to 20000x
    alpha_raw = alpha0 * (1.0 + 19999.0 * t**2)
    
    beta1 = 0.0
    beta2 = 0.992
    
    # Final surgical squeeze
    is_squeeze = (t > 0.99)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr)
    alpha = jnp.where(is_squeeze, 1000000.0 * alpha0, alpha_raw)
    
    return lr, alpha, beta1, beta2
