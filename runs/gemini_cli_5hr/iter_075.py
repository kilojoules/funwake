import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 075: Multi-cycle restart (5 cycles).
    Uses multiple cycles of LR decay with decreasing peak height to 
    systematically refine the layout while maintaining global discovery.
    """
    t = step / (total_steps - 1)
    
    # 5 Cycles
    num_cycles = 5
    cycle_idx = jnp.minimum((t * num_cycles).astype(int), num_cycles - 1)
    t_cycle = (t * num_cycles) % 1.0
    
    # Cycle-dependent peaks
    peaks = jnp.array([1.6, 1.0, 0.6, 0.3, 0.1])
    lr_peak = peaks[cycle_idx]
    
    # LR within each cycle
    lr_min = 0.002 * lr0
    lr = lr_min + (lr_peak * lr0 - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * t_cycle))
    
    # Alpha increases steadily across all cycles
    alpha_raw = alpha0 * (1.0 + 14999.0 * t**2.5)
    
    beta1 = 0.0
    beta2 = 0.992
    
    # Surgical squeeze in the very final steps
    is_squeeze = (t > 0.99)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr)
    alpha = jnp.where(is_squeeze, 2000000.0 * alpha0, alpha_raw)
    
    return lr, alpha, beta1, beta2
